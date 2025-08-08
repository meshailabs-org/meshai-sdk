"""
Anthropic Claude adapter for MeshAI SDK
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from anthropic import AsyncAnthropic, Anthropic
    from anthropic.types import MessageParam, TextBlock, ToolUseBlock
except ImportError:
    raise ImportError(
        "Anthropic is not installed. Install with: pip install meshai-sdk[anthropic]"
    )

import structlog

from ..core.agent import MeshAgent
from ..core.config import MeshConfig
from ..core.context import MeshContext
from ..core.schemas import TaskData, TaskResult
from ..exceptions.base import TaskExecutionError, ConfigurationError

logger = structlog.get_logger(__name__)


class AnthropicMeshAgent(MeshAgent):
    """
    MeshAI adapter for Anthropic Claude models.
    
    Provides integration with Claude models through the Anthropic API,
    making them available as MeshAI agents with full context management
    and tool usage capabilities.
    
    Features:
    - Support for all Claude models
    - Tool calling capabilities
    - Context-aware conversations
    - Streaming responses (optional)
    - System prompt management
    """
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        """
        Initialize Anthropic Claude MeshAI agent.
        
        Args:
            model: Claude model to use
            agent_id: Unique agent identifier
            name: Human-readable agent name
            capabilities: List of agent capabilities
            api_key: Anthropic API key
            system_prompt: System prompt for the agent
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0-1)
            tools: Tool definitions for Claude
            config: MeshAI configuration
            **kwargs: Additional metadata
        """
        # Set defaults
        if not agent_id:
            agent_id = f"claude-{model.replace('-', '_')}"
        if not name:
            name = f"Claude {model}"
        if not capabilities:
            capabilities = ["text-generation", "reasoning", "analysis", "coding", "math"]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="anthropic",
            config=config,
            **kwargs
        )
        
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        
        # Initialize Anthropic client
        api_key = api_key or self._get_api_key()
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.sync_client = Anthropic(api_key=api_key)
        
        logger.info(f"Anthropic Claude agent {agent_id} initialized with model {model}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get Anthropic API key from config or environment"""
        import os
        return (
            getattr(self.config, 'anthropic_api_key', None) or
            os.getenv('ANTHROPIC_API_KEY')
        )
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt"""
        return (
            "You are a helpful AI assistant integrated into the MeshAI platform. "
            "You can collaborate with other AI agents and use tools to help users "
            "complete tasks. Be concise, accurate, and helpful in your responses."
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """
        Handle task execution with Claude.
        
        Args:
            task_data: Task to execute
            context: Shared context
            
        Returns:
            Task execution result
        """
        try:
            # Extract task information
            task_input = task_data.input
            parameters = task_data.parameters
            
            # Prepare messages
            messages = await self._prepare_messages(task_input, context)
            
            # Prepare tools if available
            tools = self._prepare_tools()
            
            # Make API call
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages,
                tools=tools if tools else None,
                **{k: v for k, v in parameters.items() if k in ['top_p', 'top_k', 'stop_sequences']}
            )
            
            # Process response
            result = await self._process_response(response, context)
            
            # Update context
            await self._update_context(context, messages, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic task execution failed: {e}")
            raise TaskExecutionError(f"Claude execution error: {e}")
    
    async def _prepare_messages(
        self, 
        task_input: Union[str, Dict[str, Any]], 
        context: MeshContext
    ) -> List[MessageParam]:
        """Prepare messages for Claude API"""
        messages = []
        
        # Get conversation history from context
        conversation_history = await context.get("conversation_history", [])
        
        # Add recent conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages
            if isinstance(msg, dict) and 'content' in msg:
                role = "user" if msg.get('type') == 'human' else "assistant"
                messages.append({
                    "role": role,
                    "content": msg['content']
                })
        
        # Add current task input
        if isinstance(task_input, dict):
            content = task_input.get("input", task_input.get("content", str(task_input)))
        else:
            content = str(task_input)
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _prepare_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Prepare tools for Claude API"""
        if not self.tools:
            return None
        
        # Add MeshAI invoke tool
        meshai_tool = {
            "name": "invoke_meshai_agent",
            "description": "Invoke another MeshAI agent with specific capabilities",
            "input_schema": {
                "type": "object",
                "properties": {
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required capabilities for the task"
                    },
                    "task": {
                        "type": "string",
                        "description": "Task description or query"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for the task",
                        "additionalProperties": True
                    }
                },
                "required": ["capabilities", "task"]
            }
        }
        
        tools = [meshai_tool] + self.tools
        return tools
    
    async def _process_response(
        self, 
        response: Any, 
        context: MeshContext
    ) -> Dict[str, Any]:
        """Process Claude API response"""
        result = {
            "type": "anthropic_response",
            "model": self.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }
        
        # Process content blocks
        content_parts = []
        tool_results = []
        
        for content_block in response.content:
            if isinstance(content_block, TextBlock):
                content_parts.append(content_block.text)
            elif isinstance(content_block, ToolUseBlock):
                # Handle tool use
                tool_result = await self._handle_tool_use(content_block, context)
                tool_results.append(tool_result)
                content_parts.append(f"[Used tool: {content_block.name}]")
        
        result["result"] = "\n".join(content_parts)
        
        if tool_results:
            result["tool_results"] = tool_results
        
        return result
    
    async def _handle_tool_use(
        self, 
        tool_block: ToolUseBlock, 
        context: MeshContext
    ) -> Dict[str, Any]:
        """Handle tool use by Claude"""
        try:
            if tool_block.name == "invoke_meshai_agent":
                # Handle MeshAI agent invocation
                input_data = tool_block.input
                capabilities = input_data.get("capabilities", [])
                task = input_data.get("task", "")
                task_context = input_data.get("context", {})
                
                # Invoke agent through MeshAI
                result = await self.invoke_agent(
                    capabilities=capabilities,
                    task={"input": task, "context": task_context},
                    routing_strategy="capability_match"
                )
                
                return {
                    "tool_name": tool_block.name,
                    "tool_id": tool_block.id,
                    "success": result.status == "completed",
                    "result": result.result if result.status == "completed" else result.error
                }
            else:
                # Handle custom tool
                return {
                    "tool_name": tool_block.name,
                    "tool_id": tool_block.id,
                    "success": False,
                    "error": f"Unknown tool: {tool_block.name}"
                }
                
        except Exception as e:
            return {
                "tool_name": tool_block.name,
                "tool_id": tool_block.id,
                "success": False,
                "error": str(e)
            }
    
    async def _update_context(
        self, 
        context: MeshContext, 
        messages: List[MessageParam], 
        result: Dict[str, Any]
    ) -> None:
        """Update MeshAI context with conversation"""
        try:
            # Add assistant response to conversation history
            conversation_entry = {
                "type": "ai",
                "content": result.get("result", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "anthropic",
                "model": self.model,
                "usage": result.get("usage", {})
            }
            
            # Get current conversation history
            conversation_history = await context.get("conversation_history", [])
            conversation_history.append(conversation_entry)
            
            # Keep only last 50 messages
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
            
            # Update context
            await context.set("conversation_history", conversation_history)
            
        except Exception as e:
            logger.warning(f"Failed to update context: {e}")
    
    def add_tool(self, tool_definition: Dict[str, Any]) -> None:
        """
        Add a tool definition for Claude to use.
        
        Args:
            tool_definition: Tool definition following Claude's schema
        """
        self.tools.append(tool_definition)
        logger.info(f"Added tool {tool_definition.get('name', 'unknown')} to Claude agent")
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the agent"""
        self.system_prompt = prompt
        logger.info(f"Updated system prompt for Claude agent {self.agent_id}")
    
    async def stream_response(
        self, 
        task_input: Union[str, Dict[str, Any]], 
        context: MeshContext
    ):
        """
        Stream response from Claude (async generator).
        
        Args:
            task_input: Input for the task
            context: Shared context
            
        Yields:
            Partial response chunks
        """
        try:
            messages = await self._prepare_messages(task_input, context)
            tools = self._prepare_tools()
            
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages,
                tools=tools if tools else None,
            ) as stream:
                async for event in stream:
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        yield event.delta.text
                        
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "provider": "anthropic",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools_count": len(self.tools),
            "system_prompt_length": len(self.system_prompt)
        }