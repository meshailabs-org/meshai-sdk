"""
OpenAI adapter for MeshAI SDK
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    raise ImportError(
        "OpenAI is not installed. Install with: pip install meshai-sdk[openai]"
    )

import structlog

from ..core.agent import MeshAgent
from ..core.config import MeshConfig
from ..core.context import MeshContext
from ..core.schemas import TaskData
from ..exceptions.base import TaskExecutionError, ConfigurationError

logger = structlog.get_logger(__name__)


class OpenAIMeshAgent(MeshAgent):
    """MeshAI adapter for OpenAI models"""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        if not agent_id:
            agent_id = f"openai-{model.replace('-', '_').replace('.', '_')}"
        if not name:
            name = f"OpenAI {model}"
        if not capabilities:
            capabilities = ["text-generation", "reasoning", "coding", "analysis"]
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="openai",
            config=config,
            **kwargs
        )
        
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = api_key or self._get_api_key()
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.sync_client = OpenAI(api_key=api_key, base_url=base_url)
        
        logger.info(f"OpenAI agent {agent_id} initialized with model {model}")
    
    def _get_api_key(self) -> Optional[str]:
        import os
        return (
            getattr(self.config, 'openai_api_key', None) or
            os.getenv('OPENAI_API_KEY')
        )
    
    def _default_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant integrated into the MeshAI platform. "
            "You can collaborate with other AI agents to help users complete tasks. "
            "Be concise, accurate, and helpful in your responses."
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        try:
            # Extract input
            if isinstance(task_data.input, dict):
                user_message = task_data.input.get("input", str(task_data.input))
            else:
                user_message = str(task_data.input)
            
            # Prepare messages
            messages = await self._prepare_messages(user_message, context)
            
            # Prepare function calls/tools if needed
            functions = self._prepare_functions()
            
            # Make API call
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            
            if functions:
                kwargs["tools"] = functions
                kwargs["tool_choice"] = "auto"
            
            # Add custom parameters
            for key, value in task_data.parameters.items():
                if key in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
                    kwargs[key] = value
            
            # Track request for analytics
            start_time = datetime.utcnow()
            
            response = await self.client.chat.completions.create(**kwargs)
            
            # Track performance metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(response, response_time)
            
            # Process response
            result = await self._process_response(response, context)
            result["response_time_seconds"] = response_time
            
            # Update context
            await self._update_context(context, user_message, result)
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI task execution failed: {e}")
            raise TaskExecutionError(f"OpenAI execution error: {e}")
    
    async def _prepare_messages(self, user_message: str, context: MeshContext) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        conversation_history = await context.get("conversation_history", [])
        for msg in conversation_history[-10:]:  # Last 10 messages
            if isinstance(msg, dict) and 'content' in msg:
                role = "user" if msg.get('type') == 'human' else "assistant"
                messages.append({"role": role, "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _prepare_functions(self) -> Optional[List[Dict[str, Any]]]:
        """Prepare function definitions for OpenAI API"""
        # Use new OpenAI tools format instead of deprecated functions
        tools = [{
            "type": "function",
            "function": {
                "name": "invoke_meshai_agent",
                "description": "Invoke another MeshAI agent with specific capabilities to help complete complex tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Required capabilities for the task (e.g., 'coding', 'analysis', 'math')"
                        },
                        "task": {
                            "type": "string",
                            "description": "Clear task description or query for the agent"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context or parameters for the task",
                            "additionalProperties": True
                        }
                    },
                    "required": ["capabilities", "task"]
                }
            }
        }]
        
        return tools
    
    async def _process_response(self, response: Any, context: MeshContext) -> Dict[str, Any]:
        """Process OpenAI API response"""
        choice = response.choices[0]
        message = choice.message
        
        result = {
            "type": "openai_response",
            "model": self.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        # Handle tool calls (new OpenAI format)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_results = []
            for tool_call in message.tool_calls:
                tool_result = await self._handle_tool_call(tool_call, context)
                tool_results.append(tool_result)
            result["result"] = "\n".join(tool_results) if tool_results else message.content or ""
            result["tools_used"] = True
            result["tool_count"] = len(message.tool_calls)
        else:
            result["result"] = message.content or ""
            result["tools_used"] = False
        
        return result
    
    async def _handle_tool_call(self, tool_call: Any, context: MeshContext) -> str:
        """Handle tool calls from OpenAI (new format)"""
        try:
            if tool_call.function.name == "invoke_meshai_agent":
                import json
                arguments = json.loads(tool_call.function.arguments)
                
                capabilities = arguments.get("capabilities", [])
                task = arguments.get("task", "")
                task_context = arguments.get("context", {})
                
                logger.info(f"OpenAI agent invoking MeshAI agent with capabilities: {capabilities}")
                
                # Invoke agent through MeshAI
                result = await self.invoke_agent(
                    capabilities=capabilities,
                    task={"input": task, "parameters": task_context},
                    routing_strategy="capability_match"
                )
                
                if result.status == "completed":
                    response = str(result.result)
                    logger.info(f"MeshAI agent invocation successful: {response[:100]}...")
                    return response
                else:
                    error_msg = f"Task failed: {result.error or 'Unknown error'}"
                    logger.warning(f"MeshAI agent invocation failed: {error_msg}")
                    return error_msg
            else:
                return f"Unknown tool: {tool_call.function.name}"
                
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments: {e}"
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return f"Tool execution error: {e}"
    
    def _update_metrics(self, response: Any, response_time: float) -> None:
        """Update internal metrics tracking"""
        if not hasattr(self, '_request_count'):
            self._request_count = 0
            self._token_count = 0
            self._total_response_time = 0.0
        
        self._request_count += 1
        if hasattr(response, 'usage'):
            self._token_count += response.usage.total_tokens
        self._total_response_time += response_time
        self._avg_response_time = self._total_response_time / self._request_count
    
    async def _update_context(self, context: MeshContext, user_message: str, result: Dict[str, Any]):
        """Update context with conversation"""
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": user_message, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": result.get("result", ""), "timestamp": datetime.utcnow().isoformat(), "source": "openai"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)
    
    async def stream_response(self, task_input: Union[str, Dict[str, Any]], context: MeshContext):
        """Stream response from OpenAI"""
        try:
            if isinstance(task_input, dict):
                user_message = task_input.get("input", str(task_input))
            else:
                user_message = str(task_input)
            
            messages = await self._prepare_messages(user_message, context)
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {e}"
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt"""
        self.system_prompt = prompt
        logger.info(f"Updated system prompt for OpenAI agent {self.agent_id}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": self.model,
            "provider": "openai",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt_length": len(self.system_prompt),
            "supports_tools": True,
            "supports_streaming": True,
            "model_family": self._get_model_family()
        }
    
    def _get_model_family(self) -> str:
        """Get the model family for better capability understanding"""
        if "gpt-4" in self.model.lower():
            return "gpt-4"
        elif "gpt-3.5" in self.model.lower():
            return "gpt-3.5"
        elif "davinci" in self.model.lower():
            return "davinci"
        else:
            return "unknown"
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this agent"""
        # This would integrate with OpenAI usage tracking
        return {
            "total_requests": getattr(self, '_request_count', 0),
            "total_tokens": getattr(self, '_token_count', 0),
            "avg_response_time": getattr(self, '_avg_response_time', 0.0)
        }