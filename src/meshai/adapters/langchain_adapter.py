"""
LangChain adapter for MeshAI SDK
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Type, Callable
from datetime import datetime
import inspect

try:
    from langchain.agents import AgentExecutor, BaseMultiActionAgent, BaseSingleActionAgent
    from langchain.agents.agent import Agent
    from langchain.tools import BaseTool, Tool
    from langchain.schema import AgentAction, AgentFinish
    from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain.memory import ConversationBufferMemory
    from langchain.schema.runnable import Runnable
except ImportError:
    raise ImportError(
        "LangChain is not installed. Install with: pip install meshai-sdk[langchain]"
    )

import structlog

from ..core.agent import MeshAgent
from ..core.config import MeshConfig
from ..core.context import MeshContext
from ..core.schemas import TaskData, TaskResult
from ..exceptions.base import TaskExecutionError, ValidationError

logger = structlog.get_logger(__name__)


class MeshAITool(BaseTool):
    """
    LangChain tool that can invoke other MeshAI agents.
    
    This allows LangChain agents to seamlessly call other agents
    in the MeshAI network through the standard LangChain tool interface.
    """
    
    name: str = "meshai_invoke"
    description: str = "Invoke another MeshAI agent with specific capabilities"
    
    def __init__(self, mesh_agent: "LangChainMeshAgent"):
        super().__init__()
        self.mesh_agent = mesh_agent
    
    def _run(self, query: str, capabilities: str = "", **kwargs) -> str:
        """Synchronous tool execution (fallback)"""
        # Run async version in sync context
        return asyncio.run(self._arun(query, capabilities, **kwargs))
    
    async def _arun(
        self,
        query: str,
        capabilities: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Asynchronous tool execution"""
        try:
            # Parse capabilities
            caps = [c.strip() for c in capabilities.split(",") if c.strip()]
            if not caps:
                caps = ["general"]
            
            # Invoke agent through MeshAI
            result = await self.mesh_agent.invoke_agent(
                capabilities=caps,
                task={"input": query, "parameters": kwargs},
                routing_strategy="capability_match"
            )
            
            if result.status == "completed" and result.result:
                if isinstance(result.result, dict):
                    return result.result.get("result", str(result.result))
                return str(result.result)
            else:
                return f"Task failed: {result.error or 'Unknown error'}"
                
        except Exception as e:
            logger.error(f"MeshAI tool execution failed: {e}")
            return f"Error invoking MeshAI agent: {e}"


class LangChainToolAdapter(BaseTool):
    """
    Adapter to make any function available as a LangChain tool
    and register it with MeshAI for cross-agent usage.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        mesh_agent: Optional["LangChainMeshAgent"] = None
    ):
        super().__init__(name=name, description=description)
        self.func = func
        self.mesh_agent = mesh_agent
        
        # Inspect function signature
        self.signature = inspect.signature(func)
        self.is_async = inspect.iscoroutinefunction(func)
    
    def _run(self, *args, **kwargs) -> str:
        """Synchronous tool execution"""
        try:
            if self.is_async:
                # Run async function in event loop
                return asyncio.run(self.func(*args, **kwargs))
            else:
                result = self.func(*args, **kwargs)
                return str(result) if result is not None else "Success"
        except Exception as e:
            return f"Tool execution failed: {e}"
    
    async def _arun(
        self,
        *args,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Asynchronous tool execution"""
        try:
            if self.is_async:
                result = await self.func(*args, **kwargs)
            else:
                result = self.func(*args, **kwargs)
            
            return str(result) if result is not None else "Success"
        except Exception as e:
            return f"Tool execution failed: {e}"


class LangChainMeshAgent(MeshAgent):
    """
    MeshAI adapter for LangChain agents.
    
    Wraps LangChain agents (AgentExecutor, Agent, or Runnable) to make them
    compatible with MeshAI while preserving their LangChain functionality.
    
    Features:
    - Automatic tool registration with MeshAI
    - Context sharing between agents
    - LangChain memory integration
    - Support for both sync and async LangChain agents
    """
    
    def __init__(
        self,
        langchain_agent: Union[AgentExecutor, Agent, Runnable],
        agent_id: str,
        name: str,
        capabilities: List[str],
        llm: Optional[BaseLanguageModel] = None,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[ConversationBufferMemory] = None,
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        """
        Initialize LangChain MeshAI adapter.
        
        Args:
            langchain_agent: LangChain agent/executor to wrap
            agent_id: Unique agent identifier
            name: Human-readable agent name
            capabilities: List of agent capabilities
            llm: Language model (if not provided by agent)
            tools: Additional tools to add
            memory: Memory component for conversations
            config: MeshAI configuration
            **kwargs: Additional metadata
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="langchain",
            config=config,
            **kwargs
        )
        
        self.langchain_agent = langchain_agent
        self.llm = llm
        self.memory = memory or ConversationBufferMemory()
        
        # Setup tools
        self.tools = list(tools) if tools else []
        self._setup_tools()
        
        # Store original agent type for handling
        self.agent_type = self._detect_agent_type()
        
        logger.info(f"LangChain agent {agent_id} initialized with {len(self.tools)} tools")
    
    def _detect_agent_type(self) -> str:
        """Detect the type of LangChain agent"""
        if isinstance(self.langchain_agent, AgentExecutor):
            return "executor"
        elif isinstance(self.langchain_agent, (BaseMultiActionAgent, BaseSingleActionAgent)):
            return "agent"
        elif isinstance(self.langchain_agent, Runnable):
            return "runnable"
        else:
            return "unknown"
    
    def _setup_tools(self) -> None:
        """Setup tools for the agent"""
        # Add MeshAI tool for cross-agent communication
        meshai_tool = MeshAITool(self)
        self.tools.append(meshai_tool)
        
        # If agent is an AgentExecutor, add tools to it
        if isinstance(self.langchain_agent, AgentExecutor):
            # Get existing tools
            existing_tools = getattr(self.langchain_agent, 'tools', [])
            all_tools = existing_tools + self.tools
            
            # Update agent with new tools
            self.langchain_agent.tools = all_tools
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """
        Handle task execution with LangChain agent.
        
        Args:
            task_data: Task to execute
            context: Shared context
            
        Returns:
            Task execution result
        """
        try:
            # Extract input and parameters
            task_input = task_data.input
            parameters = task_data.parameters
            
            # Prepare input for LangChain
            if isinstance(task_input, dict):
                if "input" in task_input:
                    langchain_input = task_input["input"]
                else:
                    langchain_input = str(task_input)
            else:
                langchain_input = str(task_input)
            
            # Update memory with context
            await self._update_memory_from_context(context)
            
            # Execute based on agent type
            if self.agent_type == "executor":
                result = await self._execute_agent_executor(langchain_input, parameters)
            elif self.agent_type == "runnable":
                result = await self._execute_runnable(langchain_input, parameters)
            elif self.agent_type == "agent":
                result = await self._execute_agent(langchain_input, parameters)
            else:
                raise TaskExecutionError(f"Unsupported agent type: {self.agent_type}")
            
            # Update context with memory
            await self._update_context_from_memory(context)
            
            # Format result
            if isinstance(result, dict):
                return result
            else:
                return {"result": result, "type": "langchain_response"}
                
        except Exception as e:
            logger.error(f"LangChain task execution failed: {e}")
            raise TaskExecutionError(f"LangChain execution error: {e}")
    
    async def _execute_agent_executor(self, input_text: str, parameters: Dict) -> Any:
        """Execute AgentExecutor"""
        try:
            # Check if agent supports async
            if hasattr(self.langchain_agent, 'ainvoke'):
                result = await self.langchain_agent.ainvoke(
                    {"input": input_text, **parameters}
                )
            elif hasattr(self.langchain_agent, 'arun'):
                result = await self.langchain_agent.arun(
                    input_text, **parameters
                )
            else:
                # Fallback to sync execution
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.langchain_agent.invoke({"input": input_text, **parameters})
                )
            
            return result
            
        except Exception as e:
            raise TaskExecutionError(f"AgentExecutor execution failed: {e}")
    
    async def _execute_runnable(self, input_text: str, parameters: Dict) -> Any:
        """Execute Runnable"""
        try:
            input_data = {"input": input_text, **parameters}
            
            # Check if runnable supports async
            if hasattr(self.langchain_agent, 'ainvoke'):
                result = await self.langchain_agent.ainvoke(input_data)
            else:
                # Fallback to sync execution  
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.langchain_agent.invoke(input_data)
                )
            
            return result
            
        except Exception as e:
            raise TaskExecutionError(f"Runnable execution failed: {e}")
    
    async def _execute_agent(self, input_text: str, parameters: Dict) -> Any:
        """Execute Agent directly"""
        try:
            # This is more complex as we need to handle the agent loop manually
            # For now, convert to string representation
            if hasattr(self.langchain_agent, 'plan'):
                actions = await self.langchain_agent.plan(
                    intermediate_steps=[],
                    **{"input": input_text, **parameters}
                )
                
                if isinstance(actions, AgentFinish):
                    return actions.return_values
                elif isinstance(actions, list):
                    # Execute actions (simplified)
                    results = []
                    for action in actions:
                        if isinstance(action, AgentAction):
                            results.append(f"Action: {action.tool}, Input: {action.tool_input}")
                    return {"actions": results}
                else:
                    return {"result": str(actions)}
            else:
                return {"result": f"Direct agent execution not fully supported: {input_text}"}
                
        except Exception as e:
            raise TaskExecutionError(f"Agent execution failed: {e}")
    
    async def _update_memory_from_context(self, context: MeshContext) -> None:
        """Update LangChain memory from MeshAI context"""
        try:
            if not self.memory:
                return
            
            # Get conversation history from context
            conversation = await context.get("conversation_history", [])
            
            # Update memory with conversation
            if conversation and hasattr(self.memory, 'chat_memory'):
                for message in conversation[-10:]:  # Last 10 messages
                    if isinstance(message, dict):
                        if message.get('type') == 'human':
                            self.memory.chat_memory.add_user_message(message['content'])
                        elif message.get('type') == 'ai':
                            self.memory.chat_memory.add_ai_message(message['content'])
            
        except Exception as e:
            logger.warning(f"Failed to update memory from context: {e}")
    
    async def _update_context_from_memory(self, context: MeshContext) -> None:
        """Update MeshAI context from LangChain memory"""
        try:
            if not self.memory or not hasattr(self.memory, 'chat_memory'):
                return
            
            # Extract conversation history
            messages = self.memory.chat_memory.messages
            conversation = []
            
            for msg in messages[-20:]:  # Last 20 messages
                if hasattr(msg, 'type'):
                    conversation.append({
                        'type': msg.type,
                        'content': msg.content,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            # Update context
            await context.set("conversation_history", conversation)
            
        except Exception as e:
            logger.warning(f"Failed to update context from memory: {e}")
    
    def add_tool(self, tool: Union[BaseTool, Callable], name: str = None, description: str = None) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool to add (BaseTool or function)
            name: Tool name (required if tool is a function)
            description: Tool description (required if tool is a function)
        """
        if isinstance(tool, BaseTool):
            self.tools.append(tool)
        elif callable(tool):
            if not name or not description:
                raise ValueError("Name and description required for function tools")
            
            adapter = LangChainToolAdapter(name, description, tool, self)
            self.tools.append(adapter)
        else:
            raise ValueError("Tool must be BaseTool or callable")
        
        # Update agent executor tools if applicable
        if isinstance(self.langchain_agent, AgentExecutor):
            self.langchain_agent.tools = self.tools
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools registered with the agent"""
        return self.tools.copy()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of agent memory"""
        if not self.memory:
            return {}
        
        try:
            if hasattr(self.memory, 'chat_memory'):
                return {
                    "message_count": len(self.memory.chat_memory.messages),
                    "memory_type": type(self.memory).__name__,
                    "buffer": self.memory.buffer if hasattr(self.memory, 'buffer') else None
                }
            else:
                return {"memory_type": type(self.memory).__name__}
        except Exception as e:
            return {"error": f"Failed to get memory summary: {e}"}
    
    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        agent_id: str,
        name: str,
        capabilities: List[str],
        agent_type: str = "zero-shot-react-description",
        **kwargs
    ) -> "LangChainMeshAgent":
        """
        Create LangChain MeshAI agent from LLM and tools.
        
        Args:
            llm: Language model
            tools: List of tools
            agent_id: Agent identifier
            name: Agent name
            capabilities: Agent capabilities
            agent_type: LangChain agent type
            **kwargs: Additional arguments
        """
        try:
            from langchain.agents import initialize_agent, AgentType
            
            # Map agent type string to enum
            agent_type_map = {
                "zero-shot-react-description": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                "conversational-react-description": AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                "chat-zero-shot-react-description": AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                "chat-conversational-react-description": AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            }
            
            agent_type_enum = agent_type_map.get(agent_type, AgentType.ZERO_SHOT_REACT_DESCRIPTION)
            
            # Initialize agent
            agent_executor = initialize_agent(
                tools=tools,
                llm=llm,
                agent=agent_type_enum,
                verbose=kwargs.get("verbose", False),
                **{k: v for k, v in kwargs.items() if k != "verbose"}
            )
            
            return cls(
                langchain_agent=agent_executor,
                agent_id=agent_id,
                name=name,
                capabilities=capabilities,
                llm=llm,
                tools=tools,
                **kwargs
            )
            
        except ImportError as e:
            raise ImportError(f"Failed to import LangChain components: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create agent: {e}")