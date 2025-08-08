"""
AutoGen adapter for MeshAI SDK
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import autogen
    from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    raise ImportError(
        "AutoGen is not installed. Install with: pip install meshai-sdk[autogen]"
    )

import structlog

from ..core.agent import MeshAgent
from ..core.config import MeshConfig
from ..core.context import MeshContext
from ..core.schemas import TaskData
from ..exceptions.base import TaskExecutionError

logger = structlog.get_logger(__name__)


class AutoGenMeshAgent(MeshAgent):
    """MeshAI adapter for AutoGen agents and group chats"""
    
    def __init__(
        self,
        autogen_component: Union[ConversableAgent, GroupChat, GroupChatManager],
        agent_id: str,
        name: str,
        capabilities: List[str],
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="autogen",
            config=config,
            **kwargs
        )
        
        self.autogen_component = autogen_component
        self.component_type = self._detect_component_type()
        
        logger.info(f"AutoGen {self.component_type} {agent_id} initialized")
    
    def _detect_component_type(self) -> str:
        """Detect AutoGen component type"""
        if isinstance(self.autogen_component, GroupChatManager):
            return "group_chat_manager"
        elif isinstance(self.autogen_component, GroupChat):
            return "group_chat"
        elif isinstance(self.autogen_component, ConversableAgent):
            return "agent"
        else:
            return "unknown"
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        try:
            # Extract input
            if isinstance(task_data.input, dict):
                message = task_data.input.get("input", str(task_data.input))
            else:
                message = str(task_data.input)
            
            # Get conversation history for context
            conversation_history = await context.get("conversation_history", [])
            
            # Execute based on component type
            if self.component_type == "agent":
                result = await self._execute_agent(message, conversation_history)
            elif self.component_type == "group_chat_manager":
                result = await self._execute_group_chat_manager(message, conversation_history)
            else:
                result = {"result": f"AutoGen component type {self.component_type} not yet fully supported"}
            
            # Update context
            await self._update_context(context, message, result)
            
            return result
            
        except Exception as e:
            logger.error(f"AutoGen task execution failed: {e}")
            raise TaskExecutionError(f"AutoGen execution error: {e}")
    
    async def _execute_agent(self, message: str, conversation_history: List) -> Dict[str, Any]:
        """Execute single AutoGen agent"""
        try:
            # Create a simple user proxy for the conversation
            user_proxy = UserProxyAgent(
                "user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                is_termination_msg=lambda x: True
            )
            
            # Start conversation
            # Note: AutoGen is typically synchronous, so this is a simplified async wrapper
            import asyncio
            
            def run_conversation():
                user_proxy.initiate_chat(self.autogen_component, message=message)
                # Get the last message from the agent
                chat_messages = user_proxy.chat_messages.get(self.autogen_component, [])
                if chat_messages:
                    return chat_messages[-1].get("content", "No response")
                return "No response generated"
            
            response = await asyncio.get_event_loop().run_in_executor(None, run_conversation)
            
            return {
                "result": response,
                "type": "autogen_agent_response",
                "agent_name": getattr(self.autogen_component, 'name', 'unknown')
            }
            
        except Exception as e:
            raise TaskExecutionError(f"AutoGen agent execution failed: {e}")
    
    async def _execute_group_chat_manager(self, message: str, conversation_history: List) -> Dict[str, Any]:
        """Execute AutoGen group chat"""
        try:
            # This is a simplified implementation
            # Real implementation would involve more complex group chat management
            
            import asyncio
            
            def run_group_chat():
                # Simplified group chat execution
                return f"Group chat processed: {message}"
            
            response = await asyncio.get_event_loop().run_in_executor(None, run_group_chat)
            
            return {
                "result": response,
                "type": "autogen_group_chat_response",
                "participants": getattr(self.autogen_component, 'agents', [])
            }
            
        except Exception as e:
            raise TaskExecutionError(f"AutoGen group chat execution failed: {e}")
    
    async def _update_context(self, context: MeshContext, message: str, result: Dict[str, Any]):
        """Update context with conversation"""
        conversation_history = await context.get("conversation_history", [])
        conversation_history.extend([
            {"type": "human", "content": message, "timestamp": datetime.utcnow().isoformat()},
            {"type": "ai", "content": result.get("result", ""), "timestamp": datetime.utcnow().isoformat(), "source": "autogen"}
        ])
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        await context.set("conversation_history", conversation_history)
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about AutoGen component"""
        info = {
            "type": self.component_type,
            "name": getattr(self.autogen_component, 'name', 'unknown')
        }
        
        if hasattr(self.autogen_component, 'system_message'):
            info["system_message"] = self.autogen_component.system_message
        
        if hasattr(self.autogen_component, 'agents'):
            info["agents"] = [
                getattr(agent, 'name', 'unknown') 
                for agent in self.autogen_component.agents
            ]
        
        return info