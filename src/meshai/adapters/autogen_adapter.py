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
    """MeshAI adapter for AutoGen agents and group chats
    
    Supports:
    - Individual ConversableAgent wrapping
    - GroupChatManager integration
    - Multi-agent conversations
    - Context preservation across interactions
    - Dynamic capability detection
    """
    
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
            # Create a user proxy with better configuration
            user_proxy = UserProxyAgent(
                "meshai_user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,  # Allow a few exchanges
                is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE") or len(x.get("content", "")) > 1000,
                code_execution_config=False,  # Disable code execution for safety
                system_message="You are a proxy for the MeshAI system."
            )
            
            # Add context to the message if available
            enhanced_message = self._enhance_message_with_context(message, conversation_history)
            
            # Start conversation in executor
            import asyncio
            
            def run_conversation():
                try:
                    # Clear any existing chat messages to avoid conflicts
                    user_proxy.reset()
                    
                    chat_result = user_proxy.initiate_chat(
                        self.autogen_component, 
                        message=enhanced_message,
                        max_turns=3
                    )
                    
                    # Extract the final response
                    if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                        # Get the last message from the agent (not user proxy)
                        for msg in reversed(chat_result.chat_history):
                            if msg.get('name') == self.autogen_component.name:
                                return msg.get('content', 'No response')
                    
                    # Fallback: get from chat_messages
                    chat_messages = user_proxy.chat_messages.get(self.autogen_component, [])
                    if chat_messages:
                        return chat_messages[-1].get("content", "No response")
                    
                    return "No response generated"
                    
                except Exception as e:
                    logger.error(f"AutoGen conversation error: {e}")
                    return f"Conversation error: {str(e)[:200]}"
            
            response = await asyncio.get_event_loop().run_in_executor(None, run_conversation)
            
            return {
                "result": response,
                "type": "autogen_agent_response",
                "agent_name": getattr(self.autogen_component, 'name', 'unknown'),
                "agent_role": getattr(self.autogen_component, 'system_message', '')[:100],
                "execution_mode": "single_agent"
            }
            
        except Exception as e:
            logger.error(f"AutoGen agent execution failed: {e}")
            raise TaskExecutionError(f"AutoGen agent execution failed: {e}")
    
    async def _execute_group_chat_manager(self, message: str, conversation_history: List) -> Dict[str, Any]:
        """Execute AutoGen group chat"""
        try:
            import asyncio
            
            def run_group_chat():
                try:
                    # Get the group chat from the manager
                    if hasattr(self.autogen_component, 'groupchat'):
                        groupchat = self.autogen_component.groupchat
                        agents = getattr(groupchat, 'agents', [])
                    else:
                        agents = getattr(self.autogen_component, 'agents', [])
                    
                    if not agents:
                        return "No agents available in group chat"
                    
                    # Create a user proxy for the group chat
                    user_proxy = UserProxyAgent(
                        "meshai_group_proxy",
                        human_input_mode="NEVER",
                        max_consecutive_auto_reply=1,
                        is_termination_msg=lambda x: True,
                        code_execution_config=False
                    )
                    
                    # Enhanced message for group context
                    enhanced_message = self._enhance_message_with_context(message, conversation_history)
                    
                    # Initiate group chat
                    chat_result = user_proxy.initiate_chat(
                        self.autogen_component,
                        message=enhanced_message,
                        max_turns=5
                    )
                    
                    # Extract responses from all agents
                    responses = []
                    if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                        for msg in chat_result.chat_history:
                            if msg.get('name') != 'meshai_group_proxy':
                                agent_name = msg.get('name', 'unknown')
                                content = msg.get('content', '')
                                if content:
                                    responses.append(f"{agent_name}: {content}")
                    
                    return "\n\n".join(responses) if responses else "No responses from group chat"
                    
                except Exception as e:
                    logger.error(f"Group chat execution error: {e}")
                    return f"Group chat execution error: {str(e)[:200]}"
            
            response = await asyncio.get_event_loop().run_in_executor(None, run_group_chat)
            
            # Get participant information
            participants = []
            if hasattr(self.autogen_component, 'groupchat'):
                groupchat = self.autogen_component.groupchat
                agents = getattr(groupchat, 'agents', [])
                participants = [getattr(agent, 'name', 'unknown') for agent in agents]
            
            return {
                "result": response,
                "type": "autogen_group_chat_response",
                "participants": participants,
                "participant_count": len(participants),
                "execution_mode": "group_chat"
            }
            
        except Exception as e:
            logger.error(f"AutoGen group chat execution failed: {e}")
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
    
    def _enhance_message_with_context(self, message: str, conversation_history: List) -> str:
        """Enhance message with relevant context"""
        if not conversation_history:
            return message
        
        # Add recent context
        context_parts = ["Previous conversation context:"]
        for msg in conversation_history[-2:]:  # Last 2 messages
            if isinstance(msg, dict) and 'content' in msg:
                role = msg.get('type', 'unknown')
                content = msg['content'][:150]  # Limit length
                context_parts.append(f"- {role}: {content}")
        
        context_summary = "\n".join(context_parts)
        return f"{context_summary}\n\nCurrent task: {message}"
    
    async def get_agent_capabilities(self) -> List[str]:
        """Get dynamic capabilities based on AutoGen component"""
        capabilities = list(self.capabilities)  # Start with base capabilities
        
        # Add capabilities based on system message or role
        if hasattr(self.autogen_component, 'system_message'):
            system_msg = self.autogen_component.system_message.lower()
            
            if 'code' in system_msg or 'programming' in system_msg:
                capabilities.append('coding')
            if 'analysis' in system_msg or 'analyze' in system_msg:
                capabilities.append('analysis')
            if 'math' in system_msg or 'calculation' in system_msg:
                capabilities.append('mathematics')
            if 'research' in system_msg:
                capabilities.append('research')
        
        # Remove duplicates
        return list(set(capabilities))