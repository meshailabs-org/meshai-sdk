"""
CrewAI adapter for MeshAI SDK
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from crewai import Agent as CrewAIAgent, Task as CrewAITask, Crew
    from crewai.tools import BaseTool as CrewAIBaseTool
except ImportError:
    raise ImportError(
        "CrewAI is not installed. Install with: pip install meshai-sdk[crewai]"
    )

import structlog

from ..core.agent import MeshAgent
from ..core.config import MeshConfig
from ..core.context import MeshContext
from ..core.schemas import TaskData, TaskResult
from ..exceptions.base import TaskExecutionError

logger = structlog.get_logger(__name__)


class MeshAICrewTool(CrewAIBaseTool):
    """
    CrewAI tool that can invoke other MeshAI agents.
    
    This allows CrewAI agents to seamlessly call other agents
    in the MeshAI network through the standard CrewAI tool interface.
    """
    
    name: str = "MeshAI Agent Invoker"
    description: str = "Invoke another MeshAI agent with specific capabilities to help complete tasks"
    
    def __init__(self, mesh_agent: "CrewAIMeshAgent"):
        super().__init__()
        self.mesh_agent = mesh_agent
    
    def _run(self, query: str, capabilities: str = "") -> str:
        """Execute the tool"""
        try:
            # Parse capabilities
            caps = [c.strip() for c in capabilities.split(",") if c.strip()]
            if not caps:
                caps = ["general"]
            
            # Run async invoke in sync context
            result = asyncio.run(
                self.mesh_agent.invoke_agent(
                    capabilities=caps,
                    task={"input": query},
                    routing_strategy="capability_match"
                )
            )
            
            if result.status == "completed" and result.result:
                if isinstance(result.result, dict):
                    return result.result.get("result", str(result.result))
                return str(result.result)
            else:
                return f"Task failed: {result.error or 'Unknown error'}"
                
        except Exception as e:
            logger.error(f"MeshAI CrewAI tool execution failed: {e}")
            return f"Error invoking MeshAI agent: {e}"


class CrewAIMeshAgent(MeshAgent):
    """
    MeshAI adapter for CrewAI agents and crews.
    
    Wraps CrewAI Crew or individual Agent to make them compatible with MeshAI
    while preserving their CrewAI functionality.
    
    Features:
    - Crew-based multi-agent coordination
    - Individual agent wrapping
    - Context sharing between agents
    - Tool integration with MeshAI network
    """
    
    def __init__(
        self,
        crewai_component: Union[Crew, CrewAIAgent],
        agent_id: str,
        name: str,
        capabilities: List[str],
        config: Optional[MeshConfig] = None,
        **kwargs
    ):
        """
        Initialize CrewAI MeshAI adapter.
        
        Args:
            crewai_component: CrewAI Crew or Agent to wrap
            agent_id: Unique agent identifier
            name: Human-readable agent name
            capabilities: List of agent capabilities
            config: MeshAI configuration
            **kwargs: Additional metadata
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="crewai",
            config=config,
            **kwargs
        )
        
        self.crewai_component = crewai_component
        self.component_type = "crew" if isinstance(crewai_component, Crew) else "agent"
        
        # Setup MeshAI integration
        self._setup_meshai_integration()
        
        logger.info(f"CrewAI {self.component_type} {agent_id} initialized")
    
    def _setup_meshai_integration(self) -> None:
        """Setup MeshAI integration with CrewAI component"""
        try:
            # Create MeshAI tool
            meshai_tool = MeshAICrewTool(self)
            
            if self.component_type == "crew":
                # Add tool to all agents in the crew
                for agent in self.crewai_component.agents:
                    if hasattr(agent, 'tools') and agent.tools is not None:
                        agent.tools.append(meshai_tool)
                    else:
                        agent.tools = [meshai_tool]
            else:
                # Add tool to single agent
                if hasattr(self.crewai_component, 'tools') and self.crewai_component.tools is not None:
                    self.crewai_component.tools.append(meshai_tool)
                else:
                    self.crewai_component.tools = [meshai_tool]
                    
        except Exception as e:
            logger.warning(f"Failed to setup MeshAI integration: {e}")
            # Continue without MeshAI integration if setup fails
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """
        Handle task execution with CrewAI component.
        
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
            
            # Prepare input for CrewAI
            if isinstance(task_input, dict):
                description = task_input.get("description", task_input.get("input", str(task_input)))
                expected_output = task_input.get("expected_output", "Complete the requested task")
            else:
                description = str(task_input)
                expected_output = "Complete the requested task"
            
            # Update component with context
            await self._update_context_to_crew(context)
            
            # Execute based on component type
            if self.component_type == "crew":
                result = await self._execute_crew(description, expected_output, parameters)
            else:
                result = await self._execute_agent(description, expected_output, parameters)
            
            # Update context from execution
            await self._update_context_from_crew(context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"CrewAI task execution failed: {e}")
            raise TaskExecutionError(f"CrewAI execution error: {e}")
    
    async def _execute_crew(
        self, 
        description: str, 
        expected_output: str, 
        parameters: Dict
    ) -> Dict[str, Any]:
        """Execute CrewAI Crew"""
        try:
            # Create a task for the crew
            crew_task = CrewAITask(
                description=description,
                expected_output=expected_output,
                **parameters
            )
            
            # Execute crew in thread pool to avoid blocking
            def run_crew():
                return self.crewai_component.kickoff(inputs=parameters)
            
            # Run in executor to avoid blocking async loop
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_crew
            )
            
            return {
                "result": str(result),
                "type": "crew_execution",
                "agents_involved": len(self.crewai_component.agents),
                "metadata": {
                    "crew_id": getattr(self.crewai_component, 'id', 'unknown'),
                    "execution_time": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            raise TaskExecutionError(f"Crew execution failed: {e}")
    
    async def _execute_agent(
        self, 
        description: str, 
        expected_output: str, 
        parameters: Dict
    ) -> Dict[str, Any]:
        """Execute single CrewAI Agent"""
        try:
            # Create a simple task for the agent
            agent_task = CrewAITask(
                description=description,
                expected_output=expected_output,
                agent=self.crewai_component
            )
            
            # Create temporary crew with single agent
            temp_crew = Crew(
                agents=[self.crewai_component],
                tasks=[agent_task],
                verbose=False
            )
            
            # Execute in thread pool
            def run_agent():
                return temp_crew.kickoff(inputs=parameters)
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_agent
            )
            
            return {
                "result": str(result),
                "type": "agent_execution",
                "agent_role": getattr(self.crewai_component, 'role', 'unknown'),
                "metadata": {
                    "agent_id": getattr(self.crewai_component, 'id', self.agent_id),
                    "execution_time": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            raise TaskExecutionError(f"Agent execution failed: {e}")
    
    async def _update_context_to_crew(self, context: MeshContext) -> None:
        """Update CrewAI component with context information"""
        try:
            # Get relevant context data
            conversation_history = await context.get("conversation_history", [])
            shared_data = await context.get("shared_data", {})
            
            # Create context summary for CrewAI
            context_summary = self._prepare_context_summary(conversation_history, shared_data)
            
            # For crews, we can set context on all agents
            if self.component_type == "crew":
                for agent in self.crewai_component.agents:
                    self._inject_context_to_agent(agent, context_summary)
            else:
                # Single agent context update
                self._inject_context_to_agent(self.crewai_component, context_summary)
                    
        except Exception as e:
            logger.warning(f"Failed to update CrewAI context: {e}")
    
    def _prepare_context_summary(self, conversation_history: List[Dict], shared_data: Dict) -> str:
        """Prepare context summary for CrewAI agents"""
        context_parts = []
        
        # Add relevant conversation history
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 messages
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                if isinstance(msg, dict) and 'content' in msg:
                    role = msg.get('type', 'unknown')
                    content = msg['content'][:200]  # Limit length
                    context_parts.append(f"- {role}: {content}")
        
        # Add shared data insights
        if shared_data:
            context_parts.append("\nShared context:")
            for key, value in list(shared_data.items())[:3]:  # Limit to 3 items
                context_parts.append(f"- {key}: {str(value)[:100]}")
        
        return "\n".join(context_parts) if context_parts else "No additional context."
    
    def _inject_context_to_agent(self, agent, context_summary: str) -> None:
        """Inject context into CrewAI agent"""
        try:
            # Try to update backstory with context
            if hasattr(agent, 'backstory'):
                original_backstory = getattr(agent, 'backstory', '')
                if context_summary and "No additional context" not in context_summary:
                    updated_backstory = f"{original_backstory}\n\nCurrent Context:\n{context_summary}"
                    agent.backstory = updated_backstory[:2000]  # Limit total length
        except Exception as e:
            logger.debug(f"Could not inject context to agent: {e}")
    
    async def _update_context_from_crew(self, context: MeshContext, result: Dict[str, Any]) -> None:
        """Update MeshAI context from CrewAI execution results"""
        try:
            # Add execution result to conversation history
            conversation_entry = {
                "type": "ai",
                "content": result.get("result", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "crewai",
                "metadata": result.get("metadata", {})
            }
            
            # Get current conversation history
            conversation_history = await context.get("conversation_history", [])
            conversation_history.append(conversation_entry)
            
            # Keep only last 50 messages
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
            
            # Update context
            await context.set("conversation_history", conversation_history)
            
            # Update shared data if result contains structured information
            if isinstance(result.get("result"), dict):
                shared_data = await context.get("shared_data", {})
                shared_data.update({f"crewai_{self.agent_id}": result})
                await context.set("shared_data", shared_data)
                
        except Exception as e:
            logger.warning(f"Failed to update context from CrewAI: {e}")
    
    def get_crew_info(self) -> Dict[str, Any]:
        """Get information about the CrewAI component"""
        if self.component_type == "crew":
            return {
                "type": "crew",
                "agent_count": len(self.crewai_component.agents),
                "agents": [
                    {
                        "role": getattr(agent, 'role', 'unknown'),
                        "goal": getattr(agent, 'goal', 'unknown'),
                        "backstory": getattr(agent, 'backstory', '')[:100] + "..." if len(getattr(agent, 'backstory', '')) > 100 else getattr(agent, 'backstory', ''),
                        "tools_count": len(getattr(agent, 'tools', []))
                    }
                    for agent in self.crewai_component.agents
                ],
                "process": getattr(self.crewai_component, 'process', 'unknown')
            }
        else:
            return {
                "type": "agent",
                "role": getattr(self.crewai_component, 'role', 'unknown'),
                "goal": getattr(self.crewai_component, 'goal', 'unknown'),
                "backstory": getattr(self.crewai_component, 'backstory', '')[:200] + "..." if len(getattr(self.crewai_component, 'backstory', '')) > 200 else getattr(self.crewai_component, 'backstory', ''),
                "tools_count": len(getattr(self.crewai_component, 'tools', []))
            }
    
    def add_agent(self, agent: CrewAIAgent) -> None:
        """Add agent to crew (only works if component is a Crew)"""
        if self.component_type != "crew":
            raise ValueError("Can only add agents to Crew components")
        
        # Add MeshAI tool to new agent
        meshai_tool = MeshAICrewTool(self)
        if hasattr(agent, 'tools') and agent.tools is not None:
            agent.tools.append(meshai_tool)
        else:
            agent.tools = [meshai_tool]
        
        # Add agent to crew
        self.crewai_component.agents.append(agent)
        
        logger.info(f"Added agent {getattr(agent, 'role', 'unknown')} to crew {self.agent_id}")
    
    def add_task(self, task: CrewAITask) -> None:
        """Add task to crew (only works if component is a Crew)"""
        if self.component_type != "crew":
            raise ValueError("Can only add tasks to Crew components")
        
        self.crewai_component.tasks.append(task)
        logger.info(f"Added task to crew {self.agent_id}")
    
    @classmethod
    def from_agents_and_tasks(
        cls,
        agents: List[CrewAIAgent],
        tasks: List[CrewAITask],
        agent_id: str,
        name: str,
        capabilities: List[str],
        process: str = "sequential",
        **kwargs
    ) -> "CrewAIMeshAgent":
        """
        Create CrewAI MeshAI agent from agents and tasks.
        
        Args:
            agents: List of CrewAI agents
            tasks: List of CrewAI tasks
            agent_id: Agent identifier
            name: Agent name
            capabilities: Agent capabilities
            process: Execution process (sequential, hierarchical)
            **kwargs: Additional arguments
        """
        try:
            from crewai import Process
            
            # Map process string to enum
            process_map = {
                "sequential": Process.sequential,
                "hierarchical": Process.hierarchical,
            }
            
            process_enum = process_map.get(process, Process.sequential)
            
            # Create crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=process_enum,
                verbose=kwargs.get("verbose", False)
            )
            
            return cls(
                crewai_component=crew,
                agent_id=agent_id,
                name=name,
                capabilities=capabilities,
                **{k: v for k, v in kwargs.items() if k != "verbose"}
            )
            
        except ImportError as e:
            raise ImportError(f"Failed to import CrewAI components: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create crew: {e}")