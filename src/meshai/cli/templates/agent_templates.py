"""
MeshAI Agent Templates

Framework-specific agent templates for scaffolding.
"""

from typing import Dict, Any, List


def get_agent_template(
    framework: str,
    agent_name: str,
    model: str,
    capabilities: List[str],
    agent_type: str = "general"
) -> str:
    """
    Generate agent code based on framework and type.
    
    Args:
        framework: AI framework (openai, anthropic, google, etc.)
        agent_name: Name of the agent
        model: Model to use
        capabilities: List of agent capabilities
        agent_type: Type of agent (general, coder, analyst, writer, etc.)
        
    Returns:
        Agent code as string
    """
    
    templates = {
        "openai": generate_openai_agent,
        "anthropic": generate_anthropic_agent,
        "google": generate_google_agent,
        "langchain": generate_langchain_agent,
        "crewai": generate_crewai_agent,
        "autogen": generate_autogen_agent,
        "amazon": generate_amazon_agent
    }
    
    template_fn = templates.get(framework, generate_openai_agent)
    return template_fn(agent_name, model, capabilities, agent_type)


def generate_openai_agent(name: str, model: str, capabilities: List[str], agent_type: str) -> str:
    """Generate OpenAI agent template"""
    
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    system_prompts = {
        "general": f"You are {name}, a helpful AI assistant with expertise in: {', '.join(capabilities)}.",
        "coder": f"You are {name}, an expert software developer specializing in: {', '.join(capabilities)}. Write clean, well-documented, production-ready code.",
        "analyst": f"You are {name}, a data analyst expert in: {', '.join(capabilities)}. Provide data-driven insights and comprehensive analysis.",
        "writer": f"You are {name}, a professional writer skilled in: {', '.join(capabilities)}. Create clear, engaging, well-structured content.",
        "researcher": f"You are {name}, a research specialist focusing on: {', '.join(capabilities)}. Provide comprehensive, accurate, fact-based research.",
        "support": f"You are {name}, a customer support specialist. Be empathetic, helpful, and solution-focused. Your expertise: {', '.join(capabilities)}."
    }
    
    system_prompt = system_prompts.get(agent_type, system_prompts["general"])
    
    template = f'''"""
{class_name} - OpenAI Agent

Powered by {model}
Capabilities: {", ".join(capabilities)}
Type: {agent_type}
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from meshai.adapters.openai_adapter import OpenAIMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry

logger = logging.getLogger(__name__)


class {class_name}(OpenAIMeshAgent):
    """
    {class_name} - {agent_type.title()} Agent
    
    Specializes in:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self, **kwargs):
        """Initialize the agent with custom configuration"""
        default_config = {{
            "model": "{model}",
            "agent_id": "{agent_id}",
            "name": "{name}",
            "capabilities": {capabilities},
            "system_prompt": (
                "{system_prompt}"
            ),
            "temperature": {0.7 if agent_type in ["writer", "general"] else 0.3},
            "max_tokens": 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }}
        
        # Merge with any provided kwargs
        config = {{**default_config, **kwargs}}
        super().__init__(**config)
        
        # Initialize agent-specific attributes
        self.task_history = []
        self.performance_metrics = {{
            "total_tasks": 0,
            "successful_tasks": 0,
            "average_response_time": 0
        }}
    
    async def handle_task(
        self, 
        task_data: TaskData, 
        context: MeshContext
    ) -> Dict[str, Any]:
        """
        Handle incoming tasks with custom processing logic.
        
        Args:
            task_data: Task information and parameters
            context: Shared context for state management
            
        Returns:
            Task result dictionary
        """
        start_time = datetime.now()
        
        try:
            # Pre-processing
            processed_input = await self._preprocess_task(task_data, context)
            
            # Update task data if preprocessing modified it
            if processed_input != task_data.input:
                task_data = TaskData(
                    input=processed_input,
                    parameters=task_data.parameters
                )
            
            # Call parent handler with OpenAI
            result = await super().handle_task(task_data, context)
            
            # Post-processing
            result = await self._postprocess_result(result, context)
            
            # Update metrics
            self._update_metrics(start_time, success=True)
            
            # Store in history
            self.task_history.append({{
                "timestamp": start_time.isoformat(),
                "task": task_data.input[:100],
                "success": True,
                "response_time": (datetime.now() - start_time).total_seconds()
            }})
            
            return result
            
        except Exception as e:
            logger.error(f"Task handling error: {{e}}")
            self._update_metrics(start_time, success=False)
            
            return {{
                "result": f"I encountered an error: {{str(e)}}",
                "error": str(e),
                "type": "error"
            }}
    
    async def _preprocess_task(
        self, 
        task_data: TaskData, 
        context: MeshContext
    ) -> str:
        """
        Preprocess the task before sending to OpenAI.
        
        Override this method for custom preprocessing logic.
        """
        # Add context from previous interactions if available
        conversation_history = await context.get("conversation_history", [])
        
        if conversation_history and len(conversation_history) > 0:
            # Include relevant context
            recent_context = conversation_history[-3:]  # Last 3 interactions
            context_summary = self._summarize_context(recent_context)
            
            if context_summary:
                return f"{{context_summary}}\\n\\nCurrent request: {{task_data.input}}"
        
        return task_data.input
    
    async def _postprocess_result(
        self, 
        result: Dict[str, Any], 
        context: MeshContext
    ) -> Dict[str, Any]:
        """
        Postprocess the result from OpenAI.
        
        Override this method for custom postprocessing logic.
        """
        # Add metadata
        result["agent_id"] = self.agent_id
        result["agent_type"] = "{agent_type}"
        result["model"] = self.model
        result["timestamp"] = datetime.now().isoformat()
        
        # Extract and store any structured data
        if "{agent_type}" == "analyst" and "data" in result.get("result", ""):
            result["extracted_data"] = self._extract_data(result["result"])
        
        return result
    
    def _summarize_context(self, conversation_history: List[Dict]) -> str:
        """Summarize conversation context"""
        if not conversation_history:
            return ""
        
        context_parts = []
        for msg in conversation_history:
            if msg.get("type") == "human":
                context_parts.append(f"User: {{msg.get('content', '')[:50]}}...")
            elif msg.get("type") == "ai":
                context_parts.append(f"Assistant: {{msg.get('content', '')[:50]}}...")
        
        if context_parts:
            return f"Previous context: {{' | '.join(context_parts)}}"
        
        return ""
    
    def _extract_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text (for analyst agents)"""
        # Simple extraction logic - override for more sophisticated parsing
        extracted = {{}}
        
        # Look for numbers
        import re
        numbers = re.findall(r'\\b\\d+\\.?\\d*\\b', text)
        if numbers:
            extracted["numbers"] = numbers
        
        # Look for percentages
        percentages = re.findall(r'\\b\\d+\\.?\\d*%', text)
        if percentages:
            extracted["percentages"] = percentages
        
        return extracted
    
    def _update_metrics(self, start_time: datetime, success: bool):
        """Update performance metrics"""
        response_time = (datetime.now() - start_time).total_seconds()
        
        self.performance_metrics["total_tasks"] += 1
        if success:
            self.performance_metrics["successful_tasks"] += 1
        
        # Update average response time
        total = self.performance_metrics["total_tasks"]
        current_avg = self.performance_metrics["average_response_time"]
        new_avg = ((current_avg * (total - 1)) + response_time) / total
        self.performance_metrics["average_response_time"] = new_avg
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {{
            **self.performance_metrics,
            "success_rate": (
                self.performance_metrics["successful_tasks"] / 
                max(self.performance_metrics["total_tasks"], 1)
            ),
            "recent_tasks": self.task_history[-10:]  # Last 10 tasks
        }}
    
    async def invoke_specialist(
        self, 
        specialist_type: str,
        task: str,
        context: MeshContext
    ) -> Dict[str, Any]:
        """
        Invoke a specialist agent for specific tasks.
        
        Args:
            specialist_type: Type of specialist needed
            task: Task for the specialist
            context: Shared context
            
        Returns:
            Specialist's response
        """
        # Map specialist types to capabilities
        specialist_map = {{
            "coder": ["coding", "programming", "debugging"],
            "analyst": ["analysis", "data-processing", "insights"],
            "writer": ["writing", "content-creation", "editing"],
            "researcher": ["research", "fact-checking", "investigation"]
        }}
        
        capabilities = specialist_map.get(specialist_type, ["general"])
        
        # Invoke through MeshAI
        result = await self.invoke_agent(
            capabilities=capabilities,
            task={{"input": task}},
            routing_strategy="capability_match"
        )
        
        return result
    
    async def collaborative_task(
        self,
        task: str,
        collaborators: List[str],
        context: MeshContext
    ) -> Dict[str, Any]:
        """
        Execute a task with multiple collaborating agents.
        
        Args:
            task: Task to execute
            collaborators: List of collaborator types
            context: Shared context
            
        Returns:
            Combined result from collaboration
        """
        results = {{}}
        
        # Execute task with each collaborator
        for collaborator in collaborators:
            specialist_result = await self.invoke_specialist(
                collaborator, 
                task, 
                context
            )
            results[collaborator] = specialist_result
        
        # Synthesize results
        synthesis_task = TaskData(
            input=f"Synthesize these results into a cohesive response: {{results}}"
        )
        
        final_result = await self.handle_task(synthesis_task, context)
        final_result["collaboration_results"] = results
        
        return final_result


async def main():
    """
    Example usage of the {class_name} agent
    """
    # Create agent instance
    agent = {class_name}()
    
    # Register with MeshAI
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    print(f"✅ {{agent.name}} registered successfully!")
    print(f"Agent ID: {{agent.agent_id}}")
    print(f"Model: {{agent.model}}")
    print(f"Capabilities: {{', '.join(agent.capabilities)}}")
    
    # Example task
    context = MeshContext()
    task = TaskData(
        input="Hello! Please demonstrate your capabilities.",
        parameters={{"verbose": True}}
    )
    
    print("\\n🔄 Processing task...")
    result = await agent.handle_task(task, context)
    
    print(f"\\n📋 Response:")
    print(result.get('result', 'No response'))
    
    # Show performance stats
    stats = await agent.get_performance_stats()
    print(f"\\n📊 Performance Stats:")
    print(f"  Total tasks: {{stats['total_tasks']}}")
    print(f"  Success rate: {{stats['success_rate']:.1%}}")
    print(f"  Avg response time: {{stats['average_response_time']:.2f}}s")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
'''
    
    return template


def generate_anthropic_agent(name: str, model: str, capabilities: List[str], agent_type: str) -> str:
    """Generate Anthropic Claude agent template"""
    
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    template = f'''"""
{class_name} - Anthropic Claude Agent

Powered by {model}
Capabilities: {", ".join(capabilities)}
Type: {agent_type}
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry

logger = logging.getLogger(__name__)


class {class_name}(AnthropicMeshAgent):
    """
    {class_name} - Claude-powered {agent_type.title()} Agent
    
    Specializes in:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self, **kwargs):
        """Initialize Claude agent with custom configuration"""
        default_config = {{
            "model": "{model}",
            "agent_id": "{agent_id}",
            "name": "{name}",
            "capabilities": {capabilities},
            "system_prompt": (
                "You are {name}, powered by Claude. "
                "Your expertise includes: {', '.join(capabilities)}. "
                "Always provide thoughtful, detailed, and accurate responses."
            ),
            "temperature": {0.7 if agent_type in ["writer", "general"] else 0.2},
            "max_tokens": 4000  # Claude supports longer outputs
        }}
        
        config = {{**default_config, **kwargs}}
        super().__init__(**config)
        
        # Claude-specific features
        self.enable_safety_ratings = True
        self.enable_tool_use = True
    
    async def handle_task(
        self, 
        task_data: TaskData, 
        context: MeshContext
    ) -> Dict[str, Any]:
        """Handle tasks with Claude's advanced capabilities"""
        
        # Leverage Claude's strengths
        if "{agent_type}" == "analyst":
            # Claude excels at detailed analysis
            task_data = TaskData(
                input=f"Provide a comprehensive analysis: {{task_data.input}}",
                parameters={{**task_data.parameters, "detail_level": "high"}}
            )
        elif "{agent_type}" == "coder":
            # Claude is excellent at explaining code
            task_data = TaskData(
                input=f"{{task_data.input}}. Include detailed explanations and best practices.",
                parameters=task_data.parameters
            )
        
        result = await super().handle_task(task_data, context)
        
        # Add Claude-specific metadata
        if self.enable_safety_ratings and "safety_ratings" in result:
            result["safety_analysis"] = self._analyze_safety(result["safety_ratings"])
        
        return result
    
    def _analyze_safety(self, ratings: Dict) -> str:
        """Analyze Claude's safety ratings"""
        # Process safety ratings if available
        return "Content verified as safe and appropriate"


async def main():
    """Example usage"""
    agent = {class_name}()
    
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    context = MeshContext()
    task = TaskData(input="Demonstrate Claude's capabilities")
    
    result = await agent.handle_task(task, context)
    print(f"Claude says: {{result['result']}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return template


def generate_google_agent(name: str, model: str, capabilities: List[str], agent_type: str) -> str:
    """Generate Google Gemini agent template"""
    
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    template = f'''"""
{class_name} - Google Gemini Agent

Powered by {model}
Capabilities: {", ".join(capabilities)}
Type: {agent_type}
"""

import asyncio
from typing import Dict, Any, List
from meshai.adapters.google_adapter import GoogleMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry


class {class_name}(GoogleMeshAgent):
    """
    {class_name} - Gemini-powered {agent_type.title()} Agent
    
    Specializes in:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self, **kwargs):
        """Initialize Gemini agent"""
        default_config = {{
            "model": "{model}",
            "agent_id": "{agent_id}",
            "name": "{name}",
            "capabilities": {capabilities},
            "temperature": {0.7 if agent_type in ["writer", "general"] else 0.3},
            "top_p": 0.95,
            "top_k": 40
        }}
        
        config = {{**default_config, **kwargs}}
        super().__init__(**config)
        
        # Gemini-specific features
        self.enable_multimodal = "vision" in model.lower()
        self.enable_function_calling = True
    
    async def handle_multimodal_task(
        self,
        text: str,
        image_path: Optional[str] = None,
        context: MeshContext = None
    ) -> Dict[str, Any]:
        """Handle multimodal tasks with Gemini Pro Vision"""
        
        if not self.enable_multimodal:
            return await self.handle_task(TaskData(input=text), context or MeshContext())
        
        # Handle image + text input
        # Implementation would include image processing
        task_data = TaskData(
            input=text,
            parameters={{"image_path": image_path}} if image_path else {{}}
        )
        
        return await self.handle_task(task_data, context or MeshContext())


async def main():
    """Example usage"""
    agent = {class_name}()
    
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    context = MeshContext()
    task = TaskData(input="Demonstrate Gemini's capabilities")
    
    result = await agent.handle_task(task, context)
    print(f"Gemini says: {{result['result']}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return template


def generate_langchain_agent(name: str, model: str, capabilities: List[str], agent_type: str) -> str:
    """Generate LangChain agent template"""
    
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    template = f'''"""
{class_name} - LangChain Agent

Powered by LangChain with {model}
Capabilities: {", ".join(capabilities)}
Type: {agent_type}
"""

import asyncio
from typing import Dict, Any, List
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

from meshai.adapters.langchain_adapter import LangChainMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry


class {class_name}(LangChainMeshAgent):
    """
    {class_name} - LangChain-powered {agent_type.title()} Agent
    
    Specializes in:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self, **kwargs):
        """Initialize LangChain agent with tools and memory"""
        
        # Create LLM
        llm = ChatOpenAI(model="{model}", temperature=0.7)
        
        # Create tools
        tools = self._create_tools()
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        langchain_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
        
        # Initialize MeshAI wrapper
        super().__init__(
            langchain_agent=langchain_agent,
            agent_id="{agent_id}",
            name="{name}",
            capabilities={capabilities}
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for the agent"""
        tools = []
        
        # Add custom tools based on agent type
        if "{agent_type}" == "coder":
            tools.append(Tool(
                name="code_executor",
                func=self._execute_code,
                description="Execute Python code and return results"
            ))
        elif "{agent_type}" == "analyst":
            tools.append(Tool(
                name="data_analyzer",
                func=self._analyze_data,
                description="Analyze data and extract insights"
            ))
        
        # Add general tools
        tools.append(Tool(
            name="web_search",
            func=self._web_search,
            description="Search the web for information"
        ))
        
        return tools
    
    def _execute_code(self, code: str) -> str:
        """Execute Python code safely"""
        # Implement safe code execution
        return f"Code execution result for: {{code[:50]}}..."
    
    def _analyze_data(self, data: str) -> str:
        """Analyze data"""
        return f"Analysis of: {{data[:50]}}..."
    
    def _web_search(self, query: str) -> str:
        """Perform web search"""
        return f"Search results for: {{query}}"


async def main():
    """Example usage"""
    agent = {class_name}()
    
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    context = MeshContext()
    task = TaskData(input="Use your tools to help me")
    
    result = await agent.handle_task(task, context)
    print(f"LangChain agent result: {{result['result']}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return template


def generate_crewai_agent(name: str, model: str, capabilities: List[str], agent_type: str) -> str:
    """Generate CrewAI agent template"""
    
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    template = f'''"""
{class_name} - CrewAI Agent

Powered by CrewAI with {model}
Capabilities: {", ".join(capabilities)}
Type: {agent_type}
"""

import asyncio
from typing import Dict, Any, List
from crewai import Agent, Task, Crew, Process

from meshai.adapters.crewai_adapter import CrewAIMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry


class {class_name}(CrewAIMeshAgent):
    """
    {class_name} - CrewAI-powered {agent_type.title()} Agent
    
    Specializes in:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self, **kwargs):
        """Initialize CrewAI agent/crew"""
        
        # Create CrewAI agents for the crew
        agents = self._create_crew_agents()
        
        # Create crew
        crew = Crew(
            agents=agents,
            tasks=[],  # Tasks will be added dynamically
            process=Process.sequential,
            verbose=True
        )
        
        # Initialize MeshAI wrapper
        super().__init__(
            crewai_component=crew,
            agent_id="{agent_id}",
            name="{name}",
            capabilities={capabilities}
        )
    
    def _create_crew_agents(self) -> List[Agent]:
        """Create specialized agents for the crew"""
        agents = []
        
        # Create role-specific agents
        if "{agent_type}" == "researcher":
            agents.append(Agent(
                role="Lead Researcher",
                goal="Conduct comprehensive research",
                backstory="Expert researcher with deep domain knowledge"
            ))
            agents.append(Agent(
                role="Fact Checker",
                goal="Verify accuracy of information",
                backstory="Meticulous fact-checker ensuring accuracy"
            ))
        elif "{agent_type}" == "analyst":
            agents.append(Agent(
                role="Data Analyst",
                goal="Analyze data and extract insights",
                backstory="Expert analyst with statistical expertise"
            ))
            agents.append(Agent(
                role="Report Writer",
                goal="Create clear analytical reports",
                backstory="Skilled at presenting complex data clearly"
            ))
        else:
            # Default crew
            agents.append(Agent(
                role="Team Lead",
                goal="Coordinate and execute tasks",
                backstory="Experienced team lead"
            ))
            agents.append(Agent(
                role="Specialist",
                goal="Provide specialized expertise",
                backstory="Domain expert"
            ))
        
        return agents


async def main():
    """Example usage"""
    agent = {class_name}()
    
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    context = MeshContext()
    task = TaskData(input="Coordinate the crew to complete this task")
    
    result = await agent.handle_task(task, context)
    print(f"CrewAI result: {{result['result']}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return template


def generate_autogen_agent(name: str, model: str, capabilities: List[str], agent_type: str) -> str:
    """Generate AutoGen agent template"""
    
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    template = f'''"""
{class_name} - AutoGen Agent

Powered by AutoGen with {model}
Capabilities: {", ".join(capabilities)}
Type: {agent_type}
"""

import asyncio
from typing import Dict, Any, List
from autogen import ConversableAgent, GroupChat, GroupChatManager

from meshai.adapters.autogen_adapter import AutoGenMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry


class {class_name}(AutoGenMeshAgent):
    """
    {class_name} - AutoGen-powered {agent_type.title()} Agent
    
    Specializes in:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self, **kwargs):
        """Initialize AutoGen conversable agent"""
        
        # Create AutoGen agent
        autogen_agent = ConversableAgent(
            name="{agent_id}",
            system_message=(
                "You are {name}, specializing in: {', '.join(capabilities)}. "
                "Collaborate effectively with other agents to complete tasks."
            ),
            llm_config={{
                "model": "{model}",
                "temperature": 0.7
            }},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5
        )
        
        # Initialize MeshAI wrapper
        super().__init__(
            autogen_component=autogen_agent,
            agent_id="{agent_id}",
            name="{name}",
            capabilities={capabilities}
        )
        
        # AutoGen-specific features
        self.enable_code_execution = "{agent_type}" == "coder"
        self.enable_group_chat = True
    
    async def create_group_chat(
        self,
        other_agents: List[Any],
        task: str
    ) -> Dict[str, Any]:
        """Create a group chat with other AutoGen agents"""
        
        all_agents = [self.autogen_component] + other_agents
        
        group_chat = GroupChat(
            agents=all_agents,
            messages=[],
            max_round=10
        )
        
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config={{"model": "{model}"}}
        )
        
        # Initiate chat
        result = await manager.initiate_chat(
            message=task,
            clear_history=False
        )
        
        return {{"result": result, "type": "group_chat"}}


async def main():
    """Example usage"""
    agent = {class_name}()
    
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    context = MeshContext()
    task = TaskData(input="Demonstrate AutoGen conversational abilities")
    
    result = await agent.handle_task(task, context)
    print(f"AutoGen result: {{result['result']}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return template


def generate_amazon_agent(name: str, model: str, capabilities: List[str], agent_type: str) -> str:
    """Generate Amazon Bedrock agent template"""
    
    agent_id = name.lower().replace(' ', '-').replace('_', '-')
    class_name = ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    template = f'''"""
{class_name} - Amazon Bedrock Agent

Powered by {model} on AWS Bedrock
Capabilities: {", ".join(capabilities)}
Type: {agent_type}
"""

import asyncio
import boto3
from typing import Dict, Any, List, Optional

from meshai.adapters.amazon_adapter import AmazonMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.core.registry import MeshRegistry


class {class_name}(AmazonMeshAgent):
    """
    {class_name} - Bedrock-powered {agent_type.title()} Agent
    
    Specializes in:
    {chr(10).join([f"    - {cap}" for cap in capabilities])}
    """
    
    def __init__(self, **kwargs):
        """Initialize Bedrock agent"""
        
        default_config = {{
            "model": "{model}",
            "agent_id": "{agent_id}",
            "name": "{name}",
            "capabilities": {capabilities},
            "region": "us-east-1",
            "temperature": {0.7 if agent_type in ["writer", "general"] else 0.3},
            "max_tokens": 2000
        }}
        
        config = {{**default_config, **kwargs}}
        super().__init__(**config)
        
        # Bedrock-specific features
        self.enable_guardrails = True
        self.enable_knowledge_base = False
    
    async def handle_task(
        self,
        task_data: TaskData,
        context: MeshContext
    ) -> Dict[str, Any]:
        """Handle tasks with Bedrock's enterprise features"""
        
        # Add enterprise context if available
        if self.enable_guardrails:
            task_data = self._apply_guardrails(task_data)
        
        result = await super().handle_task(task_data, context)
        
        # Add compliance metadata
        result["compliance_checked"] = self.enable_guardrails
        result["region"] = self.region
        
        return result
    
    def _apply_guardrails(self, task_data: TaskData) -> TaskData:
        """Apply AWS guardrails to the task"""
        # Implement guardrail logic
        return task_data


async def main():
    """Example usage"""
    agent = {class_name}()
    
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    context = MeshContext()
    task = TaskData(input="Demonstrate Bedrock enterprise capabilities")
    
    result = await agent.handle_task(task, context)
    print(f"Bedrock result: {{result['result']}}")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return template