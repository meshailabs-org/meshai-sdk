#!/usr/bin/env python3
"""
MeshAI Interactive Quickstart Tutorial

This interactive tutorial guides new users through MeshAI concepts and features:

1. Basic Setup and Installation
2. Creating Your First Agent
3. Agent Registration and Discovery
4. Cross-Framework Communication
5. Building Multi-Agent Workflows
6. Monitoring and Debugging
7. Production Deployment

The tutorial is designed to be run step-by-step with explanations and
hands-on exercises at each stage.
"""

import asyncio
import os
import sys
from typing import Optional, Dict, Any
import time
from datetime import datetime

# Import MeshAI components
from meshai.core.context import MeshContext
from meshai.core.registry import MeshRegistry
from meshai.core.schemas import TaskData

print("🎓 MeshAI Interactive Quickstart Tutorial")
print("=" * 45)
print("Welcome to MeshAI! This tutorial will guide you through the key concepts")
print("and help you build your first multi-agent application.\n")


class TutorialGuide:
    """Interactive tutorial guide with progress tracking"""
    
    def __init__(self):
        self.current_step = 0
        self.completed_steps = set()
        self.tutorial_context = MeshContext()
        self.registry = MeshRegistry()
        
    def print_step_header(self, step_num: int, title: str, description: str):
        """Print formatted step header"""
        print(f"\n{'='*60}")
        print(f"📚 STEP {step_num}: {title}")
        print(f"{'='*60}")
        print(f"{description}\n")
        
    def print_code_example(self, title: str, code: str):
        """Print formatted code example"""
        print(f"💻 {title}:")
        print("```python")
        print(code)
        print("```\n")
        
    def wait_for_user(self, message: str = "Press Enter to continue..."):
        """Wait for user input to proceed"""
        try:
            input(f"⏳ {message}")
        except KeyboardInterrupt:
            print("\n👋 Tutorial interrupted. You can resume anytime!")
            sys.exit(0)
            
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if prerequisites are installed"""
        prereqs = {
            "Python 3.8+": sys.version_info >= (3, 8),
            "MeshAI Core": True,  # If we're running this, core is installed
        }
        
        # Check optional frameworks
        frameworks = {}
        try:
            import openai
            frameworks["OpenAI"] = bool(os.getenv('OPENAI_API_KEY'))
        except ImportError:
            frameworks["OpenAI"] = False
            
        try:
            import anthropic
            frameworks["Anthropic"] = bool(os.getenv('ANTHROPIC_API_KEY'))
        except ImportError:
            frameworks["Anthropic"] = False
            
        try:
            import google.generativeai
            frameworks["Google AI"] = bool(os.getenv('GOOGLE_API_KEY'))
        except ImportError:
            frameworks["Google AI"] = False
            
        try:
            from langchain.agents import AgentExecutor
            frameworks["LangChain"] = True
        except ImportError:
            frameworks["LangChain"] = False
            
        return {"prerequisites": prereqs, "frameworks": frameworks}


async def step_1_setup_and_installation(guide: TutorialGuide):
    """Step 1: Setup and Installation"""
    guide.print_step_header(
        1, 
        "Setup and Installation",
        "Let's start by checking your environment and installing MeshAI."
    )
    
    # Check prerequisites
    print("🔍 Checking your environment...")
    checks = guide.check_prerequisites()
    
    print("✅ Prerequisites:")
    for name, status in checks["prerequisites"].items():
        print(f"   • {name}: {'✅' if status else '❌'}")
        
    print("\n🧩 Available Frameworks:")
    available_frameworks = []
    for name, status in checks["frameworks"].items():
        status_icon = "✅" if status else "⚠️"
        print(f"   • {name}: {status_icon}")
        if status:
            available_frameworks.append(name.lower())
    
    if not available_frameworks:
        print("\n⚠️ No AI frameworks detected with API keys.")
        print("For the best experience, set up at least one:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("   export ANTHROPIC_API_KEY='your-key-here'") 
        print("   export GOOGLE_API_KEY='your-key-here'")
        
    # Installation guide
    guide.print_code_example(
        "Installation Options",
        """# Install core MeshAI
pip install meshai-sdk

# Or install with specific frameworks
pip install meshai-sdk[openai,anthropic,google]

# Or install everything
pip install meshai-sdk[all]"""
    )
    
    print("📦 MeshAI includes these core components:")
    print("   • Agent Registry - Discover and manage agents")
    print("   • Runtime Engine - Execute tasks and route messages")
    print("   • Framework Adapters - Connect different AI platforms")
    print("   • Context Management - Share state across agents")
    
    await guide.tutorial_context.set("available_frameworks", available_frameworks)
    guide.completed_steps.add(1)
    guide.wait_for_user()


async def step_2_first_agent(guide: TutorialGuide):
    """Step 2: Creating Your First Agent"""
    guide.print_step_header(
        2,
        "Creating Your First Agent", 
        "Let's create your first MeshAI agent using an available framework."
    )
    
    available_frameworks = await guide.tutorial_context.get("available_frameworks", [])
    
    if not available_frameworks:
        print("📝 Since no API keys are configured, we'll create a mock agent for demonstration:")
        
        guide.print_code_example(
            "Mock Agent Example",
            """from meshai.core.agent import MeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData

class TutorialAgent(MeshAgent):
    async def handle_task(self, task_data: TaskData, context: MeshContext):
        return {
            "result": f"Hello! I processed: {task_data.input}",
            "type": "tutorial_response",
            "timestamp": "2024-01-01T12:00:00Z"
        }

# Create the agent
tutorial_agent = TutorialAgent(
    agent_id="tutorial-agent",
    name="My First Agent", 
    capabilities=["greeting", "demo"],
    framework="tutorial"
)

print(f"✅ Created agent: {tutorial_agent.name}")"""
        )
        
        # Actually create the mock agent
        class TutorialAgent:
            def __init__(self, agent_id, name, capabilities, framework):
                self.agent_id = agent_id
                self.name = name
                self.capabilities = capabilities
                self.framework = framework
                
            async def handle_task(self, task_data, context):
                return {
                    "result": f"Hello! I processed: {task_data.input}",
                    "type": "tutorial_response", 
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        agent = TutorialAgent(
            agent_id="tutorial-agent",
            name="My First Agent",
            capabilities=["greeting", "demo"], 
            framework="tutorial"
        )
        
        print(f"✅ Created mock agent: {agent.name}")
        
    else:
        # Create real agent with available framework
        framework = available_frameworks[0]
        print(f"🚀 Creating a real agent using {framework}...")
        
        if "openai" in framework:
            guide.print_code_example(
                "OpenAI Agent Creation",
                """from meshai.adapters.openai_adapter import OpenAIMeshAgent

# Create an OpenAI agent
agent = OpenAIMeshAgent(
    model="gpt-3.5-turbo",
    agent_id="tutorial-openai-agent",
    name="Tutorial Assistant",
    capabilities=["conversation", "help", "explanation"],
    system_prompt="You are a helpful tutorial assistant for MeshAI.",
    temperature=0.7
)

print(f"✅ OpenAI agent created: {agent.name}")"""
            )
            
            try:
                from meshai.adapters.openai_adapter import OpenAIMeshAgent
                agent = OpenAIMeshAgent(
                    model="gpt-3.5-turbo",
                    agent_id="tutorial-openai-agent",
                    name="Tutorial Assistant",
                    capabilities=["conversation", "help", "explanation"],
                    system_prompt="You are a helpful tutorial assistant for MeshAI.",
                    temperature=0.7
                )
                print(f"✅ Real OpenAI agent created: {agent.name}")
                
            except Exception as e:
                print(f"❌ Error creating OpenAI agent: {e}")
                agent = None
                
        elif "anthropic" in framework:
            guide.print_code_example(
                "Anthropic Agent Creation",
                """from meshai.adapters.anthropic_adapter import AnthropicMeshAgent

# Create a Claude agent
agent = AnthropicMeshAgent(
    model="claude-3-sonnet-20240229",
    agent_id="tutorial-claude-agent", 
    name="Tutorial Claude",
    capabilities=["analysis", "explanation", "help"],
    system_prompt="You are Claude, helping users learn MeshAI.",
    temperature=0.3
)

print(f"✅ Claude agent created: {agent.name}")"""
            )
            
            try:
                from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
                agent = AnthropicMeshAgent(
                    model="claude-3-sonnet-20240229",
                    agent_id="tutorial-claude-agent",
                    name="Tutorial Claude",
                    capabilities=["analysis", "explanation", "help"],
                    system_prompt="You are Claude, helping users learn MeshAI.",
                    temperature=0.3
                )
                print(f"✅ Real Claude agent created: {agent.name}")
                
            except Exception as e:
                print(f"❌ Error creating Claude agent: {e}")
                agent = None
    
    print("\n🎯 Key Agent Concepts:")
    print("   • agent_id: Unique identifier for the agent")
    print("   • name: Human-readable name")
    print("   • capabilities: List of what the agent can do") 
    print("   • framework: Which AI platform powers the agent")
    
    await guide.tutorial_context.set("first_agent", agent if 'agent' in locals() else None)
    guide.completed_steps.add(2)
    guide.wait_for_user()


async def step_3_agent_registration(guide: TutorialGuide):
    """Step 3: Agent Registration and Discovery"""
    guide.print_step_header(
        3,
        "Agent Registration and Discovery",
        "Learn how to register agents and discover them in the MeshAI network."
    )
    
    agent = await guide.tutorial_context.get("first_agent")
    
    guide.print_code_example(
        "Agent Registration",
        """from meshai.core.registry import MeshRegistry

# Create a registry instance
registry = MeshRegistry()

# Register your agent
await registry.register_agent(agent)

print(f"✅ Agent {agent.name} registered successfully!")

# Discover agents by capability
coding_agents = await registry.discover_agents(capabilities=["coding"])
analysis_agents = await registry.discover_agents(capabilities=["analysis"])

print(f"Found {len(coding_agents)} coding agents")
print(f"Found {len(analysis_agents)} analysis agents")"""
    )
    
    # Actually register the agent if we have one
    if agent:
        try:
            await guide.registry.register_agent(agent)
            print(f"✅ Successfully registered: {agent.name}")
            
            # Test discovery
            all_agents = await guide.registry.discover_agents()
            print(f"📊 Total agents in registry: {len(all_agents)}")
            
            # Discover by capability
            if hasattr(agent, 'capabilities'):
                for capability in agent.capabilities[:2]:  # Test first 2 capabilities
                    matching_agents = await guide.registry.discover_agents(capabilities=[capability])
                    print(f"   • Agents with '{capability}' capability: {len(matching_agents)}")
                    
        except Exception as e:
            print(f"❌ Registration failed: {e}")
    else:
        print("ℹ️ No agent available for registration (API keys not configured)")
    
    print("\n🔍 Agent Discovery Features:")
    print("   • Find agents by capabilities")
    print("   • Filter by framework type") 
    print("   • Health status checking")
    print("   • Load balancing and routing")
    
    print("\n💡 Pro Tip:")
    print("   MeshAI automatically handles agent discovery and routing")
    print("   based on the capabilities you specify in your requests!")
    
    guide.completed_steps.add(3)
    guide.wait_for_user()


async def step_4_cross_framework_communication(guide: TutorialGuide):
    """Step 4: Cross-Framework Communication"""
    guide.print_step_header(
        4,
        "Cross-Framework Communication",
        "See how agents from different frameworks can work together seamlessly."
    )
    
    available_frameworks = await guide.tutorial_context.get("available_frameworks", [])
    
    guide.print_code_example(
        "Agent-to-Agent Communication",
        """# Any agent can invoke another agent using the invoke_agent method
result = await agent.invoke_agent(
    capabilities=["coding", "python"],
    task={"input": "Write a hello world function"},
    routing_strategy="capability_match"
)

print(f"Cross-agent result: {result.result}")

# Or using the MeshAI tool from within agent frameworks
# This works automatically when agents have MeshAI tools registered"""
    )
    
    if len(available_frameworks) >= 2:
        print(f"🎉 You have {len(available_frameworks)} frameworks available!")
        print("This means you can create agents that collaborate across frameworks:")
        
        for i, fw in enumerate(available_frameworks[:3]):
            print(f"   {i+1}. {fw.title()} Agent → Can call agents from other frameworks")
            
        guide.print_code_example(
            "Multi-Framework Workflow Example",
            """# Create agents from different frameworks
openai_agent = OpenAIMeshAgent(...)  # Strategic thinking
claude_agent = AnthropicMeshAgent(...)  # Analysis  
langchain_agent = LangChainMeshAgent(...)  # Tool execution

# Register all agents
await registry.register_agent(openai_agent)
await registry.register_agent(claude_agent) 
await registry.register_agent(langchain_agent)

# Now any agent can invoke others!
# OpenAI agent can ask Claude agent to analyze something
analysis_result = await openai_agent.invoke_agent(
    capabilities=["analysis"],
    task={"input": "Analyze this data..."}
)

# Claude agent can ask LangChain agent to execute tools
tool_result = await claude_agent.invoke_agent(
    capabilities=["tools", "execution"],
    task={"input": "Calculate 2+2 using calculator tool"}
)"""
        )
        
    else:
        print("ℹ️ With just one framework, you can still see the concept:")
        print("   • Agents expose their capabilities to MeshAI")
        print("   • Other agents can discover and invoke them")
        print("   • MeshAI handles routing and communication")
        print("   • Works the same whether agents use OpenAI, Claude, LangChain, etc.")
    
    print("\n🌐 Communication Features:")
    print("   • Automatic agent discovery and routing")
    print("   • Context sharing between agents")
    print("   • Error handling and fallbacks")
    print("   • Load balancing across multiple agents")
    
    guide.completed_steps.add(4)
    guide.wait_for_user()


async def step_5_multi_agent_workflows(guide: TutorialGuide):
    """Step 5: Building Multi-Agent Workflows"""
    guide.print_step_header(
        5,
        "Building Multi-Agent Workflows",
        "Create sophisticated workflows with multiple agents working together."
    )
    
    guide.print_code_example(
        "Simple Workflow Example",
        """async def content_creation_workflow(topic: str):
    # Step 1: Research the topic
    research = await research_agent.handle_task(
        TaskData(input=f"Research information about {topic}"),
        context
    )
    
    # Step 2: Write content based on research
    content = await writing_agent.handle_task(
        TaskData(input=f"Write an article about {topic} using this research: {research['result']}"),
        context  
    )
    
    # Step 3: Review and improve
    final_content = await review_agent.handle_task(
        TaskData(input=f"Review and improve this content: {content['result']}"),
        context
    )
    
    return final_content

# Run the workflow
result = await content_creation_workflow("AI Agent Orchestration")
print(f"Workflow result: {result['result']}")"""
    )
    
    # Demonstrate with available agents
    agent = await guide.tutorial_context.get("first_agent")
    if agent:
        print("🚀 Let's run a simple workflow with your agent:")
        
        try:
            context = MeshContext()
            
            # Step 1: Simple task
            task1 = TaskData(input="What are the benefits of using AI agents?")
            result1 = await agent.handle_task(task1, context)
            print(f"✅ Step 1 result: {result1['result'][:100]}...")
            
            # Step 2: Follow-up task (simulates agent-to-agent communication)
            task2 = TaskData(input="Can you elaborate on the scalability benefits mentioned?")
            result2 = await agent.handle_task(task2, context)
            print(f"✅ Step 2 result: {result2['result'][:100]}...")
            
            # Show context preservation
            conversation_history = await context.get("conversation_history", [])
            print(f"📝 Conversation history: {len(conversation_history)} messages")
            
        except Exception as e:
            print(f"❌ Workflow demo failed: {e}")
    
    guide.print_code_example(
        "Advanced Workflow Patterns",
        """# Parallel execution
async def parallel_analysis(data):
    tasks = [
        statistical_agent.handle_task(TaskData(input=data), context),
        sentiment_agent.handle_task(TaskData(input=data), context),
        keyword_agent.handle_task(TaskData(input=data), context)
    ]
    
    results = await asyncio.gather(*tasks)
    return combine_results(results)

# Conditional routing
async def smart_routing(query):
    if "code" in query.lower():
        return await coding_agent.handle_task(TaskData(input=query), context)
    elif "analyze" in query.lower():
        return await analysis_agent.handle_task(TaskData(input=query), context)
    else:
        return await general_agent.handle_task(TaskData(input=query), context)

# Error handling with fallbacks
async def resilient_task(task_data, primary_agent, fallback_agents):
    for agent in [primary_agent] + fallback_agents:
        try:
            return await agent.handle_task(task_data, context)
        except Exception as e:
            print(f"Agent {agent.name} failed: {e}")
            continue
    raise Exception("All agents failed")"""
    )
    
    print("\n🎯 Workflow Patterns:")
    print("   • Sequential: Steps that build on each other")
    print("   • Parallel: Multiple agents working simultaneously")
    print("   • Conditional: Dynamic routing based on content")
    print("   • Hierarchical: Agents managing other agents")
    
    guide.completed_steps.add(5)
    guide.wait_for_user()


async def step_6_monitoring_debugging(guide: TutorialGuide):
    """Step 6: Monitoring and Debugging"""
    guide.print_step_header(
        6,
        "Monitoring and Debugging",
        "Learn how to monitor agent performance and debug issues."
    )
    
    guide.print_code_example(
        "Monitoring Agents",
        """# Check agent health
async def check_agent_health(agent):
    try:
        test_task = TaskData(input="Health check")
        result = await asyncio.wait_for(
            agent.handle_task(test_task, MeshContext()),
            timeout=10.0
        )
        return True
    except Exception:
        return False

# Get performance metrics
if hasattr(agent, 'get_usage_stats'):
    stats = await agent.get_usage_stats()
    print(f"Total requests: {stats.get('total_requests', 0)}")
    print(f"Average response time: {stats.get('avg_response_time', 0)}s")

# Registry monitoring
registry_stats = await registry.get_stats()
print(f"Active agents: {registry_stats.get('active_agents', 0)}")
print(f"Total requests: {registry_stats.get('total_requests', 0)}")"""
    )
    
    # Demonstrate monitoring with our agent
    agent = await guide.tutorial_context.get("first_agent")
    if agent:
        print("📊 Monitoring your agent:")
        
        # Health check
        try:
            start_time = time.time()
            test_task = TaskData(input="Health check - respond with OK")
            test_context = MeshContext()
            result = await asyncio.wait_for(
                agent.handle_task(test_task, test_context),
                timeout=10.0
            )
            response_time = time.time() - start_time
            
            print(f"✅ Health check passed")
            print(f"⏱️ Response time: {response_time:.2f}s")
            print(f"📝 Response: {result.get('result', 'No result')[:50]}...")
            
        except asyncio.TimeoutError:
            print("❌ Health check timed out")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            
        # Performance stats
        try:
            if hasattr(agent, 'get_usage_stats'):
                stats = await agent.get_usage_stats()
                print(f"📈 Usage stats: {stats}")
            else:
                print("ℹ️ Agent doesn't provide usage statistics")
        except Exception as e:
            print(f"⚠️ Could not get usage stats: {e}")
    
    guide.print_code_example(
        "Debugging Techniques",
        """# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use structured logging
import structlog
logger = structlog.get_logger(__name__)
logger.info("Agent task started", agent_id=agent.agent_id, task_id="123")

# Context inspection
print("Context contents:")
for key in await context.keys():
    value = await context.get(key)
    print(f"  {key}: {type(value).__name__}")

# Error handling
try:
    result = await agent.handle_task(task_data, context)
except TaskExecutionError as e:
    logger.error(f"Task execution failed: {e}")
    # Implement fallback logic
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors"""
    )
    
    print("\n🔧 Debugging Tools:")
    print("   • Structured logging with context")
    print("   • Performance metrics and timing")
    print("   • Context inspection and debugging")
    print("   • Error classification and handling")
    print("   • Health checks and monitoring")
    
    guide.completed_steps.add(6)
    guide.wait_for_user()


async def step_7_production_deployment(guide: TutorialGuide):
    """Step 7: Production Deployment"""
    guide.print_step_header(
        7,
        "Production Deployment",
        "Best practices for deploying MeshAI in production environments."
    )
    
    guide.print_code_example(
        "Production Configuration",
        """# config.py
from meshai.core.config import MeshConfig

config = MeshConfig(
    # Security
    api_key_encryption=True,
    enable_audit_logging=True,
    
    # Performance
    max_concurrent_tasks=100,
    default_timeout=30,
    connection_pool_size=20,
    
    # Reliability
    enable_circuit_breaker=True,
    max_retries=3,
    fallback_agents=True,
    
    # Monitoring
    metrics_enabled=True,
    log_level="INFO",
    performance_tracking=True,
    
    # Scaling
    auto_scaling_enabled=True,
    max_agents_per_capability=5,
    load_balancing_strategy="round_robin"
)"""
    )
    
    guide.print_code_example(
        "Docker Deployment",
        """# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set environment variables
ENV MESHAI_LOG_LEVEL=INFO
ENV MESHAI_METRICS_ENABLED=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000
CMD ["python", "-m", "meshai.server"]"""
    )
    
    guide.print_code_example(
        "Kubernetes Deployment",
        """# meshai-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: meshai-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: meshai-platform
  template:
    metadata:
      labels:
        app: meshai-platform
    spec:
      containers:
      - name: meshai
        image: meshai-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: openai-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10"""
    )
    
    print("🚀 Production Checklist:")
    print("   ✅ Security: API key management, encryption, audit logging")
    print("   ✅ Reliability: Circuit breakers, retries, fallback agents") 
    print("   ✅ Performance: Connection pooling, caching, load balancing")
    print("   ✅ Monitoring: Metrics, logging, alerting, health checks")
    print("   ✅ Scaling: Auto-scaling, multiple replicas, resource limits")
    
    print("\n🌐 Deployment Options:")
    print("   • Docker containers for consistent environments")
    print("   • Kubernetes for orchestration and scaling")
    print("   • Cloud platforms (AWS, GCP, Azure)")
    print("   • Serverless functions for event-driven workflows")
    
    print("\n📊 Monitoring in Production:")
    print("   • Application metrics (response time, error rate)")
    print("   • Resource metrics (CPU, memory, network)")
    print("   • Business metrics (tasks completed, user satisfaction)")
    print("   • Custom dashboards and alerting rules")
    
    guide.completed_steps.add(7)
    guide.wait_for_user()


async def tutorial_completion(guide: TutorialGuide):
    """Tutorial completion and next steps"""
    print(f"\n{'🎉'*20}")
    print("CONGRATULATIONS! You've completed the MeshAI tutorial!")
    print(f"{'🎉'*20}")
    
    print(f"\n📚 What you've learned:")
    for step_num in sorted(guide.completed_steps):
        steps = {
            1: "✅ Setup and Installation",
            2: "✅ Creating Your First Agent", 
            3: "✅ Agent Registration and Discovery",
            4: "✅ Cross-Framework Communication",
            5: "✅ Building Multi-Agent Workflows",
            6: "✅ Monitoring and Debugging",
            7: "✅ Production Deployment"
        }
        print(f"   {steps.get(step_num, f'✅ Step {step_num}')}")
    
    completion_rate = (len(guide.completed_steps) / 7) * 100
    print(f"\n📊 Tutorial completion: {completion_rate:.0f}%")
    
    print(f"\n🚀 Next Steps:")
    print("1. 📖 Explore the advanced workflow examples:")
    print("   python examples/advanced_workflows.py")
    print("2. 🔧 Try framework-specific examples:")
    print("   python examples/framework_adapters.py")
    print("3. 📚 Read the comprehensive documentation:")
    print("   docs/framework-adapters.md")
    print("4. 🌐 Check out the online docs:")
    print("   https://docs.meshai.dev")
    print("5. 💬 Join the community:")
    print("   https://github.com/meshailabs/meshai-sdk/discussions")
    
    print(f"\n💡 Pro Tips for Your MeshAI Journey:")
    print("   • Start simple with one agent, then add complexity")
    print("   • Use capability-based routing for flexibility")
    print("   • Implement proper error handling and fallbacks")
    print("   • Monitor performance and costs in production")
    print("   • Contribute back to the community!")
    
    print(f"\n📧 Need Help?")
    print("   • Documentation: https://docs.meshai.dev")
    print("   • GitHub Issues: https://github.com/meshailabs/meshai-sdk/issues")
    print("   • Community: https://github.com/meshailabs/meshai-sdk/discussions")
    print("   • Email: support@meshai.dev")
    
    print(f"\n👏 Thank you for learning MeshAI!")
    print("Happy building! 🚀")


async def main():
    """Run the interactive tutorial"""
    guide = TutorialGuide()
    
    print("This tutorial will take approximately 15-20 minutes.")
    print("You can interrupt at any time with Ctrl+C and resume later.\n")
    
    try:
        guide.wait_for_user("Ready to start? Press Enter...")
        
        # Run all tutorial steps
        await step_1_setup_and_installation(guide)
        await step_2_first_agent(guide)
        await step_3_agent_registration(guide)
        await step_4_cross_framework_communication(guide)
        await step_5_multi_agent_workflows(guide)
        await step_6_monitoring_debugging(guide)
        await step_7_production_deployment(guide)
        
        await tutorial_completion(guide)
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Tutorial paused!")
        print(f"Progress saved: {len(guide.completed_steps)}/7 steps completed")
        print("Run this script again anytime to continue!")


if __name__ == "__main__":
    # Run the interactive tutorial
    asyncio.run(main())