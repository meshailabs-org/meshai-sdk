# MeshAI SDK API Reference

Complete reference documentation for the MeshAI SDK with practical examples and usage patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Agent Management](#agent-management)
4. [Framework Adapters](#framework-adapters)
5. [Context and State](#context-and-state)
6. [Task Execution](#task-execution)
7. [Registry Operations](#registry-operations)
8. [Runtime Engine](#runtime-engine)
9. [Error Handling](#error-handling)
10. [Configuration](#configuration)
11. [Monitoring and Metrics](#monitoring-and-metrics)
12. [Examples](#examples)

## Quick Start

### Installation

```bash
# Install core SDK
pip install meshai-sdk

# Install with specific frameworks
pip install meshai-sdk[openai,anthropic,google]

# Install everything
pip install meshai-sdk[all]
```

### Basic Usage

```python
import asyncio
from meshai.adapters.openai_adapter import OpenAIMeshAgent
from meshai.core.registry import MeshRegistry
from meshai.core.schemas import TaskData

async def main():
    # Create an agent
    agent = OpenAIMeshAgent(
        model="gpt-3.5-turbo",
        agent_id="my-assistant",
        name="My AI Assistant",
        capabilities=["conversation", "help"]
    )
    
    # Register the agent
    registry = MeshRegistry()
    await registry.register_agent(agent)
    
    # Execute a task
    task = TaskData(input="Hello, how can you help me?")
    result = await agent.handle_task(task, context)
    
    print(result['result'])

asyncio.run(main())
```

## Core Components

### MeshAgent Base Class

The foundation class for all MeshAI agents.

```python
from meshai.core.agent import MeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData

class CustomAgent(MeshAgent):
    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        super().__init__(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            framework="custom"
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        # Custom task processing logic
        return {
            "result": f"Processed: {task_data.input}",
            "type": "custom_response",
            "timestamp": datetime.utcnow().isoformat()
        }

# Usage
agent = CustomAgent(
    agent_id="custom-001",
    name="Custom Agent",
    capabilities=["custom-processing", "text-analysis"]
)
```

#### Methods

**`__init__(agent_id, name, capabilities, framework, config=None, **kwargs)`**

Initialize a new agent.

- `agent_id` (str): Unique identifier for the agent
- `name` (str): Human-readable name
- `capabilities` (List[str]): List of agent capabilities
- `framework` (str): Framework identifier
- `config` (Optional[MeshConfig]): Configuration object
- `**kwargs`: Additional metadata

**`async handle_task(task_data, context)`**

Process a task and return results.

- `task_data` (TaskData): Task information and parameters
- `context` (MeshContext): Shared context for state management
- Returns: `Dict[str, Any]` with task results

**`async invoke_agent(capabilities, task, routing_strategy="capability_match")`**

Invoke another agent through the MeshAI runtime.

- `capabilities` (List[str]): Required capabilities
- `task` (Dict[str, Any]): Task data to send
- `routing_strategy` (str): How to select target agent
- Returns: `TaskResult` with execution results

### TaskData Schema

Represents a task to be executed by an agent.

```python
from meshai.core.schemas import TaskData

# Simple task
task = TaskData(
    input="Write a hello world function",
    parameters={"language": "python", "style": "functional"}
)

# Complex task with metadata
task = TaskData(
    input={
        "operation": "analyze_data",
        "dataset": "customer_data.csv",
        "metrics": ["revenue", "churn", "satisfaction"]
    },
    parameters={
        "time_period": "last_quarter",
        "include_predictions": True,
        "output_format": "json"
    }
)
```

#### Fields

- `input` (Union[str, Dict[str, Any]]): Task input data
- `parameters` (Dict[str, Any]): Additional task parameters
- `metadata` (Optional[Dict[str, Any]]): Task metadata

### TaskResult Schema

Represents the result of task execution.

```python
from meshai.core.schemas import TaskResult

# TaskResult is automatically created by the runtime
result = TaskResult(
    task_id="task-123",
    agent_id="agent-456", 
    status="completed",  # "pending", "running", "completed", "failed"
    result={"output": "Hello, World!", "metrics": {...}},
    error=None,
    start_time=datetime.utcnow(),
    end_time=datetime.utcnow(),
    execution_time=0.5
)
```

## Agent Management

### Creating Agents

#### OpenAI Agent

```python
from meshai.adapters.openai_adapter import OpenAIMeshAgent

agent = OpenAIMeshAgent(
    model="gpt-4",
    agent_id="openai-expert",
    name="OpenAI Expert",
    capabilities=["reasoning", "coding", "analysis"],
    api_key="your-api-key",  # Optional, uses env var
    temperature=0.7,
    max_tokens=2000,
    system_prompt="You are an expert AI assistant."
)

# Get model information
info = agent.get_model_info()
print(f"Model: {info['model']}, Provider: {info['provider']}")

# Stream response
async for chunk in agent.stream_response("Tell me about AI", context):
    print(chunk, end="")
```

#### Anthropic Claude Agent

```python
from meshai.adapters.anthropic_adapter import AnthropicMeshAgent

agent = AnthropicMeshAgent(
    model="claude-3-sonnet-20240229",
    agent_id="claude-analyst",
    name="Claude Analyst", 
    capabilities=["analysis", "research", "writing"],
    api_key="your-api-key",
    max_tokens=4000,
    temperature=0.3
)

# Add custom tool
tool_def = {
    "name": "data_processor",
    "description": "Process structured data",
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {"type": "string"},
            "operation": {"type": "string"}
        },
        "required": ["data", "operation"]
    }
}
agent.add_tool(tool_def)
```

#### LangChain Agent

```python
from meshai.adapters.langchain_adapter import LangChainMeshAgent
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Create tools
def calculator(expression: str) -> str:
    return str(eval(expression))

calc_tool = Tool(
    name="calculator",
    description="Calculate mathematical expressions",
    func=calculator
)

# Create LangChain agent
llm = ChatOpenAI()
langchain_agent = initialize_agent(
    tools=[calc_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Wrap with MeshAI
mesh_agent = LangChainMeshAgent(
    langchain_agent=langchain_agent,
    agent_id="langchain-calculator",
    name="Calculator Agent",
    capabilities=["mathematics", "calculations"]
)

# Add custom function as tool
def text_analyzer(text: str) -> str:
    return f"Analysis: {len(text)} characters, {len(text.split())} words"

mesh_agent.add_tool(text_analyzer, "text_analyzer", "Analyze text statistics")
```

#### CrewAI Agent

```python
from meshai.adapters.crewai_adapter import CrewAIMeshAgent
from crewai import Agent, Task, Crew

# Create CrewAI agents
researcher = Agent(
    role="Research Analyst",
    goal="Research and analyze topics thoroughly",
    backstory="Expert researcher with analytical skills"
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging content",
    backstory="Skilled writer with technical expertise"
)

# Create tasks
research_task = Task(
    description="Research AI agent frameworks",
    agent=researcher,
    expected_output="Comprehensive research report"
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task],
    verbose=False
)

# Wrap with MeshAI
mesh_agent = CrewAIMeshAgent(
    crewai_component=crew,
    agent_id="research-team",
    name="Research Team",
    capabilities=["research", "writing", "collaboration"]
)

# Get crew information
crew_info = mesh_agent.get_crew_info()
print(f"Crew type: {crew_info['type']}, Agents: {crew_info['agent_count']}")
```

### Agent Registration

```python
from meshai.core.registry import MeshRegistry

registry = MeshRegistry()

# Register single agent
await registry.register_agent(agent)

# Register multiple agents
agents = [openai_agent, claude_agent, langchain_agent]
for agent in agents:
    await registry.register_agent(agent)

# Check registration status
is_registered = await registry.is_agent_registered("agent-id")

# Unregister agent
await registry.unregister_agent("agent-id")
```

### Agent Discovery

```python
# Find all agents
all_agents = await registry.discover_agents()

# Find by capabilities
coding_agents = await registry.discover_agents(capabilities=["coding"])
analysis_agents = await registry.discover_agents(capabilities=["analysis"])

# Find by framework
openai_agents = await registry.discover_agents(framework="openai")

# Find by multiple criteria
expert_agents = await registry.discover_agents(
    capabilities=["expert", "analysis"],
    framework=["anthropic", "openai"],
    match_all_capabilities=False  # OR matching
)

# Get agent details
agent_info = await registry.get_agent_info("agent-id")
print(f"Name: {agent_info['name']}, Status: {agent_info['status']}")
```

## Framework Adapters

### Available Adapters

```python
from meshai.adapters import get_available_adapters, is_adapter_available

# Check what's available
adapters = get_available_adapters()
print(f"Available adapters: {list(adapters.keys())}")

# Check specific framework
if is_adapter_available("openai"):
    from meshai.adapters.openai_adapter import OpenAIMeshAgent
```

### Framework-Specific Features

#### OpenAI Function Calling

```python
agent = OpenAIMeshAgent(model="gpt-4")

# Function calls are handled automatically
task = TaskData(
    input="Find an agent that can write Python code and ask it to create a web scraper",
    parameters={}
)

result = await agent.handle_task(task, context)
# Agent automatically uses invoke_meshai_agent function if needed
```

#### Anthropic Tool Use

```python
agent = AnthropicMeshAgent(model="claude-3-sonnet-20240229")

# Tools are automatically prepared
result = await agent.handle_task(task, context)
if "tool_results" in result:
    for tool_result in result["tool_results"]:
        print(f"Tool: {tool_result['tool_name']}, Success: {tool_result['success']}")
```

#### Google Function Calling

```python
agent = GoogleMeshAgent(model="gemini-pro")

# Multimodal task with function calling
task = TaskData(
    input="Analyze this data and find an expert agent to provide insights",
    parameters={"temperature": 0.4}
)

result = await agent.handle_task(task, context)
```

## Context and State

### MeshContext Usage

```python
from meshai.core.context import MeshContext

context = MeshContext()

# Set simple values
await context.set("user_id", "user123")
await context.set("session_data", {"started": datetime.utcnow()})

# Get values
user_id = await context.get("user_id")
session_data = await context.get("session_data", {})  # Default value

# Check existence
has_user = await context.exists("user_id")

# Remove values
await context.remove("temporary_data")

# List all keys
keys = await context.keys()

# Clear all data
await context.clear()
```

### Conversation History

```python
# Conversation history is automatically managed
conversation = await context.get("conversation_history", [])

# Each message has structure:
# {
#     "type": "human" | "ai",
#     "content": "message content",
#     "timestamp": "2024-01-01T12:00:00Z",
#     "source": "framework_name",
#     "metadata": {...}
# }

# Manually add to conversation
await context.set("conversation_history", conversation + [
    {
        "type": "human",
        "content": "Hello",
        "timestamp": datetime.utcnow().isoformat()
    }
])
```

### Shared Data

```python
# Store data accessible to all agents
await context.set("shared_data", {
    "project_info": {
        "name": "AI Assistant",
        "version": "1.0.0"
    },
    "user_preferences": {
        "language": "en",
        "format": "detailed"
    }
})

# Agents can access shared data
shared = await context.get("shared_data", {})
project_name = shared.get("project_info", {}).get("name")
```

## Task Execution

### Direct Task Execution

```python
from meshai.core.schemas import TaskData

# Simple task
task = TaskData(input="Hello, how are you?")
result = await agent.handle_task(task, context)

# Complex task
task = TaskData(
    input={
        "operation": "code_generation",
        "requirements": [
            "Create a REST API",
            "Use FastAPI framework",
            "Include authentication",
            "Add database integration"
        ]
    },
    parameters={
        "programming_language": "python",
        "database": "postgresql",
        "auth_method": "jwt",
        "include_tests": True
    }
)

result = await agent.handle_task(task, context)
```

### Cross-Agent Task Routing

```python
# Any agent can invoke other agents
result = await agent.invoke_agent(
    capabilities=["coding", "python"],
    task={
        "input": "Create a data analysis script",
        "requirements": ["pandas", "matplotlib", "statistical analysis"]
    },
    routing_strategy="capability_match"
)

# Check result
if result.status == "completed":
    print(f"Result: {result.result}")
else:
    print(f"Error: {result.error}")
```

### Batch Task Processing

```python
async def process_tasks_batch(agent, tasks, context):
    """Process multiple tasks concurrently"""
    
    async def process_one(task):
        try:
            return await agent.handle_task(task, context)
        except Exception as e:
            return {"error": str(e), "task": task.input}
    
    # Limit concurrency
    semaphore = asyncio.Semaphore(5)
    
    async def limited_process(task):
        async with semaphore:
            return await process_one(task)
    
    # Process all tasks
    results = await asyncio.gather(
        *[limited_process(task) for task in tasks],
        return_exceptions=True
    )
    
    return results

# Usage
tasks = [
    TaskData(input=f"Analyze data point {i}") 
    for i in range(10)
]

results = await process_tasks_batch(agent, tasks, context)
```

## Registry Operations

### MeshRegistry API

```python
from meshai.core.registry import MeshRegistry

registry = MeshRegistry()

# Agent management
await registry.register_agent(agent)
await registry.unregister_agent("agent-id")
agents = await registry.get_all_agents()

# Discovery
matching_agents = await registry.discover_agents(
    capabilities=["coding"],
    status="active",
    limit=5
)

# Health checking
health_status = await registry.check_agent_health("agent-id")
await registry.update_agent_status("agent-id", "active")

# Statistics
stats = await registry.get_stats()
print(f"Active agents: {stats['active_agents']}")
print(f"Total requests: {stats['total_requests']}")
```

### Registry Configuration

```python
from meshai.core.config import MeshConfig

config = MeshConfig(
    registry_url="http://localhost:8000",
    max_agents_per_capability=10,
    health_check_interval=30,
    agent_timeout=60
)

registry = MeshRegistry(config=config)
```

## Runtime Engine

### MeshRuntime Usage

```python
from meshai.core.runtime import MeshRuntime

runtime = MeshRuntime()

# Execute task with automatic agent selection
result = await runtime.execute_task(
    capabilities=["analysis", "data-science"],
    task_data=TaskData(input="Analyze customer churn data"),
    context=context,
    routing_strategy="performance_optimized"
)

# Execute with specific agent
result = await runtime.execute_task_with_agent(
    agent_id="data-analyst-001",
    task_data=task_data,
    context=context
)

# Streaming execution
async for chunk in runtime.execute_task_stream(
    capabilities=["text-generation"],
    task_data=TaskData(input="Write a story"),
    context=context
):
    print(chunk, end="")
```

### Routing Strategies

```python
# Available routing strategies:
# - "round_robin": Distribute evenly across agents
# - "capability_match": Best capability match
# - "performance_optimized": Route to fastest agent
# - "load_balanced": Consider current load
# - "sticky_session": Same agent for related tasks

result = await runtime.execute_task(
    capabilities=["coding"],
    task_data=task_data,
    context=context,
    routing_strategy="performance_optimized"
)
```

### Load Balancing

```python
# Configure load balancing
runtime.configure_load_balancing(
    strategy="weighted_round_robin",
    weights={
        "fast-agent": 3.0,
        "balanced-agent": 2.0, 
        "slow-agent": 1.0
    }
)

# Monitor agent performance
performance_stats = await runtime.get_agent_performance("agent-id")
print(f"Average response time: {performance_stats['avg_response_time']}")
print(f"Success rate: {performance_stats['success_rate']}")
```

## Error Handling

### Exception Types

```python
from meshai.exceptions.base import (
    TaskExecutionError,
    ConfigurationError,
    ValidationError,
    AgentNotFoundError,
    RegistryError
)

try:
    result = await agent.handle_task(task_data, context)
except TaskExecutionError as e:
    print(f"Task failed: {e}")
    # Implement retry logic
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    # Check API keys, settings
except AgentNotFoundError as e:
    print(f"Agent not available: {e}")
    # Try alternative agent
```

### Retry Logic

```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3)
)
async def resilient_task_execution(agent, task_data, context):
    return await agent.handle_task(task_data, context)

# Usage
try:
    result = await resilient_task_execution(agent, task_data, context)
except Exception as e:
    print(f"All retries exhausted: {e}")
```

### Fallback Strategies

```python
async def execute_with_fallback(primary_agent, fallback_agents, task_data, context):
    """Execute task with fallback agents"""
    
    agents = [primary_agent] + fallback_agents
    
    for i, agent in enumerate(agents):
        try:
            result = await agent.handle_task(task_data, context)
            if i > 0:
                print(f"Fallback agent {agent.name} succeeded")
            return result
        except Exception as e:
            if i == len(agents) - 1:  # Last agent
                raise e
            print(f"Agent {agent.name} failed, trying fallback: {e}")

# Usage
result = await execute_with_fallback(
    primary_agent=openai_agent,
    fallback_agents=[claude_agent, local_agent],
    task_data=task_data,
    context=context
)
```

## Configuration

### MeshConfig Class

```python
from meshai.core.config import MeshConfig

config = MeshConfig(
    # API Keys (optional, can use env vars)
    openai_api_key="your-key",
    anthropic_api_key="your-key",
    google_api_key="your-key",
    
    # Service URLs
    registry_url="https://api.meshai.dev/registry",
    runtime_url="https://api.meshai.dev/runtime",
    
    # Performance Settings
    default_timeout=30,
    max_retries=3,
    connection_pool_size=10,
    
    # Behavioral Settings
    default_temperature=0.7,
    default_max_tokens=1000,
    enable_streaming=True,
    
    # Logging
    log_level="INFO",
    enable_debug=False,
    log_format="structured",
    
    # Security
    enable_api_key_encryption=True,
    audit_logging=True,
    
    # Features
    enable_metrics=True,
    enable_tracing=True,
    cache_enabled=True
)

# Use config with agents
agent = OpenAIMeshAgent(config=config)
```

### Environment Variables

```bash
# Core Configuration
export MESHAI_LOG_LEVEL=INFO
export MESHAI_REGISTRY_URL=https://api.meshai.dev/registry
export MESHAI_RUNTIME_URL=https://api.meshai.dev/runtime

# API Keys
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key
export GOOGLE_API_KEY=your-google-key

# AWS (for Bedrock)
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-east-1

# Performance
export MESHAI_DEFAULT_TIMEOUT=30
export MESHAI_MAX_RETRIES=3
export MESHAI_CONNECTION_POOL_SIZE=20

# Features
export MESHAI_ENABLE_METRICS=true
export MESHAI_ENABLE_CACHING=true
export MESHAI_CACHE_TTL=3600
```

## Monitoring and Metrics

### Performance Metrics

```python
# Get agent usage statistics
stats = await agent.get_usage_stats()
print(f"Total requests: {stats.get('total_requests', 0)}")
print(f"Average response time: {stats.get('avg_response_time', 0):.2f}s")
print(f"Success rate: {stats.get('success_rate', 0):.1%}")

# Registry statistics
registry_stats = await registry.get_stats()
print(f"Active agents: {registry_stats.get('active_agents', 0)}")
print(f"Total capabilities: {registry_stats.get('total_capabilities', 0)}")
```

### Health Checks

```python
async def health_check_agent(agent):
    """Check if agent is healthy"""
    try:
        test_task = TaskData(input="health check")
        result = await asyncio.wait_for(
            agent.handle_task(test_task, MeshContext()),
            timeout=10.0
        )
        return True
    except Exception:
        return False

# Check all agents
for agent in agents:
    healthy = await health_check_agent(agent)
    print(f"{agent.name}: {'✅' if healthy else '❌'}")
```

### Custom Metrics

```python
from meshai.utils.metrics import MetricsCollector

metrics = MetricsCollector()

# Track custom metrics
@metrics.track_latency("custom_operation")
async def custom_operation():
    # Your operation here
    await asyncio.sleep(1)
    metrics.increment("operations_completed")

# View metrics
print(f"Operations completed: {metrics.get_counter('operations_completed')}")
```

## Examples

### Basic Agent Interaction

```python
import asyncio
from meshai.adapters.openai_adapter import OpenAIMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData

async def basic_example():
    # Create agent
    agent = OpenAIMeshAgent(
        model="gpt-3.5-turbo",
        agent_id="assistant",
        name="Assistant",
        capabilities=["conversation", "help"]
    )
    
    # Create context
    context = MeshContext()
    
    # Execute task
    task = TaskData(input="Explain machine learning in simple terms")
    result = await agent.handle_task(task, context)
    
    print(f"Response: {result['result']}")

asyncio.run(basic_example())
```

### Multi-Agent Workflow

```python
async def workflow_example():
    # Create different agents
    researcher = OpenAIMeshAgent(
        model="gpt-4",
        agent_id="researcher",
        capabilities=["research", "analysis"]
    )
    
    writer = OpenAIMeshAgent(
        model="gpt-3.5-turbo",
        agent_id="writer", 
        capabilities=["writing", "content-creation"]
    )
    
    # Register agents
    registry = MeshRegistry()
    await registry.register_agent(researcher)
    await registry.register_agent(writer)
    
    # Shared context
    context = MeshContext()
    await context.set("project", "AI Agent Guide")
    
    # Step 1: Research
    research_task = TaskData(
        input="Research the latest trends in AI agent development"
    )
    research_result = await researcher.handle_task(research_task, context)
    
    # Step 2: Write based on research
    writing_task = TaskData(
        input="Write an executive summary based on the research findings"
    )
    final_result = await writer.handle_task(writing_task, context)
    
    print(f"Final result: {final_result['result']}")

asyncio.run(workflow_example())
```

### Cross-Framework Integration

```python
async def cross_framework_example():
    # OpenAI for planning
    planner = OpenAIMeshAgent(
        model="gpt-4",
        capabilities=["planning", "strategy"]
    )
    
    # Claude for analysis
    analyst = AnthropicMeshAgent(
        model="claude-3-sonnet-20240229", 
        capabilities=["analysis", "evaluation"]
    )
    
    # LangChain for tool execution
    from langchain_openai import ChatOpenAI
    from langchain.tools import Tool
    
    def calculator(expr: str) -> str:
        return str(eval(expr))
    
    tools = [Tool(name="calc", func=calculator, description="Calculate")]
    llm = ChatOpenAI()
    langchain_agent = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    
    executor = LangChainMeshAgent(
        langchain_agent=langchain_agent,
        capabilities=["tools", "calculation"]
    )
    
    # Register all
    registry = MeshRegistry()
    await registry.register_agent(planner)
    await registry.register_agent(analyst)
    await registry.register_agent(executor)
    
    # Workflow: Plan → Analyze → Execute
    context = MeshContext()
    
    # Planner creates strategy
    plan_task = TaskData(input="Create a plan to analyze quarterly sales data")
    plan_result = await planner.handle_task(plan_task, context)
    
    # Analyst reviews plan
    analysis_task = TaskData(input="Evaluate the proposed analysis plan")
    analysis_result = await analyst.handle_task(analysis_task, context)
    
    # Executor performs calculations  
    calc_task = TaskData(input="Calculate: (450000 - 380000) / 380000 * 100")
    calc_result = await executor.handle_task(calc_task, context)
    
    print(f"Growth rate: {calc_result['result']}")

asyncio.run(cross_framework_example())
```

### Production Deployment Example

```python
from meshai.core.config import MeshConfig
from meshai.utils.monitoring import setup_monitoring
import logging

async def production_example():
    # Production configuration
    config = MeshConfig(
        log_level="INFO",
        enable_metrics=True,
        max_retries=5,
        default_timeout=60,
        connection_pool_size=20,
        enable_api_key_encryption=True
    )
    
    # Setup monitoring
    setup_monitoring(
        metrics_endpoint="/metrics",
        health_endpoint="/health"
    )
    
    # Create production agents with error handling
    try:
        agent = OpenAIMeshAgent(
            model="gpt-4",
            config=config,
            capabilities=["production", "reliable"]
        )
        
        # Production registry with health checks
        registry = MeshRegistry(config=config)
        await registry.register_agent(agent)
        
        # Verify agent health before serving
        if await registry.check_agent_health(agent.agent_id):
            logging.info("Agent healthy and ready for production")
        else:
            logging.error("Agent health check failed")
            
    except Exception as e:
        logging.error(f"Production setup failed: {e}")
        # Implement fallback logic

asyncio.run(production_example())
```

## Best Practices

### 1. Agent Design

```python
# Good: Specific capabilities
agent = OpenAIMeshAgent(
    capabilities=["python-coding", "web-scraping", "data-analysis"]
)

# Avoid: Generic capabilities  
agent = OpenAIMeshAgent(
    capabilities=["general", "anything"]
)
```

### 2. Error Handling

```python
# Always implement proper error handling
try:
    result = await agent.handle_task(task_data, context)
except TaskExecutionError as e:
    # Log error and implement fallback
    logging.error(f"Task failed: {e}")
    result = await fallback_agent.handle_task(task_data, context)
```

### 3. Context Management

```python
# Clean up context periodically
conversation_history = await context.get("conversation_history", [])
if len(conversation_history) > 100:
    # Keep only recent messages
    await context.set("conversation_history", conversation_history[-50:])
```

### 4. Resource Management

```python
# Use connection pooling for high throughput
config = MeshConfig(connection_pool_size=50)

# Implement proper timeout handling
task_data = TaskData(
    input="Long running task",
    parameters={"timeout": 300}
)
```

### 5. Monitoring

```python
# Track performance metrics
start_time = time.time()
result = await agent.handle_task(task_data, context)
execution_time = time.time() - start_time

if execution_time > 10.0:
    logging.warning(f"Slow task execution: {execution_time:.2f}s")
```

## Support and Resources

- **Documentation**: [docs.meshai.dev](https://docs.meshai.dev)
- **GitHub**: [github.com/meshailabs/meshai-sdk](https://github.com/meshailabs/meshai-sdk)
- **Examples**: [github.com/meshailabs/meshai-examples](https://github.com/meshailabs/meshai-examples)
- **Community**: [GitHub Discussions](https://github.com/meshailabs/meshai-sdk/discussions)
- **Issues**: [GitHub Issues](https://github.com/meshailabs/meshai-sdk/issues)
- **Email**: support@meshai.dev

---

This API reference provides comprehensive coverage of the MeshAI SDK. For additional examples and tutorials, see the `examples/` directory in the SDK repository.