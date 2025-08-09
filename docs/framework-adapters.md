# MeshAI Framework Adapters

MeshAI provides comprehensive adapters for popular AI frameworks, enabling seamless integration and cross-framework communication. This document covers all available adapters, their features, and usage examples.

## Overview

MeshAI framework adapters act as bridges between different AI frameworks and the MeshAI platform. They provide:

- **Unified Interface**: All frameworks work through the same MeshAI agent interface
- **Cross-Framework Communication**: Agents can invoke each other regardless of framework
- **Context Sharing**: Conversation history and data shared across frameworks
- **Tool Integration**: Each adapter provides MeshAI tools for cross-agent collaboration
- **Native Features**: Framework-specific features are preserved and enhanced

## Available Adapters

### 1. LangChain Adapter (`langchain_adapter.py`)

**Supported Components:**
- `AgentExecutor` - Full LangChain agents with tool execution
- `Runnable` - LangChain expression language components
- Individual `Agent` classes - Direct agent implementations

**Key Features:**
- Automatic tool registration with MeshAI
- Memory integration with LangChain's conversation buffer
- Support for both sync and async LangChain agents
- Tool adapter for custom functions

**Example:**
```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from meshai.adapters.langchain_adapter import LangChainMeshAgent

# Create LangChain agent
llm = ChatOpenAI()
langchain_agent = initialize_agent(
    tools=[calculator_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Wrap with MeshAI
mesh_agent = LangChainMeshAgent(
    langchain_agent=langchain_agent,
    agent_id="langchain-calculator",
    name="Calculator Agent",
    capabilities=["mathematics", "reasoning"]
)
```

**Dependencies:**
```bash
pip install meshai-sdk[langchain]
# Or manually: pip install langchain langchain-openai
```

### 2. CrewAI Adapter (`crewai_adapter.py`)

**Supported Components:**
- `Crew` - Multi-agent crews with coordinated tasks
- Individual `Agent` - Single CrewAI agents
- `Task` integration - CrewAI task management

**Key Features:**
- Full crew execution with multiple agents
- Individual agent wrapping for single-agent scenarios
- Context injection into agent backstories
- Cross-agent tool integration

**Example:**
```python
from crewai import Agent, Task, Crew
from meshai.adapters.crewai_adapter import CrewAIMeshAgent

# Create CrewAI crew
researcher = Agent(
    role="Research Analyst",
    goal="Research and analyze topics",
    backstory="Expert researcher with analytical skills"
)

crew = Crew(agents=[researcher], tasks=[research_task])

# Wrap with MeshAI
mesh_agent = CrewAIMeshAgent(
    crewai_component=crew,
    agent_id="research-crew",
    name="Research Team",
    capabilities=["research", "collaboration"]
)
```

**Dependencies:**
```bash
pip install meshai-sdk[crewai]
# Or manually: pip install crewai
```

### 3. AutoGen Adapter (`autogen_adapter.py`)

**Supported Components:**
- `ConversableAgent` - Standard AutoGen agents
- `GroupChatManager` - Multi-agent group chats
- `GroupChat` - Group conversation coordination

**Key Features:**
- Single agent and group chat support
- Dynamic capability detection from system messages
- Context enhancement for better conversations
- Automatic conversation management

**Example:**
```python
from autogen import ConversableAgent
from meshai.adapters.autogen_adapter import AutoGenMeshAgent

# Create AutoGen agent
assistant = ConversableAgent(
    name="helpful_assistant",
    system_message="You are a helpful AI assistant",
    human_input_mode="NEVER"
)

# Wrap with MeshAI
mesh_agent = AutoGenMeshAgent(
    autogen_component=assistant,
    agent_id="autogen-assistant", 
    name="AutoGen Assistant",
    capabilities=["conversation", "assistance"]
)
```

**Dependencies:**
```bash
pip install meshai-sdk[autogen]
# Or manually: pip install pyautogen
```

### 4. OpenAI Adapter (`openai_adapter.py`)

**Supported Models:**
- GPT-4 (all variants)
- GPT-3.5-turbo
- Custom OpenAI-compatible endpoints

**Key Features:**
- Function/tool calling support
- Streaming responses
- Usage tracking and metrics
- Custom system prompts
- Parameter customization (temperature, max_tokens, etc.)

**Example:**
```python
from meshai.adapters.openai_adapter import OpenAIMeshAgent

# Create OpenAI agent
mesh_agent = OpenAIMeshAgent(
    model="gpt-4",
    agent_id="openai-expert",
    name="OpenAI Expert",
    capabilities=["reasoning", "coding", "analysis"],
    temperature=0.7,
    max_tokens=1000,
    system_prompt="You are an AI expert specializing in agent architectures."
)
```

**Dependencies:**
```bash
pip install meshai-sdk[openai]
# Or manually: pip install openai
```

**Environment Variables:**
```bash
export OPENAI_API_KEY="your-api-key"
```

### 5. Anthropic Adapter (`anthropic_adapter.py`)

**Supported Models:**
- Claude 3 (Haiku, Sonnet, Opus)
- Claude 2.1, Claude 2.0
- Claude Instant

**Key Features:**
- Tool calling with Claude's native format
- Streaming responses
- Safety ratings analysis
- Custom tool definitions
- Advanced reasoning capabilities

**Example:**
```python
from meshai.adapters.anthropic_adapter import AnthropicMeshAgent

# Create Claude agent
mesh_agent = AnthropicMeshAgent(
    model="claude-3-sonnet-20240229",
    agent_id="claude-analyst",
    name="Claude Analyst",
    capabilities=["analysis", "reasoning", "research"],
    max_tokens=4000,
    temperature=0.3
)

# Add custom tool
custom_tool = {
    "name": "data_processor",
    "description": "Process structured data",
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {"type": "string"},
            "operation": {"type": "string"}
        }
    }
}
mesh_agent.add_tool(custom_tool)
```

**Dependencies:**
```bash
pip install meshai-sdk[anthropic]
# Or manually: pip install anthropic
```

**Environment Variables:**
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### 6. Google Adapter (`google_adapter.py`)

**Supported Models:**
- Gemini Pro, Gemini Pro Vision
- Gemini 1.5 Pro (with extended context)
- PaLM 2 models
- Vertex AI models

**Key Features:**
- Multimodal capabilities (text + images)
- Function calling support
- Safety ratings and content filtering
- Streaming responses
- Enterprise Vertex AI integration

**Example:**
```python
from meshai.adapters.google_adapter import GoogleMeshAgent, VertexAIMeshAgent

# Gemini agent
gemini_agent = GoogleMeshAgent(
    model="gemini-pro",
    agent_id="gemini-multimodal",
    name="Gemini Agent",
    capabilities=["multimodal", "reasoning", "vision"]
)

# Vertex AI agent (for enterprise)
vertex_agent = VertexAIMeshAgent(
    model="text-bison",
    project_id="your-gcp-project",
    location="us-central1",
    agent_id="vertex-enterprise",
    name="Vertex AI Agent",
    capabilities=["enterprise-ai", "scalable-inference"]
)
```

**Dependencies:**
```bash
pip install meshai-sdk[google]
# Or manually: pip install google-generativeai google-cloud-aiplatform
```

**Environment Variables:**
```bash
export GOOGLE_API_KEY="your-api-key"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 7. Amazon Adapter (`amazon_adapter.py`)

**Supported Services:**
- Amazon Bedrock (Claude, Titan, Jurassic, Cohere, Llama)
- Amazon SageMaker endpoints
- Custom model deployments

**Key Features:**
- Multi-model support with automatic format detection
- Tool calling for Claude models in Bedrock
- Streaming responses
- Enterprise security and compliance
- Custom SageMaker endpoint integration

**Example:**
```python
from meshai.adapters.amazon_adapter import BedrockMeshAgent, SageMakerMeshAgent

# Bedrock agent
bedrock_agent = BedrockMeshAgent(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    agent_id="bedrock-claude",
    name="Bedrock Claude",
    capabilities=["enterprise-ai", "analysis"],
    region="us-east-1"
)

# SageMaker agent
sagemaker_agent = SageMakerMeshAgent(
    endpoint_name="my-custom-model",
    agent_id="sagemaker-custom",
    name="Custom Model",
    capabilities=["custom-inference", "ml-models"],
    region="us-west-2"
)
```

**Dependencies:**
```bash
pip install meshai-sdk[amazon]
# Or manually: pip install boto3
```

**AWS Configuration:**
```bash
# Configure AWS credentials
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

## Common Features Across All Adapters

### 1. Cross-Framework Tool Integration

All adapters include a MeshAI tool that allows agents to invoke other agents:

```python
# This works from any framework adapter
result = await agent.invoke_agent(
    capabilities=["coding", "python"],
    task={"input": "Write a data analysis script"},
    routing_strategy="capability_match"
)
```

### 2. Context Management

Shared context across all frameworks:

```python
context = MeshContext()

# Set shared data
await context.set("project_info", {
    "name": "AI Agent System",
    "requirements": ["scalable", "secure"]
})

# All agents can access this context
result = await any_agent.handle_task(task_data, context)
```

### 3. Conversation History

Automatic conversation tracking:

```python
# Conversation history is maintained automatically
conversation = await context.get("conversation_history", [])

# Each message includes:
# - type: "human" or "ai"  
# - content: message text
# - timestamp: ISO format timestamp
# - source: framework name (e.g., "openai", "anthropic")
```

### 4. Performance Metrics

Built-in performance tracking:

```python
# Get usage statistics
stats = await agent.get_usage_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Average response time: {stats['avg_response_time']}")
```

### 5. Streaming Support

Most adapters support streaming responses:

```python
# Stream response chunks
async for chunk in agent.stream_response("Tell me about AI", context):
    print(chunk, end="", flush=True)
```

## Advanced Usage Patterns

### 1. Multi-Framework Workflows

```python
# Create agents from different frameworks
openai_agent = OpenAIMeshAgent(model="gpt-4", capabilities=["planning"])
claude_agent = AnthropicMeshAgent(model="claude-3-sonnet", capabilities=["analysis"])
crew_agent = CrewAIMeshAgent(crew=research_crew, capabilities=["research"])

# Register all agents
registry = MeshRegistry()
await registry.register_agent(openai_agent)
await registry.register_agent(claude_agent)
await registry.register_agent(crew_agent)

# Create workflow
context = MeshContext()

# Step 1: Planning with OpenAI
plan = await openai_agent.handle_task(
    TaskData(input="Create a research plan for AI safety"),
    context
)

# Step 2: Research with CrewAI
research = await crew_agent.handle_task(
    TaskData(input="Execute the research plan"),
    context
)

# Step 3: Analysis with Claude
analysis = await claude_agent.handle_task(
    TaskData(input="Analyze the research findings"),
    context
)
```

### 2. Custom Tool Integration

```python
# Define custom tools for specific frameworks

# LangChain tool
def custom_calculator(expression: str) -> str:
    return str(eval(expression))

langchain_agent.add_tool(
    custom_calculator,
    name="calculator",
    description="Calculate mathematical expressions"
)

# Anthropic tool
anthropic_tool = {
    "name": "file_analyzer",
    "description": "Analyze file contents",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "analysis_type": {"type": "string"}
        }
    }
}
claude_agent.add_tool(anthropic_tool)
```

### 3. Error Handling and Fallbacks

```python
async def robust_agent_call(agents: List[MeshAgent], task: TaskData, context: MeshContext):
    """Try multiple agents with fallbacks"""
    
    for agent in agents:
        try:
            result = await agent.handle_task(task, context)
            return result
        except TaskExecutionError as e:
            logger.warning(f"Agent {agent.agent_id} failed: {e}")
            continue
    
    raise TaskExecutionError("All agents failed")

# Usage
result = await robust_agent_call(
    [openai_agent, claude_agent, gemini_agent],
    task_data,
    context
)
```

## Configuration and Environment Setup

### Installation Options

```bash
# Install all adapters
pip install meshai-sdk[all]

# Install specific adapters
pip install meshai-sdk[openai,anthropic,google]

# Install core only
pip install meshai-sdk
```

### Environment Configuration

Create a `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Google
GOOGLE_API_KEY=your_google_key
GOOGLE_CLOUD_PROJECT=your_project_id

# AWS
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1
```

### Configuration Class

```python
from meshai.core.config import MeshConfig

config = MeshConfig(
    # API keys
    openai_api_key="your_key",
    anthropic_api_key="your_key",
    google_api_key="your_key",
    
    # Default settings
    default_temperature=0.7,
    default_max_tokens=1000,
    
    # Logging
    log_level="INFO",
    
    # Performance
    timeout_seconds=30,
    max_retries=3
)

# Use with agents
agent = OpenAIMeshAgent(config=config)
```

## Testing and Debugging

### Unit Tests

```python
import pytest
from meshai.adapters import OpenAIMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData

@pytest.mark.asyncio
async def test_openai_adapter():
    agent = OpenAIMeshAgent()
    context = MeshContext()
    task = TaskData(input="Hello world")
    
    result = await agent.handle_task(task, context)
    
    assert "result" in result
    assert result["type"] == "openai_response"
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use MeshAI's structured logging
import structlog
logger = structlog.get_logger(__name__)
logger.info("Debug information", agent_id="test", task="sample")
```

### Health Checks

```python
async def check_agent_health(agent: MeshAgent) -> bool:
    """Check if agent is responsive"""
    try:
        context = MeshContext()
        task = TaskData(input="Hello", parameters={})
        result = await asyncio.wait_for(
            agent.handle_task(task, context),
            timeout=10.0
        )
        return "result" in result
    except Exception:
        return False

# Check all agents
for agent in [openai_agent, claude_agent, gemini_agent]:
    healthy = await check_agent_health(agent)
    print(f"{agent.name}: {'✅' if healthy else '❌'}")
```

## Performance Optimization

### 1. Connection Pooling

```python
# Use connection pooling for high-throughput applications
from meshai.adapters.openai_adapter import OpenAIMeshAgent

agent = OpenAIMeshAgent(
    model="gpt-3.5-turbo",
    # Connection settings for high throughput
    max_connections=100,
    timeout=30.0
)
```

### 2. Caching

```python
from functools import lru_cache
from meshai.core.schemas import TaskData

@lru_cache(maxsize=1000)
async def cached_agent_call(agent_id: str, input_text: str):
    """Cache agent responses for identical inputs"""
    agent = get_agent_by_id(agent_id)
    context = MeshContext()
    task = TaskData(input=input_text)
    return await agent.handle_task(task, context)
```

### 3. Async Batch Processing

```python
async def process_tasks_batch(agent: MeshAgent, tasks: List[TaskData], context: MeshContext):
    """Process multiple tasks concurrently"""
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(10)
    
    async def process_one_task(task):
        async with semaphore:
            return await agent.handle_task(task, context)
    
    # Process all tasks concurrently
    tasks = [process_one_task(task) for task in tasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

## Best Practices

### 1. Framework Selection

- **LangChain**: Best for complex tool chains and established workflows
- **CrewAI**: Ideal for multi-agent collaboration and role-based tasks  
- **AutoGen**: Great for conversational multi-agent systems
- **OpenAI**: Reliable for general-purpose text generation and reasoning
- **Anthropic**: Excellent for analysis, safety-conscious applications
- **Google**: Best for multimodal tasks and enterprise deployments
- **Amazon**: Preferred for enterprise/regulated environments

### 2. Resource Management

```python
# Use context managers for proper cleanup
class ManagedAgent:
    def __init__(self, agent: MeshAgent):
        self.agent = agent
    
    async def __aenter__(self):
        return self.agent
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        if hasattr(self.agent, 'client'):
            await self.agent.client.close()

# Usage
async with ManagedAgent(openai_agent) as agent:
    result = await agent.handle_task(task, context)
```

### 3. Error Recovery

```python
class ResilientAgent:
    def __init__(self, primary: MeshAgent, fallbacks: List[MeshAgent]):
        self.primary = primary
        self.fallbacks = fallbacks
    
    async def handle_task(self, task: TaskData, context: MeshContext):
        agents = [self.primary] + self.fallbacks
        
        for i, agent in enumerate(agents):
            try:
                result = await agent.handle_task(task, context)
                if i > 0:  # Used fallback
                    logger.info(f"Fallback agent {agent.agent_id} succeeded")
                return result
            except Exception as e:
                if i == len(agents) - 1:  # Last agent failed
                    raise e
                logger.warning(f"Agent {agent.agent_id} failed, trying fallback: {e}")
```

### 4. Monitoring and Observability

```python
from meshai.utils.metrics import MetricsCollector

metrics = MetricsCollector()

# Track agent performance
@metrics.track_latency("agent_call")
@metrics.count_calls("agent_requests")
async def tracked_agent_call(agent: MeshAgent, task: TaskData, context: MeshContext):
    try:
        result = await agent.handle_task(task, context)
        metrics.increment("agent_success", tags={"agent_id": agent.agent_id})
        return result
    except Exception as e:
        metrics.increment("agent_error", tags={"agent_id": agent.agent_id, "error": type(e).__name__})
        raise
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   ConfigurationError: OpenAI API key required
   ```
   - Set environment variables or pass keys directly to agents
   - Verify key validity and permissions

2. **Import Errors**
   ```bash
   ImportError: LangChain is not installed
   ```
   - Install framework-specific dependencies: `pip install meshai-sdk[langchain]`

3. **Timeout Issues**
   ```python
   # Increase timeout for slow models
   agent = OpenAIMeshAgent(timeout=60.0)
   ```

4. **Memory Issues**
   ```python
   # Limit conversation history
   context = MeshContext(max_history_length=20)
   ```

5. **Rate Limiting**
   ```python
   # Add retry logic
   from tenacity import retry, wait_exponential
   
   @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
   async def resilient_call(agent, task, context):
       return await agent.handle_task(task, context)
   ```

### Debug Commands

```bash
# Test adapter availability
python -c "from meshai.adapters import get_available_adapters; print(get_available_adapters())"

# Test specific adapter
python -c "
from meshai.adapters.openai_adapter import OpenAIMeshAgent
agent = OpenAIMeshAgent()
print(f'Agent created: {agent.agent_id}')
"

# Run example
python examples/framework_adapters.py
```

## Contributing

To contribute new framework adapters:

1. Follow the `MeshAgent` interface in `src/meshai/core/agent.py`
2. Implement required methods: `handle_task()`, `__init__()`
3. Add optional methods: `stream_response()`, `get_model_info()`
4. Include comprehensive error handling
5. Add tool integration for cross-agent communication
6. Write tests and documentation
7. Update `__init__.py` to register the adapter

See existing adapters as reference implementations.

## Support

- 📖 Documentation: [docs.meshai.dev](https://docs.meshai.dev)
- 💬 Community: [GitHub Discussions](https://github.com/meshailabs/meshai-sdk/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/meshailabs/meshai-sdk/issues)
- 📧 Email: support@meshai.dev

---

For more information, see the [MeshAI SDK Documentation](../README.md).