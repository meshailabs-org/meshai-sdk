# MeshAI SDK Examples

This directory contains examples showing how to use the MeshAI SDK with different frameworks and use cases.

## Getting Started

1. **Install the SDK**:
   ```bash
   pip install meshai-sdk
   
   # Or with specific framework support
   pip install meshai-sdk[langchain,openai,anthropic]
   ```

2. **Set up environment variables**:
   ```bash
   export MESHAI_REGISTRY_URL="http://localhost:8001"
   export MESHAI_RUNTIME_URL="http://localhost:8002"
   
   # API keys for AI services (optional)
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. **Start MeshAI services** (see main repository for deployment instructions)

## Examples

### 1. Basic Agent (`basic_agent.py`)

Shows how to create a simple MeshAI agent from scratch:

```bash
python basic_agent.py
```

Features:
- Custom agent implementation
- Task handling
- Context management
- Agent registration

### 2. LangChain Integration (`langchain_example.py`)

Demonstrates wrapping existing LangChain agents:

```bash
python langchain_example.py
```

Features:
- LangChain agent wrapping
- Tool integration
- Memory management
- Cross-framework communication

### 3. Multi-Agent Workflow (`multi_agent_workflow.py`)

Shows multiple agents coordinating on complex tasks:

```bash
python multi_agent_workflow.py
```

Features:
- Workflow coordination
- Agent collaboration
- AI service integration
- Context sharing

## Configuration

You can customize agent behavior through environment variables or configuration files:

```python
from meshai import MeshConfig

config = MeshConfig(
    registry_url="http://localhost:8001",
    runtime_url="http://localhost:8002",
    agent_port=8000,
    debug_mode=True
)
```

## Testing Examples

1. **Start an example agent**:
   ```bash
   python basic_agent.py
   ```

2. **In another terminal, test the agent**:
   ```bash
   curl -X POST http://localhost:8000/execute \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "test1",
       "task_type": "process", 
       "payload": {
         "text": "Hello MeshAI!"
       }
     }'
   ```

3. **Check agent health**:
   ```bash
   curl http://localhost:8000/health
   ```

## Framework-Specific Examples

### LangChain
- Custom tools and chains
- Memory integration
- Agent executors

### CrewAI  
- Multi-agent crews
- Role-based collaboration
- Task delegation

### AI Services
- OpenAI GPT models
- Anthropic Claude
- Google Gemini
- Amazon Bedrock

## Advanced Usage

### Custom Capabilities

```python
@register_agent(
    capabilities=["data-analysis", "visualization", "reporting"],
    name="Data Analyst Agent"
)
class DataAnalystAgent(MeshAgent):
    async def handle_task(self, task_data, context):
        # Custom data analysis logic
        return {"analysis": "results"}
```

### Context Sharing

```python
# Store data for other agents
await context.set("analysis_results", results)

# Retrieve shared data
previous_results = await context.get("analysis_results")
```

### Agent Invocation

```python
# Invoke other agents
result = await self.invoke_agent(
    capabilities=["text-generation", "summarization"],
    task={"input": "Summarize these findings..."}
)
```

## CLI Usage

The SDK includes a CLI for management tasks:

```bash
# List available agents
meshai agents list

# Submit a task
meshai tasks submit --task-file task.json

# Check task status  
meshai tasks status --task-id abc123

# Test connectivity
meshai test --full
```

## Troubleshooting

1. **Connection Issues**: Check that MeshAI services are running and accessible
2. **Registration Failures**: Verify agent configuration and network connectivity
3. **Task Timeouts**: Increase timeout values or check agent performance
4. **Import Errors**: Install required framework dependencies

For more help, see the [MeshAI Documentation](https://docs.meshai.dev) or check the [GitHub Issues](https://github.com/meshailabs/meshai-sdk/issues).