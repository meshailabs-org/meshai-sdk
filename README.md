# MeshAI SDK

The official Python SDK for MeshAI - AI Agent Interoperability Platform.

MeshAI enables seamless communication and orchestration between AI agents built on different frameworks (LangChain, CrewAI, AutoGen, etc.).

## Quick Start

### Installation

```bash
# Core SDK
pip install meshai-sdk

# With LangChain support
pip install meshai-sdk[langchain]

# With CrewAI support  
pip install meshai-sdk[crewai]

# With all framework support
pip install meshai-sdk[all]
```

### Basic Usage

```python
from meshai import MeshAgent, register_agent
from langchain.agents import create_openai_functions_agent

# Create your agent using any framework
langchain_agent = create_openai_functions_agent(...)

# Wrap it with MeshAI
@register_agent(
    capabilities=["text-analysis", "summarization"],
    name="My LangChain Agent"
)
class MyAgent(MeshAgent):
    async def handle_task(self, task_data, context):
        # Your agent logic here
        result = await self.langchain_agent.arun(task_data["input"])
        return {"result": result}

# Agent automatically registers with MeshAI registry
agent = MyAgent()

# Invoke other agents through MeshAI
result = await agent.invoke_agent(
    capabilities=["code-generation"],
    task={"input": "Create a Python function to sort a list"}
)
```

## Framework Support

### LangChain
```python
from meshai.adapters import LangChainMeshAgent

agent = LangChainMeshAgent(
    langchain_agent=your_agent,
    agent_id="langchain-analyzer",
    capabilities=["text-analysis"]
)
```

### CrewAI
```python
from meshai.adapters import CrewAIMeshAgent

agent = CrewAIMeshAgent(
    crew=your_crew,
    agent_id="crewai-team",
    capabilities=["multi-step-analysis"]
)
```

### Direct API Integration
```python
from meshai.adapters import AnthropicMeshAgent, OpenAIMeshAgent

# Anthropic Claude
claude_agent = AnthropicMeshAgent(
    model="claude-3-sonnet-20240229",
    capabilities=["reasoning", "analysis"]
)

# OpenAI GPT
openai_agent = OpenAIMeshAgent(
    model="gpt-4-turbo-preview", 
    capabilities=["generation", "coding"]
)
```

## Features

- **Framework Agnostic**: Works with LangChain, CrewAI, AutoGen, and custom agents
- **Automatic Registration**: Agents self-register with the MeshAI registry
- **Smart Routing**: Intelligent agent selection based on capabilities and performance
- **Context Sharing**: Seamless context and memory management across agents
- **Built-in Monitoring**: Performance metrics and health monitoring
- **Easy Integration**: Minimal code changes to existing agents

## Architecture

```
Your Agent Framework (LangChain, CrewAI, etc.)
                 ↓
         MeshAI SDK Adapter
                 ↓
            MeshAI Runtime
                 ↓
        Agent Registry & Router
                 ↓
         Target Agent Framework
```

## Documentation

- [Getting Started Guide](https://docs.meshai.dev/getting-started)
- [Framework Adapters](https://docs.meshai.dev/adapters)
- [API Reference](https://docs.meshai.dev/api)
- [Examples](https://github.com/meshailabs/meshai-examples)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.