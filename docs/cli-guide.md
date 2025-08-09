# MeshAI CLI Guide

The MeshAI CLI (`meshai`) is a comprehensive command-line tool for developing, testing, and deploying AI agents across multiple frameworks.

## Installation

```bash
# Install MeshAI SDK with CLI
pip install meshai-sdk

# Or install from source
pip install -e .
```

After installation, the `meshai` command will be available globally.

## Quick Start

```bash
# Show help
meshai --help

# Create a new project
meshai init my-ai-project --template multi-agent

# Create a new agent
meshai agent create code-reviewer --framework anthropic

# Start development server
meshai dev server --watch

# Test an agent
meshai agent test my-agent-id

# Deploy to production
meshai deploy --environment production
```

## Commands Overview

### Project Management

#### `meshai init [PROJECT_NAME]`
Initialize a new MeshAI project with scaffolding and templates.

**Options:**
- `--template`, `-t`: Project template (simple, multi-agent, production, research)
- `--force`: Overwrite existing files

**Templates:**
- `simple`: Single-agent application
- `multi-agent`: Multi-agent workflow application  
- `production`: Production-ready with Docker and monitoring
- `research`: Research and analysis workflows

**Example:**
```bash
meshai init my-ai-app --template production
cd my-ai-app
cp .env.example .env  # Configure API keys
meshai dev server
```

### Agent Management

#### `meshai agent create NAME`
Create a new AI agent with framework-specific template.

**Options:**
- `--framework`, `-f`: AI framework (openai, anthropic, google, langchain, crewai, autogen)
- `--model`, `-m`: Model to use (optional, will prompt if not provided)
- `--capabilities`, `-c`: Agent capabilities (multiple allowed)
- `--output`, `-o`: Output directory (default: agents/)

**Example:**
```bash
meshai agent create data-analyst \
  --framework openai \
  --model gpt-4 \
  --capabilities analysis,visualization,insights
```

#### `meshai agent list`
List all registered agents in the current environment.

**Options:**
- `--format`, `-f`: Output format (table, json, yaml)

**Example:**
```bash
meshai agent list --format table
```

#### `meshai agent test AGENT_ID`
Test an agent with a sample message to verify functionality.

**Options:**
- `--message`, `-m`: Test message (default: "Hello, this is a test message")
- `--verbose`, `-v`: Show detailed output and timing

**Example:**
```bash
meshai agent test my-agent --verbose
```

### Development Tools

#### `meshai dev server`
Start a development server with hot-reload capabilities.

**Options:**
- `--port`, `-p`: Server port (default: 8080)
- `--watch`, `-w`: Watch for file changes and auto-restart
- `--debug`: Enable debug mode with detailed logging

**Example:**
```bash
meshai dev server --port 3000 --watch --debug
```

### Configuration

#### `meshai config validate`
Validate the current MeshAI configuration file.

**Options:**
- `--fix`: Automatically fix common configuration issues

**Example:**
```bash
meshai config validate --fix
```

### Deployment

#### `meshai deploy`
Deploy the current project to a target environment.

**Options:**
- `--environment`, `-e`: Target environment (development, staging, production)
- `--config`, `-c`: Custom configuration file

**Example:**
```bash
meshai deploy --environment production
```

### Monitoring

#### `meshai logs`
View application logs with filtering and following options.

**Options:**
- `--follow`, `-f`: Follow log output in real-time
- `--service`, `-s`: Filter by service name
- `--level`, `-l`: Log level filter (DEBUG, INFO, WARNING, ERROR)
- `--lines`, `-n`: Number of lines to show (default: 100)

**Example:**
```bash
meshai logs --follow --service registry --level INFO
```

## Configuration

MeshAI CLI uses YAML configuration files for project settings:

### `config.yaml`
```yaml
# MeshAI Configuration
meshai:
  # Registry settings
  registry:
    url: "http://localhost:8000"
    timeout: 30
  
  # Runtime settings  
  runtime:
    url: "http://localhost:8001"
    max_concurrent_tasks: 10
  
  # Logging
  logging:
    level: INFO
    format: structured
  
  # Performance
  performance:
    enable_metrics: true
    enable_caching: true
    cache_ttl: 3600

# Agent configurations
agents:
  default:
    temperature: 0.7
    max_tokens: 2000
    timeout: 60
```

### Environment Variables

The CLI respects standard MeshAI environment variables:

```bash
# API Keys
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key  
export GOOGLE_API_KEY=your-google-key

# MeshAI Settings
export MESHAI_LOG_LEVEL=INFO
export MESHAI_REGISTRY_URL=https://api.meshai.dev/registry
export MESHAI_RUNTIME_URL=https://api.meshai.dev/runtime

# Development
export MESHAI_DEV_MODE=true
export MESHAI_DEBUG=false
```

## Project Templates

### Simple Template
Basic single-agent application structure:
```
my-project/
├── main.py              # Entry point
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies  
└── .env.example         # Environment template
```

### Multi-Agent Template  
Multi-agent workflow application:
```
my-project/
├── main.py              # Entry point
├── agents/              # Agent definitions
│   ├── __init__.py
│   └── example_agent.py
├── workflows/           # Workflow definitions
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── .env.example         # Environment template
```

### Production Template
Production-ready application with Docker and monitoring:
```
my-project/
├── main.py              # Entry point
├── agents/              # Agent definitions
├── workflows/           # Workflow definitions
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service setup
├── monitoring/          # Monitoring configs
└── .env.example         # Environment template
```

## Framework Support

The CLI supports all major AI frameworks with specific templates and configurations:

| Framework | Models | Features |
|-----------|---------|----------|
| **OpenAI** | GPT-4, GPT-3.5-turbo | Function calling, streaming |
| **Anthropic** | Claude-3 variants | Tool use, safety ratings |
| **Google AI** | Gemini Pro | Multimodal, function calling |
| **LangChain** | Any LLM | Full ecosystem, tools, memory |
| **CrewAI** | Any LLM | Multi-agent crews, role-based |
| **AutoGen** | Any LLM | Conversational agents, groups |

## Development Workflow

### 1. Project Setup
```bash
# Create new project
meshai init my-ai-project --template multi-agent
cd my-ai-project

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt
```

### 2. Agent Development
```bash
# Create agents for different tasks
meshai agent create researcher --framework google
meshai agent create analyst --framework anthropic  
meshai agent create writer --framework openai

# Test agents
meshai agent test researcher --verbose
meshai agent test analyst --verbose
```

### 3. Development Server
```bash
# Start with hot-reload
meshai dev server --watch --debug

# View logs
meshai logs --follow --level DEBUG
```

### 4. Configuration & Validation
```bash
# Validate configuration
meshai config validate --fix

# Check agent status
meshai agent list
```

### 5. Deployment
```bash
# Deploy to staging
meshai deploy --environment staging

# Deploy to production  
meshai deploy --environment production

# Monitor production
meshai logs --follow --service runtime
```

## Advanced Usage

### Custom Agent Templates

You can create custom agent templates by extending the base agent classes:

```python
# agents/custom_agent.py
from meshai.adapters.openai_adapter import OpenAIMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData

class CustomAgent(OpenAIMeshAgent):
    def __init__(self):
        super().__init__(
            model="gpt-4",
            agent_id="custom-agent",
            name="Custom Agent",
            capabilities=["custom-processing"],
            system_prompt="You are a specialized agent..."
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext):
        # Custom processing logic
        result = await super().handle_task(task_data, context)
        
        # Post-processing
        return result
```

### Configuration Validation

The CLI includes built-in validation for common configuration issues:

```bash
# Validate and fix configuration
meshai config validate --fix

# Common fixes applied:
# - Missing required sections
# - Invalid URL formats  
# - Incorrect timeout values
# - Missing environment variables
```

### Performance Monitoring

Monitor your agents and workflows:

```bash
# View performance metrics
meshai agent list --format json | jq '.[] | {id: .agent_id, requests: .stats.requests}'

# Follow application logs
meshai logs --follow --service all

# Check specific agent performance
meshai agent test my-agent --verbose
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```bash
   # Check environment variables
   env | grep API_KEY
   
   # Or use .env file
   cat .env
   ```

2. **Configuration Errors**
   ```bash
   # Validate configuration
   meshai config validate
   
   # Reset to defaults
   rm config.yaml
   meshai config validate --fix
   ```

3. **Agent Registration Failures**
   ```bash
   # Check registry status
   curl http://localhost:8000/health
   
   # View detailed logs
   meshai logs --service registry --level DEBUG
   ```

4. **Import Errors**
   ```bash
   # Install all dependencies
   pip install -r requirements.txt
   
   # Or install with specific framework
   pip install meshai-sdk[openai,anthropic]
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set environment variable
export MESHAI_DEBUG=true

# Or use CLI flag
meshai dev server --debug

# View debug logs
meshai logs --level DEBUG --follow
```

## Contributing

The MeshAI CLI is open source. Contributions are welcome!

### Adding New Commands

1. Create command in `src/meshai/cli/main.py`
2. Add tests in `tests/test_cli.py`
3. Update documentation
4. Submit pull request

### Adding Framework Support

1. Create adapter in `src/meshai/adapters/`
2. Add to `AGENT_FRAMEWORKS` in CLI
3. Create template and examples
4. Add tests and documentation

## Support

- **Documentation**: [docs.meshai.dev](https://docs.meshai.dev)
- **GitHub Issues**: [github.com/meshailabs/meshai-sdk/issues](https://github.com/meshailabs/meshai-sdk/issues)  
- **Community**: [GitHub Discussions](https://github.com/meshailabs/meshai-sdk/discussions)
- **Email**: support@meshai.dev

---

**Happy building with MeshAI CLI! 🚀**