# MeshAI Development Server

The MeshAI Development Server provides a comprehensive development environment with hot-reload, real-time testing, and interactive debugging capabilities for multi-agent applications.

## Features

### 🔥 Hot Reload
- **Intelligent Reloading**: Automatically detects file changes and reloads only affected modules
- **Dependency Tracking**: Understands module dependencies and reloads in correct order
- **State Preservation**: Maintains agent state across reloads when possible
- **Safe Rollback**: Automatic rollback on reload failures

### 🧪 Real-time Testing  
- **Interactive Agent Testing**: Test agents directly from the web dashboard
- **WebSocket Updates**: Real-time notifications of task completions and errors
- **Performance Monitoring**: Track response times and success rates
- **Task History**: View recent agent interactions and results

### 📊 Live Dashboard
- **Agent Management**: View registered agents and their capabilities
- **Performance Metrics**: Monitor request counts, response times, and uptime
- **Activity Log**: Real-time feed of development events
- **Configuration Validation**: Live validation of agent configurations

### 🐛 Development Tools
- **File Watching**: Monitor Python files, YAML configs, and other project files
- **Error Recovery**: Graceful handling of agent failures and reload errors
- **Memory Management**: Automatic cleanup and garbage collection
- **Debug Logging**: Comprehensive logging with adjustable levels

## Quick Start

### Start Development Server

```bash
# Start with default settings
meshai dev server

# Start with custom port and file watching
meshai dev server --port 3000 --watch --debug

# Start without hot-reload
meshai dev server --port 8080 --no-reload
```

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port`, `-p` | Server port | 8080 |
| `--watch`, `-w` | Enable file watching | true |
| `--debug` | Enable debug logging | false |
| `--no-reload` | Disable hot-reload | false |

### Access Dashboard

Once started, open your browser to:
- **Dashboard**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

## Project Structure

The development server automatically discovers agents in these locations:

```
my-project/
├── agents/              # Primary agent directory
│   ├── researcher.py    # Individual agent files
│   ├── analyst.py
│   └── writer.py
├── src/agents/          # Source-based agents
├── app/agents/          # App-based agents
├── main.py              # Main application entry point
├── config.yaml          # Configuration file
└── .env                 # Environment variables
```

## Agent Discovery

The server automatically discovers and loads agents that:

1. **Inherit from MeshAgent**: Extend any MeshAI agent class
2. **Have Required Attributes**: Include `agent_id` and `handle_task` method
3. **Are in Watch Paths**: Located in monitored directories
4. **Follow Naming**: Use descriptive filenames (not starting with `_`)

### Example Agent Structure

```python
# agents/my_agent.py
from meshai.adapters.openai_adapter import OpenAIMeshAgent

class MyAgent(OpenAIMeshAgent):
    def __init__(self):
        super().__init__(
            model="gpt-3.5-turbo",
            agent_id="my-agent",
            name="My Agent",
            capabilities=["conversation", "help"]
        )
    
    async def handle_task(self, task_data, context):
        # Agent logic here
        result = await super().handle_task(task_data, context)
        return result
```

## Hot Reload System

### How It Works

1. **File Monitoring**: Watches for changes in Python files, configs, and templates
2. **Dependency Analysis**: Analyzes import relationships between modules
3. **Safe Reloading**: Reloads modules in dependency order with rollback protection
4. **State Preservation**: Maintains agent instances and their state when possible
5. **Notification**: Updates dashboard and logs with reload status

### Reload Triggers

Hot reload is triggered by changes to:
- **Python Files** (`.py`): Agent definitions, utilities, configurations
- **Config Files** (`.yaml`, `.yml`, `.json`): Application and agent configurations
- **Environment Files** (`.env`): Environment variables and API keys

### Manual Reload

You can manually trigger a reload:

```bash
# Via API
curl -X POST http://localhost:8080/reload

# Via dashboard
# Click "Reload" button in the web interface
```

## Web Dashboard

### Overview Section
- **Active Agents**: Number of registered agents
- **Request Count**: Total API requests processed
- **Response Time**: Average agent response time
- **Uptime**: Development server uptime

### Agent Management
- **Agent List**: View all registered agents with capabilities
- **Health Status**: Real-time agent health monitoring
- **Registration Status**: Track agent registration/deregistration

### Testing Interface
- **Agent Selection**: Choose agent to test from dropdown
- **Message Input**: Enter test messages for agents
- **Response Display**: View agent responses with timing
- **History**: Track testing history and results

### Activity Log
- **File Changes**: Real-time notification of file modifications
- **Reload Events**: Hot reload success/failure notifications
- **Task Results**: Agent task completion status
- **Error Reports**: Detailed error information and stack traces

## API Endpoints

### Health and Status

```bash
# Health check
GET /health
# Response: {"status": "healthy", "uptime": "1h 23m 45s"}

# Server metrics
GET /metrics
# Response: {"requests": 150, "errors": 2, "avg_response_time": 0.85, ...}
```

### Agent Management

```bash
# List all agents
GET /agents
# Response: {"agents": [{"id": "agent-1", "name": "Agent 1", ...}]}

# Test specific agent
POST /agents/{agent_id}/test
Content-Type: application/json
{"message": "Hello, test message"}
```

### Development Operations

```bash
# Trigger manual reload
POST /reload
# Response: {"message": "Reload triggered"}

# Get recent logs
GET /logs
# Response: {"logs": ["2024-01-01 12:00:00 - INFO - Agent loaded", ...]}
```

## Configuration

### Server Configuration

```yaml
# config.yaml
development:
  server:
    port: 8080
    debug: true
    auto_reload: true
    
  file_watching:
    paths: [".", "agents", "src"]
    ignore_patterns: ["*.pyc", "__pycache__", ".git"]
    debounce_ms: 500
    
  hot_reload:
    preserve_state: true
    safe_mode: true
    rollback_on_error: true
```

### Environment Variables

```bash
# Development settings
export MESHAI_DEV_PORT=8080
export MESHAI_DEV_DEBUG=true
export MESHAI_DEV_AUTO_RELOAD=true

# File watching
export MESHAI_WATCH_PATHS=".,agents,src"
export MESHAI_IGNORE_PATTERNS="*.pyc,__pycache__,.git"

# Logging
export MESHAI_LOG_LEVEL=DEBUG
export MESHAI_DEV_LOG_FORMAT=structured
```

## Advanced Features

### Custom Reload Logic

```python
# Custom reload callback
async def my_reload_callback():
    """Custom logic after reload"""
    print("🔄 Custom reload logic executed")
    
    # Reinitialize custom resources
    await reinitialize_database()
    await refresh_external_connections()

# Use with development server
from meshai.dev.server import DevServer

server = DevServer(
    port=8080,
    auto_reload=True,
    reload_callback=my_reload_callback
)

await server.start()
```

### State Preservation

```python
# Register instances for state preservation
from meshai.dev.reloader import HotReloader

reloader = HotReloader(watch_paths=["."])

# Register agent instance
reloader.register_instance(my_agent, "agents.my_agent")

# Custom state preservation
@dataclass
class MyAgentState:
    conversation_history: List[str]
    user_preferences: Dict[str, Any]
    
def preserve_state(agent):
    return MyAgentState(
        conversation_history=agent.history,
        user_preferences=agent.preferences
    )

def restore_state(agent, state):
    agent.history = state.conversation_history
    agent.preferences = state.user_preferences
```

### File Watcher Customization

```python
from meshai.dev.watcher import FileWatcher

# Custom file watcher
async def my_file_callback(file_path: str, event_type: str):
    print(f"File {event_type}: {file_path}")
    
    # Custom logic for different file types
    if file_path.endswith('.yaml'):
        await reload_configuration(file_path)
    elif file_path.endswith('.py'):
        await reload_agents(file_path)

watcher = FileWatcher(
    paths=[".", "config", "templates"],
    callback=my_file_callback,
    file_patterns=["*.py", "*.yaml", "*.json"],
    ignore_patterns=["*.pyc", "__pycache__", ".git"],
    debounce_ms=1000  # Wait 1 second before processing
)

watcher.start()
```

## Performance Optimization

### Memory Management

The development server includes automatic memory management:

- **Weak References**: Uses weak references to prevent memory leaks
- **Garbage Collection**: Automatic cleanup after reloads
- **Module Cleanup**: Removes unused modules from memory
- **Instance Tracking**: Monitors agent instances across reloads

### File Watching Optimization

- **Intelligent Filtering**: Ignores irrelevant files (`.pyc`, `__pycache__`, etc.)
- **Debouncing**: Prevents duplicate reload events
- **Selective Watching**: Only monitors relevant file types
- **Efficient Polling**: Falls back to polling if `watchdog` unavailable

### Hot Reload Optimization

- **Dependency Analysis**: Only reloads affected modules
- **Incremental Loading**: Loads changed modules incrementally
- **State Preservation**: Maintains state to avoid reinitialization
- **Rollback Protection**: Safe rollback on failures

## Troubleshooting

### Common Issues

#### Agent Not Discovered

**Problem**: Agents not appearing in dashboard

**Solutions**:
```bash
# Check agent file location
ls agents/
ls src/agents/

# Verify agent class structure
python -c "from agents.my_agent import MyAgent; print(MyAgent().agent_id)"

# Check server logs
meshai logs --service registry --level DEBUG
```

#### Hot Reload Fails

**Problem**: Reload errors or rollbacks

**Solutions**:
```bash
# Check for syntax errors
python -m py_compile agents/my_agent.py

# Review reload logs
meshai logs --service reloader --level DEBUG

# Disable safe mode temporarily
# In config.yaml: hot_reload: { safe_mode: false }
```

#### File Changes Not Detected

**Problem**: File watching not working

**Solutions**:
```bash
# Install watchdog
pip install watchdog

# Check file patterns
# Ensure your files match the watch patterns

# Verify permissions
ls -la agents/

# Test manual reload
curl -X POST http://localhost:8080/reload
```

#### High Memory Usage

**Problem**: Memory leaks during development

**Solutions**:
```python
# Enable garbage collection logging
import gc
gc.set_debug(gc.DEBUG_STATS)

# Check for circular references
from meshai.dev.reloader import HotReloader
stats = reloader.get_stats()
print(f"Preserved instances: {stats['preserved_instances']}")

# Restart server periodically
# Or disable state preservation temporarily
```

### Debug Commands

```bash
# View all server stats
curl http://localhost:8080/metrics

# Check file watcher status
curl http://localhost:8080/debug/watcher

# View module dependencies
curl http://localhost:8080/debug/dependencies

# Force garbage collection
curl -X POST http://localhost:8080/debug/gc
```

## Best Practices

### Development Workflow

1. **Start Simple**: Begin with basic agent structure
2. **Incremental Changes**: Make small, testable changes
3. **Use Dashboard**: Leverage web interface for testing
4. **Monitor Performance**: Watch response times and memory usage
5. **Handle Errors**: Implement proper error handling in agents

### File Organization

```
project/
├── agents/                    # Main agent definitions
│   ├── __init__.py           # Package initialization
│   ├── base/                 # Base classes and utilities
│   ├── specialized/          # Specialized agent types
│   └── tests/                # Agent tests
├── config/                   # Configuration files
│   ├── development.yaml      # Dev environment config
│   ├── production.yaml       # Prod environment config
│   └── agents.yaml           # Agent-specific config
├── utils/                    # Shared utilities
├── tests/                    # Integration tests
└── main.py                   # Application entry point
```

### Error Handling

```python
class RobustAgent(OpenAIMeshAgent):
    async def handle_task(self, task_data, context):
        try:
            result = await super().handle_task(task_data, context)
            return result
        except Exception as e:
            # Log error for debugging
            logger.error(f"Task failed: {e}", exc_info=True)
            
            # Return graceful error response
            return {
                "result": "I encountered an error processing your request.",
                "error": str(e),
                "type": "error"
            }
```

### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper

class MonitoredAgent(OpenAIMeshAgent):
    @monitor_performance
    async def handle_task(self, task_data, context):
        return await super().handle_task(task_data, context)
```

## Integration Examples

### VS Code Integration

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Start MeshAI Dev Server",
            "type": "shell",
            "command": "meshai",
            "args": ["dev", "server", "--watch", "--debug"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            }
        }
    ]
}
```

### Docker Development

```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install development dependencies
RUN pip install watchdog

COPY . .

EXPOSE 8080

CMD ["meshai", "dev", "server", "--host", "0.0.0.0", "--watch"]
```

### CI/CD Integration

```yaml
# .github/workflows/dev-test.yml
name: Development Testing

on:
  pull_request:
    branches: [main]

jobs:
  dev-server-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-asyncio
    
    - name: Test dev server startup
      run: |
        timeout 30s meshai dev server --port 8081 &
        sleep 5
        curl -f http://localhost:8081/health
        
    - name: Test agent discovery
      run: |
        curl -f http://localhost:8081/agents
```

---

The MeshAI Development Server provides a powerful, feature-rich environment for developing and testing multi-agent applications with real-time feedback and intelligent hot-reload capabilities.