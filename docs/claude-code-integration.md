# MeshAI Integration with Claude Code

This guide shows you how to run MeshAI inside the Claude Code environment for powerful multi-agent development workflows.

## 🚀 **Running MeshAI in Claude Code**

### **Step 1: Set Up Python Virtual Environment**
```bash
# Create a virtual environment to avoid system Python conflicts
python3 -m venv meshai-env

# Use the virtual environment's pip for installations
meshai-env/bin/pip install --upgrade pip
```

### **Step 2: Install MeshAI SDK**
```bash
# Install the MeshAI SDK in editable mode
meshai-env/bin/pip install -e ./meshai-sdk

# Install development dependencies
cd meshai-sdk
../meshai-env/bin/pip install -r requirements.txt
cd ..
```

### **Step 3: Initialize a MeshAI Project**
```bash
# Create a new MeshAI project in Claude Code
meshai-env/bin/meshai init claude-code-project --template multi-agent

# Navigate to the project
cd claude-code-project
```

### **Step 4: Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# ANTHROPIC_API_KEY=your-anthropic-key
# OPENAI_API_KEY=your-openai-key
```

### **Step 5: Start MeshAI Development Server**
```bash
# Start the hot-reload development server
meshai-env/bin/meshai dev server --port 8080 --watch --debug
```

This will give you:
- **Web Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Real-time agent testing interface**
- **Hot-reload when you modify agents**

### **Step 6: Create Agents for Claude Code Tasks**

Create agents that help with code development:

```bash
# Create a code reviewer agent
meshai-env/bin/meshai agent create code-reviewer --framework anthropic --capabilities code-analysis,debugging,optimization

# Create a documentation agent  
meshai-env/bin/meshai agent create doc-writer --framework openai --capabilities documentation,commenting,explanation

# Create a test generator agent
meshai-env/bin/meshai agent create test-generator --framework anthropic --capabilities testing,unit-tests,integration-tests
```

### **Step 7: Test Your Setup**
```bash
# Test an agent
meshai-env/bin/meshai agent test code-reviewer --message "Review this Python function for potential bugs"

# List all agents
meshai-env/bin/meshai agent list --format table
```

## 🔧 **Example: Code Review Workflow**

Create a `agents/code_reviewer.py`:

```python
from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData

class CodeReviewerAgent(AnthropicMeshAgent):
    def __init__(self):
        super().__init__(
            model="claude-3-sonnet-20240229",
            agent_id="code-reviewer",
            name="Code Reviewer",
            capabilities=["code-analysis", "debugging", "optimization", "security"],
            system_prompt=(
                "You are an expert code reviewer. Analyze code for:"
                "- Bugs and potential issues"
                "- Performance optimizations" 
                "- Security vulnerabilities"
                "- Code quality and best practices"
                "- Suggest specific improvements with examples"
            )
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext):
        # Add file reading capabilities for Claude Code integration
        code_content = task_data.input
        
        # Enhanced prompt for code review
        enhanced_input = f"""
        Please review this code:
        
        ```
        {code_content}
        ```
        
        Provide:
        1. Issue identification
        2. Severity assessment  
        3. Specific fix recommendations
        4. Improved code examples
        """
        
        task_data.input = enhanced_input
        return await super().handle_task(task_data, context)
```

## 🔗 **Integration Options**

### **1. Using MeshAI CLI (Recommended)**
The comprehensive CLI tool provides direct integration:

```bash
# Using the virtual environment's MeshAI CLI
# Initialize a new project
meshai-env/bin/meshai init my-claude-project --template multi-agent

# Create agents that work with Claude
meshai-env/bin/meshai agent create claude-assistant --framework anthropic --model claude-3-sonnet-20240229

# Start development server
meshai-env/bin/meshai dev server --port 8080 --watch --debug

# Or activate the virtual environment for easier access
source meshai-env/bin/activate
meshai --help  # Now you can use meshai directly
```

### **2. Direct SDK Integration**
Create agents that use Claude models through MeshAI:

```python
from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
from meshai.core.registry import MeshRegistry

# Create Claude-powered agent
claude_agent = AnthropicMeshAgent(
    model="claude-3-sonnet-20240229",
    agent_id="claude-code-assistant", 
    name="Claude Code Assistant",
    capabilities=["code-analysis", "debugging", "documentation"],
    api_key="your-anthropic-api-key"
)

# Register with MeshAI
registry = MeshRegistry()
await registry.register_agent(claude_agent)
```

### **3. Claude Code as a Tool/Agent**
Create a MeshAI adapter that interfaces with Claude Code:

```python
class ClaudeCodeAgent(MeshAgent):
    """Agent that interfaces with Claude Code"""
    
    async def handle_task(self, task_data, context):
        # Use Claude Code API or subprocess calls
        # to execute code analysis, generation, etc.
        pass
```

## 📊 **Development Dashboard Features**

Once running, you'll have access to:

- **Agent Testing Interface**: Test agents with sample code
- **Performance Monitoring**: Track response times and success rates  
- **Real-time Logs**: See agent activity as you develop
- **WebSocket Updates**: Live notifications of agent responses
- **File Change Detection**: Automatic reload when you modify agents

## 🎯 **Claude Code Integration Tips**

1. **Use the development server** for rapid agent iteration
2. **Create task-specific agents** for different coding needs
3. **Leverage hot-reload** to test changes instantly
4. **Use the web dashboard** to debug agent interactions
5. **Monitor performance** to optimize agent responses

## 🛠 **Common Use Cases**

### **Code Review Agent**
```python
# Review code for bugs, performance, and best practices
result = await code_reviewer.handle_task(
    TaskData(input=your_code_content),
    context
)
```

### **Documentation Generator**
```python
# Generate comprehensive documentation
result = await doc_writer.handle_task(
    TaskData(input="Generate docs for this function: " + function_code),
    context
)
```

### **Test Generator**
```python
# Create unit tests for your code
result = await test_generator.handle_task(
    TaskData(input="Create pytest tests for: " + class_code),
    context
)
```

## 🔧 **Advanced Configuration**

### **Custom Agent Templates**
```bash
# Create custom agent with specific capabilities
meshai-env/bin/meshai agent create my-custom-agent \
  --framework anthropic \
  --model claude-3-opus-20240229 \
  --capabilities custom-analysis,code-generation,refactoring
```

### **Multi-Agent Workflows**
```python
# Chain multiple agents together
class CodeWorkflowAgent(MeshAgent):
    async def handle_task(self, task_data, context):
        # 1. Code review
        review_result = await context.route_to_agent("code-reviewer", task_data)
        
        # 2. Generate tests
        test_result = await context.route_to_agent("test-generator", task_data)
        
        # 3. Create documentation
        doc_result = await context.route_to_agent("doc-writer", task_data)
        
        return {
            "review": review_result,
            "tests": test_result,
            "documentation": doc_result
        }
```

### **Environment Configuration**
```bash
# .env file for Claude Code integration
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
MESHAI_DEV_PORT=8080
MESHAI_DEV_DEBUG=true
MESHAI_DEV_AUTO_RELOAD=true
MESHAI_LOG_LEVEL=INFO
```

## 🚨 **Troubleshooting**

### **Python Environment Issues**
```bash
# If you encounter "externally-managed-environment" error:
# Always use virtual environment
python3 -m venv meshai-env
meshai-env/bin/pip install -e ./meshai-sdk

# If pip is missing in virtual environment:
meshai-env/bin/python -m ensurepip --upgrade
```

### **Missing Registry Module**
If you encounter `ModuleNotFoundError: No module named 'meshai.core.registry'`:
```bash
# The registry module needs to be created
# Check if it exists:
ls meshai-sdk/src/meshai/core/registry.py

# If missing, the module needs to be added to the SDK
# This should be fixed in the latest version
```

### **Agent Not Starting**
```bash
# Check agent registration
meshai-env/bin/meshai agent list

# Validate configuration
meshai-env/bin/meshai config validate --fix

# Check logs
meshai-env/bin/meshai logs --service registry --level DEBUG
```

### **Development Server Issues**
```bash
# Check if port is available
netstat -an | grep :8080

# Restart with different port
meshai-env/bin/meshai dev server --port 3000

# Check file permissions
ls -la agents/
```

### **API Key Issues**
```bash
# Verify environment variables
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" https://api.anthropic.com/v1/messages
```

## 📚 **Additional Resources**

- [MeshAI CLI Guide](./cli-guide.md)
- [Development Server Documentation](./development-server.md)
- [Framework Adapters Guide](./framework-adapters.md)
- [API Reference](./api-reference.md)

This setup gives you a powerful multi-agent development environment running directly in your Claude Code workspace!