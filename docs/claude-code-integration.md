# MeshAI Integration with Claude Code

This guide shows you how to run MeshAI inside the Claude Code environment for powerful multi-agent development workflows.

## 🚀 **Running MeshAI in Claude Code**

### **Step 1: Install MeshAI SDK**
```bash
# Install the MeshAI SDK with all dependencies
pip install -e ./meshai-sdk

# Or install development dependencies
cd meshai-sdk
pip install -r requirements.txt
```

### **Step 2: Initialize a MeshAI Project**
```bash
# Create a new MeshAI project in Claude Code
meshai init claude-code-project --template multi-agent

# Navigate to the project
cd claude-code-project
```

### **Step 3: Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# ANTHROPIC_API_KEY=your-anthropic-key
# OPENAI_API_KEY=your-openai-key
```

### **Step 4: Start MeshAI Development Server**
```bash
# Start the hot-reload development server
meshai dev server --port 8080 --watch --debug
```

This will give you:
- **Web Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Real-time agent testing interface**
- **Hot-reload when you modify agents**

### **Step 5: Create Agents for Claude Code Tasks**

Create agents that help with code development:

```bash
# Create a code reviewer agent
meshai agent create code-reviewer --framework anthropic --capabilities code-analysis,debugging,optimization

# Create a documentation agent  
meshai agent create doc-writer --framework openai --capabilities documentation,commenting,explanation

# Create a test generator agent
meshai agent create test-generator --framework anthropic --capabilities testing,unit-tests,integration-tests
```

### **Step 6: Test Your Setup**
```bash
# Test an agent
meshai agent test code-reviewer --message "Review this Python function for potential bugs"

# List all agents
meshai agent list --format table
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
# Install MeshAI SDK
pip install meshai-sdk

# Initialize a new project
meshai init my-claude-project --template multi-agent

# Create agents that work with Claude
meshai agent create claude-assistant --framework anthropic --model claude-3-sonnet-20240229

# Start development server
meshai dev server --port 8080 --watch --debug
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
meshai agent create my-custom-agent \
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

### **Agent Not Starting**
```bash
# Check agent registration
meshai agent list

# Validate configuration
meshai config validate --fix

# Check logs
meshai logs --service registry --level DEBUG
```

### **Development Server Issues**
```bash
# Check if port is available
netstat -an | grep :8080

# Restart with different port
meshai dev server --port 3000

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