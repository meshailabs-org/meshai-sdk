# MeshAI SDK - Quick Start Guide

## 🚀 Start MeshAI Services

1. **Start all services:**
   ```bash
   python scripts/start-all.py
   ```

2. **Or start individual services:**
   ```bash
   # Registry Service (port 8001)
   python scripts/start-registry.py
   
   # Runtime Service (port 8002) 
   python scripts/start-runtime.py
   ```

3. **Development mode with hot reload:**
   ```bash
   ./scripts/start-dev.sh
   ```

## 🧪 Test the System

1. **Run integration test:**
   ```bash
   python scripts/integration-test.py
   ```

2. **Run complete demo:**
   ```bash
   python examples/complete_system_demo.py
   ```

## 📊 Service Endpoints

### Registry Service (localhost:8001)
- **Health:** http://localhost:8001/health
- **Metrics:** http://localhost:8001/metrics  
- **API Docs:** http://localhost:8001/docs
- **Agents:** http://localhost:8001/api/v1/agents

### Runtime Service (localhost:8002)
- **Health:** http://localhost:8002/health
- **Metrics:** http://localhost:8002/metrics
- **API Docs:** http://localhost:8002/docs
- **Tasks:** http://localhost:8002/api/v1/tasks

## 🤖 Create Your First Agent

```python
from meshai.core.agent import MeshAgent
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData

class MyAgent(MeshAgent):
    def __init__(self):
        super().__init__(
            agent_id="my-agent",
            name="My First Agent", 
            capabilities=["text-processing"],
            framework="custom"
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext):
        return {"result": "Hello from MeshAI!"}

# Run your agent
async def main():
    agent = MyAgent()
    await agent.run()  # Starts server and registers with MeshAI

asyncio.run(main())
```

## ⚡ Submit Tasks

```python
from meshai.clients.runtime import RuntimeClient
from meshai.core.schemas import TaskData

runtime = RuntimeClient()

# Submit a task
task = TaskData(
    task_type="greeting",
    input={"message": "Hello MeshAI!"},
    required_capabilities=["text-processing"]
)

result = await runtime.submit_and_wait(task)
print(result.result)
```

## 📈 System Status

- **Registry Health:** `curl http://localhost:8001/health`
- **Runtime Health:** `curl http://localhost:8002/health`
- **List Agents:** `curl http://localhost:8001/api/v1/agents`
- **List Tasks:** `curl http://localhost:8002/api/v1/tasks`

---

**🎉 You're ready to build with MeshAI!**

Check out more examples in the `examples/` directory.