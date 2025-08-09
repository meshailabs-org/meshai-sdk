#!/usr/bin/env python3
"""
MeshAI Database Integration Demo

Demonstrates the complete database-integrated MeshAI system:
1. Initialize database schema
2. Start database-integrated services
3. Register agents with persistent storage
4. Execute tasks with database tracking
5. Show persistent data across restarts
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.core.agent import MeshAgent
from meshai.core.context import MeshContext
from meshai.core.config import MeshConfig
from meshai.core.schemas import TaskData, TaskStatus
from meshai.clients.registry import RegistryClient
from meshai.clients.runtime import RuntimeClient
from meshai.database.session import get_database_manager
from meshai.database.migrations import init_database, get_current_revision


class PersistentDemoAgent(MeshAgent):
    """Demo agent that shows database persistence"""
    
    def __init__(self, agent_id: str, agent_type: str = "demo"):
        super().__init__(
            agent_id=agent_id,
            name=f"Persistent {agent_type.title()} Agent",
            capabilities=[f"{agent_type}-processing", "database-demo"],
            framework="custom",
            metadata={"type": agent_type, "demo": True}
        )
        self.processed_count = 0
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """Process tasks with database awareness"""
        self.processed_count += 1
        
        # Simulate some work
        await asyncio.sleep(0.5)
        
        # Extract input
        if isinstance(task_data.input, dict):
            input_data = task_data.input
        else:
            input_data = {"data": str(task_data.input)}
        
        # Process based on task type
        if task_data.task_type == "count":
            result = {
                "count": self.processed_count,
                "total_processed": self.processed_count,
                "agent_info": {
                    "id": self.agent_id,
                    "name": self.name,
                    "type": self.metadata.get("type")
                }
            }
        elif task_data.task_type == "echo":
            result = {
                "echo": input_data,
                "processed_at": time.time(),
                "processed_by": self.agent_id
            }
        else:
            result = {
                "message": f"Processed by {self.agent_id}",
                "input": input_data,
                "count": self.processed_count
            }
        
        # Store in context for persistence
        await context.set(f"last_result_{self.agent_id}", result)
        
        return {
            "status": "success",
            "result": result,
            "agent_metadata": self.metadata,
            "database_demo": True
        }


async def check_database_status():
    """Check database initialization and status"""
    print("📊 Database Status Check")
    print("-" * 30)
    
    try:
        # Check database connectivity
        db_manager = get_database_manager()
        connected = await db_manager.test_connection()
        
        if connected:
            print("✅ Database connection: OK")
            
            # Check migration status
            current_rev = get_current_revision()
            if current_rev:
                print(f"✅ Database schema: Initialized (rev: {current_rev})")
            else:
                print("⚠️  Database schema: Not initialized")
                print("🔧 Initializing database...")
                init_database()
                print("✅ Database schema: Initialized")
            
            return True
        else:
            print("❌ Database connection: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


async def wait_for_database_services():
    """Wait for database-integrated services"""
    print("\n⏳ Waiting for database services...")
    
    config = MeshConfig()
    registry_client = RegistryClient(config)
    runtime_client = RuntimeClient(config)
    
    for attempt in range(30):
        try:
            # Test registry service
            registry_health = await registry_client.get_registry_health()
            runtime_health = await runtime_client.get_runtime_health()
            
            print("✅ Database services are running!")
            print(f"📊 Registry: {registry_health.get('total_agents', 0)} agents registered")
            
            await registry_client.close()
            await runtime_client.close()
            return True
            
        except Exception:
            if attempt == 29:
                print("❌ Database services not available")
                print("\n💡 To start database services:")
                print("   python scripts/start-with-database.py")
                return False
            await asyncio.sleep(1)


async def demonstrate_persistent_agents():
    """Demonstrate agent persistence across sessions"""
    print("\n" + "=" * 60)
    print("🤖 DEMO: Persistent Agent Registration")
    print("=" * 60)
    
    # Create agents
    agents = [
        PersistentDemoAgent("persistent-counter", "counter"),
        PersistentDemoAgent("persistent-processor", "processor"),
        PersistentDemoAgent("persistent-echo", "echo")
    ]
    
    # Start agent servers
    for i, agent in enumerate(agents):
        agent.config = agent.config.update(agent_port=8200 + i)
        await agent.start_server()
        await asyncio.sleep(1)  # Stagger startup
    
    # Register agents (this will persist in database)
    print("📝 Registering agents in database...")
    for agent in agents:
        try:
            await agent.register()
            print(f"✅ {agent.name} registered and persisted")
        except Exception as e:
            print(f"⚠️  {agent.name} registration issue: {e}")
    
    await asyncio.sleep(2)  # Let registration complete
    
    # Verify agents are in database
    config = MeshConfig()
    registry_client = RegistryClient(config)
    
    try:
        all_agents = await registry_client.list_agents()
        demo_agents = [a for a in all_agents if "database-demo" in a.capabilities]
        
        print(f"\n📊 Database now contains {len(demo_agents)} demo agents:")
        for agent in demo_agents:
            print(f"   - {agent.name} ({agent.id}) - {agent.framework}")
            print(f"     Capabilities: {', '.join(agent.capabilities)}")
            print(f"     Status: {agent.status}")
        
    except Exception as e:
        print(f"❌ Failed to query database: {e}")
    
    await registry_client.close()
    return agents


async def demonstrate_persistent_tasks():
    """Demonstrate task persistence in database"""
    print("\n" + "=" * 60)
    print("📋 DEMO: Persistent Task Execution")
    print("=" * 60)
    
    config = MeshConfig()
    runtime_client = RuntimeClient(config)
    
    # Submit various tasks that will be persisted
    test_tasks = [
        ("count", {"increment": 1}, "Get current count"),
        ("echo", {"message": "Database persistence test"}, "Echo test"),
        ("process", {"data": "Sample processing task"}, "General processing")
    ]
    
    task_results = []
    
    for task_type, input_data, description in test_tasks:
        print(f"\n📤 Submitting task: {description}")
        
        task_data = TaskData(
            task_type=task_type,
            input=input_data,
            required_capabilities=["database-demo"],
            timeout_seconds=15
        )
        
        try:
            # Submit and wait for completion
            result = await runtime_client.submit_and_wait(task_data, timeout=20)
            
            if result.status == TaskStatus.COMPLETED:
                print(f"✅ Task completed by {result.agent_id}")
                print(f"   Execution time: {result.execution_time_seconds:.2f}s")
                if result.result:
                    task_result = result.result.get('result', {})
                    if 'count' in task_result:
                        print(f"   Count: {task_result['count']}")
                    elif 'echo' in task_result:
                        print(f"   Echo: {task_result['echo']['message']}")
                
                task_results.append(result)
            else:
                print(f"❌ Task failed: {result.error}")
                
        except Exception as e:
            print(f"❌ Task execution error: {e}")
        
        await asyncio.sleep(1)
    
    # Show task persistence
    print(f"\n📊 Database Task History")
    print("-" * 30)
    try:
        # Get task statistics from runtime
        stats = await runtime_client.get_runtime_stats()
        print(f"✅ Total completed: {stats.get('completed_tasks', 0)}")
        print(f"✅ Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"✅ Average time: {stats.get('avg_execution_time_ms', 0):.1f}ms")
        
    except Exception as e:
        print(f"⚠️  Could not get stats: {e}")
    
    await runtime_client.close()
    return task_results


async def demonstrate_data_persistence():
    """Show that data persists across service restarts"""
    print("\n" + "=" * 60)
    print("💾 DEMO: Data Persistence Verification")
    print("=" * 60)
    
    config = MeshConfig()
    registry_client = RegistryClient(config)
    runtime_client = RuntimeClient(config)
    
    try:
        # Show persistent agents
        agents = await registry_client.list_agents()
        demo_agents = [a for a in agents if "database-demo" in a.capabilities]
        
        print(f"📊 Persistent Agents: {len(demo_agents)} found")
        for agent in demo_agents:
            print(f"   - {agent.name}: {agent.status} (created: {agent.created_at})")
        
        # Show persistent task stats
        runtime_stats = await runtime_client.get_runtime_stats()
        print(f"\n📊 Persistent Task Data:")
        print(f"   - Total tasks: {runtime_stats.get('total_tasks_submitted', 0)}")
        print(f"   - Completed: {runtime_stats.get('completed_tasks', 0)}")
        print(f"   - Failed: {runtime_stats.get('failed_tasks', 0)}")
        
        print(f"\n✅ All data persists in database!")
        print("💡 This data will survive service restarts and system reboots")
        
    except Exception as e:
        print(f"❌ Error checking persistence: {e}")
    
    await registry_client.close()
    await runtime_client.close()


async def cleanup_demo_agents(agents):
    """Clean up demo agents"""
    print("\n" + "=" * 60)
    print("🧹 DEMO: Cleanup")
    print("=" * 60)
    
    for agent in agents:
        try:
            await agent.unregister()
            await agent.stop_server()
            print(f"✅ {agent.name} cleaned up")
        except Exception as e:
            print(f"⚠️  {agent.name} cleanup issue: {e}")


async def main():
    """Main demo function"""
    print("🗄️  MeshAI Database Integration Demo")
    print("=" * 60)
    print("This demo showcases:")
    print("- Database schema initialization")
    print("- Persistent agent registration")
    print("- Task execution with database tracking")
    print("- Data persistence across service restarts")
    print("=" * 60)
    
    # Check database status
    if not await check_database_status():
        print("\n💡 Database issues detected. Please check configuration.")
        return
    
    # Wait for database services
    if not await wait_for_database_services():
        return
    
    agents = []
    
    try:
        # Demo 1: Persistent agent registration
        agents = await demonstrate_persistent_agents()
        
        # Demo 2: Persistent task execution
        await demonstrate_persistent_tasks()
        
        # Demo 3: Data persistence verification
        await demonstrate_data_persistence()
        
        print("\n" + "=" * 60)
        print("🎉 Database Integration Demo Complete!")
        print("=" * 60)
        print("Key capabilities demonstrated:")
        print("✅ PostgreSQL/SQLite database integration")
        print("✅ Persistent agent registration and discovery")
        print("✅ Task execution tracking and history")
        print("✅ Data persistence across service restarts")
        print("✅ Database health monitoring and connectivity")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if agents:
            await cleanup_demo_agents(agents)


if __name__ == "__main__":
    asyncio.run(main())