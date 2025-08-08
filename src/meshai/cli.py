"""
MeshAI SDK Command Line Interface
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import structlog

from .core.config import MeshConfig, get_config, set_config
from .clients.registry import RegistryClient
from .clients.runtime import RuntimeClient
from .core.schemas import TaskData, AgentRegistration, DiscoveryQuery

logger = structlog.get_logger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup CLI logging"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


async def cmd_config(args):
    """Configuration management commands"""
    config = get_config()
    
    if args.action == "show":
        config_dict = config.to_dict()
        print(json.dumps(config_dict, indent=2, default=str))
    
    elif args.action == "set":
        if not args.key or not args.value:
            print("Error: --key and --value required for set action")
            sys.exit(1)
        
        # Update configuration
        updates = {args.key: args.value}
        new_config = config.update(**updates)
        set_config(new_config)
        
        print(f"Configuration updated: {args.key} = {args.value}")
    
    elif args.action == "validate":
        try:
            # Test connections
            async with RegistryClient(config) as registry:
                health = await registry.get_registry_health()
                print(f"Registry: {health['status']}")
            
            async with RuntimeClient(config) as runtime:
                health = await runtime.get_runtime_health()
                print(f"Runtime: {health['status']}")
            
            print("Configuration is valid")
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            sys.exit(1)


async def cmd_agents(args):
    """Agent management commands"""
    config = get_config()
    
    async with RegistryClient(config) as registry:
        if args.action == "list":
            agents = await registry.list_agents(limit=args.limit or 50)
            
            if args.format == "json":
                print(json.dumps([agent.model_dump() for agent in agents], indent=2, default=str))
            else:
                print(f"{'ID':<20} {'Name':<25} {'Framework':<12} {'Status':<10} {'Capabilities'}")
                print("-" * 100)
                for agent in agents:
                    caps = ", ".join(agent.capabilities[:3])
                    if len(agent.capabilities) > 3:
                        caps += f" (+{len(agent.capabilities) - 3} more)"
                    print(f"{agent.id:<20} {agent.name:<25} {agent.framework:<12} {agent.status:<10} {caps}")
        
        elif args.action == "show":
            if not args.agent_id:
                print("Error: --agent-id required for show action")
                sys.exit(1)
            
            try:
                agent = await registry.get_agent(args.agent_id)
                if args.format == "json":
                    print(json.dumps(agent.model_dump(), indent=2, default=str))
                else:
                    print(f"Agent: {agent.name} ({agent.id})")
                    print(f"Framework: {agent.framework}")
                    print(f"Status: {agent.status}")
                    print(f"Capabilities: {', '.join(agent.capabilities)}")
                    print(f"Endpoint: {agent.endpoint}")
                    print(f"Created: {agent.created_at}")
                    if agent.metadata:
                        print("Metadata:")
                        for key, value in agent.metadata.items():
                            print(f"  {key}: {value}")
                            
            except Exception as e:
                print(f"Error retrieving agent: {e}")
                sys.exit(1)
        
        elif args.action == "discover":
            capabilities = args.capabilities.split(",") if args.capabilities else None
            
            query = DiscoveryQuery(
                capabilities=capabilities,
                framework=args.framework,
                limit=args.limit
            )
            
            agents = await registry.discover_agents(query)
            
            print(f"Found {len(agents)} matching agents:")
            for agent in agents:
                print(f"  - {agent.name} ({agent.id}): {', '.join(agent.capabilities)}")
        
        elif args.action == "register":
            if not args.config_file:
                print("Error: --config-file required for register action")
                sys.exit(1)
            
            try:
                with open(args.config_file, 'r') as f:
                    agent_config = json.load(f)
                
                registration = AgentRegistration(**agent_config)
                agent = await registry.register_agent(registration)
                
                print(f"Agent registered successfully: {agent.id}")
                
            except Exception as e:
                print(f"Error registering agent: {e}")
                sys.exit(1)


async def cmd_tasks(args):
    """Task management commands"""
    config = get_config()
    
    async with RuntimeClient(config) as runtime:
        if args.action == "submit":
            if not args.task_file:
                print("Error: --task-file required for submit action")
                sys.exit(1)
            
            try:
                with open(args.task_file, 'r') as f:
                    task_config = json.load(f)
                
                task_data = TaskData(**task_config)
                result = await runtime.submit_task(task_data)
                
                if args.format == "json":
                    print(json.dumps(result.model_dump(), indent=2, default=str))
                else:
                    print(f"Task submitted: {result.task_id}")
                    print(f"Status: {result.status}")
                    if result.result:
                        print(f"Result: {result.result}")
                        
            except Exception as e:
                print(f"Error submitting task: {e}")
                sys.exit(1)
        
        elif args.action == "status":
            if not args.task_id:
                print("Error: --task-id required for status action")
                sys.exit(1)
            
            try:
                result = await runtime.get_task_status(args.task_id)
                
                if args.format == "json":
                    print(json.dumps(result.model_dump(), indent=2, default=str))
                else:
                    print(f"Task: {result.task_id}")
                    print(f"Status: {result.status}")
                    if result.agent_id:
                        print(f"Agent: {result.agent_id}")
                    if result.execution_time:
                        print(f"Execution Time: {result.execution_time}s")
                    if result.result:
                        print(f"Result: {result.result}")
                    if result.error:
                        print(f"Error: {result.error}")
                        
            except Exception as e:
                print(f"Error getting task status: {e}")
                sys.exit(1)
        
        elif args.action == "cancel":
            if not args.task_id:
                print("Error: --task-id required for cancel action")
                sys.exit(1)
            
            try:
                success = await runtime.cancel_task(args.task_id)
                if success:
                    print(f"Task {args.task_id} cancelled successfully")
                else:
                    print(f"Task {args.task_id} could not be cancelled (may be already completed)")
                    
            except Exception as e:
                print(f"Error cancelling task: {e}")
                sys.exit(1)
        
        elif args.action == "stats":
            try:
                stats = await runtime.get_runtime_stats()
                
                if args.format == "json":
                    print(json.dumps(stats, indent=2))
                else:
                    print("Runtime Statistics:")
                    print(f"  Queue Length: {stats.get('queue_length', 'N/A')}")
                    print(f"  Active Tasks: {stats.get('active_tasks', 'N/A')}")
                    
                    task_counts = stats.get('task_counts', {})
                    if task_counts:
                        print("  Task Counts by Status:")
                        for status, count in task_counts.items():
                            print(f"    {status}: {count}")
                    
                    agent_loads = stats.get('agent_loads', {})
                    if agent_loads:
                        print("  Agent Load:")
                        for agent_id, load in agent_loads.items():
                            print(f"    {agent_id}: {load}")
                            
            except Exception as e:
                print(f"Error getting runtime stats: {e}")
                sys.exit(1)


async def cmd_test(args):
    """Test connectivity and basic functionality"""
    config = get_config()
    
    print("Testing MeshAI connectivity...")
    
    try:
        # Test Registry
        async with RegistryClient(config) as registry:
            health = await registry.get_registry_health()
            print(f"✓ Registry: {health['status']}")
        
        # Test Runtime
        async with RuntimeClient(config) as runtime:
            health = await runtime.get_runtime_health()
            print(f"✓ Runtime: {health['status']}")
        
        # Test basic workflow
        if args.full:
            print("\nRunning full connectivity test...")
            
            async with RuntimeClient(config) as runtime:
                # Submit a simple test task
                test_task = TaskData(
                    task_type="test",
                    input="Hello MeshAI!",
                    required_capabilities=["test"],
                    timeout_seconds=10
                )
                
                result = await runtime.submit_task(test_task)
                print(f"✓ Test task submitted: {result.task_id}")
                
                # Check status
                final_result = await runtime.get_task_status(result.task_id)
                print(f"✓ Test task status: {final_result.status}")
        
        print("\nAll tests passed! 🎉")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="MeshAI SDK CLI")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--format", default="table", choices=["table", "json"], help="Output format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("action", choices=["show", "set", "validate"])
    config_parser.add_argument("--key", help="Configuration key")
    config_parser.add_argument("--value", help="Configuration value")
    
    # Agent commands
    agents_parser = subparsers.add_parser("agents", help="Agent management")
    agents_parser.add_argument("action", choices=["list", "show", "discover", "register"])
    agents_parser.add_argument("--agent-id", help="Agent ID")
    agents_parser.add_argument("--capabilities", help="Comma-separated capabilities")
    agents_parser.add_argument("--framework", help="Framework filter")
    agents_parser.add_argument("--limit", type=int, help="Result limit")
    agents_parser.add_argument("--config-file", help="Agent configuration file")
    
    # Task commands
    tasks_parser = subparsers.add_parser("tasks", help="Task management")
    tasks_parser.add_argument("action", choices=["submit", "status", "cancel", "stats"])
    tasks_parser.add_argument("--task-id", help="Task ID")
    tasks_parser.add_argument("--task-file", help="Task configuration file")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test connectivity")
    test_parser.add_argument("--full", action="store_true", help="Run full test suite")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            config = MeshConfig.from_dict(config_data)
            set_config(config)
    
    # Run command
    if args.command == "config":
        asyncio.run(cmd_config(args))
    elif args.command == "agents":
        asyncio.run(cmd_agents(args))
    elif args.command == "tasks":
        asyncio.run(cmd_tasks(args))
    elif args.command == "test":
        asyncio.run(cmd_test(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()