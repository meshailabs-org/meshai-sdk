#!/usr/bin/env python3
"""
Start All MeshAI Services

This script starts both Registry and Runtime services for development.
"""

import asyncio
import sys
import os
import signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.services.registry_service import run_server as run_registry
from meshai.services.runtime_service import run_server as run_runtime
from meshai.core.config import MeshConfig


class ServiceOrchestrator:
    """Orchestrates multiple MeshAI services"""
    
    def __init__(self):
        self.config = MeshConfig.from_env()
        self.running = True
        self.tasks = []
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⚠️  Received signal {signum}, shutting down services...")
        self.running = False
        for task in self.tasks:
            task.cancel()
    
    async def start_services(self):
        """Start all services concurrently"""
        print("=" * 60)
        print("🚀 MeshAI Service Orchestrator")
        print("=" * 60)
        print("Starting Registry Service on localhost:8001")
        print("Starting Runtime Service on localhost:8002")
        print("=" * 60)
        print("Press Ctrl+C to stop all services")
        print("=" * 60)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start services concurrently
            registry_task = asyncio.create_task(
                run_registry(self.config, "0.0.0.0", 8001)
            )
            runtime_task = asyncio.create_task(
                run_runtime(self.config, "0.0.0.0", 8002)
            )
            
            self.tasks = [registry_task, runtime_task]
            
            # Wait for all services
            await asyncio.gather(*self.tasks)
            
        except asyncio.CancelledError:
            print("✅ All services stopped successfully")
        except Exception as e:
            print(f"❌ Service orchestrator failed: {e}")
            sys.exit(1)


async def main():
    """Main entry point"""
    orchestrator = ServiceOrchestrator()
    await orchestrator.start_services()


if __name__ == "__main__":
    asyncio.run(main())