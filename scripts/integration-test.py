#!/usr/bin/env python3
"""
MeshAI Integration Test

Quick integration test to verify the complete system is working.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.core.config import MeshConfig
from meshai.clients.registry import RegistryClient
from meshai.clients.runtime import RuntimeClient


async def test_services():
    """Test that services are running and responding"""
    config = MeshConfig()
    registry_client = RegistryClient(config)
    runtime_client = RuntimeClient(config)
    
    print("🔍 Testing service connectivity...")
    
    try:
        # Test Registry service
        registry_health = await registry_client.get_registry_health()
        print(f"✅ Registry service: {registry_health.get('status', 'unknown')}")
        
        # Test Runtime service
        runtime_health = await runtime_client.get_runtime_health()
        print(f"✅ Runtime service: {runtime_health.get('status', 'unknown')}")
        
        # List agents
        agents = await registry_client.list_agents()
        print(f"✅ Found {len(agents)} registered agents")
        
        # Get runtime stats
        stats = await runtime_client.get_runtime_stats()
        print(f"✅ Runtime stats - Active: {stats.get('active_tasks', 0)}, "
              f"Completed: {stats.get('completed_tasks', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False
    
    finally:
        await registry_client.close()
        await runtime_client.close()


async def main():
    """Main test function"""
    print("🧪 MeshAI Integration Test")
    print("=" * 40)
    
    success = await test_services()
    
    if success:
        print("=" * 40)
        print("🎉 Integration test PASSED!")
        print("✅ All services are running correctly")
        print("✅ API endpoints are responsive")
        print("✅ System is ready for use")
        print("=" * 40)
        print("💡 Try running the complete demo:")
        print("   python examples/complete_system_demo.py")
    else:
        print("=" * 40)
        print("❌ Integration test FAILED!")
        print("⚠️  Services may not be running")
        print("💡 Start services with:")
        print("   python scripts/start-all.py")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())