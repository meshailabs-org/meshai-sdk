#!/usr/bin/env python3
"""
Start MeshAI Runtime Service

This script starts the MeshAI Runtime service for task orchestration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.services.runtime_service import run_server
from meshai.core.config import MeshConfig


async def main():
    """Start the runtime service"""
    # Load configuration
    config = MeshConfig.from_env()
    
    # Override with command line args if provided
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8002
    
    print("=" * 60)
    print("🚀 MeshAI Runtime Service")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug Mode: {config.debug_mode}")
    print(f"Log Level: {config.log_level}")
    print(f"Registry URL: {config.registry_url}")
    print("=" * 60)
    
    try:
        await run_server(config, host, port)
    except KeyboardInterrupt:
        print("\n⚠️  Runtime service stopped by user")
    except Exception as e:
        print(f"❌ Runtime service failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())