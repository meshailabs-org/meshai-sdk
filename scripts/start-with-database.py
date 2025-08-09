#!/usr/bin/env python3
"""
Start MeshAI Services with Database Integration

This script initializes the database and starts services with persistent storage.
"""

import asyncio
import sys
import os
import signal
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.services.database_registry_service import run_server as run_registry
from meshai.services.runtime_service import run_server as run_runtime
from meshai.database.migrations import init_database, get_current_revision
from meshai.database.session import DatabaseManager
from meshai.core.config import MeshConfig


class DatabaseServiceOrchestrator:
    """Orchestrates MeshAI services with database integration"""
    
    def __init__(self):
        self.config = MeshConfig.from_env()
        self.running = True
        self.tasks = []
        
        # Database configuration info
        self.db_url = os.getenv("MESHAI_DATABASE_URL")
        self.pg_host = os.getenv("MESHAI_POSTGRES_HOST")
        self.sqlite_path = os.getenv("MESHAI_SQLITE_PATH", "meshai.db")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⚠️  Received signal {signum}, shutting down services...")
        self.running = False
        for task in self.tasks:
            task.cancel()
    
    async def initialize_database(self):
        """Initialize database schema"""
        print("🗄️  Database Initialization")
        print("-" * 40)
        
        # Show database configuration
        if self.db_url:
            print(f"📍 Database URL: {self.db_url}")
        elif self.pg_host:
            print(f"📍 PostgreSQL Host: {self.pg_host}")
            print(f"📍 Database: {os.getenv('MESHAI_POSTGRES_DATABASE', 'meshai')}")
        else:
            print(f"📍 SQLite Path: {self.sqlite_path}")
        
        try:
            # Test database connection
            print("🔍 Testing database connectivity...")
            db_manager = DatabaseManager()
            
            if await db_manager.test_connection():
                print("✅ Database connection successful")
            else:
                print("❌ Database connection failed")
                return False
            
            # Check current revision
            current_rev = get_current_revision()
            if current_rev:
                print(f"📊 Current revision: {current_rev}")
            else:
                print("📊 Database not initialized")
            
            # Initialize database
            print("🚀 Initializing database schema...")
            init_database()
            
            # Verify initialization
            final_rev = get_current_revision()
            print(f"✅ Database initialized (revision: {final_rev})")
            
            await db_manager.close()
            return True
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            if "connection" in str(e).lower():
                print("\n💡 Database connection issues:")
                if self.pg_host:
                    print("   - Verify PostgreSQL server is running")
                    print("   - Check host, port, credentials, and network connectivity")
                    print("   - Ensure database exists")
                else:
                    print("   - Check SQLite file permissions")
                    print("   - Ensure parent directory exists and is writable")
            return False
    
    async def start_services(self):
        """Start all services with database integration"""
        print("\n" + "=" * 60)
        print("🚀 MeshAI Service Orchestrator (Database Edition)")
        print("=" * 60)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Initialize database first
            if not await self.initialize_database():
                print("\n❌ Database initialization failed. Cannot start services.")
                return
            
            print("\n📡 Starting Services")
            print("-" * 40)
            print("✅ Registry Service (Database): http://localhost:8001")
            print("✅ Runtime Service: http://localhost:8002")
            print("✅ Health Checks: /health endpoints available")
            print("✅ API Documentation: /docs endpoints available")
            print("✅ Metrics: /metrics endpoints available")
            print("-" * 40)
            print("Press Ctrl+C to stop all services")
            print("=" * 60)
            
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
            import traceback
            traceback.print_exc()
            sys.exit(1)


def show_configuration():
    """Show current configuration"""
    print("⚙️  Configuration")
    print("-" * 40)
    
    # Environment variables
    env_vars = [
        ("MESHAI_DATABASE_URL", "Database connection URL"),
        ("MESHAI_POSTGRES_HOST", "PostgreSQL host"),
        ("MESHAI_POSTGRES_USER", "PostgreSQL user"),
        ("MESHAI_POSTGRES_DATABASE", "PostgreSQL database"),
        ("MESHAI_POSTGRES_PORT", "PostgreSQL port"),
        ("MESHAI_SQLITE_PATH", "SQLite database path"),
        ("MESHAI_DEBUG", "Debug mode"),
        ("MESHAI_LOG_LEVEL", "Logging level")
    ]
    
    for var, description in env_vars:
        value = os.getenv(var)
        if value:
            # Mask passwords
            if "PASSWORD" in var and value:
                value = "*" * len(value)
            print(f"✅ {var}: {value}")
    
    print()


async def main():
    """Main entry point"""
    show_configuration()
    
    orchestrator = DatabaseServiceOrchestrator()
    await orchestrator.start_services()


if __name__ == "__main__":
    # Set some development defaults
    if not os.getenv("MESHAI_DEBUG"):
        os.environ["MESHAI_DEBUG"] = "true"
    if not os.getenv("MESHAI_LOG_LEVEL"):
        os.environ["MESHAI_LOG_LEVEL"] = "INFO"
    
    asyncio.run(main())