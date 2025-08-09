#!/usr/bin/env python3
"""
Initialize MeshAI Database

This script initializes the database schema for MeshAI.
Supports both SQLite (development) and PostgreSQL (production).
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.database.migrations import init_database, get_current_revision
from meshai.database.session import DatabaseManager


async def test_database_connection():
    """Test database connectivity"""
    print("🔍 Testing database connection...")
    
    db_manager = DatabaseManager()
    
    try:
        is_connected = await db_manager.test_connection()
        
        if is_connected:
            print("✅ Database connection successful")
            
            # Get current revision
            current_rev = get_current_revision()
            if current_rev:
                print(f"📊 Current database revision: {current_rev}")
            else:
                print("📊 Database is not initialized")
            
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False
    finally:
        await db_manager.close()


def main():
    """Main initialization function"""
    print("🗄️  MeshAI Database Initialization")
    print("=" * 50)
    
    # Show configuration
    import os
    db_url = os.getenv("MESHAI_DATABASE_URL")
    pg_host = os.getenv("MESHAI_POSTGRES_HOST")
    sqlite_path = os.getenv("MESHAI_SQLITE_PATH", "meshai.db")
    
    if db_url:
        print(f"📍 Database URL: {db_url}")
    elif pg_host:
        print(f"📍 PostgreSQL Host: {pg_host}")
    else:
        print(f"📍 SQLite Path: {sqlite_path}")
    
    print("=" * 50)
    
    try:
        # Test connection first
        connection_ok = asyncio.run(test_database_connection())
        
        if not connection_ok:
            print("\n💡 Database connection failed. Check your configuration:")
            print("   - For SQLite: Ensure the directory is writable")
            print("   - For PostgreSQL: Verify host, credentials, and network")
            sys.exit(1)
        
        print("\n🚀 Initializing database schema...")
        
        # Initialize database
        init_database()
        
        print("✅ Database initialization completed successfully!")
        
        # Test connection again
        print("\n🔍 Final connectivity test...")
        final_test = asyncio.run(test_database_connection())
        
        if final_test:
            print("🎉 Database is ready for use!")
            print("\n💡 Next steps:")
            print("   - Start MeshAI services: python scripts/start-all.py")
            print("   - Run integration test: python scripts/integration-test.py")
        else:
            print("⚠️  Database initialized but connection test failed")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()