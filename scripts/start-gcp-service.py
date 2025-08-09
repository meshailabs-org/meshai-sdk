#!/usr/bin/env python3
"""
GCP Cloud Run Service Starter

Starts individual Registry or Runtime services based on SERVICE_TYPE environment variable.
This enables running separate Cloud Run services from the same Docker image.
"""

import os
import sys
import asyncio
import signal
from typing import Optional

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from meshai.services.database_registry_service import run_server as run_registry_server
from meshai.services.runtime_service import run_server as run_runtime_server
from meshai.core.config import get_config
from meshai.database.migrations import init_database
from meshai.utils.logging import setup_logging
import structlog

logger = structlog.get_logger(__name__)


def setup_environment():
    """Setup environment and logging"""
    config = get_config()
    setup_logging(config.log_level, "gcp-starter")
    
    # Initialize database (creates tables if they don't exist)
    try:
        init_database()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue anyway - might be a connection issue that resolves


async def run_service():
    """Run the appropriate service based on SERVICE_TYPE environment variable"""
    service_type = os.getenv("SERVICE_TYPE", "registry").lower()
    
    # Cloud Run sets PORT environment variable automatically
    port = int(os.getenv("PORT", "8080"))  # Cloud Run default
    host = os.getenv("HOST", "0.0.0.0")
    
    if service_type == "registry":
        logger.info(f"Starting Registry Service on {host}:{port}")
        await run_registry_server(host=host, port=port)
        
    elif service_type == "runtime":
        logger.info(f"Starting Runtime Service on {host}:{port}")
        await run_runtime_server(host=host, port=port)
        
    else:
        logger.error(f"Invalid SERVICE_TYPE: {service_type}. Must be 'registry' or 'runtime'")
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting MeshAI GCP Service...")
    
    # Setup environment
    setup_environment()
    
    # Display configuration
    service_type = os.getenv("SERVICE_TYPE", "registry").lower()
    port = int(os.getenv("PORT", "8080"))  # Cloud Run sets this automatically
    
    logger.info("Service Configuration:", extra={
        "service_type": service_type,
        "port": port,
        "host": os.getenv("HOST", "0.0.0.0"),
        "database_url": os.getenv("MESHAI_DATABASE_URL", "Not set"),
        "redis_url": os.getenv("MESHAI_REDIS_URL", "Not set"),
        "environment": os.getenv("ENVIRONMENT", "dev")
    })
    
    # Start the service
    try:
        asyncio.run(run_service())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()