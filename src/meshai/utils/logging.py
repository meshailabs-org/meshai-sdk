"""
Logging utilities for MeshAI SDK
"""

import sys
import logging
from typing import Optional

import structlog


def setup_logging(level: str = "INFO", agent_id: Optional[str] = None) -> None:
    """
    Setup structured logging for MeshAI SDK.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        agent_id: Optional agent ID to include in logs
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO", utc=True),
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Add agent_id to context if provided
    if agent_id:
        structlog.contextvars.bind_contextvars(agent_id=agent_id)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger
    """
    return structlog.get_logger(name)