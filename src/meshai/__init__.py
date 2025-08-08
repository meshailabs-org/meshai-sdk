"""
MeshAI SDK - AI Agent Interoperability Platform

This SDK enables seamless integration and communication between AI agents
built on different frameworks (LangChain, CrewAI, AutoGen, etc.).
"""

from .core.agent import MeshAgent, register_agent
from .core.config import MeshConfig
from .core.context import MeshContext
from .clients.registry import RegistryClient
from .clients.runtime import RuntimeClient
from .exceptions.base import MeshAIError, AgentNotFoundError, TaskExecutionError

__version__ = "0.1.0"
__author__ = "MeshAI Labs"
__email__ = "dev@meshai.dev"

__all__ = [
    # Core classes
    "MeshAgent",
    "register_agent", 
    "MeshConfig",
    "MeshContext",
    
    # Clients
    "RegistryClient",
    "RuntimeClient",
    
    # Exceptions
    "MeshAIError",
    "AgentNotFoundError", 
    "TaskExecutionError",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
]