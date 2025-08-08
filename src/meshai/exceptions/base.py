"""
Base exceptions for MeshAI SDK
"""

from typing import Optional, Dict, Any


class MeshAIError(Exception):
    """Base exception for all MeshAI SDK errors"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AgentNotFoundError(MeshAIError):
    """Raised when an agent cannot be found or discovered"""
    
    def __init__(
        self, 
        message: str = "Agent not found",
        agent_id: Optional[str] = None,
        capabilities: Optional[list] = None
    ):
        super().__init__(message, "AGENT_NOT_FOUND")
        self.agent_id = agent_id
        self.capabilities = capabilities


class TaskExecutionError(MeshAIError):
    """Raised when task execution fails"""
    
    def __init__(
        self,
        message: str = "Task execution failed", 
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message, "TASK_EXECUTION_FAILED")
        self.task_id = task_id
        self.agent_id = agent_id
        self.original_error = original_error


class RegistrationError(MeshAIError):
    """Raised when agent registration fails"""
    
    def __init__(
        self,
        message: str = "Agent registration failed",
        agent_id: Optional[str] = None
    ):
        super().__init__(message, "REGISTRATION_FAILED")
        self.agent_id = agent_id


class RoutingError(MeshAIError):
    """Raised when task routing fails"""
    
    def __init__(
        self,
        message: str = "Task routing failed",
        task_id: Optional[str] = None,
        capabilities: Optional[list] = None
    ):
        super().__init__(message, "ROUTING_FAILED")
        self.task_id = task_id
        self.capabilities = capabilities


class ContextError(MeshAIError):
    """Raised when context management operations fail"""
    
    def __init__(
        self,
        message: str = "Context operation failed",
        context_id: Optional[str] = None
    ):
        super().__init__(message, "CONTEXT_ERROR")
        self.context_id = context_id


class AuthenticationError(MeshAIError):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_FAILED")


class ValidationError(MeshAIError):
    """Raised when data validation fails"""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Optional[Any] = None
    ):
        super().__init__(message, "VALIDATION_FAILED")
        self.field = field
        self.value = value


class TimeoutError(MeshAIError):
    """Raised when operations timeout"""
    
    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None
    ):
        super().__init__(message, "TIMEOUT")
        self.timeout_seconds = timeout_seconds


class ConfigurationError(MeshAIError):
    """Raised when configuration is invalid or missing"""
    
    def __init__(
        self,
        message: str = "Configuration error", 
        config_key: Optional[str] = None
    ):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key