"""
Configuration management for MeshAI SDK
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator


@dataclass
class MeshConfig:
    """MeshAI SDK Configuration"""
    
    # Service URLs
    registry_url: str = field(default_factory=lambda: os.getenv("MESHAI_REGISTRY_URL", "http://localhost:8001"))
    runtime_url: str = field(default_factory=lambda: os.getenv("MESHAI_RUNTIME_URL", "http://localhost:8002"))
    
    # Authentication
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("MESHAI_API_KEY"))
    auth_token: Optional[str] = field(default_factory=lambda: os.getenv("MESHAI_AUTH_TOKEN"))
    
    # Agent configuration
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_port: int = field(default_factory=lambda: int(os.getenv("MESHAI_AGENT_PORT", "8000")))
    agent_host: str = field(default_factory=lambda: os.getenv("MESHAI_AGENT_HOST", "0.0.0.0"))
    
    # Timeouts and retries
    default_timeout: int = field(default_factory=lambda: int(os.getenv("MESHAI_DEFAULT_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MESHAI_MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("MESHAI_RETRY_DELAY", "1.0")))
    
    # Health checks
    health_check_interval: int = field(default_factory=lambda: int(os.getenv("MESHAI_HEALTH_CHECK_INTERVAL", "30")))
    health_check_timeout: int = field(default_factory=lambda: int(os.getenv("MESHAI_HEALTH_CHECK_TIMEOUT", "5")))
    
    # Logging and monitoring
    log_level: str = field(default_factory=lambda: os.getenv("MESHAI_LOG_LEVEL", "INFO"))
    enable_metrics: bool = field(default_factory=lambda: os.getenv("MESHAI_ENABLE_METRICS", "true").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("MESHAI_METRICS_PORT", "9090")))
    
    # Context management
    context_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("MESHAI_CONTEXT_TTL", "3600")))
    enable_context_sharing: bool = field(default_factory=lambda: os.getenv("MESHAI_ENABLE_CONTEXT_SHARING", "true").lower() == "true")
    
    # Performance
    max_concurrent_tasks: int = field(default_factory=lambda: int(os.getenv("MESHAI_MAX_CONCURRENT_TASKS", "10")))
    task_queue_size: int = field(default_factory=lambda: int(os.getenv("MESHAI_TASK_QUEUE_SIZE", "100")))
    
    # Development settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("MESHAI_DEBUG", "false").lower() == "true")
    auto_register: bool = field(default_factory=lambda: os.getenv("MESHAI_AUTO_REGISTER", "true").lower() == "true")
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.default_timeout <= 0:
            raise ValueError("default_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.agent_port <= 0 or self.agent_port > 65535:
            raise ValueError("agent_port must be between 1 and 65535")
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
    
    @classmethod
    def from_env(cls, **overrides) -> "MeshConfig":
        """Create configuration from environment variables with optional overrides"""
        config = cls()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MeshConfig":
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def update(self, **updates) -> "MeshConfig":
        """Create new configuration with updates"""
        config_dict = self.to_dict()
        config_dict.update(updates)
        return self.from_dict(config_dict)
    
    @property 
    def registry_base_url(self) -> str:
        """Get base registry URL without trailing slash"""
        return self.registry_url.rstrip("/")
    
    @property
    def runtime_base_url(self) -> str:
        """Get base runtime URL without trailing slash"""
        return self.runtime_url.rstrip("/")
    
    @property
    def agent_endpoint(self) -> str:
        """Get agent endpoint URL"""
        if self.agent_host == "0.0.0.0":
            # Use localhost for external access
            return f"http://localhost:{self.agent_port}"
        return f"http://{self.agent_host}:{self.agent_port}"
    
    def get(self, key: str, default=None):
        """Get configuration value by key with optional default"""
        return getattr(self, key, default)
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"meshai-sdk/0.1.0"
        }
        
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        return headers


class ConfigManager:
    """Global configuration manager"""
    
    _instance: Optional[MeshConfig] = None
    
    @classmethod
    def get_config(cls) -> MeshConfig:
        """Get global configuration instance"""
        if cls._instance is None:
            cls._instance = MeshConfig.from_env()
        return cls._instance
    
    @classmethod
    def set_config(cls, config: MeshConfig) -> None:
        """Set global configuration instance"""
        cls._instance = config
    
    @classmethod
    def reset_config(cls) -> None:
        """Reset configuration to default"""
        cls._instance = None


# Global configuration instance
def get_config() -> MeshConfig:
    """Get the global MeshAI configuration"""
    return ConfigManager.get_config()


def set_config(config: MeshConfig) -> None:
    """Set the global MeshAI configuration"""
    ConfigManager.set_config(config)