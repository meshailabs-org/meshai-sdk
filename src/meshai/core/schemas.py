"""
Core data schemas for MeshAI SDK
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    ROUTING = "routing" 
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RoutingStrategy(str, Enum):
    """Task routing strategy"""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE_BASED = "performance_based"
    LEAST_LOADED = "least_loaded"
    STICKY_SESSION = "sticky_session"


class TaskData(BaseModel):
    """Task data structure"""
    model_config = ConfigDict(extra="allow")
    
    task_id: Optional[str] = None
    task_type: str = Field(..., description="Type of task to execute")
    input: Union[str, Dict[str, Any]] = Field(..., description="Task input data")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    
    # Execution requirements
    required_capabilities: List[str] = Field(default_factory=list)
    preferred_framework: Optional[str] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH
    timeout_seconds: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)
    
    # Metadata
    created_at: Optional[datetime] = None
    source_agent: Optional[str] = None
    correlation_id: Optional[str] = None


class TaskResult(BaseModel):
    """Task execution result"""
    model_config = ConfigDict(extra="allow")
    
    task_id: str
    status: TaskStatus
    result: Optional[Union[str, Dict[str, Any]]] = None
    error: Optional[str] = None
    
    # Execution info
    agent_id: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    retry_count: int = 0
    routing_strategy_used: Optional[RoutingStrategy] = None
    context_updates: Dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    """Agent information and metadata"""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    framework: str = Field(..., description="Agent framework (langchain, crewai, etc.)")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    
    # Endpoints and configuration
    endpoint: Optional[str] = None
    health_endpoint: Optional[str] = None
    
    # Status and performance
    status: str = "active"
    health_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_response_time_ms: Optional[int] = Field(None, ge=0)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    current_load: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRegistration(BaseModel):
    """Agent registration data"""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name") 
    framework: str = Field(..., description="Agent framework")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    
    # Configuration
    endpoint: Optional[str] = None
    health_endpoint: Optional[str] = None
    max_concurrent_tasks: int = Field(default=10, gt=0)
    
    # Schemas (optional JSON schemas for validation)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    # Metadata
    description: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextData(BaseModel):
    """Shared context data structure"""
    model_config = ConfigDict(extra="allow")
    
    context_id: str
    session_id: Optional[str] = None
    
    # Memory components
    shared_memory: Dict[str, Any] = Field(default_factory=dict)
    agent_memory: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    access_count: int = 0


class DiscoveryQuery(BaseModel):
    """Agent discovery query"""
    model_config = ConfigDict(extra="allow")
    
    capabilities: Optional[List[str]] = None
    framework: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Performance filters
    min_success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_response_time_ms: Optional[int] = Field(None, gt=0)
    max_load: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Query options
    exclude_agents: List[str] = Field(default_factory=list)
    limit: Optional[int] = Field(None, gt=0)
    include_offline: bool = False


class WebhookEvent(BaseModel):
    """Webhook event structure"""
    model_config = ConfigDict(extra="allow")
    
    event_type: str
    agent_id: str
    timestamp: datetime
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Event metadata
    event_id: Optional[str] = None
    source: Optional[str] = None
    correlation_id: Optional[str] = None