"""
SQLAlchemy Database Models for MeshAI

Defines all database tables and relationships for persistent storage.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Float, Boolean, JSON, ForeignKey,
    Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID
import sqlalchemy.dialects.postgresql as postgresql
import uuid

Base = declarative_base()


class Agent(Base):
    """Agent registration and metadata table"""
    __tablename__ = "agents"
    
    # Primary identification
    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    framework = Column(String(100), nullable=False)
    
    # Configuration
    endpoint = Column(String(500), nullable=True)
    health_endpoint = Column(String(500), nullable=True)
    max_concurrent_tasks = Column(Integer, default=10)
    
    # Capabilities and metadata
    capabilities = Column(JSON, nullable=False, default=list)
    tags = Column(JSON, nullable=False, default=list)
    agent_metadata = Column(JSON, nullable=False, default=dict)
    
    # Schema definitions
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)
    
    # Status and performance
    status = Column(String(50), default="active")
    health_score = Column(Float, nullable=True)
    avg_response_time_ms = Column(Integer, nullable=True)
    success_rate = Column(Float, nullable=True)
    current_load = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=True)
    
    # Version and description
    version = Column(String(50), default="1.0.0")
    description = Column(Text, nullable=True)
    
    # Relationships
    tasks = relationship("Task", back_populates="agent", cascade="all, delete-orphan")
    heartbeats = relationship("AgentHeartbeat", back_populates="agent", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_agents_status", "status"),
        Index("ix_agents_framework", "framework"),
        Index("ix_agents_created_at", "created_at"),
        Index("ix_agents_last_seen", "last_seen_at"),
        CheckConstraint("health_score >= 0 AND health_score <= 1", name="check_health_score"),
        CheckConstraint("success_rate >= 0 AND success_rate <= 1", name="check_success_rate"),
        CheckConstraint("current_load >= 0 AND current_load <= 1", name="check_current_load"),
    )
    
    @validates('capabilities')
    def validate_capabilities(self, key, capabilities):
        if not isinstance(capabilities, list):
            raise ValueError("Capabilities must be a list")
        return capabilities
    
    @validates('tags')
    def validate_tags(self, key, tags):
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list")
        return tags
    
    @validates('agent_metadata')
    def validate_agent_metadata(self, key, agent_metadata):
        if not isinstance(agent_metadata, dict):
            raise ValueError("Agent metadata must be a dictionary")
        return agent_metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "name": self.name,
            "framework": self.framework,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "health_endpoint": self.health_endpoint,
            "status": self.status,
            "health_score": self.health_score,
            "avg_response_time_ms": self.avg_response_time_ms,
            "success_rate": self.success_rate,
            "current_load": self.current_load,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "metadata": self.agent_metadata
        }


class Task(Base):
    """Task execution records table"""
    __tablename__ = "tasks"
    
    # Primary identification
    task_id = Column(String(255), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_type = Column(String(255), nullable=False)
    
    # Task data
    input_data = Column(JSON, nullable=False)
    parameters = Column(JSON, nullable=False, default=dict)
    context_data = Column(JSON, nullable=False, default=dict)
    
    # Execution requirements
    required_capabilities = Column(JSON, nullable=False, default=list)
    preferred_framework = Column(String(100), nullable=True)
    routing_strategy = Column(String(100), default="capability_match")
    timeout_seconds = Column(Integer, default=30)
    max_retries = Column(Integer, default=3)
    
    # Status and results
    status = Column(String(50), default="pending")
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Execution info
    agent_id = Column(String(255), ForeignKey("agents.id"), nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    retry_count = Column(Integer, default=0)
    routing_strategy_used = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Metadata
    source_agent = Column(String(255), nullable=True)
    correlation_id = Column(String(255), nullable=True)
    context_updates = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    agent = relationship("Agent", back_populates="tasks")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_tasks_status", "status"),
        Index("ix_tasks_agent_id", "agent_id"),
        Index("ix_tasks_created_at", "created_at"),
        Index("ix_tasks_source_agent", "source_agent"),
        Index("ix_tasks_routing_strategy", "routing_strategy"),
        Index("ix_tasks_correlation_id", "correlation_id"),
        CheckConstraint("timeout_seconds > 0", name="check_timeout_positive"),
        CheckConstraint("max_retries >= 0", name="check_max_retries_non_negative"),
    )
    
    @validates('required_capabilities')
    def validate_required_capabilities(self, key, capabilities):
        if not isinstance(capabilities, list):
            raise ValueError("Required capabilities must be a list")
        return capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result_data,
            "error": self.error_message,
            "agent_id": self.agent_id,
            "execution_time_seconds": self.execution_time_seconds,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "routing_strategy_used": self.routing_strategy_used,
            "context_updates": self.context_updates
        }


class Context(Base):
    """Shared context storage table"""
    __tablename__ = "contexts"
    
    # Primary identification
    context_id = Column(String(255), primary_key=True)
    session_id = Column(String(255), nullable=True)
    
    # Memory components
    shared_memory = Column(JSON, nullable=False, default=dict)
    agent_memory = Column(JSON, nullable=False, default=dict)
    
    # Access control
    owner_agent_id = Column(String(255), nullable=True)
    access_permissions = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)
    
    # Version management
    version = Column(Integer, default=1)
    parent_version = Column(Integer, nullable=True)
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_contexts_session_id", "session_id"),
        Index("ix_contexts_owner", "owner_agent_id"),
        Index("ix_contexts_created_at", "created_at"),
        Index("ix_contexts_expires_at", "expires_at"),
        CheckConstraint("version > 0", name="check_version_positive"),
    )
    
    @validates('shared_memory')
    def validate_shared_memory(self, key, memory):
        if not isinstance(memory, dict):
            raise ValueError("Shared memory must be a dictionary")
        return memory
    
    @validates('agent_memory')
    def validate_agent_memory(self, key, memory):
        if not isinstance(memory, dict):
            raise ValueError("Agent memory must be a dictionary")
        return memory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "shared_memory": self.shared_memory,
            "agent_memory": self.agent_memory,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count
        }


class AgentHeartbeat(Base):
    """Agent heartbeat and health monitoring table"""
    __tablename__ = "agent_heartbeats"
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(255), ForeignKey("agents.id"), nullable=False)
    
    # Heartbeat data
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), nullable=False)
    
    # Performance metrics
    response_time_ms = Column(Float, nullable=True)
    cpu_usage = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    active_tasks = Column(Integer, default=0)
    
    # Health indicators
    health_score = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)
    
    # Additional metrics
    metrics_data = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    agent = relationship("Agent", back_populates="heartbeats")
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_heartbeats_agent_timestamp", "agent_id", "timestamp"),
        Index("ix_heartbeats_timestamp", "timestamp"),
        Index("ix_heartbeats_status", "status"),
        CheckConstraint("health_score >= 0 AND health_score <= 1", name="check_heartbeat_health_score"),
        CheckConstraint("error_rate >= 0 AND error_rate <= 1", name="check_heartbeat_error_rate"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "response_time_ms": self.response_time_ms,
            "cpu_usage": self.cpu_usage,
            "memory_usage_mb": self.memory_usage_mb,
            "active_tasks": self.active_tasks,
            "health_score": self.health_score,
            "error_rate": self.error_rate,
            "metrics_data": self.metrics_data
        }


class ContextHistory(Base):
    """Context version history table"""
    __tablename__ = "context_history"
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    context_id = Column(String(255), nullable=False)
    version_id = Column(String(255), nullable=False)
    
    # Version data
    version_number = Column(Integer, nullable=False)
    change_type = Column(String(50), nullable=False)  # create, update, delete
    changed_by = Column(String(255), nullable=False)  # agent_id
    
    # Data snapshot
    data_snapshot = Column(JSON, nullable=False)
    changes_made = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_context_history_context_version", "context_id", "version_number"),
        Index("ix_context_history_timestamp", "timestamp"),
        Index("ix_context_history_changed_by", "changed_by"),
        UniqueConstraint("context_id", "version_number", name="uq_context_version"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "context_id": self.context_id,
            "version_id": self.version_id,
            "version_number": self.version_number,
            "change_type": self.change_type,
            "changed_by": self.changed_by,
            "data_snapshot": self.data_snapshot,
            "changes_made": self.changes_made,
            "timestamp": self.timestamp,
            "description": self.description
        }