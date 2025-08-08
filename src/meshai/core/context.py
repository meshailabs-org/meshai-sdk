"""
Context management for MeshAI agents
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import asyncio
from contextlib import asynccontextmanager

import structlog
from pydantic import BaseModel

from .config import MeshConfig
from .schemas import ContextData
from ..exceptions.base import ContextError

logger = structlog.get_logger(__name__)


class MeshContext:
    """
    Context manager for sharing data and state between agents.
    
    Provides functionality for:
    - Shared memory across agent interactions
    - Agent-specific memory isolation
    - Context persistence and expiration
    - Thread-safe access to context data
    """
    
    def __init__(
        self,
        config: Optional[MeshConfig] = None,
        context_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        from .config import get_config
        self.config = config or get_config()
        self.context_id = context_id or str(uuid.uuid4())
        self.session_id = session_id or str(uuid.uuid4())
        
        # In-memory storage for context data
        self._shared_memory: Dict[str, Any] = {}
        self._agent_memory: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        
        # Context metadata
        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()
        self._access_count = 0
        self._locks: Dict[str, asyncio.Lock] = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[MeshConfig] = None) -> "MeshContext":
        """Create context from dictionary data"""
        context = cls(
            config=config,
            context_id=data.get("context_id"),
            session_id=data.get("session_id")
        )
        
        context._shared_memory = data.get("shared_memory", {})
        context._agent_memory = data.get("agent_memory", {})
        context._metadata = data.get("metadata", {})
        
        # Parse timestamps if present
        if "created_at" in data:
            context._created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            context._updated_at = datetime.fromisoformat(data["updated_at"])
        
        context._access_count = data.get("access_count", 0)
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "shared_memory": self._shared_memory.copy(),
            "agent_memory": self._agent_memory.copy(),
            "metadata": self._metadata.copy(),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "access_count": self._access_count
        }
    
    async def get(self, key: str, default: Any = None, agent_scope: bool = False) -> Any:
        """
        Get value from context.
        
        Args:
            key: The key to retrieve
            default: Default value if key not found
            agent_scope: If True, get from agent-specific memory
            
        Returns:
            The value associated with the key
        """
        await self._track_access()
        
        if agent_scope:
            return self._agent_memory.get(key, default)
        else:
            return self._shared_memory.get(key, default)
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        agent_scope: bool = False,
        agent_id: Optional[str] = None
    ) -> None:
        """
        Set value in context.
        
        Args:
            key: The key to set
            value: The value to store
            agent_scope: If True, store in agent-specific memory
            agent_id: Agent ID for agent-specific storage
        """
        async with self._get_lock(key):
            await self._track_access()
            
            if agent_scope:
                if agent_id:
                    # Store in specific agent's memory
                    if agent_id not in self._agent_memory:
                        self._agent_memory[agent_id] = {}
                    self._agent_memory[agent_id][key] = value
                else:
                    # Store in current agent's memory
                    self._agent_memory[key] = value
            else:
                # Store in shared memory
                self._shared_memory[key] = value
            
            self._updated_at = datetime.utcnow()
    
    async def update(
        self, 
        updates: Dict[str, Any], 
        agent_scope: bool = False,
        agent_id: Optional[str] = None
    ) -> None:
        """
        Update multiple values in context.
        
        Args:
            updates: Dictionary of key-value pairs to update
            agent_scope: If True, update agent-specific memory
            agent_id: Agent ID for agent-specific updates
        """
        for key, value in updates.items():
            await self.set(key, value, agent_scope=agent_scope, agent_id=agent_id)
    
    async def delete(self, key: str, agent_scope: bool = False) -> bool:
        """
        Delete key from context.
        
        Args:
            key: The key to delete
            agent_scope: If True, delete from agent-specific memory
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        async with self._get_lock(key):
            await self._track_access()
            
            if agent_scope:
                if key in self._agent_memory:
                    del self._agent_memory[key]
                    self._updated_at = datetime.utcnow()
                    return True
            else:
                if key in self._shared_memory:
                    del self._shared_memory[key]
                    self._updated_at = datetime.utcnow()
                    return True
            
            return False
    
    async def has(self, key: str, agent_scope: bool = False) -> bool:
        """
        Check if key exists in context.
        
        Args:
            key: The key to check
            agent_scope: If True, check agent-specific memory
            
        Returns:
            True if key exists
        """
        if agent_scope:
            return key in self._agent_memory
        else:
            return key in self._shared_memory
    
    async def keys(self, agent_scope: bool = False) -> List[str]:
        """
        Get all keys in context.
        
        Args:
            agent_scope: If True, get agent-specific keys
            
        Returns:
            List of keys
        """
        if agent_scope:
            return list(self._agent_memory.keys())
        else:
            return list(self._shared_memory.keys())
    
    async def clear(self, agent_scope: bool = False) -> None:
        """
        Clear context data.
        
        Args:
            agent_scope: If True, clear only agent-specific memory
        """
        await self._track_access()
        
        if agent_scope:
            self._agent_memory.clear()
        else:
            self._shared_memory.clear()
            self._agent_memory.clear()  # Clear all agent memory too
        
        self._updated_at = datetime.utcnow()
    
    def get_shared_memory(self) -> Dict[str, Any]:
        """Get copy of shared memory"""
        return self._shared_memory.copy()
    
    def get_agent_memory(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get agent-specific memory.
        
        Args:
            agent_id: Specific agent ID to get memory for
            
        Returns:
            Dictionary of agent memory
        """
        if agent_id and agent_id in self._agent_memory:
            return self._agent_memory[agent_id].copy()
        return self._agent_memory.copy()
    
    async def share_with_agent(
        self, 
        target_agent_id: str, 
        keys: List[str],
        source_agent_id: Optional[str] = None
    ) -> None:
        """
        Share specific context data with another agent.
        
        Args:
            target_agent_id: ID of agent to share data with
            keys: List of keys to share
            source_agent_id: Source agent ID (if sharing agent-specific data)
        """
        await self._track_access()
        
        if target_agent_id not in self._agent_memory:
            self._agent_memory[target_agent_id] = {}
        
        source_memory = self._shared_memory
        if source_agent_id and source_agent_id in self._agent_memory:
            source_memory = self._agent_memory[source_agent_id]
        
        for key in keys:
            if key in source_memory:
                self._agent_memory[target_agent_id][key] = source_memory[key]
        
        self._updated_at = datetime.utcnow()
    
    async def merge_context(self, other_context: "MeshContext") -> None:
        """
        Merge another context into this one.
        
        Args:
            other_context: Context to merge from
        """
        await self._track_access()
        
        # Merge shared memory
        other_shared = other_context.get_shared_memory()
        self._shared_memory.update(other_shared)
        
        # Merge agent memory
        other_agent = other_context.get_agent_memory()
        for agent_id, memory in other_agent.items():
            if agent_id not in self._agent_memory:
                self._agent_memory[agent_id] = {}
            self._agent_memory[agent_id].update(memory)
        
        self._updated_at = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if context has expired"""
        if self.config.context_ttl_seconds <= 0:
            return False  # No expiration
        
        expiry_time = self._created_at + timedelta(seconds=self.config.context_ttl_seconds)
        return datetime.utcnow() > expiry_time
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get context metadata"""
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "access_count": self._access_count,
            "is_expired": self.is_expired(),
            "shared_keys_count": len(self._shared_memory),
            "agent_memory_count": len(self._agent_memory)
        }
    
    async def _track_access(self) -> None:
        """Track context access for metrics"""
        self._access_count += 1
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for specific key"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for atomic context operations.
        
        Example:
            async with context.transaction():
                await context.set("key1", "value1")
                await context.set("key2", "value2")
                # Both operations committed together
        """
        # Create snapshot for rollback
        snapshot = self.to_dict()
        
        try:
            yield self
        except Exception as e:
            # Rollback on error
            restored_context = MeshContext.from_dict(snapshot, self.config)
            self._shared_memory = restored_context._shared_memory
            self._agent_memory = restored_context._agent_memory
            self._metadata = restored_context._metadata
            logger.warning(f"Context transaction rolled back due to error: {e}")
            raise
    
    async def save_to_persistent_store(self) -> None:
        """
        Save context to persistent storage (placeholder for future implementation).
        
        This would integrate with Redis or other persistent storage
        in a production deployment.
        """
        # TODO: Implement Redis/database persistence
        logger.debug(f"Context {self.context_id} saved to persistent store")
    
    @classmethod
    async def load_from_persistent_store(
        cls, 
        context_id: str,
        config: Optional[MeshConfig] = None
    ) -> Optional["MeshContext"]:
        """
        Load context from persistent storage (placeholder for future implementation).
        
        Args:
            context_id: Context ID to load
            config: MeshAI configuration
            
        Returns:
            Loaded context or None if not found
        """
        # TODO: Implement Redis/database loading
        logger.debug(f"Loading context {context_id} from persistent store")
        return None
    
    def __str__(self) -> str:
        return f"MeshContext(id={self.context_id}, session={self.session_id})"
    
    def __repr__(self) -> str:
        return (
            f"MeshContext("
            f"context_id='{self.context_id}', "
            f"session_id='{self.session_id}', "
            f"shared_keys={len(self._shared_memory)}, "
            f"agent_keys={len(self._agent_memory)}"
            f")"
        )