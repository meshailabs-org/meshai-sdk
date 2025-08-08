"""
Advanced Context Management System for MeshAI

This module provides sophisticated context storage, retrieval, and management
capabilities including agent-specific memory, conflict resolution, and versioning.
"""

import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

import structlog
from pydantic import BaseModel

from .config import MeshConfig
from .schemas import ContextData
from ..exceptions.base import ContextError, ConfigurationError
from ..utils.serialization import serialize_data, deserialize_data

logger = structlog.get_logger(__name__)


class ConflictResolution(str, Enum):
    """Conflict resolution strategies"""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins" 
    MERGE_STRATEGY = "merge_strategy"
    AGENT_PRIORITY = "agent_priority"
    USER_PROMPT = "user_prompt"
    VERSION_BRANCH = "version_branch"


class AccessLevel(str, Enum):
    """Context access levels"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    OWNER = "owner"


class ContextScope(str, Enum):
    """Context visibility scopes"""
    PRIVATE = "private"        # Only creating agent
    SHARED = "shared"          # Specified agents
    PUBLIC = "public"          # All agents
    TEMPORARY = "temporary"    # Session-based


@dataclass
class ContextVersion:
    """Represents a version of context data"""
    version_id: str
    timestamp: datetime
    agent_id: str
    changes: Dict[str, Any]
    parent_version: Optional[str] = None
    is_current: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPolicy:
    """Context access and sharing policies"""
    context_id: str
    owner_agent: str
    scope: ContextScope = ContextScope.PRIVATE
    access_permissions: Dict[str, AccessLevel] = field(default_factory=dict)
    allowed_operations: Set[str] = field(default_factory=lambda: {"read", "write"})
    ttl_seconds: Optional[int] = None
    max_versions: int = 10
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextChange:
    """Represents a change to context data"""
    change_id: str
    context_id: str
    agent_id: str
    operation: str  # create, update, delete, merge
    key_path: str
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedContextManager:
    """
    Advanced Context Management System with:
    - Persistent Redis storage
    - Agent-specific memory isolation
    - Conflict resolution mechanisms
    - Context versioning and rollback
    - Access control and sharing policies
    - Real-time change notifications
    """
    
    def __init__(self, config: MeshConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.context_locks: Dict[str, asyncio.Lock] = {}
        self.change_subscribers: Dict[str, List] = {}
        
        # Context storage
        self.active_contexts: Dict[str, ContextData] = {}
        self.context_policies: Dict[str, ContextPolicy] = {}
        self.context_versions: Dict[str, List[ContextVersion]] = {}
        
        # Initialize Redis if available
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection if configured"""
        if not redis:
            logger.warning("Redis not available - using in-memory context storage")
            return
        
        redis_url = getattr(self.config, 'redis_url', None)
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Redis context storage initialized")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
        else:
            logger.info("No Redis URL configured - using in-memory storage")
    
    async def create_context(
        self,
        context_id: str,
        owner_agent: str,
        initial_data: Optional[Dict[str, Any]] = None,
        policy: Optional[ContextPolicy] = None,
        session_id: Optional[str] = None
    ) -> ContextData:
        """
        Create a new context with advanced features.
        
        Args:
            context_id: Unique context identifier
            owner_agent: Agent creating the context
            initial_data: Initial context data
            policy: Context access policy
            session_id: Optional session identifier
            
        Returns:
            Created context data
        """
        if await self.context_exists(context_id):
            raise ContextError(f"Context {context_id} already exists", context_id)
        
        # Create context data
        context_data = ContextData(
            context_id=context_id,
            session_id=session_id or str(uuid.uuid4()),
            shared_memory=initial_data or {},
            agent_memory={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Create policy
        if not policy:
            policy = ContextPolicy(
                context_id=context_id,
                owner_agent=owner_agent
            )
        
        # Store context and policy
        await self._store_context(context_data)
        await self._store_policy(policy)
        
        # Create initial version
        version = ContextVersion(
            version_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            agent_id=owner_agent,
            changes={"action": "create", "data": initial_data or {}}
        )
        
        await self._store_version(context_id, version)
        
        # Log creation
        change = ContextChange(
            change_id=str(uuid.uuid4()),
            context_id=context_id,
            agent_id=owner_agent,
            operation="create",
            key_path="",
            new_value=initial_data
        )
        
        await self._log_change(change)
        
        logger.info(f"Context {context_id} created by {owner_agent}")
        return context_data
    
    async def get_context(
        self,
        context_id: str,
        agent_id: str,
        include_agent_memory: bool = True
    ) -> Optional[ContextData]:
        """
        Retrieve context with access control.
        
        Args:
            context_id: Context identifier
            agent_id: Requesting agent ID
            include_agent_memory: Include agent-specific memory
            
        Returns:
            Context data if authorized
        """
        # Check access permissions
        if not await self._check_access(context_id, agent_id, "read"):
            raise ContextError(f"Agent {agent_id} not authorized to read context {context_id}")
        
        # Get context data
        context_data = await self._load_context(context_id)
        if not context_data:
            return None
        
        # Filter agent memory based on permissions
        if include_agent_memory:
            agent_memory = context_data.agent_memory.get(agent_id, {})
            context_data.agent_memory = {agent_id: agent_memory}
        else:
            context_data.agent_memory = {}
        
        # Track access
        await self._track_access(context_id, agent_id, "read")
        
        return context_data
    
    async def update_context(
        self,
        context_id: str,
        agent_id: str,
        updates: Dict[str, Any],
        target_scope: str = "shared",
        conflict_resolution: Optional[ConflictResolution] = None
    ) -> bool:
        """
        Update context with conflict resolution.
        
        Args:
            context_id: Context identifier
            agent_id: Updating agent ID
            updates: Updates to apply
            target_scope: Where to apply updates (shared, agent)
            conflict_resolution: Override default conflict resolution
            
        Returns:
            True if successful
        """
        # Check write permissions
        if not await self._check_access(context_id, agent_id, "write"):
            raise ContextError(f"Agent {agent_id} not authorized to write to context {context_id}")
        
        # Acquire context lock
        async with await self._get_context_lock(context_id):
            # Load current context
            context_data = await self._load_context(context_id)
            if not context_data:
                raise ContextError(f"Context {context_id} not found")
            
            # Apply updates with conflict resolution
            resolved_updates = await self._resolve_conflicts(
                context_id, agent_id, updates, target_scope, conflict_resolution
            )
            
            # Create new version before applying changes
            old_version = await self._create_checkpoint(context_id, agent_id)
            
            try:
                # Apply resolved updates
                if target_scope == "shared":
                    context_data.shared_memory.update(resolved_updates)
                elif target_scope == "agent":
                    if agent_id not in context_data.agent_memory:
                        context_data.agent_memory[agent_id] = {}
                    context_data.agent_memory[agent_id].update(resolved_updates)
                
                context_data.updated_at = datetime.utcnow()
                
                # Store updated context
                await self._store_context(context_data)
                
                # Log changes
                for key, value in resolved_updates.items():
                    change = ContextChange(
                        change_id=str(uuid.uuid4()),
                        context_id=context_id,
                        agent_id=agent_id,
                        operation="update",
                        key_path=f"{target_scope}.{key}",
                        new_value=value
                    )
                    await self._log_change(change)
                
                logger.info(f"Context {context_id} updated by {agent_id}")
                return True
                
            except Exception as e:
                # Rollback on error
                await self._rollback_to_version(context_id, old_version.version_id)
                raise ContextError(f"Failed to update context: {e}")
    
    async def share_context(
        self,
        context_id: str,
        owner_agent: str,
        target_agents: List[str],
        access_level: AccessLevel = AccessLevel.READ_WRITE,
        keys: Optional[List[str]] = None
    ) -> bool:
        """
        Share context with other agents.
        
        Args:
            context_id: Context to share
            owner_agent: Current owner
            target_agents: Agents to share with
            access_level: Access level to grant
            keys: Specific keys to share (None = all)
            
        Returns:
            True if successful
        """
        # Check ownership
        policy = await self._load_policy(context_id)
        if not policy or policy.owner_agent != owner_agent:
            raise ContextError(f"Agent {owner_agent} not authorized to share context {context_id}")
        
        # Update policy
        for agent_id in target_agents:
            policy.access_permissions[agent_id] = access_level
        
        if keys:
            # Selective sharing - create filtered context
            context_data = await self._load_context(context_id)
            if context_data:
                filtered_data = {k: v for k, v in context_data.shared_memory.items() if k in keys}
                
                # Create shared context
                shared_context_id = f"{context_id}_shared_{uuid.uuid4().hex[:8]}"
                await self.create_context(
                    shared_context_id,
                    owner_agent,
                    filtered_data,
                    ContextPolicy(
                        context_id=shared_context_id,
                        owner_agent=owner_agent,
                        scope=ContextScope.SHARED,
                        access_permissions={agent: access_level for agent in target_agents}
                    )
                )
        else:
            # Full sharing
            policy.scope = ContextScope.SHARED
            await self._store_policy(policy)
        
        logger.info(f"Context {context_id} shared by {owner_agent} with {target_agents}")
        return True
    
    async def get_context_history(
        self,
        context_id: str,
        agent_id: str,
        limit: int = 50
    ) -> List[ContextVersion]:
        """Get context version history"""
        if not await self._check_access(context_id, agent_id, "read"):
            raise ContextError(f"Agent {agent_id} not authorized to access context history")
        
        versions = await self._load_versions(context_id)
        return versions[-limit:] if versions else []
    
    async def rollback_context(
        self,
        context_id: str,
        agent_id: str,
        version_id: str
    ) -> bool:
        """Rollback context to a previous version"""
        if not await self._check_access(context_id, agent_id, "write"):
            raise ContextError(f"Agent {agent_id} not authorized to rollback context")
        
        return await self._rollback_to_version(context_id, version_id)
    
    async def delete_context(
        self,
        context_id: str,
        agent_id: str
    ) -> bool:
        """Delete context (owner only)"""
        policy = await self._load_policy(context_id)
        if not policy or policy.owner_agent != agent_id:
            raise ContextError(f"Agent {agent_id} not authorized to delete context {context_id}")
        
        # Delete from storage
        await self._delete_context_data(context_id)
        
        # Log deletion
        change = ContextChange(
            change_id=str(uuid.uuid4()),
            context_id=context_id,
            agent_id=agent_id,
            operation="delete",
            key_path=""
        )
        await self._log_change(change)
        
        logger.info(f"Context {context_id} deleted by {agent_id}")
        return True
    
    async def get_agent_contexts(
        self,
        agent_id: str,
        include_shared: bool = True
    ) -> List[str]:
        """Get all contexts accessible to an agent"""
        contexts = []
        
        # Get owned contexts
        for context_id, policy in self.context_policies.items():
            if policy.owner_agent == agent_id:
                contexts.append(context_id)
            elif include_shared and agent_id in policy.access_permissions:
                contexts.append(context_id)
        
        return contexts
    
    async def context_exists(self, context_id: str) -> bool:
        """Check if context exists"""
        if self.redis_client:
            return bool(await self.redis_client.exists(f"context:{context_id}"))
        else:
            return context_id in self.active_contexts
    
    # Private implementation methods
    
    async def _resolve_conflicts(
        self,
        context_id: str,
        agent_id: str,
        updates: Dict[str, Any],
        target_scope: str,
        resolution_strategy: Optional[ConflictResolution]
    ) -> Dict[str, Any]:
        """Resolve conflicts in context updates"""
        policy = await self._load_policy(context_id)
        strategy = resolution_strategy or policy.conflict_resolution
        
        current_data = await self._load_context(context_id)
        if not current_data:
            return updates
        
        target_data = (current_data.shared_memory if target_scope == "shared" 
                      else current_data.agent_memory.get(agent_id, {}))
        
        resolved = {}
        
        for key, new_value in updates.items():
            old_value = target_data.get(key)
            
            if old_value is None:
                # No conflict - new key
                resolved[key] = new_value
            elif strategy == ConflictResolution.LAST_WRITE_WINS:
                resolved[key] = new_value
            elif strategy == ConflictResolution.FIRST_WRITE_WINS:
                resolved[key] = old_value
            elif strategy == ConflictResolution.MERGE_STRATEGY:
                resolved[key] = await self._merge_values(old_value, new_value)
            elif strategy == ConflictResolution.AGENT_PRIORITY:
                # Use agent priority (owner > high priority agents > others)
                if policy.owner_agent == agent_id:
                    resolved[key] = new_value
                else:
                    # Could implement priority system here
                    resolved[key] = new_value
            elif strategy == ConflictResolution.VERSION_BRANCH:
                # Create branched version for conflicts
                branch_id = await self._create_conflict_branch(context_id, key, old_value, new_value, agent_id)
                resolved[key] = {"_conflict_branch": branch_id, "values": [old_value, new_value]}
            else:
                # Default to last write wins
                resolved[key] = new_value
        
        return resolved
    
    async def _merge_values(self, old_value: Any, new_value: Any) -> Any:
        """Merge two values intelligently"""
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            # Deep merge dictionaries
            merged = old_value.copy()
            for key, value in new_value.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = await self._merge_values(merged[key], value)
                else:
                    merged[key] = value
            return merged
        elif isinstance(old_value, list) and isinstance(new_value, list):
            # Merge lists (append unique items)
            merged = old_value.copy()
            for item in new_value:
                if item not in merged:
                    merged.append(item)
            return merged
        else:
            # For primitives, return new value
            return new_value
    
    async def _check_access(self, context_id: str, agent_id: str, operation: str) -> bool:
        """Check if agent has permission for operation"""
        policy = await self._load_policy(context_id)
        if not policy:
            return False
        
        # Owner has all permissions
        if policy.owner_agent == agent_id:
            return True
        
        # Check explicit permissions
        access_level = policy.access_permissions.get(agent_id)
        if not access_level:
            # Check if context is public
            if policy.scope == ContextScope.PUBLIC:
                access_level = AccessLevel.READ_ONLY
            else:
                return False
        
        # Check operation permissions
        if operation == "read":
            return access_level in [AccessLevel.READ_ONLY, AccessLevel.READ_WRITE, AccessLevel.ADMIN]
        elif operation == "write":
            return access_level in [AccessLevel.READ_WRITE, AccessLevel.ADMIN]
        elif operation == "admin":
            return access_level == AccessLevel.ADMIN
        
        return False
    
    async def _get_context_lock(self, context_id: str) -> asyncio.Lock:
        """Get or create lock for context"""
        if context_id not in self.context_locks:
            self.context_locks[context_id] = asyncio.Lock()
        return self.context_locks[context_id]
    
    async def _create_checkpoint(self, context_id: str, agent_id: str) -> ContextVersion:
        """Create a checkpoint version before changes"""
        context_data = await self._load_context(context_id)
        if not context_data:
            raise ContextError(f"Context {context_id} not found")
        
        version = ContextVersion(
            version_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            changes={"checkpoint": context_data.model_dump()}
        )
        
        await self._store_version(context_id, version)
        return version
    
    # Storage implementation methods
    
    async def _store_context(self, context_data: ContextData) -> None:
        """Store context data"""
        if self.redis_client:
            key = f"context:{context_data.context_id}"
            data = serialize_data(context_data.model_dump())
            await self.redis_client.setex(key, 86400, data)  # 24h TTL
        else:
            self.active_contexts[context_data.context_id] = context_data
    
    async def _load_context(self, context_id: str) -> Optional[ContextData]:
        """Load context data"""
        if self.redis_client:
            key = f"context:{context_id}"
            data = await self.redis_client.get(key)
            if data:
                context_dict = deserialize_data(data)
                return ContextData(**context_dict)
        else:
            return self.active_contexts.get(context_id)
        
        return None
    
    async def _store_policy(self, policy: ContextPolicy) -> None:
        """Store context policy"""
        if self.redis_client:
            key = f"policy:{policy.context_id}"
            data = serialize_data(policy.__dict__)
            await self.redis_client.set(key, data)
        else:
            self.context_policies[policy.context_id] = policy
    
    async def _load_policy(self, context_id: str) -> Optional[ContextPolicy]:
        """Load context policy"""
        if self.redis_client:
            key = f"policy:{context_id}"
            data = await self.redis_client.get(key)
            if data:
                policy_dict = deserialize_data(data)
                return ContextPolicy(**policy_dict)
        else:
            return self.context_policies.get(context_id)
        
        return None
    
    async def _store_version(self, context_id: str, version: ContextVersion) -> None:
        """Store context version"""
        if self.redis_client:
            key = f"versions:{context_id}"
            await self.redis_client.lpush(key, serialize_data(version.__dict__))
            await self.redis_client.ltrim(key, 0, 99)  # Keep last 100 versions
        else:
            if context_id not in self.context_versions:
                self.context_versions[context_id] = []
            self.context_versions[context_id].append(version)
            # Keep last 100 versions
            if len(self.context_versions[context_id]) > 100:
                self.context_versions[context_id] = self.context_versions[context_id][-100:]
    
    async def _load_versions(self, context_id: str) -> List[ContextVersion]:
        """Load context versions"""
        if self.redis_client:
            key = f"versions:{context_id}"
            version_data = await self.redis_client.lrange(key, 0, -1)
            versions = []
            for data in version_data:
                version_dict = deserialize_data(data)
                versions.append(ContextVersion(**version_dict))
            return versions
        else:
            return self.context_versions.get(context_id, [])
    
    async def _log_change(self, change: ContextChange) -> None:
        """Log context change"""
        if self.redis_client:
            key = f"changes:{change.context_id}"
            await self.redis_client.lpush(key, serialize_data(change.__dict__))
            await self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 changes
            await self.redis_client.expire(key, 604800)  # 7 days
    
    async def _track_access(self, context_id: str, agent_id: str, operation: str) -> None:
        """Track context access for analytics"""
        if self.redis_client:
            key = f"access:{context_id}:{datetime.utcnow().strftime('%Y-%m-%d')}"
            await self.redis_client.hincrby(key, f"{agent_id}:{operation}", 1)
            await self.redis_client.expire(key, 2592000)  # 30 days
    
    async def _rollback_to_version(self, context_id: str, version_id: str) -> bool:
        """Rollback context to specific version"""
        versions = await self._load_versions(context_id)
        target_version = next((v for v in versions if v.version_id == version_id), None)
        
        if not target_version:
            raise ContextError(f"Version {version_id} not found for context {context_id}")
        
        # Restore context from checkpoint
        checkpoint_data = target_version.changes.get("checkpoint")
        if checkpoint_data:
            context_data = ContextData(**checkpoint_data)
            await self._store_context(context_data)
            return True
        
        return False
    
    async def _create_conflict_branch(
        self,
        context_id: str,
        key: str,
        old_value: Any,
        new_value: Any,
        agent_id: str
    ) -> str:
        """Create a branch for conflicting values"""
        branch_id = str(uuid.uuid4())
        branch_data = {
            "branch_id": branch_id,
            "context_id": context_id,
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "agent_id": agent_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if self.redis_client:
            key = f"conflict_branch:{branch_id}"
            await self.redis_client.set(key, serialize_data(branch_data))
            await self.redis_client.expire(key, 86400)  # 24h TTL
        
        return branch_id
    
    async def _delete_context_data(self, context_id: str) -> None:
        """Delete all context-related data"""
        if self.redis_client:
            keys = [
                f"context:{context_id}",
                f"policy:{context_id}",
                f"versions:{context_id}",
                f"changes:{context_id}"
            ]
            await self.redis_client.delete(*keys)
        else:
            self.active_contexts.pop(context_id, None)
            self.context_policies.pop(context_id, None)
            self.context_versions.pop(context_id, None)
    
    async def close(self) -> None:
        """Close connections and cleanup"""
        if self.redis_client:
            await self.redis_client.close()
        
        self.context_locks.clear()
        self.change_subscribers.clear()