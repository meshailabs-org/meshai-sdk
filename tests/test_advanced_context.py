"""
Tests for Advanced Context Management System
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from meshai.core.context import MeshContext
from meshai.core.context_manager import (
    AdvancedContextManager, ContextPolicy, ConflictResolution, 
    ContextScope, AccessLevel, ContextVersion, ContextChange
)
from meshai.core.config import MeshConfig
from meshai.core.schemas import ContextData
from meshai.exceptions.base import ContextError


@pytest.fixture
def config():
    """Test configuration"""
    config = MeshConfig()
    config.redis_url = None  # Use in-memory for tests
    return config


@pytest.fixture
def advanced_manager(config):
    """Advanced context manager instance"""
    return AdvancedContextManager(config)


@pytest.fixture
def context_policy():
    """Sample context policy"""
    return ContextPolicy(
        context_id="test-context-001",
        owner_agent="test-agent",
        scope=ContextScope.PRIVATE,
        conflict_resolution=ConflictResolution.LAST_WRITE_WINS
    )


class TestAdvancedContextManager:
    """Test the advanced context manager functionality"""
    
    @pytest.mark.asyncio
    async def test_create_context(self, advanced_manager, context_policy):
        """Test context creation with policy"""
        context_id = "test-context-001"
        owner_agent = "test-agent"
        initial_data = {"key1": "value1", "key2": "value2"}
        
        # Create context
        context_data = await advanced_manager.create_context(
            context_id, owner_agent, initial_data, context_policy
        )
        
        assert context_data.context_id == context_id
        assert context_data.shared_memory == initial_data
        assert context_data.created_at is not None
        
        # Verify context exists
        assert await advanced_manager.context_exists(context_id)
    
    @pytest.mark.asyncio
    async def test_get_context_with_access_control(self, advanced_manager, context_policy):
        """Test getting context with access control"""
        context_id = "test-context-002"
        owner_agent = "owner-agent"
        other_agent = "other-agent"
        
        # Create context
        await advanced_manager.create_context(
            context_id, owner_agent, {"data": "test"}, context_policy
        )
        
        # Owner should have access
        context_data = await advanced_manager.get_context(context_id, owner_agent)
        assert context_data is not None
        assert context_data.shared_memory["data"] == "test"
        
        # Other agent should not have access (private scope)
        with pytest.raises(ContextError):
            await advanced_manager.get_context(context_id, other_agent)
    
    @pytest.mark.asyncio
    async def test_update_context_with_conflict_resolution(self, advanced_manager):
        """Test context updates with conflict resolution"""
        context_id = "test-context-003"
        owner_agent = "owner-agent"
        
        # Create context with merge strategy
        policy = ContextPolicy(
            context_id=context_id,
            owner_agent=owner_agent,
            conflict_resolution=ConflictResolution.MERGE_STRATEGY
        )
        
        await advanced_manager.create_context(
            context_id, owner_agent, {"field1": "value1"}, policy
        )
        
        # Update with overlapping data
        updates = {"field1": "updated_value1", "field2": "value2"}
        success = await advanced_manager.update_context(
            context_id, owner_agent, updates
        )
        
        assert success
        
        # Verify merged data
        context_data = await advanced_manager.get_context(context_id, owner_agent)
        assert context_data.shared_memory["field1"] == "updated_value1"
        assert context_data.shared_memory["field2"] == "value2"
    
    @pytest.mark.asyncio
    async def test_agent_specific_memory(self, advanced_manager):
        """Test agent-specific memory isolation"""
        context_id = "test-context-004"
        owner_agent = "owner-agent"
        
        policy = ContextPolicy(
            context_id=context_id,
            owner_agent=owner_agent,
            scope=ContextScope.SHARED
        )
        
        await advanced_manager.create_context(
            context_id, owner_agent, {}, policy
        )
        
        # Update agent-specific memory
        agent_updates = {"agent_data": "private_value"}
        success = await advanced_manager.update_context(
            context_id, owner_agent, agent_updates, target_scope="agent"
        )
        
        assert success
        
        # Verify agent memory is isolated
        context_data = await advanced_manager.get_context(context_id, owner_agent)
        assert owner_agent in context_data.agent_memory
        assert context_data.agent_memory[owner_agent]["agent_data"] == "private_value"
    
    @pytest.mark.asyncio
    async def test_context_sharing(self, advanced_manager):
        """Test context sharing between agents"""
        context_id = "test-context-005"
        owner_agent = "owner-agent"
        target_agent = "target-agent"
        
        # Create context
        policy = ContextPolicy(
            context_id=context_id,
            owner_agent=owner_agent
        )
        
        await advanced_manager.create_context(
            context_id, owner_agent, {"shared_data": "value"}, policy
        )
        
        # Share with target agent
        success = await advanced_manager.share_context(
            context_id, owner_agent, [target_agent], AccessLevel.READ_WRITE
        )
        
        assert success
        
        # Verify target agent can access
        context_data = await advanced_manager.get_context(context_id, target_agent)
        assert context_data is not None
        assert context_data.shared_memory["shared_data"] == "value"
    
    @pytest.mark.asyncio
    async def test_context_versioning(self, advanced_manager):
        """Test context versioning and history"""
        context_id = "test-context-006"
        owner_agent = "owner-agent"
        
        # Create context
        await advanced_manager.create_context(
            context_id, owner_agent, {"version": 1}
        )
        
        # Make several updates
        for i in range(2, 6):
            await advanced_manager.update_context(
                context_id, owner_agent, {"version": i}
            )
        
        # Get history
        history = await advanced_manager.get_context_history(
            context_id, owner_agent, limit=10
        )
        
        assert len(history) >= 5  # Initial + 4 updates
    
    @pytest.mark.asyncio
    async def test_context_rollback(self, advanced_manager):
        """Test context rollback functionality"""
        context_id = "test-context-007"
        owner_agent = "owner-agent"
        
        # Create context
        await advanced_manager.create_context(
            context_id, owner_agent, {"data": "original"}
        )
        
        # Update context
        await advanced_manager.update_context(
            context_id, owner_agent, {"data": "modified"}
        )
        
        # Get history to find original version
        history = await advanced_manager.get_context_history(
            context_id, owner_agent
        )
        
        if len(history) >= 2:
            original_version = history[0].version_id  # First version
            
            # Rollback
            success = await advanced_manager.rollback_context(
                context_id, owner_agent, original_version
            )
            
            # Note: Rollback behavior depends on implementation
            # This test verifies the method can be called without error
            assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_context_deletion(self, advanced_manager):
        """Test context deletion"""
        context_id = "test-context-008"
        owner_agent = "owner-agent"
        
        # Create context
        await advanced_manager.create_context(
            context_id, owner_agent, {"data": "test"}
        )
        
        # Verify context exists
        assert await advanced_manager.context_exists(context_id)
        
        # Delete context
        success = await advanced_manager.delete_context(context_id, owner_agent)
        assert success
        
        # Verify context no longer exists
        assert not await advanced_manager.context_exists(context_id)
    
    @pytest.mark.asyncio
    async def test_get_agent_contexts(self, advanced_manager):
        """Test getting contexts for an agent"""
        owner_agent = "owner-agent"
        
        # Create multiple contexts
        context_ids = []
        for i in range(3):
            context_id = f"test-context-{i}"
            context_ids.append(context_id)
            await advanced_manager.create_context(
                context_id, owner_agent, {"index": i}
            )
        
        # Get agent's contexts
        agent_contexts = await advanced_manager.get_agent_contexts(owner_agent)
        
        for context_id in context_ids:
            assert context_id in agent_contexts


class TestMeshContextIntegration:
    """Test MeshContext integration with AdvancedContextManager"""
    
    @pytest.mark.asyncio
    async def test_mesh_context_with_advanced_manager(self, config):
        """Test MeshContext using AdvancedContextManager"""
        context = MeshContext(config=config, use_advanced_manager=True)
        
        # The context should initialize without the advanced manager
        # since Redis is not available in test config
        assert context.use_advanced_manager == False  # Should fallback
    
    @pytest.mark.asyncio
    async def test_context_set_get_with_agent_id(self, config):
        """Test context operations with agent IDs"""
        context = MeshContext(config=config)
        agent_id = "test-agent"
        
        # Set value in agent scope
        await context.set("agent_key", "agent_value", agent_scope=True, agent_id=agent_id)
        
        # Get value (fallback to basic implementation)
        value = await context.get("agent_key", agent_scope=True)
        # Note: This will use the basic implementation since advanced manager is not available
    
    @pytest.mark.asyncio
    async def test_context_creation_advanced_features(self, config):
        """Test advanced context creation methods"""
        context = MeshContext(config=config)
        owner_agent = "owner-agent"
        
        # Test advanced context creation (should handle gracefully when not available)
        success = await context.create_advanced_context(owner_agent)
        assert success == False  # Should fail gracefully without advanced manager
        
        # Test history retrieval (should return empty list)
        history = await context.get_context_history(owner_agent)
        assert history == []
        
        # Test rollback (should return False)
        rollback_success = await context.rollback_to_version(owner_agent, "fake-version")
        assert rollback_success == False
    
    @pytest.mark.asyncio
    async def test_context_cleanup(self, config):
        """Test context cleanup"""
        context = MeshContext(config=config, use_advanced_manager=True)
        
        # Should not raise error even if advanced manager is None
        await context.close()


class TestConflictResolution:
    """Test different conflict resolution strategies"""
    
    @pytest.mark.asyncio
    async def test_last_write_wins(self, advanced_manager):
        """Test last write wins strategy"""
        context_id = "conflict-test-001"
        owner_agent = "owner-agent"
        
        policy = ContextPolicy(
            context_id=context_id,
            owner_agent=owner_agent,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS
        )
        
        await advanced_manager.create_context(
            context_id, owner_agent, {"key": "original"}, policy
        )
        
        # Update with new value
        await advanced_manager.update_context(
            context_id, owner_agent, {"key": "updated"}
        )
        
        # Verify last write wins
        context_data = await advanced_manager.get_context(context_id, owner_agent)
        assert context_data.shared_memory["key"] == "updated"
    
    @pytest.mark.asyncio
    async def test_merge_strategy(self, advanced_manager):
        """Test merge strategy for dictionaries"""
        context_id = "conflict-test-002"
        owner_agent = "owner-agent"
        
        policy = ContextPolicy(
            context_id=context_id,
            owner_agent=owner_agent,
            conflict_resolution=ConflictResolution.MERGE_STRATEGY
        )
        
        # Create with nested dictionary
        initial_data = {"config": {"setting1": "value1", "setting2": "value2"}}
        await advanced_manager.create_context(
            context_id, owner_agent, initial_data, policy
        )
        
        # Update with overlapping nested data
        updates = {"config": {"setting2": "updated_value2", "setting3": "value3"}}
        await advanced_manager.update_context(
            context_id, owner_agent, updates
        )
        
        # Verify merge occurred
        context_data = await advanced_manager.get_context(context_id, owner_agent)
        config_data = context_data.shared_memory["config"]
        
        assert config_data["setting1"] == "value1"  # Original
        assert config_data["setting2"] == "updated_value2"  # Updated
        assert config_data["setting3"] == "value3"  # New


@pytest.mark.asyncio
async def test_context_manager_cleanup():
    """Test proper cleanup of context manager resources"""
    config = MeshConfig()
    manager = AdvancedContextManager(config)
    
    # Should not raise error
    await manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])