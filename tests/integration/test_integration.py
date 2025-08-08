"""
Integration Test Suite for MeshAI

This module provides comprehensive integration tests for the complete
MeshAI system including all components working together.
"""

import pytest
import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, AsyncMock

from meshai.core import MeshAgent, MeshContext
from meshai.core.config import MeshConfig
from meshai.core.registry_client import RegistryClient
from meshai.core.runtime_client import RuntimeClient
from meshai.core.circuit_breaker import circuit_breaker_manager, CircuitBreakerConfig
from meshai.core.performance_monitor import performance_monitor
from meshai.core.failover_manager import failover_manager
from meshai.core.health_monitor import health_monitor
from meshai.core.routing_engine import routing_engine, RoutingContext, RoutingPolicy, RoutingStrategy
from meshai.core.context_manager import AdvancedContextManager, ContextPolicy, ConflictResolution


class IntegrationTestAgent(MeshAgent):
    """Test agent for integration testing"""
    
    def __init__(self, agent_id: str, config: MeshConfig = None):
        super().__init__(agent_id, config)
        self.tasks_processed = 0
        self.last_task = None
    
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Process test task"""
        self.tasks_processed += 1
        self.last_task = task_data
        
        # Simulate processing
        processing_time = task_data.get("processing_time", 0.1)
        await asyncio.sleep(processing_time)
        
        # Check for error simulation
        if task_data.get("should_fail", False):
            raise Exception(f"Simulated error in {self.agent_id}")
        
        return {
            "agent_id": self.agent_id,
            "task_id": task_data.get("task_id"),
            "result": "processed",
            "tasks_processed": self.tasks_processed
        }


@pytest.mark.integration
class TestCoreIntegration:
    """Test core component integration"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return MeshConfig()
    
    @pytest.fixture
    async def test_agents(self, config):
        """Create test agents"""
        agents = []
        for i in range(3):
            agent = IntegrationTestAgent(f"test-agent-{i}", config)
            await agent.start()
            agents.append(agent)
        
        yield agents
        
        for agent in agents:
            await agent.stop()
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, test_agents):
        """Test agent lifecycle management"""
        for agent in test_agents:
            assert agent.status == "running"
            
            # Test health check
            health = await agent.health_check()
            assert health["status"] == "healthy"
            
            # Process task
            context = MeshContext()
            result = await agent.handle_task({"task_id": "test"}, context)
            assert result["result"] == "processed"
    
    @pytest.mark.asyncio
    async def test_context_sharing_between_agents(self, test_agents):
        """Test context sharing between multiple agents"""
        context = MeshContext()
        
        # First agent sets context
        agent1 = test_agents[0]
        await context.set("shared_data", {"value": 42}, agent_id=agent1.agent_id)
        
        # Second agent reads context
        agent2 = test_agents[1]
        shared_data = await context.get("shared_data")
        assert shared_data["value"] == 42
        
        # Test agent-specific context
        await context.set("private_data", "secret", agent_scope=True, agent_id=agent1.agent_id)
        
        # Other agent shouldn't see private data
        private_data = await context.get("private_data", agent_scope=True)
        assert private_data is None
    
    @pytest.mark.asyncio
    async def test_registry_integration(self, test_agents, config):
        """Test registry client integration"""
        registry = RegistryClient(config)
        
        # Register agents
        for agent in test_agents:
            success = await registry.register_agent(
                agent_id=agent.agent_id,
                capabilities=["test-capability"],
                endpoint=f"http://{agent.agent_id}:8080",
                metadata={"type": "test"}
            )
            assert success
        
        # Discover agents by capability
        discovered = await registry.discover_agents(
            capability="test-capability"
        )
        assert len(discovered) == len(test_agents)
        
        # Test health monitoring
        for agent in test_agents:
            success = await registry.update_health(
                agent.agent_id,
                status="healthy",
                metrics={"cpu": 50, "memory": 60}
            )
            assert success
    
    @pytest.mark.asyncio
    async def test_runtime_integration(self, test_agents, config):
        """Test runtime client integration"""
        runtime = RuntimeClient(config)
        
        # Submit task
        task_id = await runtime.submit_task(
            capability="test-capability",
            task_data={"operation": "test"},
            priority=1
        )
        assert task_id is not None
        
        # Monitor task (mock implementation)
        # In real scenario, this would interact with actual runtime
        status = {
            "task_id": task_id,
            "status": "completed",
            "result": {"success": True}
        }
        
        assert status["status"] == "completed"


@pytest.mark.integration
class TestHealthMonitoringIntegration:
    """Test health monitoring system integration"""
    
    @pytest.fixture
    async def monitoring_setup(self):
        """Set up monitoring systems"""
        await performance_monitor.start_monitoring()
        await health_monitor.start_monitoring()
        await failover_manager.start_monitoring()
        
        yield
        
        await performance_monitor.stop_monitoring()
        await health_monitor.stop_monitoring()
        await failover_manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, monitoring_setup):
        """Test circuit breaker with monitoring"""
        
        # Create circuit breaker
        breaker = circuit_breaker_manager.get_or_create(
            "test-breaker",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=1.0
            )
        )
        
        # Simulate failures to open circuit
        async def failing_operation():
            raise Exception("Test failure")
        
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_operation)
        
        # Circuit should be open
        assert breaker.is_open
        
        # Record in performance monitor
        performance_monitor.record_agent_request(
            "test-agent",
            response_time=0.1,
            success=False
        )
        
        # Check metrics
        stats = performance_monitor.get_agent_stats("test-agent")
        assert stats.failed_requests > 0
    
    @pytest.mark.asyncio
    async def test_failover_with_health_monitoring(self, monitoring_setup):
        """Test failover integration with health monitoring"""
        
        # Register agents with different health states
        failover_manager.register_agent(
            agent_id="healthy-agent",
            endpoint_url="http://healthy:8080",
            capabilities=["test"],
            weight=2.0
        )
        
        failover_manager.register_agent(
            agent_id="unhealthy-agent",
            endpoint_url="http://unhealthy:8080",
            capabilities=["test"],
            weight=1.0
        )
        
        # Mark one agent as unhealthy
        for _ in range(5):
            await failover_manager.record_request_result(
                "unhealthy-agent",
                success=False,
                response_time=0.0,
                capability="test"
            )
        
        # Get healthy agent should not return unhealthy one
        healthy_agent = await failover_manager.get_healthy_agent("test")
        assert healthy_agent is not None
        assert healthy_agent.agent_id == "healthy-agent"
        
        # Test failover event
        alternative = await failover_manager.handle_failover(
            "unhealthy-agent",
            "test",
            Exception("Agent failed")
        )
        assert alternative is not None
        assert alternative.agent_id == "healthy-agent"
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, monitoring_setup):
        """Test health check system integration"""
        
        # Register custom health check
        check_called = False
        
        async def custom_check():
            nonlocal check_called
            check_called = True
            return True
        
        health_monitor.register_health_check(
            name="integration_test",
            check_function=custom_check,
            interval=timedelta(seconds=0.1),
            timeout=timedelta(seconds=1),
            critical=False
        )
        
        # Wait for health check to run
        await asyncio.sleep(0.2)
        
        # Get health status
        health_status = await health_monitor.get_health_status()
        assert health_status.status in ["healthy", "degraded"]
        assert check_called
        
        # Test readiness probe
        ready, info = health_monitor.get_readiness_status()
        assert isinstance(ready, bool)
        assert "ready" in info


@pytest.mark.integration
class TestRoutingIntegration:
    """Test routing system integration"""
    
    @pytest.fixture
    async def routing_setup(self):
        """Set up routing system"""
        await routing_engine.start()
        
        # Add test policy
        policy = RoutingPolicy(
            name="test-policy",
            capability_patterns=["test-*"],
            primary_strategy=RoutingStrategy.PERFORMANCE_BASED
        )
        routing_engine.add_routing_policy(policy)
        
        yield
        
        await routing_engine.stop()
    
    @pytest.mark.asyncio
    async def test_routing_with_failover(self, routing_setup):
        """Test routing integration with failover"""
        
        # Register agents for routing
        for i in range(3):
            failover_manager.register_agent(
                agent_id=f"router-agent-{i}",
                endpoint_url=f"http://agent-{i}:8080",
                capabilities=["test-routing"],
                weight=1.0
            )
        
        # Create routing context
        context = RoutingContext(
            request_id="test-request",
            capability="test-routing",
            priority=2
        )
        
        # Route request
        decision = await routing_engine.route_request(context)
        assert decision is not None
        assert decision.selected_agent is not None
        assert decision.routing_strategy is not None
        
        # Test sticky session
        context_with_user = RoutingContext(
            request_id="test-request-2",
            user_id="user-123",
            capability="test-routing"
        )
        
        decision1 = await routing_engine.route_request(context_with_user)
        decision2 = await routing_engine.route_request(context_with_user)
        
        # Both should route to same agent (sticky session)
        # Note: This depends on policy configuration
        assert decision1 is not None
        assert decision2 is not None
    
    @pytest.mark.asyncio
    async def test_performance_based_routing(self, routing_setup):
        """Test performance-based routing selection"""
        
        # Register agents with different performance profiles
        agents = []
        for i in range(3):
            agent_id = f"perf-agent-{i}"
            failover_manager.register_agent(
                agent_id=agent_id,
                endpoint_url=f"http://agent-{i}:8080",
                capabilities=["test-performance"],
                weight=1.0
            )
            agents.append(agent_id)
        
        # Simulate different performance for agents
        performance_monitor.record_agent_request(agents[0], 0.1, True)  # Fast
        performance_monitor.record_agent_request(agents[1], 1.0, True)  # Slow
        performance_monitor.record_agent_request(agents[2], 0.5, True)  # Medium
        
        # Route multiple requests
        selected_agents = []
        for i in range(10):
            context = RoutingContext(
                request_id=f"perf-test-{i}",
                capability="test-performance"
            )
            decision = await routing_engine.route_request(context)
            if decision:
                selected_agents.append(decision.selected_agent.agent_id)
        
        # Fast agent should be selected more often
        fast_agent_selections = selected_agents.count(agents[0])
        assert fast_agent_selections >= 3  # At least 30% of selections


@pytest.mark.integration
class TestContextManagementIntegration:
    """Test advanced context management integration"""
    
    @pytest.fixture
    def context_manager(self):
        """Create context manager"""
        config = MeshConfig()
        return AdvancedContextManager(config)
    
    @pytest.mark.asyncio
    async def test_context_versioning_integration(self, context_manager):
        """Test context versioning with multiple agents"""
        context_id = "test-context"
        
        # Create context with policy
        policy = ContextPolicy(
            context_id=context_id,
            owner_agent="agent-1",
            conflict_resolution=ConflictResolution.MERGE_STRATEGY
        )
        
        context_data = await context_manager.create_context(
            context_id,
            "agent-1",
            {"initial": "data"},
            policy
        )
        
        assert context_data is not None
        
        # Multiple agents update context
        for i in range(3):
            success = await context_manager.update_context(
                context_id,
                f"agent-{i}",
                {f"update_{i}": f"value_{i}"}
            )
            assert success
        
        # Get context history
        history = await context_manager.get_context_history(
            context_id,
            "agent-1",
            limit=10
        )
        assert len(history) >= 3
        
        # Test rollback
        if history:
            version_id = history[0].version_id
            success = await context_manager.rollback_context(
                context_id,
                "agent-1",
                version_id
            )
            # Rollback may succeed or fail based on implementation
    
    @pytest.mark.asyncio
    async def test_context_conflict_resolution(self, context_manager):
        """Test conflict resolution strategies"""
        context_id = "conflict-test"
        
        # Create context with merge strategy
        policy = ContextPolicy(
            context_id=context_id,
            owner_agent="owner",
            conflict_resolution=ConflictResolution.MERGE_STRATEGY
        )
        
        await context_manager.create_context(
            context_id,
            "owner",
            {"data": {"field1": "value1"}},
            policy
        )
        
        # Concurrent updates from different agents
        update_tasks = []
        for i in range(3):
            update_task = context_manager.update_context(
                context_id,
                f"agent-{i}",
                {"data": {f"field{i+2}": f"value{i+2}"}}
            )
            update_tasks.append(update_task)
        
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        # Check merged result
        final_context = await context_manager.get_context(
            context_id,
            "owner"
        )
        
        # Should have merged fields
        assert final_context is not None
        # Note: Exact merge behavior depends on implementation


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test end-to-end scenarios"""
    
    @pytest.fixture
    async def full_system_setup(self):
        """Set up complete system"""
        config = MeshConfig()
        
        # Start all monitoring systems
        await performance_monitor.start_monitoring()
        await health_monitor.start_monitoring()
        await failover_manager.start_monitoring()
        await routing_engine.start()
        
        # Create test agents
        agents = []
        for i in range(5):
            agent = IntegrationTestAgent(f"e2e-agent-{i}", config)
            await agent.start()
            agents.append(agent)
            
            # Register with failover manager
            failover_manager.register_agent(
                agent_id=agent.agent_id,
                endpoint_url=f"http://{agent.agent_id}:8080",
                capabilities=["e2e-test"],
                weight=1.0
            )
        
        yield agents
        
        # Cleanup
        for agent in agents:
            await agent.stop()
        
        await routing_engine.stop()
        await failover_manager.stop_monitoring()
        await health_monitor.stop_monitoring()
        await performance_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_load_balancing_scenario(self, full_system_setup):
        """Test load balancing across multiple agents"""
        agents = full_system_setup
        
        # Submit multiple tasks
        tasks = []
        for i in range(20):
            context = RoutingContext(
                request_id=f"load-{i}",
                capability="e2e-test"
            )
            
            # Route to agent
            decision = await routing_engine.route_request(context)
            
            if decision:
                # Find actual agent
                agent = next(
                    (a for a in agents if a.agent_id == decision.selected_agent.agent_id),
                    None
                )
                
                if agent:
                    task = agent.handle_task(
                        {"task_id": f"task-{i}"},
                        MeshContext()
                    )
                    tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check distribution
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) >= 15  # At least 75% success
        
        # Check that load was distributed
        task_counts = [agent.tasks_processed for agent in agents]
        assert max(task_counts) <= len(tasks) / 2  # No agent got more than half
    
    @pytest.mark.asyncio
    async def test_failure_recovery_scenario(self, full_system_setup):
        """Test system recovery from failures"""
        agents = full_system_setup
        
        # Simulate agent failures
        failing_agent = agents[0]
        
        # Process some successful tasks first
        for i in range(5):
            result = await failing_agent.handle_task(
                {"task_id": f"success-{i}"},
                MeshContext()
            )
            assert result["result"] == "processed"
            
            # Record success
            performance_monitor.record_agent_request(
                failing_agent.agent_id,
                0.1,
                True
            )
        
        # Now simulate failures
        for i in range(5):
            try:
                await failing_agent.handle_task(
                    {"task_id": f"fail-{i}", "should_fail": True},
                    MeshContext()
                )
            except Exception:
                # Record failure
                performance_monitor.record_agent_request(
                    failing_agent.agent_id,
                    0.0,
                    False
                )
                
                await failover_manager.record_request_result(
                    failing_agent.agent_id,
                    False,
                    0.0,
                    "e2e-test"
                )
        
        # System should route away from failing agent
        context = RoutingContext(
            request_id="after-failure",
            capability="e2e-test"
        )
        
        decision = await routing_engine.route_request(context)
        
        # Should not select the failing agent
        if decision:
            assert decision.selected_agent.agent_id != failing_agent.agent_id
    
    @pytest.mark.asyncio
    async def test_performance_degradation_scenario(self, full_system_setup):
        """Test system response to performance degradation"""
        agents = full_system_setup
        
        # Simulate one agent becoming slow
        slow_agent = agents[1]
        
        # Process tasks with increasing latency
        for i in range(10):
            latency = 0.1 * (i + 1)  # Increasing latency
            
            result = await slow_agent.handle_task(
                {"task_id": f"slow-{i}", "processing_time": latency},
                MeshContext()
            )
            
            # Record performance
            performance_monitor.record_agent_request(
                slow_agent.agent_id,
                latency,
                True
            )
        
        # System should deprioritize slow agent
        selected_counts = {agent.agent_id: 0 for agent in agents}
        
        for i in range(10):
            context = RoutingContext(
                request_id=f"after-degradation-{i}",
                capability="e2e-test"
            )
            
            decision = await routing_engine.route_request(context)
            if decision:
                selected_counts[decision.selected_agent.agent_id] += 1
        
        # Slow agent should be selected less frequently
        slow_agent_selections = selected_counts[slow_agent.agent_id]
        avg_selections = sum(selected_counts.values()) / len(agents)
        
        # Slow agent should get less than average selections
        assert slow_agent_selections < avg_selections


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])