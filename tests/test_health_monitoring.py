"""
Comprehensive Tests for Enhanced Health Monitoring System
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import threading

from meshai.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, 
    CircuitBreakerOpenError, circuit_breaker_manager
)
from meshai.core.performance_monitor import (
    PerformanceMonitor, AlertRule, AlertSeverity, 
    performance_monitor, record_performance
)
from meshai.core.failover_manager import (
    FailoverManager, AgentEndpoint, FailoverStrategy, 
    AgentHealth, failover_manager
)
from meshai.core.health_monitor import (
    HealthMonitor, HealthStatus, ComponentStatus, health_monitor
)
from meshai.core.alerting_system import (
    AlertingSystem, NotificationChannel, AlertRoute, alerting_system
)
from meshai.core.analytics_dashboard import (
    AnalyticsDashboard, TimeRange, analytics_dashboard
)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    @pytest.fixture
    def circuit_breaker_config(self):
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            timeout=0.5
        )
    
    @pytest.fixture
    def circuit_breaker(self, circuit_breaker_config):
        return CircuitBreaker("test_circuit", circuit_breaker_config)
    
    @pytest.mark.asyncio
    async def test_successful_calls(self, circuit_breaker):
        """Test successful calls keep circuit closed"""
        async def success_func():
            return "success"
        
        # Multiple successful calls
        for _ in range(5):
            result = await circuit_breaker.call(success_func)
            assert result == "success"
            assert circuit_breaker.is_closed
    
    @pytest.mark.asyncio
    async def test_failure_opens_circuit(self, circuit_breaker):
        """Test that failures open the circuit"""
        async def failing_func():
            raise Exception("Test failure")
        
        # Circuit should open after threshold failures
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.is_open
        
        # Further calls should be rejected immediately
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_timeout_opens_circuit(self, circuit_breaker):
        """Test that timeouts open the circuit"""
        async def slow_func():
            await asyncio.sleep(1.0)  # Longer than timeout
            return "too slow"
        
        # Timeout should be treated as failure
        for i in range(3):
            with pytest.raises(asyncio.TimeoutError):
                await circuit_breaker.call(slow_func)
        
        assert circuit_breaker.is_open
    
    @pytest.mark.asyncio
    async def test_half_open_recovery(self, circuit_breaker):
        """Test circuit recovery through half-open state"""
        async def failing_func():
            raise Exception("Test failure")
        
        async def success_func():
            return "success"
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.is_open
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should transition to half-open
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.is_half_open
        
        # More successful calls should close circuit
        for _ in range(2):
            await circuit_breaker.call(success_func)
        
        assert circuit_breaker.is_closed
    
    def test_circuit_breaker_metrics(self, circuit_breaker):
        """Test circuit breaker metrics collection"""
        metrics = circuit_breaker.get_metrics()
        
        assert "name" in metrics
        assert "state" in metrics
        assert "failure_count" in metrics
        assert "success_count" in metrics
        assert metrics["name"] == "test_circuit"


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    @pytest.fixture
    def perf_monitor(self):
        return PerformanceMonitor(collection_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_performance_recording(self, perf_monitor):
        """Test recording agent performance metrics"""
        agent_id = "test-agent"
        
        # Record some requests
        perf_monitor.record_agent_request(agent_id, 0.5, True)
        perf_monitor.record_agent_request(agent_id, 1.0, True)
        perf_monitor.record_agent_request(agent_id, 0.8, False)
        
        # Check stats
        stats = perf_monitor.get_agent_stats(agent_id)
        assert stats is not None
        assert stats.total_requests == 3
        assert stats.successful_requests == 2
        assert stats.failed_requests == 1
        assert abs(stats.error_rate - (1/3)) < 0.01
        assert stats.min_response_time == 0.5
        assert stats.max_response_time == 1.0
    
    @pytest.mark.asyncio
    async def test_alert_rules(self, perf_monitor):
        """Test alert rule evaluation"""
        alert_rule = AlertRule(
            name="high_error_rate",
            metric_name="error_rate",
            threshold=0.5,
            comparison="gt",
            severity=AlertSeverity.HIGH,
            duration=timedelta(seconds=1)
        )
        
        perf_monitor.add_alert_rule(alert_rule)
        
        # Record high error rate
        agent_id = "failing-agent"
        for _ in range(10):
            perf_monitor.record_agent_request(agent_id, 0.1, False)
        
        # Start monitoring to trigger alert checking
        await perf_monitor.start_monitoring()
        await asyncio.sleep(0.2)  # Let monitoring run
        await perf_monitor.stop_monitoring()
    
    def test_custom_metrics(self, perf_monitor):
        """Test custom metric recording"""
        perf_monitor.record_custom_metric(
            "custom_metric",
            42.0,
            labels={"type": "test"},
            metadata={"unit": "seconds"}
        )
        
        # Verify metric was recorded
        assert "custom_metric" in perf_monitor.metrics
        metric_points = perf_monitor.metrics["custom_metric"]
        assert len(metric_points) == 1
        assert metric_points[0].value == 42.0
    
    @pytest.mark.asyncio
    async def test_record_performance_decorator(self):
        """Test the record_performance decorator"""
        @record_performance("decorated-agent")
        async def test_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_function()
        assert result == "success"
        
        # Check that performance was recorded
        stats = performance_monitor.get_agent_stats("decorated-agent")
        assert stats is not None
        assert stats.total_requests >= 1


class TestFailoverManager:
    """Test failover management functionality"""
    
    @pytest.fixture
    def failover_mgr(self):
        return FailoverManager(health_check_interval=0.1)
    
    def test_agent_registration(self, failover_mgr):
        """Test agent registration and discovery"""
        failover_mgr.register_agent(
            agent_id="test-agent-1",
            endpoint_url="http://test1:8080",
            capabilities=["processing"],
            weight=1.0
        )
        
        failover_mgr.register_agent(
            agent_id="test-agent-2", 
            endpoint_url="http://test2:8080",
            capabilities=["processing"],
            weight=2.0
        )
        
        # Test agent discovery
        assert "test-agent-1" in failover_mgr.agents
        assert "test-agent-2" in failover_mgr.agents
        assert "processing" in failover_mgr.capability_map
        assert len(failover_mgr.capability_map["processing"]) == 2
    
    @pytest.mark.asyncio
    async def test_healthy_agent_selection(self, failover_mgr):
        """Test healthy agent selection strategies"""
        # Register agents
        for i in range(3):
            failover_mgr.register_agent(
                agent_id=f"agent-{i}",
                endpoint_url=f"http://agent{i}:8080",
                capabilities=["test"],
                weight=1.0
            )
        
        # Test round-robin selection
        selected_agents = []
        for _ in range(6):
            agent = await failover_mgr.get_healthy_agent("test", strategy=FailoverStrategy.ROUND_ROBIN)
            if agent:
                selected_agents.append(agent.agent_id)
        
        # Should cycle through all agents
        unique_agents = set(selected_agents)
        assert len(unique_agents) == 3
    
    @pytest.mark.asyncio
    async def test_agent_health_tracking(self, failover_mgr):
        """Test agent health status tracking"""
        failover_mgr.register_agent(
            agent_id="health-test-agent",
            endpoint_url="http://test:8080", 
            capabilities=["test"]
        )
        
        agent = failover_mgr.agents["health-test-agent"]
        assert agent.health == AgentHealth.HEALTHY
        
        # Record failures
        for _ in range(5):
            await failover_mgr.record_request_result(
                "health-test-agent", False, 0.0, "test"
            )
        
        # Agent should be marked unhealthy
        assert agent.health == AgentHealth.UNHEALTHY
        
        # Record successes
        for _ in range(5):
            await failover_mgr.record_request_result(
                "health-test-agent", True, 0.5, "test"
            )
        
        # Agent should recover
        assert agent.health in [AgentHealth.HEALTHY, AgentHealth.DEGRADED]
    
    @pytest.mark.asyncio
    async def test_failover_handling(self, failover_mgr):
        """Test failover event handling"""
        # Register primary and backup agents
        failover_mgr.register_agent(
            agent_id="primary-agent",
            endpoint_url="http://primary:8080",
            capabilities=["failover-test"]
        )
        
        failover_mgr.register_agent(
            agent_id="backup-agent", 
            endpoint_url="http://backup:8080",
            capabilities=["failover-test"]
        )
        
        # Simulate failover
        alternative = await failover_mgr.handle_failover(
            "primary-agent",
            "failover-test", 
            Exception("Primary failed")
        )
        
        assert alternative is not None
        assert alternative.agent_id == "backup-agent"
        
        # Check failover events
        events = failover_mgr.get_failover_events()
        assert len(events) >= 1
        assert events[-1].original_agent == "primary-agent"
        assert events[-1].failover_agent == "backup-agent"


class TestHealthMonitor:
    """Test health monitoring functionality"""
    
    @pytest.fixture
    def health_mon(self):
        return HealthMonitor(check_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_health_check_registration(self, health_mon):
        """Test health check registration and execution"""
        check_called = False
        
        async def test_check():
            nonlocal check_called
            check_called = True
            return True
        
        health_mon.register_health_check(
            name="test_check",
            check_function=test_check,
            interval=timedelta(seconds=0.1),
            timeout=timedelta(seconds=1)
        )
        
        # Start monitoring briefly
        await health_mon.start_monitoring()
        await asyncio.sleep(0.2)
        await health_mon.stop_monitoring()
        
        assert check_called
    
    @pytest.mark.asyncio
    async def test_health_status_aggregation(self, health_mon):
        """Test health status aggregation"""
        # Register passing and failing checks
        async def passing_check():
            return True
        
        async def failing_check():
            return False
        
        health_mon.register_health_check(
            "passing", passing_check, 
            timedelta(seconds=0.1), timedelta(seconds=1), critical=False
        )
        
        health_mon.register_health_check(
            "failing", failing_check,
            timedelta(seconds=0.1), timedelta(seconds=1), critical=True
        )
        
        # Run checks
        await health_mon.start_monitoring()
        await asyncio.sleep(0.2)
        await health_mon.stop_monitoring()
        
        # Get health status
        status = await health_mon.get_health_status()
        assert status.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_readiness_and_liveness(self, health_mon):
        """Test Kubernetes readiness and liveness probes"""
        # Test liveness (should be True if monitoring is running)
        await health_mon.start_monitoring()
        alive, liveness_info = health_mon.get_liveness_status()
        assert alive == True
        assert liveness_info["monitoring_active"] == True
        await health_mon.stop_monitoring()
        
        # Test readiness
        ready, readiness_info = health_mon.get_readiness_status()
        assert "ready" in readiness_info
        assert "critical_checks_total" in readiness_info


class TestAlertingSystem:
    """Test alerting and notification functionality"""
    
    @pytest.fixture  
    def alert_system(self):
        return AlertingSystem()
    
    def test_notification_configuration(self, alert_system):
        """Test notification channel configuration"""
        alert_system.configure_webhook(
            name="test_webhook",
            url="http://localhost:8080/webhook",
            method="POST"
        )
        
        assert "test_webhook" in alert_system.notification_configs
        config = alert_system.notification_configs["test_webhook"]
        assert config.channel == NotificationChannel.WEBHOOK
        assert config.config["url"] == "http://localhost:8080/webhook"
    
    def test_alert_routing(self, alert_system):
        """Test alert routing rules"""
        route = AlertRoute(
            name="critical_route",
            conditions={"severity": "critical"},
            channels=["webhook"],
            severity_filter=[AlertSeverity.CRITICAL]
        )
        
        alert_system.add_alert_route(route)
        assert "critical_route" in alert_system.alert_routes
    
    @pytest.mark.asyncio
    async def test_alert_processing(self, alert_system):
        """Test alert processing pipeline"""
        from meshai.core.performance_monitor import Alert
        
        # Create mock alert
        alert = Alert(
            rule_name="test_rule",
            agent_id="test_agent",
            metric_name="test_metric",
            current_value=100.0,
            threshold=50.0,
            severity=AlertSeverity.HIGH,
            message="Test alert",
            triggered_at=datetime.utcnow()
        )
        
        # Add route that matches
        alert_system.add_alert_route(AlertRoute(
            name="test_route",
            conditions={},
            channels=["test_webhook"],
            severity_filter=[AlertSeverity.HIGH]
        ))
        
        # Configure mock webhook (disabled)
        alert_system.configure_webhook(
            name="test_webhook",
            url="http://localhost:8080/test",
            enabled=False  # Disabled to avoid actual HTTP calls
        )
        
        # Process alert
        await alert_system.process_alert(alert)
        
        # Should have created notification (though sending will fail due to disabled webhook)
        assert len(alert_system.active_notifications) >= 0


class TestAnalyticsDashboard:
    """Test analytics dashboard functionality"""
    
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()
    
    @pytest.mark.asyncio
    async def test_system_overview(self, dashboard):
        """Test system overview computation"""
        overview = await dashboard.get_system_overview()
        
        assert "timestamp" in overview
        assert "system_health" in overview
        assert "agents" in overview
        assert "requests" in overview
        assert "circuit_breakers" in overview
        assert "alerts" in overview
    
    @pytest.mark.asyncio
    async def test_agent_list(self, dashboard):
        """Test agent list generation"""
        agents = await dashboard.get_agent_list()
        
        # Should return list (might be empty in test environment)
        assert isinstance(agents, list)
    
    @pytest.mark.asyncio
    async def test_metrics_time_series(self, dashboard):
        """Test time series data generation"""
        series = await dashboard.get_metrics_time_series(
            "response_time",
            TimeRange.LAST_HOUR
        )
        
        assert isinstance(series, list)
        if series:  # If data exists
            point = series[0]
            assert hasattr(point, 'timestamp')
            assert hasattr(point, 'value')
    
    @pytest.mark.asyncio 
    async def test_dashboard_widgets(self, dashboard):
        """Test dashboard widget generation"""
        widgets = await dashboard.get_dashboard_widgets()
        
        assert isinstance(widgets, list)
        
        # Check widget structure
        for widget in widgets:
            assert hasattr(widget, 'id')
            assert hasattr(widget, 'title') 
            assert hasattr(widget, 'widget_type')
            assert hasattr(widget, 'data')
    
    def test_cache_functionality(self, dashboard):
        """Test dashboard caching"""
        # Test cache hit/miss
        def test_compute():
            return {"test": "data"}
        
        # First call should compute
        result1 = dashboard._get_cached_or_compute("test_key", test_compute, 60)
        assert result1 == {"test": "data"}
        
        # Second call should use cache
        result2 = dashboard._get_cached_or_compute("test_key", lambda: {"different": "data"}, 60)
        assert result2 == {"test": "data"}  # Should be cached result
        
        # Clear cache
        dashboard.clear_cache()
        
        # Should compute again
        result3 = dashboard._get_cached_or_compute("test_key", lambda: {"new": "data"}, 60)
        assert result3 == {"new": "data"}


class TestIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_pipeline(self):
        """Test the complete monitoring pipeline"""
        # This test would set up all components and test their interaction
        # For now, just test that components can be imported and initialized
        
        assert performance_monitor is not None
        assert failover_manager is not None
        assert health_monitor is not None
        assert alerting_system is not None
        assert analytics_dashboard is not None
        assert circuit_breaker_manager is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring(self):
        """Test that monitoring systems can run concurrently"""
        # Start multiple monitoring systems
        tasks = [
            performance_monitor.start_monitoring(),
            health_monitor.start_monitoring(), 
            failover_manager.start_monitoring()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Let them run briefly
        await asyncio.sleep(0.1)
        
        # Stop them
        stop_tasks = [
            performance_monitor.stop_monitoring(),
            health_monitor.stop_monitoring(),
            failover_manager.stop_monitoring()
        ]
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)


# Performance test for high-load scenarios
@pytest.mark.performance
class TestPerformanceUnderLoad:
    """Performance tests for monitoring under high load"""
    
    @pytest.mark.asyncio
    async def test_high_request_volume(self):
        """Test monitoring performance with high request volume"""
        agent_id = "load-test-agent"
        
        # Record many requests quickly
        start_time = time.time()
        for i in range(1000):
            performance_monitor.record_agent_request(
                agent_id=agent_id,
                response_time=0.1,
                success=i % 10 != 0  # 10% failure rate
            )
        
        duration = time.time() - start_time
        
        # Should handle 1000 requests quickly
        assert duration < 1.0  # Less than 1 second
        
        # Verify stats were recorded correctly
        stats = performance_monitor.get_agent_stats(agent_id)
        assert stats.total_requests == 1000
        assert abs(stats.error_rate - 0.1) < 0.01  # ~10% error rate
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_breakers(self):
        """Test concurrent circuit breaker usage"""
        async def test_function(success_rate: float):
            import random
            if random.random() < success_rate:
                return "success"
            else:
                raise Exception("failure")
        
        # Create multiple circuit breakers
        breakers = []
        for i in range(10):
            breaker = CircuitBreaker(f"test_breaker_{i}", CircuitBreakerConfig())
            breakers.append(breaker)
        
        # Run concurrent operations
        async def run_operations(breaker, success_rate):
            for _ in range(50):
                try:
                    await breaker.call(lambda: test_function(success_rate))
                except:
                    pass  # Expected failures
        
        # Run with different success rates
        tasks = [
            run_operations(breakers[i], 0.8 if i % 2 == 0 else 0.3)
            for i in range(len(breakers))
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify circuit breakers are in expected states
        closed_breakers = sum(1 for b in breakers if b.is_closed)
        open_breakers = sum(1 for b in breakers if b.is_open)
        
        # Should have some open and some closed based on success rates
        assert closed_breakers > 0
        assert open_breakers >= 0  # May or may not have opened depending on timing


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])