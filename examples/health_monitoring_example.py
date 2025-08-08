#!/usr/bin/env python3
"""
Enhanced Health Monitoring Example

This example demonstrates the comprehensive health monitoring capabilities
of MeshAI, including:
- Circuit breaker protection
- Performance metrics collection
- Automatic failover mechanisms
- Health check endpoints
- Alerting and notifications
- Performance analytics dashboard
"""

import asyncio
import logging
import time
import random
from datetime import timedelta
from typing import Any, Dict

from meshai.core import MeshAgent, MeshContext
from meshai.core.config import MeshConfig
from meshai.core.circuit_breaker import (
    CircuitBreakerConfig, circuit_breaker, circuit_breaker_manager
)
from meshai.core.performance_monitor import (
    performance_monitor, record_performance, AlertRule, AlertSeverity
)
from meshai.core.failover_manager import (
    failover_manager, FailoverStrategy, FailoverRule
)
from meshai.core.health_monitor import health_monitor
from meshai.core.alerting_system import alerting_system, AlertRoute, NotificationChannel
from meshai.core.analytics_dashboard import analytics_dashboard, TimeRange

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessingAgent(MeshAgent):
    """Agent that processes data with varying success rates"""
    
    def __init__(self, agent_id: str, failure_rate: float = 0.1, config: MeshConfig = None):
        super().__init__(agent_id, config)
        self.failure_rate = failure_rate
        self.processing_delay = random.uniform(0.5, 2.0)
    
    @record_performance("data_processing_agent")
    @circuit_breaker(
        name=None,  # Will use function name
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            timeout=5.0
        )
    )
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Process data with potential failures"""
        
        # Simulate processing time
        processing_time = self.processing_delay + random.uniform(-0.2, 0.2)
        await asyncio.sleep(processing_time)
        
        # Simulate failures based on failure rate
        if random.random() < self.failure_rate:
            error_types = [
                "Network timeout",
                "Database connection failed",
                "Invalid input format",
                "Memory allocation error"
            ]
            raise Exception(f"Processing failed: {random.choice(error_types)}")
        
        # Successful processing
        result = {
            "status": "success",
            "processed_items": task_data.get("items", 0),
            "processing_time": processing_time,
            "agent_id": self.agent_id
        }
        
        return result


class AnalyticsAgent(MeshAgent):
    """Agent that performs analytics with slower responses"""
    
    def __init__(self, agent_id: str, config: MeshConfig = None):
        super().__init__(agent_id, config)
        self.slow_threshold = 0.3  # 30% chance of slow response
    
    @record_performance("analytics_agent")
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Perform analytics with variable response times"""
        
        # Simulate slow responses occasionally
        if random.random() < self.slow_threshold:
            # Slow response (6-10 seconds)
            processing_time = random.uniform(6.0, 10.0)
            await asyncio.sleep(processing_time)
        else:
            # Normal response (0.5-2 seconds)
            processing_time = random.uniform(0.5, 2.0)
            await asyncio.sleep(processing_time)
        
        result = {
            "status": "success",
            "analysis": {
                "trend": random.choice(["increasing", "decreasing", "stable"]),
                "confidence": random.uniform(0.7, 0.95),
                "processing_time": processing_time
            },
            "agent_id": self.agent_id
        }
        
        return result


class ReportingAgent(MeshAgent):
    """Reliable agent for report generation"""
    
    @record_performance("reporting_agent")
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Generate reports reliably"""
        
        # Simulate consistent processing
        processing_time = random.uniform(1.0, 2.0)
        await asyncio.sleep(processing_time)
        
        result = {
            "status": "success",
            "report": {
                "type": task_data.get("report_type", "standard"),
                "format": task_data.get("format", "pdf"),
                "pages": random.randint(5, 20)
            },
            "processing_time": processing_time,
            "agent_id": self.agent_id
        }
        
        return result


async def setup_health_monitoring():
    """Set up comprehensive health monitoring"""
    
    print("=== Setting Up Health Monitoring ===")
    
    # 1. Configure Performance Monitoring
    print("1. Starting performance monitoring...")
    await performance_monitor.start_monitoring()
    
    # Add performance alert rules
    performance_monitor.add_alert_rule(AlertRule(
        name="high_error_rate",
        metric_name="error_rate",
        threshold=0.2,  # 20% error rate
        comparison="gt",
        severity=AlertSeverity.HIGH,
        duration=timedelta(minutes=2),
        agent_filter=None  # Apply to all agents
    ))
    
    performance_monitor.add_alert_rule(AlertRule(
        name="slow_response_time",
        metric_name="response_time",
        threshold=5.0,  # 5 seconds
        comparison="gt",
        severity=AlertSeverity.MEDIUM,
        duration=timedelta(minutes=1)
    ))
    
    # 2. Configure Failover Management
    print("2. Starting failover management...")
    await failover_manager.start_monitoring()
    
    # Add failover rules
    failover_manager.add_failover_rule(FailoverRule(
        name="data_processing_failover",
        capability_filter=["data-processing"],
        max_failures=3,
        failure_window=timedelta(minutes=5),
        strategy=FailoverStrategy.HEALTH_BASED
    ))
    
    # 3. Configure Health Monitoring
    print("3. Starting health monitoring...")
    await health_monitor.start_monitoring()
    
    # Add custom health check
    async def check_external_api():
        """Custom health check for external API"""
        try:
            # Simulate external API check
            await asyncio.sleep(0.1)
            return random.random() > 0.1  # 10% failure rate
        except:
            return False
    
    health_monitor.register_health_check(
        name="external_api",
        check_function=check_external_api,
        interval=timedelta(seconds=30),
        timeout=timedelta(seconds=5),
        critical=False,
        description="Check external API availability"
    )
    
    # 4. Configure Alerting System
    print("4. Setting up alerting system...")
    
    # Configure Slack notifications (example)
    alerting_system.configure_slack(
        name="slack_alerts",
        webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",  # Replace with real URL
        channel="#meshai-alerts",
        enabled=True  # Set to False in this example since we don't have real webhook
    )
    
    # Configure webhook notifications
    alerting_system.configure_webhook(
        name="webhook_alerts",
        url="http://localhost:8080/webhook/alerts",  # Example webhook
        enabled=False  # Disabled for demo
    )
    
    # Add alert routing
    alerting_system.add_alert_route(AlertRoute(
        name="critical_alerts",
        conditions={"severity": "critical"},
        channels=["slack_alerts"],
        escalation_timeout=timedelta(minutes=5),
        severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.HIGH]
    ))
    
    await alerting_system.start_processing()
    
    print("✅ Health monitoring setup complete!")


async def register_agents():
    """Register agents with failover management"""
    
    print("\n=== Registering Agents ===")
    
    # Register multiple instances of each agent type for failover
    agents = []
    
    # Data processing agents (some unreliable)
    for i in range(3):
        failure_rate = 0.05 if i == 0 else 0.2  # First agent more reliable
        agent = DataProcessingAgent(f"data-processor-{i+1}", failure_rate)
        agents.append(agent)
        
        # Register with failover manager
        failover_manager.register_agent(
            agent_id=agent.agent_id,
            endpoint_url=f"http://agent-{agent.agent_id}:8080",
            capabilities=["data-processing"],
            weight=2.0 if i == 0 else 1.0,  # Higher weight for reliable agent
            priority=1,
            metadata={"type": "data_processor", "failure_rate": failure_rate}
        )
    
    # Analytics agents (some slow)
    for i in range(2):
        agent = AnalyticsAgent(f"analytics-{i+1}")
        agents.append(agent)
        
        failover_manager.register_agent(
            agent_id=agent.agent_id,
            endpoint_url=f"http://agent-{agent.agent_id}:8080",
            capabilities=["analytics"],
            weight=1.0,
            priority=1,
            metadata={"type": "analytics"}
        )
    
    # Reporting agents (reliable)
    for i in range(2):
        agent = ReportingAgent(f"reporter-{i+1}")
        agents.append(agent)
        
        failover_manager.register_agent(
            agent_id=agent.agent_id,
            endpoint_url=f"http://agent-{agent.agent_id}:8080",
            capabilities=["reporting"],
            weight=1.0,
            priority=1,
            metadata={"type": "reporting"}
        )
    
    print(f"✅ Registered {len(agents)} agents")
    return agents


async def simulate_workload(agents, duration_minutes=10):
    """Simulate realistic workload with various scenarios"""
    
    print(f"\n=== Simulating Workload for {duration_minutes} minutes ===")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    request_count = 0
    
    while time.time() < end_time:
        try:
            # Select random task type
            task_types = [
                ("data-processing", {"items": random.randint(100, 1000)}),
                ("analytics", {"dataset": "customer_data", "analysis_type": "trend"}),
                ("reporting", {"report_type": "weekly", "format": "pdf"})
            ]
            
            capability, task_data = random.choice(task_types)
            
            # Get healthy agent for this capability
            selected_agent = await failover_manager.get_healthy_agent(capability)
            
            if selected_agent:
                # Find the actual agent instance
                agent_instance = next(
                    (a for a in agents if a.agent_id == selected_agent.agent_id),
                    None
                )
                
                if agent_instance:
                    try:
                        # Execute task
                        context = MeshContext()
                        result = await agent_instance.handle_task(task_data, context)
                        
                        # Record successful request
                        await failover_manager.record_request_result(
                            selected_agent.agent_id,
                            success=True,
                            response_time=result.get("processing_time", 1.0),
                            capability=capability
                        )
                        
                        request_count += 1
                        
                        if request_count % 20 == 0:
                            print(f"Processed {request_count} requests...")
                        
                    except Exception as e:
                        # Record failed request
                        await failover_manager.record_request_result(
                            selected_agent.agent_id,
                            success=False,
                            response_time=0.0,
                            capability=capability
                        )
                        
                        # Try failover
                        alternative = await failover_manager.handle_failover(
                            selected_agent.agent_id,
                            capability,
                            e
                        )
                        
                        if alternative:
                            print(f"Failed over from {selected_agent.agent_id} to {alternative.agent_id}")
                        else:
                            print(f"No failover available for {capability}")
            
            # Vary request rate
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
        except Exception as e:
            logger.error(f"Error in workload simulation: {e}")
            await asyncio.sleep(1)
    
    print(f"✅ Workload simulation complete. Processed {request_count} requests.")


async def demonstrate_dashboard_analytics():
    """Demonstrate analytics dashboard capabilities"""
    
    print("\n=== Dashboard Analytics Demo ===")
    
    # Get system overview
    print("1. System Overview:")
    overview = await analytics_dashboard.get_system_overview()
    print(f"   System Health: {overview.get('system_health')}")
    print(f"   Total Agents: {overview.get('agents', {}).get('total', 0)}")
    print(f"   Healthy Agents: {overview.get('agents', {}).get('healthy', 0)}")
    print(f"   Total Requests: {overview.get('requests', {}).get('total', 0)}")
    print(f"   Error Rate: {overview.get('requests', {}).get('error_rate', 0):.2%}")
    
    # Get agent status
    print("\n2. Agent Status:")
    agents = await analytics_dashboard.get_agent_list()
    for agent in agents[:5]:  # Show first 5 agents
        print(f"   {agent.agent_id}: {agent.status} (Health: {agent.health}, "
              f"Requests: {agent.total_requests}, Error Rate: {agent.error_rate:.2%})")
    
    # Get performance metrics
    print("\n3. Performance Dashboard:")
    performance = await analytics_dashboard.get_performance_dashboard()
    summary = performance.get('summary', {})
    print(f"   Average Response Time: {summary.get('avg_response_time', 0):.2f}ms")
    print(f"   P95 Response Time: {summary.get('p95_response_time', 0):.2f}ms")
    print(f"   Total Throughput: {summary.get('total_throughput', 0):.2f} req/s")
    
    # Get alerts
    print("\n4. Active Alerts:")
    alert_data = await analytics_dashboard.get_alert_dashboard()
    active_alerts = alert_data.get('active_alerts', {})
    print(f"   Total Alerts: {active_alerts.get('total', 0)}")
    by_severity = active_alerts.get('by_severity', {})
    for severity, count in by_severity.items():
        if count > 0:
            print(f"   {severity.upper()}: {count}")
    
    # Get dashboard widgets
    print("\n5. Dashboard Widgets:")
    widgets = await analytics_dashboard.get_dashboard_widgets()
    for widget in widgets:
        print(f"   - {widget.title} ({widget.widget_type})")


async def demonstrate_health_endpoints():
    """Demonstrate health check endpoints"""
    
    print("\n=== Health Check Endpoints ===")
    
    # Get overall health status
    print("1. Overall Health Status:")
    health_status = await health_monitor.get_health_status()
    print(f"   Status: {health_status.status.value}")
    print(f"   Uptime: {health_status.uptime_seconds:.0f} seconds")
    print(f"   Components: {len(health_status.components)}")
    
    # Get detailed status
    print("\n2. Detailed System Status:")
    detailed_status = await health_monitor.get_detailed_status()
    
    # Show component health
    components = detailed_status.get('components', {})
    for name, component in components.items():
        status = component.get('status', 'unknown')
        message = component.get('message', 'No message')
        print(f"   {name}: {status} - {message}")
    
    # Get readiness status (for Kubernetes)
    print("\n3. Readiness Status (K8s):")
    ready, readiness_info = health_monitor.get_readiness_status()
    print(f"   Ready: {ready}")
    if not ready:
        failed_checks = readiness_info.get('failed_checks', [])
        print(f"   Failed Checks: {failed_checks}")
    
    # Get liveness status (for Kubernetes)
    print("\n4. Liveness Status (K8s):")
    alive, liveness_info = health_monitor.get_liveness_status()
    print(f"   Alive: {alive}")
    print(f"   Monitoring Active: {liveness_info.get('monitoring_active', False)}")


async def cleanup_monitoring():
    """Clean up monitoring resources"""
    
    print("\n=== Cleaning Up ===")
    
    await performance_monitor.stop_monitoring()
    await failover_manager.stop_monitoring()
    await health_monitor.stop_monitoring()
    await alerting_system.stop_processing()
    
    print("✅ Monitoring cleanup complete")


async def main():
    """Run the comprehensive health monitoring demonstration"""
    
    print("🏥 MeshAI Enhanced Health Monitoring Demo")
    print("=" * 50)
    
    try:
        # Set up health monitoring
        await setup_health_monitoring()
        
        # Register agents
        agents = await register_agents()
        
        # Let monitoring systems initialize
        print("\n⏱️  Allowing monitoring systems to initialize...")
        await asyncio.sleep(5)
        
        # Simulate workload
        await simulate_workload(agents, duration_minutes=3)  # 3 minutes for demo
        
        # Let metrics stabilize
        print("\n⏱️  Allowing metrics to stabilize...")
        await asyncio.sleep(10)
        
        # Demonstrate analytics
        await demonstrate_dashboard_analytics()
        
        # Demonstrate health endpoints
        await demonstrate_health_endpoints()
        
        print("\n🎉 Health monitoring demonstration complete!")
        print("\nIn a real deployment, you would:")
        print("1. Set up Prometheus/Grafana for metrics visualization")
        print("2. Configure real Slack/PagerDuty integrations")
        print("3. Deploy health check endpoints for load balancers")
        print("4. Set up log aggregation and analysis")
        print("5. Configure automated remediation actions")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        await cleanup_monitoring()


if __name__ == "__main__":
    asyncio.run(main())