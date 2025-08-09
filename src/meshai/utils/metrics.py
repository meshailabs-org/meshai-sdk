"""
Metrics collection utilities for MeshAI SDK
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time

from prometheus_client import (
    Counter, 
    Histogram, 
    Gauge, 
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)


class MetricsCollector:
    """
    Prometheus metrics collector for MeshAI agents.
    
    Tracks:
    - Task execution metrics
    - Performance metrics  
    - Error rates
    - System health
    """
    
    # Class-level metrics to avoid duplication
    _metrics_initialized = False
    _shared_metrics = {}
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Initialize shared metrics only once
        if not MetricsCollector._metrics_initialized:
            self._init_shared_metrics()
            MetricsCollector._metrics_initialized = True
        
        # Reference shared metrics
        self.tasks_submitted = MetricsCollector._shared_metrics["tasks_submitted"]
        self.tasks_completed = MetricsCollector._shared_metrics["tasks_completed"]
        self.tasks_failed = MetricsCollector._shared_metrics["tasks_failed"]
        self.task_duration = MetricsCollector._shared_metrics["task_duration"]
        self.task_success_rate = MetricsCollector._shared_metrics["task_success_rate"]
        self.response_time = MetricsCollector._shared_metrics["response_time"]
        self.active_tasks = MetricsCollector._shared_metrics["active_tasks"]
        self.task_timeout_count = MetricsCollector._shared_metrics["task_timeout_count"]
        self.connection_errors = MetricsCollector._shared_metrics["connection_errors"]
        self.agent_uptime = MetricsCollector._shared_metrics["agent_uptime"]
        self.agent_info = MetricsCollector._shared_metrics["agent_info"]
        self.context_operations = MetricsCollector._shared_metrics["context_operations"]
        self.context_size = MetricsCollector._shared_metrics["context_size"]
        self.total_registrations = MetricsCollector._shared_metrics["total_registrations"]
        self.total_discoveries = MetricsCollector._shared_metrics["total_discoveries"]
        
        self._start_time = time.time()
        
        # Set agent info
        self.agent_info.labels(agent_id=agent_id).info({
            "sdk_version": "0.1.0",
            "start_time": datetime.utcnow().isoformat()
        })
    
    @classmethod
    def _init_shared_metrics(cls):
        """Initialize shared metrics once"""
        cls._shared_metrics = {
            "tasks_submitted": Counter(
                "meshai_tasks_submitted_total",
                "Total tasks submitted",
                ["agent_id"]
            ),
            "tasks_completed": Counter(
                "meshai_tasks_completed_total", 
                "Total tasks completed successfully",
                ["agent_id"]
            ),
            "tasks_failed": Counter(
                "meshai_tasks_failed_total",
                "Total tasks failed", 
                ["agent_id", "error_type"]
            ),
            "task_duration": Histogram(
                "meshai_task_duration_seconds",
                "Task execution duration",
                ["agent_id"]
            ),
            "task_success_rate": Counter(
                "meshai_task_success_rate",
                "Task success rate counter",
                ["agent_id"]
            ),
            "response_time": Histogram(
                "meshai_response_time_seconds", 
                "Response time for requests",
                ["agent_id", "endpoint"]
            ),
            "active_tasks": Gauge(
                "meshai_active_tasks",
                "Number of currently active tasks",
                ["agent_id"]
            ),
            "task_timeout_count": Counter(
                "meshai_task_timeouts_total",
                "Total task timeouts",
                ["agent_id"]
            ),
            "connection_errors": Counter(
                "meshai_connection_errors_total",
                "Total connection errors",
                ["agent_id", "service"]
            ),
            "agent_uptime": Gauge(
                "meshai_agent_uptime_seconds",
                "Agent uptime in seconds", 
                ["agent_id"]
            ),
            "agent_info": Info(
                "meshai_agent_info",
                "Agent information",
                ["agent_id"]
            ),
            "context_operations": Counter(
                "meshai_context_operations_total",
                "Total context operations",
                ["agent_id", "operation"]
            ),
            "context_size": Histogram(
                "meshai_context_size_bytes",
                "Context data size in bytes",
                ["agent_id"]
            ),
            "total_registrations": Counter(
                "meshai_registrations_total",
                "Total agent registrations",
                ["service_id"]
            ),
            "total_discoveries": Counter(
                "meshai_discoveries_total",
                "Total agent discoveries",
                ["service_id"]
            )
        }
    
    def record_task_submitted(self) -> None:
        """Record a task submission"""
        self.tasks_submitted.labels(agent_id=self.agent_id).inc()
    
    def record_task_completed(self, duration: float) -> None:
        """Record a successful task completion"""
        self.tasks_completed.labels(agent_id=self.agent_id).inc()
        self.task_duration.labels(agent_id=self.agent_id).observe(duration)
    
    def record_task_failed(self, error_type: str) -> None:
        """Record a failed task"""
        self.tasks_failed.labels(agent_id=self.agent_id, error_type=error_type).inc()
    
    def record_task_timeout(self) -> None:
        """Record a task timeout"""
        self.task_timeout_count.labels(agent_id=self.agent_id).inc()
    
    def record_connection_error(self, service: str) -> None:
        """Record a connection error"""
        self.connection_errors.labels(agent_id=self.agent_id, service=service).inc()
    
    def set_active_tasks(self, count: int) -> None:
        """Set the number of active tasks"""
        self.active_tasks.labels(agent_id=self.agent_id).set(count)
    
    def record_response_time(self, endpoint: str, duration: float) -> None:
        """Record response time for an endpoint"""
        self.response_time.labels(agent_id=self.agent_id, endpoint=endpoint).observe(duration)
    
    def update_uptime(self) -> None:
        """Update agent uptime"""
        uptime = time.time() - self._start_time
        self.agent_uptime.labels(agent_id=self.agent_id).set(uptime)
    
    def record_context_operation(self, operation: str) -> None:
        """Record a context operation"""
        self.context_operations.labels(agent_id=self.agent_id, operation=operation).inc()
    
    def record_context_size(self, size_bytes: int) -> None:
        """Record context size"""
        self.context_size.labels(agent_id=self.agent_id).observe(size_bytes)
    
    def record_registration(self, service_id: str = None) -> None:
        """Record an agent registration"""
        service_id = service_id or self.agent_id
        self.total_registrations.labels(service_id=service_id).inc()
    
    def record_discovery(self, service_id: str = None) -> None:
        """Record an agent discovery"""
        service_id = service_id or self.agent_id
        self.total_discoveries.labels(service_id=service_id).inc()
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest()
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format (alias for get_metrics)"""
        return self.get_metrics()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for JSON responses"""
        return {
            "agent_id": self.agent_id,
            "uptime_seconds": time.time() - self._start_time,
            "metrics_available": True
        }


# Global metrics collector for system-wide metrics
system_metrics = None

def get_system_metrics() -> MetricsCollector:
    """Get or create system metrics collector"""
    global system_metrics
    if system_metrics is None:
        system_metrics = MetricsCollector("system")
    return system_metrics