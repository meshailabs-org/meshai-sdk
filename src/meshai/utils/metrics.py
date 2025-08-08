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
    CONTENT_TYPE_LATEST
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
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Task metrics
        self.tasks_submitted = Counter(
            "meshai_tasks_submitted_total",
            "Total tasks submitted",
            ["agent_id"]
        )
        
        self.tasks_completed = Counter(
            "meshai_tasks_completed_total", 
            "Total tasks completed successfully",
            ["agent_id"]
        )
        
        self.tasks_failed = Counter(
            "meshai_tasks_failed_total",
            "Total tasks failed", 
            ["agent_id", "error_type"]
        )
        
        self.task_duration = Histogram(
            "meshai_task_duration_seconds",
            "Task execution duration",
            ["agent_id"]
        )
        
        # Performance metrics
        self.task_success_rate = Counter(
            "meshai_task_success_rate",
            "Task success rate counter",
            ["agent_id"]
        )
        
        self.response_time = Histogram(
            "meshai_response_time_seconds", 
            "Response time for requests",
            ["agent_id", "endpoint"]
        )
        
        self.active_tasks = Gauge(
            "meshai_active_tasks",
            "Number of currently active tasks",
            ["agent_id"]
        )
        
        # Error metrics
        self.task_timeout_count = Counter(
            "meshai_task_timeouts_total",
            "Total task timeouts",
            ["agent_id"]
        )
        
        self.connection_errors = Counter(
            "meshai_connection_errors_total",
            "Total connection errors",
            ["agent_id", "service"]
        )
        
        # System metrics
        self.agent_uptime = Gauge(
            "meshai_agent_uptime_seconds",
            "Agent uptime in seconds", 
            ["agent_id"]
        )
        
        self.agent_info = Info(
            "meshai_agent_info",
            "Agent information",
            ["agent_id"]
        )
        
        # Context metrics
        self.context_operations = Counter(
            "meshai_context_operations_total",
            "Total context operations",
            ["agent_id", "operation"]
        )
        
        self.context_size = Histogram(
            "meshai_context_size_bytes",
            "Context data size in bytes",
            ["agent_id"]
        )
        
        self._start_time = time.time()
        
        # Set agent info
        self.agent_info.labels(agent_id=agent_id).info({
            "agent_id": agent_id,
            "sdk_version": "0.1.0",
            "start_time": datetime.utcnow().isoformat()
        })
    
    def record_task_submitted(self) -> None:
        """Record a task submission"""
        self.tasks_submitted.labels(agent_id=self.agent_id).inc()
    
    def record_task_completed(self, duration: float) -> None:
        """Record a successful task completion"""
        self.tasks_completed.labels(agent_id=self.agent_id).inc()
        self.task_duration.labels(agent_id=self.agent_id).observe(duration)
        self.task_success_rate.labels(agent_id=self.agent_id).inc()
    
    def record_task_failed(self, error_type: str = "unknown") -> None:
        """Record a task failure"""
        self.tasks_failed.labels(agent_id=self.agent_id, error_type=error_type).inc()
    
    def record_task_timeout(self) -> None:
        """Record a task timeout"""
        self.task_timeout_count.labels(agent_id=self.agent_id).inc()
    
    def record_response_time(self, endpoint: str, duration: float) -> None:
        """Record response time for an endpoint"""
        self.response_time.labels(agent_id=self.agent_id, endpoint=endpoint).observe(duration)
    
    def record_connection_error(self, service: str) -> None:
        """Record a connection error"""
        self.connection_errors.labels(agent_id=self.agent_id, service=service).inc()
    
    def set_active_tasks(self, count: int) -> None:
        """Set the number of active tasks"""
        self.active_tasks.labels(agent_id=self.agent_id).set(count)
    
    def record_context_operation(self, operation: str) -> None:
        """Record a context operation"""
        self.context_operations.labels(agent_id=self.agent_id, operation=operation).inc()
    
    def record_context_size(self, size_bytes: int) -> None:
        """Record context data size"""
        self.context_size.labels(agent_id=self.agent_id).observe(size_bytes)
    
    def update_uptime(self) -> None:
        """Update agent uptime"""
        uptime = time.time() - self._start_time
        self.agent_uptime.labels(agent_id=self.agent_id).set(uptime)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        # Update uptime
        self.update_uptime()
        
        # Get current values (simplified for basic metrics)
        return {
            "tasks_submitted": self.tasks_submitted.labels(agent_id=self.agent_id)._value._value,
            "tasks_completed": self.tasks_completed.labels(agent_id=self.agent_id)._value._value,
            "tasks_failed": sum(
                metric._value._value 
                for metric in self.tasks_failed._metrics.values()
                if metric._labelvalues[0] == self.agent_id
            ),
            "active_tasks": self.active_tasks.labels(agent_id=self.agent_id)._value._value,
            "uptime_seconds": time.time() - self._start_time,
            "success_rate": self._calculate_success_rate(),
            "avg_response_time": self._calculate_avg_response_time()
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        return generate_latest().decode('utf-8')
    
    def get_prometheus_content_type(self) -> str:
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST
    
    def _calculate_success_rate(self) -> float:
        """Calculate current success rate"""
        completed = self.tasks_completed.labels(agent_id=self.agent_id)._value._value
        total_failed = sum(
            metric._value._value 
            for metric in self.tasks_failed._metrics.values()
            if metric._labelvalues[0] == self.agent_id
        )
        
        total = completed + total_failed
        if total == 0:
            return 1.0
        
        return completed / total
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        # This is a simplified calculation
        # In practice, you'd want to maintain a sliding window
        try:
            histogram = self.response_time.labels(agent_id=self.agent_id, endpoint="execute")
            if histogram._count._value == 0:
                return 0.0
            return histogram._sum._value / histogram._count._value
        except:
            return 0.0
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)"""
        # Note: Prometheus metrics don't support reset in the client library
        # This is a placeholder for future implementation
        pass


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, metrics: MetricsCollector, operation: str):
        self.metrics = metrics
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            # Operation succeeded
            self.metrics.record_response_time(self.operation, duration)
        else:
            # Operation failed
            self.metrics.record_task_failed(str(exc_type.__name__))
    
    def elapsed(self) -> float:
        """Get elapsed time since start"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0