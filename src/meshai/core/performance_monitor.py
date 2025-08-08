"""
Performance Monitoring and Metrics Collection for MeshAI

This module provides comprehensive performance monitoring capabilities
including metrics collection, analysis, and alerting for AI agents.
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics we collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceStats:
    """Performance statistics for an agent"""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    error_rate: float = 0.0
    throughput_per_second: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class AlertRule:
    """Performance alert rule"""
    name: str
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    severity: AlertSeverity
    duration: timedelta  # How long condition must persist
    agent_filter: Optional[str] = None  # Filter by agent ID
    enabled: bool = True
    cooldown: timedelta = timedelta(minutes=5)
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Performance alert"""
    rule_name: str
    agent_id: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for MeshAI agents.
    
    Features:
    - Real-time metrics collection
    - Performance analytics
    - Alerting and notifications
    - Resource usage tracking
    - SLA monitoring
    """
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        
        # Metrics storage
        self.agent_stats: Dict[str, AgentPerformanceStats] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Callback handlers
        self.alert_callbacks: List[Callable] = []
        
        logger.info("Performance monitor initialized", 
                   interval=collection_interval)
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Agent performance metrics
        self.agent_requests = Counter(
            'meshai_agent_requests_total',
            'Total agent requests',
            ['agent_id', 'status']
        )
        
        self.agent_response_time = Histogram(
            'meshai_agent_response_time_seconds',
            'Agent response time distribution',
            ['agent_id']
        )
        
        self.agent_error_rate = Gauge(
            'meshai_agent_error_rate',
            'Agent error rate',
            ['agent_id']
        )
        
        self.agent_throughput = Gauge(
            'meshai_agent_throughput_per_second',
            'Agent throughput per second',
            ['agent_id']
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'meshai_system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'meshai_system_memory_usage_percent',
            'System memory usage percentage'
        )
        
        # Alert metrics
        self.active_alerts_count = Gauge(
            'meshai_active_alerts_total',
            'Number of active alerts',
            ['severity']
        )
    
    async def start_monitoring(self):
        """Start the background monitoring task"""
        if self._running:
            logger.warning("Performance monitoring already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop the background monitoring task"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._update_agent_analytics()
                await self._check_alert_rules()
                await self._cleanup_old_data()
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    def record_agent_request(
        self,
        agent_id: str,
        response_time: float,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record an agent request for performance tracking"""
        with self._lock:
            # Get or create agent stats
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = AgentPerformanceStats(agent_id=agent_id)
            
            stats = self.agent_stats[agent_id]
            
            # Update basic counters
            stats.total_requests += 1
            if success:
                stats.successful_requests += 1
                self.agent_requests.labels(agent_id=agent_id, status='success').inc()
            else:
                stats.failed_requests += 1
                self.agent_requests.labels(agent_id=agent_id, status='failure').inc()
            
            # Update response time stats
            stats.response_times.append(response_time)
            stats.min_response_time = min(stats.min_response_time, response_time)
            stats.max_response_time = max(stats.max_response_time, response_time)
            
            # Calculate percentiles
            if len(stats.response_times) >= 10:
                sorted_times = sorted(stats.response_times)
                stats.p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
                stats.p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
            
            stats.avg_response_time = sum(stats.response_times) / len(stats.response_times)
            stats.error_rate = stats.failed_requests / stats.total_requests
            stats.last_updated = datetime.utcnow()
            
            # Update Prometheus metrics
            self.agent_response_time.labels(agent_id=agent_id).observe(response_time)
            self.agent_error_rate.labels(agent_id=agent_id).set(stats.error_rate)
            
            # Store metric for trend analysis
            metric = PerformanceMetric(
                name="agent_response_time",
                value=response_time,
                timestamp=datetime.utcnow(),
                labels={"agent_id": agent_id, "success": str(success), **(labels or {})}
            )
            self.metrics[f"agent_{agent_id}_response_time"].append(metric)
    
    def record_custom_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a custom metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            self.record_custom_metric("system_cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_memory_usage.set(memory_percent)
            self.record_custom_metric("system_memory_usage", memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_custom_metric("system_disk_usage", disk_percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.record_custom_metric("network_bytes_sent", net_io.bytes_sent)
            self.record_custom_metric("network_bytes_recv", net_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _update_agent_analytics(self):
        """Update agent performance analytics"""
        with self._lock:
            current_time = datetime.utcnow()
            
            for agent_id, stats in self.agent_stats.items():
                # Calculate throughput (requests per second)
                time_window = timedelta(seconds=self.collection_interval)
                recent_metrics = [
                    m for m in self.metrics[f"agent_{agent_id}_response_time"]
                    if current_time - m.timestamp <= time_window
                ]
                
                stats.throughput_per_second = len(recent_metrics) / self.collection_interval
                self.agent_throughput.labels(agent_id=agent_id).set(stats.throughput_per_second)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a performance alert rule"""
        with self._lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    async def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed"""
        with self._lock:
            current_time = datetime.utcnow()
            
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown period
                if (rule.last_triggered and 
                    current_time - rule.last_triggered < rule.cooldown):
                    continue
                
                # Get current metric value
                current_value = self._get_current_metric_value(rule)
                if current_value is None:
                    continue
                
                # Check threshold
                should_alert = self._evaluate_threshold(
                    current_value, rule.threshold, rule.comparison
                )
                
                if should_alert:
                    await self._trigger_alert(rule, current_value)
    
    def _get_current_metric_value(self, rule: AlertRule) -> Optional[float]:
        """Get current value for a metric"""
        if rule.metric_name == "error_rate" and rule.agent_filter:
            stats = self.agent_stats.get(rule.agent_filter)
            return stats.error_rate if stats else None
        elif rule.metric_name == "response_time" and rule.agent_filter:
            stats = self.agent_stats.get(rule.agent_filter)
            return stats.avg_response_time if stats else None
        elif rule.metric_name == "throughput" and rule.agent_filter:
            stats = self.agent_stats.get(rule.agent_filter)
            return stats.throughput_per_second if stats else None
        elif rule.metric_name in self.metrics:
            recent_metrics = self.metrics[rule.metric_name]
            if recent_metrics:
                return recent_metrics[-1].value
        
        return None
    
    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate if value meets threshold condition"""
        if comparison == "gt":
            return value > threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "eq":
            return abs(value - threshold) < 0.001  # Float equality with tolerance
        return False
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert"""
        agent_id = rule.agent_filter or "system"
        alert_id = f"{rule.name}_{agent_id}"
        
        # Check if alert already exists
        if alert_id in self.active_alerts:
            return
        
        alert = Alert(
            rule_name=rule.name,
            agent_id=agent_id,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f"{rule.metric_name} {current_value:.2f} {rule.comparison} {rule.threshold}",
            triggered_at=datetime.utcnow()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update rule state
        rule.last_triggered = datetime.utcnow()
        
        # Update Prometheus metrics
        self.active_alerts_count.labels(severity=rule.severity.value).inc()
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(
            f"Alert triggered: {rule.name}",
            agent_id=agent_id,
            metric=rule.metric_name,
            value=current_value,
            threshold=rule.threshold,
            severity=rule.severity.value
        )
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.utcnow()
                
                # Update Prometheus metrics
                self.active_alerts_count.labels(severity=alert.severity.value).dec()
                
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert_id}")
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an active alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self._lock:
            # Clean up old metrics
            for name, metrics in self.metrics.items():
                # Remove metrics older than 24 hours
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()
            
            # Clean up resolved alerts older than 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)
            self.alert_history = deque([
                alert for alert in self.alert_history
                if not alert.resolved_at or alert.resolved_at > week_ago
            ], maxlen=10000)
    
    def get_agent_stats(self, agent_id: str) -> Optional[AgentPerformanceStats]:
        """Get performance stats for an agent"""
        return self.agent_stats.get(agent_id)
    
    def get_all_agent_stats(self) -> Dict[str, AgentPerformanceStats]:
        """Get performance stats for all agents"""
        with self._lock:
            return self.agent_stats.copy()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        latest_metrics = {}
        
        with self._lock:
            for name, metrics in self.metrics.items():
                if metrics and name.startswith("system_"):
                    latest_metrics[name] = metrics[-1].value
        
        return latest_metrics
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        with self._lock:
            return list(self.alert_history)[-limit:]
    
    def get_metrics_for_export(self) -> str:
        """Get metrics in Prometheus format for scraping"""
        return generate_latest()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def record_performance(agent_id: str):
    """
    Decorator to automatically record performance metrics for agent methods.
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                performance_monitor.record_agent_request(
                    agent_id=agent_id,
                    response_time=duration,
                    success=success
                )
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                performance_monitor.record_agent_request(
                    agent_id=agent_id,
                    response_time=duration,
                    success=success
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator