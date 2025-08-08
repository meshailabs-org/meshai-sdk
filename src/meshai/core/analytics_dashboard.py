"""
Analytics Dashboard Data Provider for MeshAI Health Monitoring

This module provides structured data for building performance analytics
dashboards and real-time monitoring interfaces.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import threading

import structlog
from .performance_monitor import performance_monitor, AgentPerformanceStats
from .circuit_breaker import circuit_breaker_manager, CircuitState
from .failover_manager import failover_manager, AgentHealth
from .health_monitor import health_monitor, HealthStatus
from .alerting_system import alerting_system

logger = structlog.get_logger(__name__)


class TimeRange(str, Enum):
    """Time range options for analytics"""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"


class MetricAggregation(str, Enum):
    """Metric aggregation methods"""
    AVERAGE = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    P95 = "p95"
    P99 = "p99"


@dataclass
class TimeSeriesPoint:
    """A single point in a time series"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration and data"""
    id: str
    title: str
    widget_type: str  # "metric", "chart", "table", "alert_list", etc.
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentOverview:
    """Agent overview for dashboard"""
    agent_id: str
    status: str
    health: str
    total_requests: int
    error_rate: float
    avg_response_time: float
    last_seen: datetime
    circuit_breaker_state: str
    current_connections: int


class AnalyticsDashboard:
    """
    Analytics dashboard data provider.
    
    Provides structured data for building monitoring dashboards including:
    - Real-time metrics
    - Time series data
    - Agent performance overviews
    - System health status
    - Alert summaries
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
        logger.info("Analytics dashboard initialized")
    
    def _get_cached_or_compute(
        self,
        cache_key: str,
        compute_func: callable,
        ttl_seconds: int = 60
    ) -> Any:
        """Get cached result or compute new value"""
        with self._lock:
            now = datetime.utcnow()
            
            # Check if we have valid cached data
            if (cache_key in self._cache and 
                cache_key in self._cache_ttl and
                now < self._cache_ttl[cache_key]):
                return self._cache[cache_key]
            
            # Compute new value
            try:
                result = compute_func()
                self._cache[cache_key] = result
                self._cache_ttl[cache_key] = now + timedelta(seconds=ttl_seconds)
                return result
            except Exception as e:
                logger.error(f"Failed to compute dashboard data for {cache_key}: {e}")
                return self._cache.get(cache_key)  # Return stale data if available
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get high-level system overview"""
        return self._get_cached_or_compute(
            "system_overview",
            self._compute_system_overview,
            ttl_seconds=30
        )
    
    def _compute_system_overview(self) -> Dict[str, Any]:
        """Compute system overview metrics"""
        try:
            # Get health status
            health_task = asyncio.create_task(health_monitor.get_health_status())
            health_status = asyncio.get_event_loop().run_until_complete(health_task)
            
            # Get agent stats
            agent_stats = performance_monitor.get_all_agent_stats()
            
            # Get circuit breaker stats
            circuit_stats = circuit_breaker_manager.get_all_metrics()
            
            # Get failover stats
            failover_stats = failover_manager.get_all_agents_status()
            
            # Get active alerts
            active_alerts = alerting_system.get_active_notifications()
            
            # Calculate aggregate metrics
            total_agents = len(agent_stats)
            healthy_agents = len([
                s for s in failover_stats.values()
                if s and s.get('health') == 'healthy'
            ])
            
            total_requests = sum(s.total_requests for s in agent_stats.values())
            total_errors = sum(s.failed_requests for s in agent_stats.values())
            
            avg_response_time = 0
            if agent_stats:
                response_times = [s.avg_response_time for s in agent_stats.values() if s.avg_response_time > 0]
                if response_times:
                    avg_response_time = statistics.mean(response_times)
            
            open_circuits = len([
                m for m in circuit_stats.values()
                if m.get('state') == 'open'
            ])
            
            # Alert summary by severity
            alert_summary = {
                'critical': len([a for a in active_alerts if a.get('severity') == 'critical']),
                'high': len([a for a in active_alerts if a.get('severity') == 'high']),
                'medium': len([a for a in active_alerts if a.get('severity') == 'medium']),
                'low': len([a for a in active_alerts if a.get('severity') == 'low'])
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system_health": health_status.status.value,
                "uptime_seconds": health_status.uptime_seconds,
                "agents": {
                    "total": total_agents,
                    "healthy": healthy_agents,
                    "health_ratio": healthy_agents / total_agents if total_agents > 0 else 0
                },
                "requests": {
                    "total": total_requests,
                    "errors": total_errors,
                    "error_rate": total_errors / total_requests if total_requests > 0 else 0,
                    "avg_response_time": avg_response_time
                },
                "circuit_breakers": {
                    "total": len(circuit_stats),
                    "open": open_circuits,
                    "open_ratio": open_circuits / len(circuit_stats) if circuit_stats else 0
                },
                "alerts": {
                    "total": len(active_alerts),
                    "by_severity": alert_summary
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to compute system overview: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def get_agent_list(self) -> List[AgentOverview]:
        """Get list of all agents with their status"""
        return self._get_cached_or_compute(
            "agent_list",
            self._compute_agent_list,
            ttl_seconds=30
        )
    
    def _compute_agent_list(self) -> List[AgentOverview]:
        """Compute agent list with status information"""
        try:
            agent_stats = performance_monitor.get_all_agent_stats()
            failover_stats = failover_manager.get_all_agents_status()
            circuit_stats = circuit_breaker_manager.get_all_metrics()
            
            agents = []
            
            # Combine data from all sources
            all_agent_ids = set()
            all_agent_ids.update(agent_stats.keys())
            all_agent_ids.update(failover_stats.keys())
            
            for agent_id in all_agent_ids:
                perf_stats = agent_stats.get(agent_id)
                failover_status = failover_stats.get(agent_id, {})
                circuit_breaker = circuit_stats.get(f"agent_{agent_id}", {})
                
                # Determine overall status
                health = failover_status.get('health', 'unknown')
                circuit_state = circuit_breaker.get('state', 'unknown')
                
                if circuit_state == 'open':
                    status = 'offline'
                elif health == 'healthy':
                    status = 'healthy'
                elif health == 'degraded':
                    status = 'degraded'
                elif health == 'unhealthy':
                    status = 'unhealthy'
                else:
                    status = 'unknown'
                
                agent_overview = AgentOverview(
                    agent_id=agent_id,
                    status=status,
                    health=health,
                    total_requests=perf_stats.total_requests if perf_stats else 0,
                    error_rate=perf_stats.error_rate if perf_stats else 0,
                    avg_response_time=perf_stats.avg_response_time if perf_stats else 0,
                    last_seen=perf_stats.last_updated if perf_stats else datetime.utcnow(),
                    circuit_breaker_state=circuit_state,
                    current_connections=failover_status.get('current_connections', 0)
                )
                
                agents.append(agent_overview)
            
            # Sort by status (healthy first, then by agent_id)
            status_priority = {'healthy': 0, 'degraded': 1, 'unhealthy': 2, 'offline': 3, 'unknown': 4}
            agents.sort(key=lambda a: (status_priority.get(a.status, 5), a.agent_id))
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to compute agent list: {e}")
            return []
    
    async def get_metrics_time_series(
        self,
        metric_name: str,
        time_range: TimeRange,
        agent_id: Optional[str] = None,
        aggregation: MetricAggregation = MetricAggregation.AVERAGE
    ) -> List[TimeSeriesPoint]:
        """Get time series data for a specific metric"""
        cache_key = f"timeseries_{metric_name}_{time_range.value}_{agent_id}_{aggregation.value}"
        
        return self._get_cached_or_compute(
            cache_key,
            lambda: self._compute_time_series(metric_name, time_range, agent_id, aggregation),
            ttl_seconds=60
        )
    
    def _compute_time_series(
        self,
        metric_name: str,
        time_range: TimeRange,
        agent_id: Optional[str],
        aggregation: MetricAggregation
    ) -> List[TimeSeriesPoint]:
        """Compute time series data"""
        try:
            # This is a simplified implementation
            # In a real system, you'd query your metrics storage (e.g., Prometheus, InfluxDB)
            
            now = datetime.utcnow()
            
            # Determine time range
            if time_range == TimeRange.LAST_HOUR:
                start_time = now - timedelta(hours=1)
                interval = timedelta(minutes=1)
            elif time_range == TimeRange.LAST_6_HOURS:
                start_time = now - timedelta(hours=6)
                interval = timedelta(minutes=5)
            elif time_range == TimeRange.LAST_24_HOURS:
                start_time = now - timedelta(hours=24)
                interval = timedelta(minutes=15)
            elif time_range == TimeRange.LAST_7_DAYS:
                start_time = now - timedelta(days=7)
                interval = timedelta(hours=1)
            else:  # LAST_30_DAYS
                start_time = now - timedelta(days=30)
                interval = timedelta(hours=6)
            
            # Generate sample data points
            points = []
            current_time = start_time
            
            while current_time <= now:
                # This would be replaced with actual data retrieval
                value = self._get_sample_metric_value(metric_name, current_time, agent_id)
                
                points.append(TimeSeriesPoint(
                    timestamp=current_time,
                    value=value,
                    labels={"agent_id": agent_id} if agent_id else {}
                ))
                
                current_time += interval
            
            return points
            
        except Exception as e:
            logger.error(f"Failed to compute time series for {metric_name}: {e}")
            return []
    
    def _get_sample_metric_value(
        self,
        metric_name: str,
        timestamp: datetime,
        agent_id: Optional[str]
    ) -> float:
        """Generate sample metric values (replace with actual data retrieval)"""
        import random
        import math
        
        # Generate realistic sample data based on metric type
        base_time = timestamp.timestamp()
        
        if metric_name == "response_time":
            # Simulate response time with some variance
            base_value = 100 + 50 * math.sin(base_time / 3600)  # Hourly pattern
            noise = random.gauss(0, 10)
            return max(0, base_value + noise)
        elif metric_name == "error_rate":
            # Simulate error rate (0-100%)
            base_value = 2 + 3 * math.sin(base_time / 7200)  # 2-hour pattern
            noise = random.gauss(0, 0.5)
            return max(0, min(100, base_value + noise))
        elif metric_name == "throughput":
            # Simulate requests per second
            base_value = 50 + 30 * math.sin(base_time / 3600)
            noise = random.gauss(0, 5)
            return max(0, base_value + noise)
        elif metric_name == "cpu_usage":
            # Simulate CPU usage
            base_value = 30 + 20 * math.sin(base_time / 1800)
            noise = random.gauss(0, 5)
            return max(0, min(100, base_value + noise))
        else:
            return random.uniform(0, 100)
    
    async def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get alert dashboard data"""
        return self._get_cached_or_compute(
            "alert_dashboard",
            self._compute_alert_dashboard,
            ttl_seconds=30
        )
    
    def _compute_alert_dashboard(self) -> Dict[str, Any]:
        """Compute alert dashboard data"""
        try:
            active_alerts = alerting_system.get_active_notifications()
            notification_stats = alerting_system.get_notification_stats()
            
            # Group alerts by severity
            alerts_by_severity = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
            
            for alert in active_alerts:
                severity = alert.get('severity', 'low')
                if severity in alerts_by_severity:
                    alerts_by_severity[severity].append(alert)
            
            # Recent alert trends (simplified)
            alert_trends = {
                'last_hour': len([a for a in active_alerts if self._is_within_last_hour(a.get('triggered_at'))]),
                'last_24h': len(active_alerts)
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "active_alerts": {
                    "total": len(active_alerts),
                    "by_severity": {k: len(v) for k, v in alerts_by_severity.items()}
                },
                "alert_list": active_alerts[:20],  # Latest 20 alerts
                "notification_stats": notification_stats,
                "trends": alert_trends
            }
            
        except Exception as e:
            logger.error(f"Failed to compute alert dashboard: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def _is_within_last_hour(self, timestamp_str: Optional[str]) -> bool:
        """Check if timestamp is within the last hour"""
        if not timestamp_str:
            return False
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return datetime.utcnow() - timestamp <= timedelta(hours=1)
        except:
            return False
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        return self._get_cached_or_compute(
            "performance_dashboard",
            self._compute_performance_dashboard,
            ttl_seconds=30
        )
    
    def _compute_performance_dashboard(self) -> Dict[str, Any]:
        """Compute performance dashboard data"""
        try:
            agent_stats = performance_monitor.get_all_agent_stats()
            system_metrics = performance_monitor.get_system_metrics()
            
            # Calculate aggregate performance metrics
            if agent_stats:
                total_requests = sum(s.total_requests for s in agent_stats.values())
                total_errors = sum(s.failed_requests for s in agent_stats.values())
                
                response_times = [s.avg_response_time for s in agent_stats.values() if s.avg_response_time > 0]
                throughputs = [s.throughput_per_second for s in agent_stats.values()]
                
                avg_response_time = statistics.mean(response_times) if response_times else 0
                total_throughput = sum(throughputs)
                
                # Calculate percentiles
                p95_response_time = 0
                p99_response_time = 0
                if response_times:
                    sorted_times = sorted(response_times)
                    p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
                    p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
            else:
                total_requests = 0
                total_errors = 0
                avg_response_time = 0
                total_throughput = 0
                p95_response_time = 0
                p99_response_time = 0
            
            # Top performing agents
            top_agents = sorted(
                agent_stats.items(),
                key=lambda x: x[1].throughput_per_second,
                reverse=True
            )[:5]
            
            # Slowest agents
            slowest_agents = sorted(
                agent_stats.items(),
                key=lambda x: x[1].avg_response_time,
                reverse=True
            )[:5]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "error_rate": total_errors / total_requests if total_requests > 0 else 0,
                    "avg_response_time": avg_response_time,
                    "p95_response_time": p95_response_time,
                    "p99_response_time": p99_response_time,
                    "total_throughput": total_throughput
                },
                "system_metrics": system_metrics,
                "top_performers": [
                    {
                        "agent_id": agent_id,
                        "throughput": stats.throughput_per_second,
                        "response_time": stats.avg_response_time,
                        "error_rate": stats.error_rate
                    }
                    for agent_id, stats in top_agents
                ],
                "slowest_agents": [
                    {
                        "agent_id": agent_id,
                        "response_time": stats.avg_response_time,
                        "throughput": stats.throughput_per_second,
                        "error_rate": stats.error_rate
                    }
                    for agent_id, stats in slowest_agents
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to compute performance dashboard: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def get_dashboard_widgets(self) -> List[DashboardWidget]:
        """Get all dashboard widgets"""
        widgets = []
        
        try:
            # System overview widget
            system_overview = await self.get_system_overview()
            widgets.append(DashboardWidget(
                id="system_overview",
                title="System Overview",
                widget_type="metrics_summary",
                data=system_overview
            ))
            
            # Agent status widget
            agent_list = await self.get_agent_list()
            widgets.append(DashboardWidget(
                id="agent_status",
                title="Agent Status",
                widget_type="agent_table",
                data=agent_list
            ))
            
            # Performance metrics widget
            performance_data = await self.get_performance_dashboard()
            widgets.append(DashboardWidget(
                id="performance_metrics",
                title="Performance Metrics",
                widget_type="performance_summary",
                data=performance_data
            ))
            
            # Alert summary widget
            alert_data = await self.get_alert_dashboard()
            widgets.append(DashboardWidget(
                id="alert_summary",
                title="Active Alerts",
                widget_type="alert_summary",
                data=alert_data
            ))
            
            # Response time chart widget
            response_time_series = await self.get_metrics_time_series(
                "response_time",
                TimeRange.LAST_24_HOURS
            )
            widgets.append(DashboardWidget(
                id="response_time_chart",
                title="Response Time Trend",
                widget_type="time_series_chart",
                data={
                    "series": response_time_series,
                    "y_axis_label": "Response Time (ms)",
                    "chart_type": "line"
                }
            ))
            
            # Throughput chart widget
            throughput_series = await self.get_metrics_time_series(
                "throughput",
                TimeRange.LAST_24_HOURS
            )
            widgets.append(DashboardWidget(
                id="throughput_chart",
                title="Throughput Trend",
                widget_type="time_series_chart",
                data={
                    "series": throughput_series,
                    "y_axis_label": "Requests/sec",
                    "chart_type": "area"
                }
            ))
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard widgets: {e}")
        
        return widgets
    
    def clear_cache(self):
        """Clear the dashboard cache"""
        with self._lock:
            self._cache.clear()
            self._cache_ttl.clear()
        logger.info("Dashboard cache cleared")


# Global analytics dashboard instance
analytics_dashboard = AnalyticsDashboard()