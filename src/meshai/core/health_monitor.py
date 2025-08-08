"""
Health Monitoring System for MeshAI Agents

This module provides comprehensive health monitoring capabilities
including health checks, status endpoints, and diagnostic information.
"""

import asyncio
import time
import psutil
import aiohttp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

import structlog
from .performance_monitor import performance_monitor
from .circuit_breaker import circuit_breaker_manager
from .failover_manager import failover_manager, AgentHealth

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Overall health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentStatus(str, Enum):
    """Individual component status"""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable
    interval: timedelta
    timeout: timedelta
    critical: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)
    last_run: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    name: str
    status: ComponentStatus
    message: str = ""
    last_checked: datetime = field(default_factory=datetime.utcnow)
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Features:
    - Configurable health checks
    - System component monitoring
    - External dependency checks
    - Health status endpoints
    - Integration with performance metrics
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.start_time = datetime.utcnow()
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # Monitoring state
        self._lock = threading.RLock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Built-in checks
        self._register_builtin_checks()
        
        logger.info("Health monitor initialized", interval=check_interval)
    
    def _register_builtin_checks(self):
        """Register built-in health checks"""
        # System resource checks
        self.register_health_check(
            "cpu_usage",
            self._check_cpu_usage,
            interval=timedelta(seconds=30),
            timeout=timedelta(seconds=5),
            critical=False,
            description="Monitor CPU usage"
        )
        
        self.register_health_check(
            "memory_usage",
            self._check_memory_usage,
            interval=timedelta(seconds=30),
            timeout=timedelta(seconds=5),
            critical=False,
            description="Monitor memory usage"
        )
        
        self.register_health_check(
            "disk_usage",
            self._check_disk_usage,
            interval=timedelta(minutes=5),
            timeout=timedelta(seconds=10),
            critical=False,
            description="Monitor disk usage"
        )
        
        # Agent system checks
        self.register_health_check(
            "circuit_breakers",
            self._check_circuit_breakers,
            interval=timedelta(seconds=30),
            timeout=timedelta(seconds=5),
            critical=True,
            description="Monitor circuit breaker states"
        )
        
        self.register_health_check(
            "agent_failover",
            self._check_agent_failover,
            interval=timedelta(seconds=30),
            timeout=timedelta(seconds=5),
            critical=True,
            description="Monitor agent failover system"
        )
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self._running:
            logger.warning("Health monitoring already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable,
        interval: timedelta,
        timeout: timedelta,
        critical: bool = True,
        description: str = "",
        tags: Optional[List[str]] = None
    ):
        """Register a health check"""
        with self._lock:
            health_check = HealthCheck(
                name=name,
                check_function=check_function,
                interval=interval,
                timeout=timeout,
                critical=critical,
                description=description,
                tags=tags or []
            )
            
            self.health_checks[name] = health_check
            logger.info(f"Registered health check: {name}")
    
    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                logger.info(f"Unregistered health check: {name}")
    
    async def _monitoring_loop(self):
        """Main health monitoring loop"""
        while self._running:
            try:
                await self._run_health_checks()
                await self._update_component_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _run_health_checks(self):
        """Run all scheduled health checks"""
        current_time = datetime.utcnow()
        
        with self._lock:
            checks_to_run = [
                check for check in self.health_checks.values()
                if (not check.last_run or 
                    current_time - check.last_run >= check.interval)
            ]
        
        # Run checks concurrently
        if checks_to_run:
            await asyncio.gather(
                *[self._run_single_check(check) for check in checks_to_run],
                return_exceptions=True
            )
    
    async def _run_single_check(self, check: HealthCheck):
        """Run a single health check"""
        start_time = time.time()
        
        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout.total_seconds()
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, check.check_function
                )
            
            # Update check state
            with self._lock:
                check.last_run = datetime.utcnow()
                check.last_result = bool(result)
                check.last_error = None
                
                if check.last_result:
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1
            
            duration = time.time() - start_time
            logger.debug(f"Health check {check.name} completed", 
                        result=result, duration=duration)
            
        except asyncio.TimeoutError:
            with self._lock:
                check.last_run = datetime.utcnow()
                check.last_result = False
                check.last_error = "Timeout"
                check.consecutive_failures += 1
            
            logger.warning(f"Health check {check.name} timed out")
            
        except Exception as e:
            with self._lock:
                check.last_run = datetime.utcnow()
                check.last_result = False
                check.last_error = str(e)
                check.consecutive_failures += 1
            
            logger.error(f"Health check {check.name} failed: {e}")
    
    async def _update_component_health(self):
        """Update overall component health status"""
        with self._lock:
            for name, check in self.health_checks.items():
                if check.last_run:
                    # Determine component status
                    if check.last_result:
                        status = ComponentStatus.UP
                        message = "OK"
                    elif check.consecutive_failures >= 3:
                        status = ComponentStatus.DOWN
                        message = f"Failed: {check.last_error or 'Unknown error'}"
                    else:
                        status = ComponentStatus.DEGRADED
                        message = f"Intermittent failures: {check.consecutive_failures}"
                    
                    self.component_health[name] = ComponentHealth(
                        name=name,
                        status=status,
                        message=message,
                        last_checked=check.last_run
                    )
    
    # Built-in health check implementations
    
    async def _check_cpu_usage(self) -> bool:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            # Consider healthy if CPU < 90%
            return cpu_percent < 90.0
        except Exception:
            return False
    
    async def _check_memory_usage(self) -> bool:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            # Consider healthy if memory usage < 90%
            return memory.percent < 90.0
        except Exception:
            return False
    
    async def _check_disk_usage(self) -> bool:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            # Consider healthy if disk usage < 85%
            return disk_percent < 85.0
        except Exception:
            return False
    
    async def _check_circuit_breakers(self) -> bool:
        """Check circuit breaker health"""
        try:
            metrics = circuit_breaker_manager.get_all_metrics()
            
            # Check if too many circuit breakers are open
            open_breakers = [
                name for name, metric in metrics.items()
                if metric.get('state') == 'open'
            ]
            
            # Consider unhealthy if more than 50% of breakers are open
            total_breakers = len(metrics)
            if total_breakers == 0:
                return True  # No breakers configured
            
            open_ratio = len(open_breakers) / total_breakers
            return open_ratio < 0.5
            
        except Exception:
            return False
    
    async def _check_agent_failover(self) -> bool:
        """Check agent failover system health"""
        try:
            # Check if failover manager is responsive
            agent_stats = failover_manager.get_all_agents_status()
            
            # Check if we have healthy agents
            total_agents = len(agent_stats)
            if total_agents == 0:
                return True  # No agents registered yet
            
            healthy_agents = sum(
                1 for status in agent_stats.values()
                if status and status.get('health') == 'healthy'
            )
            
            # Consider healthy if at least 25% of agents are healthy
            healthy_ratio = healthy_agents / total_agents
            return healthy_ratio >= 0.25
            
        except Exception:
            return False
    
    async def check_external_dependency(
        self,
        name: str,
        url: str,
        timeout: float = 10.0,
        expected_status: int = 200
    ) -> ComponentHealth:
        """Check health of external dependency"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == expected_status:
                        return ComponentHealth(
                            name=name,
                            status=ComponentStatus.UP,
                            message=f"OK ({response.status})",
                            response_time=response_time,
                            metadata={"url": url, "status_code": response.status}
                        )
                    else:
                        return ComponentHealth(
                            name=name,
                            status=ComponentStatus.DEGRADED,
                            message=f"Unexpected status: {response.status}",
                            response_time=response_time,
                            metadata={"url": url, "status_code": response.status}
                        )
                        
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.DOWN,
                message="Timeout",
                response_time=time.time() - start_time,
                metadata={"url": url, "error": "timeout"}
            )
            
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.DOWN,
                message=str(e),
                response_time=time.time() - start_time,
                metadata={"url": url, "error": str(e)}
            )
    
    async def get_health_status(self) -> SystemHealth:
        """Get overall system health status"""
        current_time = datetime.utcnow()
        uptime = (current_time - self.start_time).total_seconds()
        
        with self._lock:
            # Determine overall health status
            critical_failures = sum(
                1 for name, check in self.health_checks.items()
                if check.critical and check.last_result is False
            )
            
            non_critical_failures = sum(
                1 for name, check in self.health_checks.items()
                if not check.critical and check.last_result is False
            )
            
            if critical_failures > 0:
                if critical_failures >= 2:
                    overall_status = HealthStatus.CRITICAL
                else:
                    overall_status = HealthStatus.UNHEALTHY
            elif non_critical_failures > 0:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Get active alerts
            active_alerts = []
            if hasattr(performance_monitor, 'get_active_alerts'):
                alerts = performance_monitor.get_active_alerts()
                active_alerts = [
                    {
                        "rule_name": alert.rule_name,
                        "agent_id": alert.agent_id,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "triggered_at": alert.triggered_at.isoformat()
                    }
                    for alert in alerts
                ]
            
            # Get system metrics
            system_metrics = {}
            if hasattr(performance_monitor, 'get_system_metrics'):
                system_metrics = performance_monitor.get_system_metrics()
            
            return SystemHealth(
                status=overall_status,
                timestamp=current_time,
                uptime_seconds=uptime,
                components=self.component_health.copy(),
                alerts=active_alerts,
                metrics=system_metrics
            )
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status including all components"""
        health_status = await self.get_health_status()
        
        with self._lock:
            health_checks_status = {}
            for name, check in self.health_checks.items():
                health_checks_status[name] = {
                    "description": check.description,
                    "critical": check.critical,
                    "interval_seconds": check.interval.total_seconds(),
                    "last_run": check.last_run.isoformat() if check.last_run else None,
                    "last_result": check.last_result,
                    "last_error": check.last_error,
                    "consecutive_failures": check.consecutive_failures,
                    "tags": check.tags
                }
        
        # Get performance statistics
        agent_stats = {}
        if hasattr(performance_monitor, 'get_all_agent_stats'):
            stats = performance_monitor.get_all_agent_stats()
            agent_stats = {
                agent_id: {
                    "total_requests": stat.total_requests,
                    "successful_requests": stat.successful_requests,
                    "failed_requests": stat.failed_requests,
                    "error_rate": stat.error_rate,
                    "avg_response_time": stat.avg_response_time,
                    "throughput_per_second": stat.throughput_per_second
                }
                for agent_id, stat in stats.items()
            }
        
        # Get circuit breaker status
        circuit_breaker_status = circuit_breaker_manager.get_all_metrics()
        
        # Get failover status
        failover_status = failover_manager.get_all_agents_status()
        
        return {
            "status": health_status.status.value,
            "timestamp": health_status.timestamp.isoformat(),
            "uptime_seconds": health_status.uptime_seconds,
            "components": {
                name: {
                    "status": component.status.value,
                    "message": component.message,
                    "last_checked": component.last_checked.isoformat(),
                    "response_time": component.response_time,
                    "metadata": component.metadata
                }
                for name, component in health_status.components.items()
            },
            "health_checks": health_checks_status,
            "alerts": health_status.alerts,
            "metrics": health_status.metrics,
            "agent_performance": agent_stats,
            "circuit_breakers": circuit_breaker_status,
            "failover_agents": failover_status
        }
    
    def get_readiness_status(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Get readiness status for Kubernetes readiness probe.
        
        Returns:
            Tuple of (ready, status_info)
        """
        with self._lock:
            # Check critical health checks
            critical_checks = [
                check for check in self.health_checks.values()
                if check.critical
            ]
            
            failed_critical = [
                check.name for check in critical_checks
                if check.last_result is False
            ]
            
            ready = len(failed_critical) == 0
            
            status_info = {
                "ready": ready,
                "timestamp": datetime.utcnow().isoformat(),
                "critical_checks_total": len(critical_checks),
                "critical_checks_failed": len(failed_critical),
                "failed_checks": failed_critical
            }
            
            return ready, status_info
    
    def get_liveness_status(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Get liveness status for Kubernetes liveness probe.
        
        Returns:
            Tuple of (alive, status_info)
        """
        # Simple liveness check - if monitoring is running, we're alive
        alive = self._running
        
        status_info = {
            "alive": alive,
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": self._running,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }
        
        return alive, status_info


# Global health monitor instance
health_monitor = HealthMonitor()