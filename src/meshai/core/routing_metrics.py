"""
Routing Metrics and Analytics for MeshAI

This module provides comprehensive metrics collection and analytics
for routing decisions and performance optimization.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import threading

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary

from .routing_engine import RoutingDecision, RoutingStrategy, RoutingContext
from .performance_monitor import performance_monitor

logger = structlog.get_logger(__name__)


class MetricPeriod(str, Enum):
    """Time periods for metric aggregation"""
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"


@dataclass
class RoutingMetricSnapshot:
    """Snapshot of routing metrics at a point in time"""
    timestamp: datetime
    period: MetricPeriod
    total_requests: int
    successful_routes: int
    failed_routes: int
    avg_routing_time: float
    avg_confidence_score: float
    strategy_distribution: Dict[str, int]
    agent_distribution: Dict[str, int]
    error_rate: float
    p95_routing_time: float
    p99_routing_time: float


@dataclass
class AgentRoutingStats:
    """Routing statistics for a specific agent"""
    agent_id: str
    total_routed: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    sticky_sessions_active: int = 0
    load_balance_score: float = 1.0
    last_selected: Optional[datetime] = None
    selection_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class StrategyPerformance:
    """Performance metrics for a routing strategy"""
    strategy: RoutingStrategy
    total_uses: int = 0
    successful_routes: int = 0
    avg_response_time: float = 0.0
    avg_confidence: float = 0.0
    error_rate: float = 0.0
    p95_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class RoutingTrend:
    """Trend analysis for routing metrics"""
    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str  # "up", "down", "stable"
    is_anomaly: bool = False
    confidence: float = 0.0


class RoutingMetricsCollector:
    """
    Comprehensive routing metrics collection and analytics.
    
    Features:
    - Real-time routing metrics collection
    - Strategy performance tracking
    - Agent utilization analytics
    - Trend analysis and anomaly detection
    - Prometheus metrics export
    - Historical data aggregation
    """
    
    def __init__(self, retention_period: timedelta = timedelta(days=7)):
        self.retention_period = retention_period
        
        # Metrics storage
        self.routing_decisions: deque = deque(maxlen=100000)
        self.agent_stats: Dict[str, AgentRoutingStats] = {}
        self.strategy_performance: Dict[RoutingStrategy, StrategyPerformance] = {}
        
        # Time-series metrics
        self.metrics_timeseries: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_snapshots: Dict[MetricPeriod, deque] = {
            period: deque(maxlen=1000) for period in MetricPeriod
        }
        
        # Aggregated metrics
        self.hourly_aggregates: deque = deque(maxlen=168)  # 7 days of hourly data
        self.daily_aggregates: deque = deque(maxlen=30)  # 30 days of daily data
        
        # Thread safety
        self._lock = threading.RLock()
        self._aggregation_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        logger.info("Routing metrics collector initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for routing"""
        self.routing_requests = Counter(
            'meshai_routing_requests_total',
            'Total routing requests',
            ['strategy', 'capability', 'status']
        )
        
        self.routing_duration = Histogram(
            'meshai_routing_duration_seconds',
            'Routing decision duration',
            ['strategy']
        )
        
        self.routing_confidence = Gauge(
            'meshai_routing_confidence',
            'Average routing confidence score',
            ['strategy']
        )
        
        self.agent_selection_count = Counter(
            'meshai_agent_selections_total',
            'Agent selection count',
            ['agent_id', 'strategy']
        )
        
        self.sticky_sessions_active = Gauge(
            'meshai_sticky_sessions_active',
            'Active sticky sessions'
        )
        
        self.routing_error_rate = Gauge(
            'meshai_routing_error_rate',
            'Routing error rate'
        )
    
    async def start(self):
        """Start metrics collection"""
        if self._running:
            logger.warning("Metrics collector already running")
            return
        
        self._running = True
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        logger.info("Routing metrics collection started")
    
    async def stop(self):
        """Stop metrics collection"""
        self._running = False
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("Routing metrics collection stopped")
    
    def record_routing_decision(
        self,
        context: RoutingContext,
        decision: RoutingDecision,
        duration: float,
        success: bool = True,
        response_time: Optional[float] = None
    ):
        """Record a routing decision and its outcome"""
        with self._lock:
            # Store decision
            self.routing_decisions.append({
                'timestamp': datetime.utcnow(),
                'context': context,
                'decision': decision,
                'duration': duration,
                'success': success,
                'response_time': response_time
            })
            
            # Update agent stats
            agent_id = decision.selected_agent.agent_id
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = AgentRoutingStats(agent_id=agent_id)
            
            stats = self.agent_stats[agent_id]
            stats.total_routed += 1
            if success:
                stats.successful_requests += 1
            else:
                stats.failed_requests += 1
            
            if response_time:
                stats.total_response_time += response_time
                stats.avg_response_time = stats.total_response_time / stats.total_routed
            
            stats.last_selected = datetime.utcnow()
            stats.selection_reasons[decision.routing_strategy.value] += 1
            
            # Update strategy performance
            strategy = decision.routing_strategy
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = StrategyPerformance(strategy=strategy)
            
            perf = self.strategy_performance[strategy]
            perf.total_uses += 1
            if success:
                perf.successful_routes += 1
            
            if response_time:
                perf.response_times.append(response_time)
                perf.avg_response_time = statistics.mean(perf.response_times)
                if len(perf.response_times) >= 10:
                    sorted_times = sorted(perf.response_times)
                    perf.p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
            
            perf.avg_confidence = (
                perf.avg_confidence * (perf.total_uses - 1) + decision.confidence_score
            ) / perf.total_uses
            
            perf.error_rate = 1.0 - (perf.successful_routes / perf.total_uses)
            
            # Update Prometheus metrics
            self.routing_requests.labels(
                strategy=strategy.value,
                capability=context.capability,
                status='success' if success else 'failure'
            ).inc()
            
            self.routing_duration.labels(strategy=strategy.value).observe(duration)
            self.routing_confidence.labels(strategy=strategy.value).set(decision.confidence_score)
            self.agent_selection_count.labels(agent_id=agent_id, strategy=strategy.value).inc()
            
            if not success:
                self.routing_error_rate.set(self._calculate_error_rate())
            
            # Store time-series data
            self.metrics_timeseries['routing_duration'].append({
                'timestamp': datetime.utcnow(),
                'value': duration
            })
            
            self.metrics_timeseries['confidence_score'].append({
                'timestamp': datetime.utcnow(),
                'value': decision.confidence_score
            })
    
    def update_sticky_sessions_count(self, count: int):
        """Update active sticky sessions count"""
        self.sticky_sessions_active.set(count)
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall routing error rate"""
        if not self.routing_decisions:
            return 0.0
        
        recent_decisions = list(self.routing_decisions)[-1000:]  # Last 1000 decisions
        failures = sum(1 for d in recent_decisions if not d['success'])
        return failures / len(recent_decisions)
    
    async def _aggregation_loop(self):
        """Periodic metrics aggregation"""
        while self._running:
            try:
                await self._aggregate_metrics()
                await self._cleanup_old_data()
                await asyncio.sleep(60)  # Aggregate every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
                await asyncio.sleep(10)
    
    async def _aggregate_metrics(self):
        """Aggregate metrics for different time periods"""
        current_time = datetime.utcnow()
        
        with self._lock:
            # Minute aggregation
            minute_snapshot = self._create_snapshot(MetricPeriod.MINUTE, timedelta(minutes=1))
            if minute_snapshot:
                self.metric_snapshots[MetricPeriod.MINUTE].append(minute_snapshot)
            
            # 5-minute aggregation
            if current_time.minute % 5 == 0:
                five_min_snapshot = self._create_snapshot(MetricPeriod.FIVE_MINUTES, timedelta(minutes=5))
                if five_min_snapshot:
                    self.metric_snapshots[MetricPeriod.FIVE_MINUTES].append(five_min_snapshot)
            
            # Hourly aggregation
            if current_time.minute == 0:
                hourly_snapshot = self._create_snapshot(MetricPeriod.HOUR, timedelta(hours=1))
                if hourly_snapshot:
                    self.metric_snapshots[MetricPeriod.HOUR].append(hourly_snapshot)
                    self.hourly_aggregates.append(hourly_snapshot)
            
            # Daily aggregation
            if current_time.hour == 0 and current_time.minute == 0:
                daily_snapshot = self._create_snapshot(MetricPeriod.DAY, timedelta(days=1))
                if daily_snapshot:
                    self.metric_snapshots[MetricPeriod.DAY].append(daily_snapshot)
                    self.daily_aggregates.append(daily_snapshot)
    
    def _create_snapshot(self, period: MetricPeriod, duration: timedelta) -> Optional[RoutingMetricSnapshot]:
        """Create a metric snapshot for a time period"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - duration
        
        # Filter decisions within period
        period_decisions = [
            d for d in self.routing_decisions
            if d['timestamp'] > cutoff_time
        ]
        
        if not period_decisions:
            return None
        
        # Calculate metrics
        total_requests = len(period_decisions)
        successful_routes = sum(1 for d in period_decisions if d['success'])
        failed_routes = total_requests - successful_routes
        
        routing_times = [d['duration'] for d in period_decisions]
        avg_routing_time = statistics.mean(routing_times)
        
        confidence_scores = [d['decision'].confidence_score for d in period_decisions]
        avg_confidence = statistics.mean(confidence_scores)
        
        # Strategy distribution
        strategy_counts = defaultdict(int)
        for d in period_decisions:
            strategy_counts[d['decision'].routing_strategy.value] += 1
        
        # Agent distribution
        agent_counts = defaultdict(int)
        for d in period_decisions:
            agent_counts[d['decision'].selected_agent.agent_id] += 1
        
        # Percentiles
        sorted_times = sorted(routing_times)
        p95_time = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        p99_time = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
        
        return RoutingMetricSnapshot(
            timestamp=current_time,
            period=period,
            total_requests=total_requests,
            successful_routes=successful_routes,
            failed_routes=failed_routes,
            avg_routing_time=avg_routing_time,
            avg_confidence_score=avg_confidence,
            strategy_distribution=dict(strategy_counts),
            agent_distribution=dict(agent_counts),
            error_rate=failed_routes / total_requests if total_requests > 0 else 0,
            p95_routing_time=p95_time,
            p99_routing_time=p99_time
        )
    
    async def _cleanup_old_data(self):
        """Clean up old metrics data"""
        cutoff_time = datetime.utcnow() - self.retention_period
        
        with self._lock:
            # Clean up old routing decisions
            while self.routing_decisions and self.routing_decisions[0]['timestamp'] < cutoff_time:
                self.routing_decisions.popleft()
            
            # Clean up old time-series data
            for series in self.metrics_timeseries.values():
                while series and series[0]['timestamp'] < cutoff_time:
                    series.popleft()
    
    def get_routing_analytics(self, period: MetricPeriod = MetricPeriod.HOUR) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""
        with self._lock:
            snapshots = list(self.metric_snapshots[period])
            
            if not snapshots:
                return {}
            
            latest_snapshot = snapshots[-1] if snapshots else None
            
            # Calculate trends
            trends = self._calculate_trends(snapshots)
            
            # Get top performing strategies
            top_strategies = sorted(
                self.strategy_performance.items(),
                key=lambda x: x[1].successful_routes / max(x[1].total_uses, 1),
                reverse=True
            )[:5]
            
            # Get agent utilization
            agent_utilization = {
                agent_id: {
                    'total_routed': stats.total_routed,
                    'success_rate': stats.successful_requests / max(stats.total_routed, 1),
                    'avg_response_time': stats.avg_response_time,
                    'last_selected': stats.last_selected.isoformat() if stats.last_selected else None
                }
                for agent_id, stats in self.agent_stats.items()
            }
            
            return {
                'period': period.value,
                'latest_snapshot': {
                    'timestamp': latest_snapshot.timestamp.isoformat(),
                    'total_requests': latest_snapshot.total_requests,
                    'success_rate': latest_snapshot.successful_routes / max(latest_snapshot.total_requests, 1),
                    'avg_routing_time': latest_snapshot.avg_routing_time,
                    'avg_confidence': latest_snapshot.avg_confidence_score,
                    'error_rate': latest_snapshot.error_rate
                } if latest_snapshot else None,
                'trends': trends,
                'top_strategies': [
                    {
                        'strategy': str(strategy),
                        'total_uses': perf.total_uses,
                        'success_rate': perf.successful_routes / max(perf.total_uses, 1),
                        'avg_response_time': perf.avg_response_time,
                        'avg_confidence': perf.avg_confidence
                    }
                    for strategy, perf in top_strategies
                ],
                'agent_utilization': agent_utilization,
                'total_agents': len(self.agent_stats),
                'active_strategies': len(self.strategy_performance)
            }
    
    def _calculate_trends(self, snapshots: List[RoutingMetricSnapshot]) -> List[RoutingTrend]:
        """Calculate trends from snapshots"""
        if len(snapshots) < 2:
            return []
        
        trends = []
        current = snapshots[-1]
        previous = snapshots[-2]
        
        # Request volume trend
        volume_change = ((current.total_requests - previous.total_requests) / 
                        max(previous.total_requests, 1)) * 100
        
        trends.append(RoutingTrend(
            metric_name='request_volume',
            current_value=current.total_requests,
            previous_value=previous.total_requests,
            change_percent=volume_change,
            trend_direction='up' if volume_change > 0 else 'down' if volume_change < 0 else 'stable',
            confidence=0.9
        ))
        
        # Error rate trend
        error_change = current.error_rate - previous.error_rate
        trends.append(RoutingTrend(
            metric_name='error_rate',
            current_value=current.error_rate,
            previous_value=previous.error_rate,
            change_percent=error_change * 100,
            trend_direction='up' if error_change > 0 else 'down' if error_change < 0 else 'stable',
            is_anomaly=abs(error_change) > 0.1,  # Flag if error rate changed by >10%
            confidence=0.85
        ))
        
        # Response time trend
        time_change = ((current.avg_routing_time - previous.avg_routing_time) / 
                      max(previous.avg_routing_time, 0.001)) * 100
        
        trends.append(RoutingTrend(
            metric_name='avg_routing_time',
            current_value=current.avg_routing_time,
            previous_value=previous.avg_routing_time,
            change_percent=time_change,
            trend_direction='up' if time_change > 0 else 'down' if time_change < 0 else 'stable',
            confidence=0.9
        ))
        
        return trends
    
    def get_agent_recommendation(self, capability: str) -> List[Tuple[str, float]]:
        """Get agent recommendations based on historical performance"""
        with self._lock:
            recommendations = []
            
            for agent_id, stats in self.agent_stats.items():
                if stats.total_routed < 10:  # Need minimum data
                    continue
                
                # Calculate recommendation score
                success_rate = stats.successful_requests / stats.total_routed
                
                # Penalize high response times
                response_score = 1.0 / (1.0 + stats.avg_response_time)
                
                # Consider recent activity
                if stats.last_selected:
                    recency = (datetime.utcnow() - stats.last_selected).total_seconds() / 3600
                    recency_score = 1.0 / (1.0 + recency)
                else:
                    recency_score = 0.5
                
                # Combined score
                total_score = (success_rate * 0.5 + response_score * 0.3 + recency_score * 0.2)
                
                recommendations.append((agent_id, total_score))
            
            # Sort by score descending
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:10]  # Top 10 recommendations
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external consumption"""
        with self._lock:
            return {
                'routing_decisions': len(self.routing_decisions),
                'agent_stats': {
                    agent_id: {
                        'total_routed': stats.total_routed,
                        'successful_requests': stats.successful_requests,
                        'failed_requests': stats.failed_requests,
                        'avg_response_time': stats.avg_response_time
                    }
                    for agent_id, stats in self.agent_stats.items()
                },
                'strategy_performance': {
                    strategy.value: {
                        'total_uses': perf.total_uses,
                        'successful_routes': perf.successful_routes,
                        'avg_response_time': perf.avg_response_time,
                        'avg_confidence': perf.avg_confidence,
                        'error_rate': perf.error_rate
                    }
                    for strategy, perf in self.strategy_performance.items()
                },
                'snapshots': {
                    period.value: len(snapshots)
                    for period, snapshots in self.metric_snapshots.items()
                }
            }


# Global routing metrics collector
routing_metrics = RoutingMetricsCollector()