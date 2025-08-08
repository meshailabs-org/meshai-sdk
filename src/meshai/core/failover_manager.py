"""
Automatic Failover Manager for MeshAI Agents

This module provides intelligent failover capabilities for maintaining
high availability and reliability in distributed agent systems.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import threading
from collections import defaultdict, deque

import structlog
from .circuit_breaker import CircuitBreaker, CircuitState, circuit_breaker_manager
from .performance_monitor import performance_monitor, AlertSeverity

logger = structlog.get_logger(__name__)


class FailoverStrategy(str, Enum):
    """Failover strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"  
    FASTEST_RESPONSE = "fastest_response"
    RANDOM = "random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"


class AgentHealth(str, Enum):
    """Agent health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class AgentEndpoint:
    """Represents an agent endpoint for failover"""
    agent_id: str
    endpoint_url: str
    capabilities: List[str]
    health: AgentHealth = AgentHealth.HEALTHY
    weight: float = 1.0
    priority: int = 1
    current_connections: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    failure_count: int = 0
    recovery_count: int = 0
    circuit_breaker: Optional[CircuitBreaker] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverRule:
    """Failover rule configuration"""
    name: str
    capability_filter: List[str]  # Apply to agents with these capabilities
    max_failures: int = 3
    failure_window: timedelta = timedelta(minutes=5)
    health_check_interval: timedelta = timedelta(seconds=30)
    recovery_threshold: int = 3  # Successful calls to mark as recovered
    strategy: FailoverStrategy = FailoverStrategy.HEALTH_BASED
    enabled: bool = True


@dataclass
class FailoverEvent:
    """Failover event record"""
    timestamp: datetime
    original_agent: str
    failover_agent: str
    reason: str
    capability: str
    success: bool
    response_time: Optional[float] = None


class FailoverManager:
    """
    Intelligent failover manager for MeshAI agents.
    
    Features:
    - Multiple failover strategies
    - Health monitoring and automatic recovery
    - Circuit breaker integration
    - Performance-based routing
    - Failure pattern analysis
    """
    
    def __init__(self, health_check_interval: float = 30.0):
        self.health_check_interval = health_check_interval
        
        # Agent registry
        self.agents: Dict[str, AgentEndpoint] = {}
        self.capability_map: Dict[str, List[str]] = defaultdict(list)
        
        # Failover state
        self.failover_rules: Dict[str, FailoverRule] = {}
        self.failover_events: deque = deque(maxlen=10000)
        self.active_connections: Dict[str, int] = defaultdict(int)
        
        # Round-robin state
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks
        self.failover_callbacks: List[Callable] = []
        self.health_change_callbacks: List[Callable] = []
        
        logger.info("Failover manager initialized", 
                   health_check_interval=health_check_interval)
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self._running:
            logger.warning("Failover monitoring already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Failover monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Failover monitoring stopped")
    
    def register_agent(
        self,
        agent_id: str,
        endpoint_url: str,
        capabilities: List[str],
        weight: float = 1.0,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register an agent for failover management"""
        with self._lock:
            # Create circuit breaker for this agent
            circuit_breaker = circuit_breaker_manager.get_or_create(
                f"agent_{agent_id}",
                on_state_change=self._on_circuit_breaker_state_change
            )
            
            agent = AgentEndpoint(
                agent_id=agent_id,
                endpoint_url=endpoint_url,
                capabilities=capabilities,
                weight=weight,
                priority=priority,
                circuit_breaker=circuit_breaker,
                metadata=metadata or {}
            )
            
            self.agents[agent_id] = agent
            
            # Update capability mapping
            for capability in capabilities:
                if agent_id not in self.capability_map[capability]:
                    self.capability_map[capability].append(agent_id)
            
            logger.info(f"Registered agent for failover: {agent_id}", 
                       capabilities=capabilities, weight=weight, priority=priority)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Remove from capability mapping
                for capability in agent.capabilities:
                    if agent_id in self.capability_map[capability]:
                        self.capability_map[capability].remove(agent_id)
                
                del self.agents[agent_id]
                logger.info(f"Unregistered agent: {agent_id}")
    
    def add_failover_rule(self, rule: FailoverRule):
        """Add a failover rule"""
        with self._lock:
            self.failover_rules[rule.name] = rule
            logger.info(f"Added failover rule: {rule.name}")
    
    async def get_healthy_agent(
        self,
        capability: str,
        exclude_agents: Optional[Set[str]] = None,
        strategy: Optional[FailoverStrategy] = None
    ) -> Optional[AgentEndpoint]:
        """
        Get a healthy agent for the specified capability.
        
        Args:
            capability: Required capability
            exclude_agents: Agents to exclude from selection
            strategy: Failover strategy to use
            
        Returns:
            Selected agent endpoint or None if no healthy agents
        """
        with self._lock:
            # Get agents with the required capability
            candidate_agents = [
                self.agents[agent_id]
                for agent_id in self.capability_map.get(capability, [])
                if agent_id in self.agents and 
                   (not exclude_agents or agent_id not in exclude_agents)
            ]
            
            if not candidate_agents:
                logger.warning(f"No agents available for capability: {capability}")
                return None
            
            # Filter healthy agents
            healthy_agents = [
                agent for agent in candidate_agents
                if self._is_agent_available(agent)
            ]
            
            if not healthy_agents:
                logger.warning(f"No healthy agents available for capability: {capability}")
                # Try degraded agents as fallback
                degraded_agents = [
                    agent for agent in candidate_agents
                    if agent.health == AgentHealth.DEGRADED and 
                       agent.circuit_breaker and not agent.circuit_breaker.is_open
                ]
                if degraded_agents:
                    healthy_agents = degraded_agents
                    logger.info(f"Using degraded agents as fallback for: {capability}")
                else:
                    return None
            
            # Apply selection strategy
            selected_strategy = strategy or self._get_strategy_for_capability(capability)
            selected_agent = self._select_agent_by_strategy(healthy_agents, selected_strategy, capability)
            
            if selected_agent:
                # Update connection count
                self.active_connections[selected_agent.agent_id] += 1
                selected_agent.current_connections += 1
                
                logger.debug(f"Selected agent {selected_agent.agent_id} for {capability} using {selected_strategy.value}")
            
            return selected_agent
    
    def _is_agent_available(self, agent: AgentEndpoint) -> bool:
        """Check if agent is available for requests"""
        # Check health status
        if agent.health == AgentHealth.OFFLINE or agent.health == AgentHealth.UNHEALTHY:
            return False
        
        # Check circuit breaker
        if agent.circuit_breaker and agent.circuit_breaker.is_open:
            return False
        
        return True
    
    def _get_strategy_for_capability(self, capability: str) -> FailoverStrategy:
        """Get failover strategy for capability"""
        # Check if there's a specific rule for this capability
        for rule in self.failover_rules.values():
            if capability in rule.capability_filter and rule.enabled:
                return rule.strategy
        
        # Default strategy
        return FailoverStrategy.HEALTH_BASED
    
    def _select_agent_by_strategy(
        self,
        agents: List[AgentEndpoint],
        strategy: FailoverStrategy,
        capability: str
    ) -> Optional[AgentEndpoint]:
        """Select agent based on strategy"""
        if not agents:
            return None
        
        if strategy == FailoverStrategy.ROUND_ROBIN:
            counter = self._round_robin_counters[capability]
            selected = agents[counter % len(agents)]
            self._round_robin_counters[capability] = counter + 1
            return selected
        
        elif strategy == FailoverStrategy.RANDOM:
            return random.choice(agents)
        
        elif strategy == FailoverStrategy.LEAST_CONNECTIONS:
            return min(agents, key=lambda a: a.current_connections)
        
        elif strategy == FailoverStrategy.FASTEST_RESPONSE:
            return min(agents, key=lambda a: a.avg_response_time or float('inf'))
        
        elif strategy == FailoverStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted selection based on agent weights
            total_weight = sum(a.weight for a in agents)
            if total_weight == 0:
                return agents[0]
            
            rand_val = random.uniform(0, total_weight)
            current_weight = 0
            for agent in agents:
                current_weight += agent.weight
                if rand_val <= current_weight:
                    return agent
            return agents[-1]
        
        elif strategy == FailoverStrategy.HEALTH_BASED:
            # Sort by health, then by performance
            def health_score(agent: AgentEndpoint) -> Tuple[int, float, float]:
                health_priority = {
                    AgentHealth.HEALTHY: 0,
                    AgentHealth.DEGRADED: 1,
                    AgentHealth.UNHEALTHY: 2,
                    AgentHealth.OFFLINE: 3
                }
                return (
                    health_priority.get(agent.health, 3),
                    agent.avg_response_time or 0,
                    1.0 - agent.success_rate  # Lower is better
                )
            
            return min(agents, key=health_score)
        
        return agents[0]  # Fallback
    
    async def record_request_result(
        self,
        agent_id: str,
        success: bool,
        response_time: float,
        capability: str
    ):
        """Record the result of a request to an agent"""
        with self._lock:
            if agent_id not in self.agents:
                return
            
            agent = self.agents[agent_id]
            
            # Update connection count
            if agent.current_connections > 0:
                agent.current_connections -= 1
            if self.active_connections[agent_id] > 0:
                self.active_connections[agent_id] -= 1
            
            # Update performance metrics
            if success:
                agent.recovery_count += 1
                agent.failure_count = max(0, agent.failure_count - 1)  # Gradual recovery
            else:
                agent.failure_count += 1
                agent.recovery_count = 0
            
            # Update response time (exponential moving average)
            if agent.avg_response_time == 0:
                agent.avg_response_time = response_time
            else:
                alpha = 0.1  # Smoothing factor
                agent.avg_response_time = (
                    alpha * response_time + (1 - alpha) * agent.avg_response_time
                )
            
            # Update success rate (sliding window)
            total_requests = agent.failure_count + agent.recovery_count
            if total_requests > 0:
                agent.success_rate = agent.recovery_count / total_requests
            
            # Check if health status should change
            await self._update_agent_health(agent)
    
    async def _update_agent_health(self, agent: AgentEndpoint):
        """Update agent health based on recent performance"""
        old_health = agent.health
        
        # Determine new health status
        if agent.circuit_breaker and agent.circuit_breaker.is_open:
            new_health = AgentHealth.OFFLINE
        elif agent.failure_count >= 5:
            new_health = AgentHealth.UNHEALTHY
        elif agent.failure_count >= 2 or agent.success_rate < 0.8:
            new_health = AgentHealth.DEGRADED
        elif agent.recovery_count >= 3 and agent.success_rate >= 0.95:
            new_health = AgentHealth.HEALTHY
        else:
            new_health = agent.health  # No change
        
        if new_health != old_health:
            agent.health = new_health
            
            logger.info(f"Agent {agent.agent_id} health changed: {old_health.value} -> {new_health.value}",
                       failure_count=agent.failure_count,
                       success_rate=agent.success_rate)
            
            # Notify callbacks
            for callback in self.health_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(agent.agent_id, old_health, new_health)
                    else:
                        callback(agent.agent_id, old_health, new_health)
                except Exception as e:
                    logger.error(f"Error in health change callback: {e}")
    
    async def handle_failover(
        self,
        failed_agent_id: str,
        capability: str,
        original_error: Exception
    ) -> Optional[AgentEndpoint]:
        """
        Handle failover when an agent fails.
        
        Args:
            failed_agent_id: ID of the failed agent
            capability: Capability that failed
            original_error: Original error that triggered failover
            
        Returns:
            Alternative agent to use or None
        """
        start_time = time.time()
        
        # Record failure
        if failed_agent_id in self.agents:
            await self.record_request_result(
                failed_agent_id, False, 0.0, capability
            )
        
        # Find alternative agent
        alternative_agent = await self.get_healthy_agent(
            capability, 
            exclude_agents={failed_agent_id}
        )
        
        # Record failover event
        event = FailoverEvent(
            timestamp=datetime.utcnow(),
            original_agent=failed_agent_id,
            failover_agent=alternative_agent.agent_id if alternative_agent else "none",
            reason=str(original_error),
            capability=capability,
            success=alternative_agent is not None,
            response_time=time.time() - start_time
        )
        
        self.failover_events.append(event)
        
        if alternative_agent:
            logger.info(f"Failover successful: {failed_agent_id} -> {alternative_agent.agent_id}",
                       capability=capability, reason=str(original_error))
            
            # Notify callbacks
            for callback in self.failover_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in failover callback: {e}")
        else:
            logger.error(f"Failover failed - no healthy agents available",
                        failed_agent=failed_agent_id, capability=capability)
        
        return alternative_agent
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while self._running:
            try:
                await self._perform_health_checks()
                await self._cleanup_old_events()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _perform_health_checks(self):
        """Perform health checks on all agents"""
        with self._lock:
            agents_to_check = list(self.agents.values())
        
        for agent in agents_to_check:
            try:
                await self._health_check_agent(agent)
            except Exception as e:
                logger.error(f"Health check failed for {agent.agent_id}: {e}")
    
    async def _health_check_agent(self, agent: AgentEndpoint):
        """Perform health check on a specific agent"""
        try:
            # Simple health check - could be expanded to actual HTTP health check
            current_time = datetime.utcnow()
            time_since_check = current_time - agent.last_health_check
            
            # If agent hasn't been used recently and is marked unhealthy, try to recover
            if (agent.health in [AgentHealth.UNHEALTHY, AgentHealth.DEGRADED] and
                time_since_check > timedelta(minutes=5)):
                
                # Gradual recovery - reduce failure count
                if agent.failure_count > 0:
                    agent.failure_count = max(0, agent.failure_count - 1)
                    await self._update_agent_health(agent)
            
            agent.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"Health check error for {agent.agent_id}: {e}")
    
    async def _cleanup_old_events(self):
        """Clean up old failover events"""
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        with self._lock:
            # Remove old events
            while (self.failover_events and 
                   self.failover_events[0].timestamp < cutoff_time):
                self.failover_events.popleft()
    
    def _on_circuit_breaker_state_change(self, name: str, old_state, new_state):
        """Handle circuit breaker state changes"""
        agent_id = name.replace("agent_", "")
        
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            if new_state == CircuitState.OPEN:
                agent.health = AgentHealth.OFFLINE
                logger.warning(f"Agent {agent_id} marked offline due to circuit breaker")
            elif new_state == CircuitState.CLOSED:
                # Agent recovered - will be re-evaluated in next health check
                logger.info(f"Agent {agent_id} circuit breaker closed - evaluating health")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an agent"""
        with self._lock:
            if agent_id not in self.agents:
                return None
            
            agent = self.agents[agent_id]
            return {
                "agent_id": agent.agent_id,
                "health": agent.health.value,
                "current_connections": agent.current_connections,
                "avg_response_time": agent.avg_response_time,
                "success_rate": agent.success_rate,
                "failure_count": agent.failure_count,
                "recovery_count": agent.recovery_count,
                "circuit_breaker_state": agent.circuit_breaker.state.value if agent.circuit_breaker else "unknown",
                "last_health_check": agent.last_health_check.isoformat()
            }
    
    def get_all_agents_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        with self._lock:
            return {
                agent_id: self.get_agent_status(agent_id)
                for agent_id in self.agents.keys()
            }
    
    def get_failover_events(self, limit: int = 100) -> List[FailoverEvent]:
        """Get recent failover events"""
        with self._lock:
            return list(self.failover_events)[-limit:]
    
    def add_failover_callback(self, callback: Callable):
        """Add callback for failover events"""
        self.failover_callbacks.append(callback)
    
    def add_health_change_callback(self, callback: Callable):
        """Add callback for health changes"""
        self.health_change_callbacks.append(callback)


# Global failover manager instance
failover_manager = FailoverManager()