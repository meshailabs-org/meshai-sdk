"""
Advanced Routing Engine for MeshAI Agents

This module provides sophisticated routing algorithms for intelligent
agent selection, load balancing, and performance optimization.
"""

import asyncio
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import statistics
import random
import math

import structlog
from .performance_monitor import performance_monitor
from .failover_manager import failover_manager, AgentEndpoint, AgentHealth
from .circuit_breaker import circuit_breaker_manager

logger = structlog.get_logger(__name__)


class RoutingStrategy(str, Enum):
    """Available routing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PERFORMANCE_BASED = "performance_based"
    STICKY_SESSION = "sticky_session"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    ML_OPTIMIZED = "ml_optimized"


class SessionAffinityType(str, Enum):
    """Types of session affinity"""
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    IP_ADDRESS = "ip_address"
    CUSTOM_KEY = "custom_key"
    CONTEXT_BASED = "context_based"


class LoadBalancingPolicy(str, Enum):
    """Load balancing policies"""
    SPREAD_LOAD = "spread_load"  # Distribute evenly
    POWER_OF_TWO = "power_of_two"  # Choose best of 2 random agents
    LEAST_LOADED = "least_loaded"  # Always choose least loaded
    WEIGHTED_RANDOM = "weighted_random"  # Random with weights
    PRIORITY_BASED = "priority_based"  # Priority tiers


@dataclass
class RoutingContext:
    """Context information for routing decisions"""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    capability: str = ""
    priority: int = 1
    timeout: float = 10.0
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    previous_agents: List[str] = field(default_factory=list)
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingDecision:
    """Result of a routing decision"""
    selected_agent: AgentEndpoint
    routing_strategy: RoutingStrategy
    decision_factors: Dict[str, Any]
    confidence_score: float
    alternative_agents: List[AgentEndpoint] = field(default_factory=list)
    routing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceProfile:
    """Performance profile for an agent"""
    agent_id: str
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    success_rate: float = 1.0
    current_load: int = 0
    capacity_utilization: float = 0.0
    resource_efficiency: float = 1.0
    recent_performance_trend: float = 1.0  # 1.0 = stable, >1.0 = improving, <1.0 = degrading
    last_updated: datetime = field(default_factory=datetime.utcnow)
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class StickySession:
    """Sticky session information"""
    session_key: str
    agent_id: str
    affinity_type: SessionAffinityType
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingPolicy:
    """Routing policy configuration"""
    name: str
    capability_patterns: List[str]  # Capabilities this policy applies to
    primary_strategy: RoutingStrategy
    fallback_strategies: List[RoutingStrategy] = field(default_factory=list)
    load_balancing_policy: LoadBalancingPolicy = LoadBalancingPolicy.SPREAD_LOAD
    session_affinity: Optional[SessionAffinityType] = None
    session_ttl: timedelta = timedelta(hours=1)
    weight_factors: Dict[str, float] = field(default_factory=dict)  # Factor weights
    constraints: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class ConsistentHashRing:
    """Consistent hash ring for agent selection"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.agents: Set[str] = set()
        self._lock = threading.RLock()
    
    def add_agent(self, agent_id: str):
        """Add agent to hash ring"""
        with self._lock:
            if agent_id in self.agents:
                return
            
            self.agents.add(agent_id)
            
            # Add virtual nodes for this agent
            for i in range(self.virtual_nodes):
                virtual_key = f"{agent_id}:{i}"
                hash_value = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                self.ring[hash_value] = agent_id
    
    def remove_agent(self, agent_id: str):
        """Remove agent from hash ring"""
        with self._lock:
            if agent_id not in self.agents:
                return
            
            self.agents.remove(agent_id)
            
            # Remove virtual nodes for this agent
            keys_to_remove = [k for k, v in self.ring.items() if v == agent_id]
            for key in keys_to_remove:
                del self.ring[key]
    
    def get_agent(self, key: str) -> Optional[str]:
        """Get agent for a given key"""
        with self._lock:
            if not self.ring:
                return None
            
            hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
            
            # Find the first agent clockwise from this hash
            sorted_keys = sorted(self.ring.keys())
            
            for ring_key in sorted_keys:
                if ring_key >= hash_value:
                    return self.ring[ring_key]
            
            # Wrap around to the first agent
            return self.ring[sorted_keys[0]]


class AdvancedRoutingEngine:
    """
    Advanced routing engine with sophisticated algorithms.
    
    Features:
    - Performance-based selection with ML optimization
    - Sticky sessions with multiple affinity types
    - Advanced load balancing algorithms
    - Consistent hashing for distributed routing
    - Adaptive weights based on real-time performance
    - Circuit breaker integration
    - Routing analytics and optimization
    """
    
    def __init__(self):
        # Agent performance tracking
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Sticky session management
        self.sticky_sessions: Dict[str, StickySession] = {}
        self.session_cleanup_interval = 300  # 5 minutes
        
        # Routing policies
        self.routing_policies: Dict[str, RoutingPolicy] = {}
        self.default_policy = RoutingPolicy(
            name="default",
            capability_patterns=["*"],
            primary_strategy=RoutingStrategy.PERFORMANCE_BASED
        )
        
        # Load balancing state
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.consistent_hash_rings: Dict[str, ConsistentHashRing] = {}
        
        # Routing metrics
        self.routing_decisions: deque = deque(maxlen=10000)
        self.strategy_performance: Dict[RoutingStrategy, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread safety
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks
        self.routing_callbacks: List[Callable] = []
        
        logger.info("Advanced routing engine initialized")
    
    async def start(self):
        """Start the routing engine"""
        if self._running:
            logger.warning("Routing engine already running")
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Initialize with existing agents from failover manager
        await self._sync_with_failover_manager()
        
        logger.info("Advanced routing engine started")
    
    async def stop(self):
        """Stop the routing engine"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced routing engine stopped")
    
    async def _sync_with_failover_manager(self):
        """Sync agent information with failover manager"""
        try:
            agents_status = failover_manager.get_all_agents_status()
            
            with self._lock:
                for agent_id, status in agents_status.items():
                    if agent_id not in self.agent_profiles:
                        self.agent_profiles[agent_id] = AgentPerformanceProfile(agent_id=agent_id)
                    
                    profile = self.agent_profiles[agent_id]
                    profile.current_load = status.get('current_connections', 0)
                    profile.avg_response_time = status.get('avg_response_time', 0)
                    
                    # Add to consistent hash rings
                    for capability in ["*"]:  # Would get actual capabilities from agent
                        if capability not in self.consistent_hash_rings:
                            self.consistent_hash_rings[capability] = ConsistentHashRing()
                        self.consistent_hash_rings[capability].add_agent(agent_id)
        
        except Exception as e:
            logger.error(f"Failed to sync with failover manager: {e}")
    
    def add_routing_policy(self, policy: RoutingPolicy):
        """Add a routing policy"""
        with self._lock:
            self.routing_policies[policy.name] = policy
            logger.info(f"Added routing policy: {policy.name}")
    
    def remove_routing_policy(self, policy_name: str):
        """Remove a routing policy"""
        with self._lock:
            if policy_name in self.routing_policies:
                del self.routing_policies[policy_name]
                logger.info(f"Removed routing policy: {policy_name}")
    
    async def route_request(self, context: RoutingContext) -> Optional[RoutingDecision]:
        """
        Route a request to the best available agent.
        
        Args:
            context: Routing context with request information
            
        Returns:
            Routing decision or None if no agents available
        """
        start_time = time.time()
        
        try:
            # Find applicable routing policy
            policy = self._find_applicable_policy(context.capability)
            
            # Get available agents
            available_agents = await self._get_available_agents(context.capability)
            if not available_agents:
                logger.warning(f"No available agents for capability: {context.capability}")
                return None
            
            # Update agent performance profiles
            await self._update_agent_profiles(available_agents)
            
            # Check for sticky session
            if policy.session_affinity:
                sticky_decision = await self._try_sticky_session(context, policy, available_agents)
                if sticky_decision:
                    await self._record_routing_decision(sticky_decision, time.time() - start_time)
                    return sticky_decision
            
            # Apply primary routing strategy
            decision = await self._apply_routing_strategy(
                context, policy.primary_strategy, available_agents, policy
            )
            
            if not decision:
                # Try fallback strategies
                for fallback_strategy in policy.fallback_strategies:
                    decision = await self._apply_routing_strategy(
                        context, fallback_strategy, available_agents, policy
                    )
                    if decision:
                        break
            
            if decision:
                # Update sticky session if needed
                if policy.session_affinity:
                    await self._update_sticky_session(context, decision, policy)
                
                # Record routing decision
                await self._record_routing_decision(decision, time.time() - start_time)
                
                # Notify callbacks
                for callback in self.routing_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(context, decision)
                        else:
                            callback(context, decision)
                    except Exception as e:
                        logger.error(f"Error in routing callback: {e}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in routing request: {e}")
            return None
    
    def _find_applicable_policy(self, capability: str) -> RoutingPolicy:
        """Find the applicable routing policy for a capability"""
        with self._lock:
            # Check for exact match first
            for policy in self.routing_policies.values():
                if capability in policy.capability_patterns:
                    return policy
            
            # Check for pattern matches
            for policy in self.routing_policies.values():
                for pattern in policy.capability_patterns:
                    if pattern == "*" or self._matches_pattern(capability, pattern):
                        return policy
            
            return self.default_policy
    
    def _matches_pattern(self, capability: str, pattern: str) -> bool:
        """Check if capability matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(capability, pattern)
    
    async def _get_available_agents(self, capability: str) -> List[AgentEndpoint]:
        """Get available agents for a capability"""
        try:
            # Get healthy agents from failover manager
            available = []
            
            # This is a simplified version - in real implementation,
            # we'd integrate more deeply with the failover manager
            agents_status = failover_manager.get_all_agents_status()
            
            for agent_id, status in agents_status.items():
                if status and status.get('health') in ['healthy', 'degraded']:
                    # Create mock endpoint - in real implementation, this would come from registry
                    endpoint = AgentEndpoint(
                        agent_id=agent_id,
                        endpoint_url=f"http://{agent_id}:8080",
                        capabilities=[capability],
                        health=AgentHealth(status.get('health', 'unknown')),
                        current_connections=status.get('current_connections', 0)
                    )
                    available.append(endpoint)
            
            return available
            
        except Exception as e:
            logger.error(f"Error getting available agents: {e}")
            return []
    
    async def _update_agent_profiles(self, agents: List[AgentEndpoint]):
        """Update agent performance profiles with latest data"""
        try:
            # Get performance stats from performance monitor
            all_stats = performance_monitor.get_all_agent_stats()
            
            with self._lock:
                for agent in agents:
                    if agent.agent_id not in self.agent_profiles:
                        self.agent_profiles[agent.agent_id] = AgentPerformanceProfile(agent_id=agent.agent_id)
                    
                    profile = self.agent_profiles[agent.agent_id]
                    
                    # Update from performance monitor
                    if agent.agent_id in all_stats:
                        stats = all_stats[agent.agent_id]
                        profile.avg_response_time = stats.avg_response_time
                        profile.success_rate = 1.0 - stats.error_rate
                        profile.current_load = agent.current_connections
                    
                    # Calculate performance trend
                    if len(profile.performance_history) >= 10:
                        recent_times = list(profile.performance_history)[-10:]
                        older_times = list(profile.performance_history)[-20:-10] if len(profile.performance_history) >= 20 else []
                        
                        if older_times:
                            recent_avg = statistics.mean(recent_times)
                            older_avg = statistics.mean(older_times)
                            if older_avg > 0:
                                profile.recent_performance_trend = older_avg / recent_avg  # Lower response time = better trend
                    
                    profile.last_updated = datetime.utcnow()
                    
        except Exception as e:
            logger.error(f"Error updating agent profiles: {e}")
    
    async def _try_sticky_session(
        self, 
        context: RoutingContext, 
        policy: RoutingPolicy,
        available_agents: List[AgentEndpoint]
    ) -> Optional[RoutingDecision]:
        """Try to route using sticky session"""
        if not policy.session_affinity:
            return None
        
        session_key = self._generate_session_key(context, policy.session_affinity)
        if not session_key:
            return None
        
        with self._lock:
            session = self.sticky_sessions.get(session_key)
            
            if session and not self._is_session_expired(session, policy.session_ttl):
                # Check if the sticky agent is still available
                sticky_agent = next(
                    (a for a in available_agents if a.agent_id == session.agent_id),
                    None
                )
                
                if sticky_agent:
                    # Update session access
                    session.last_accessed = datetime.utcnow()
                    session.access_count += 1
                    
                    return RoutingDecision(
                        selected_agent=sticky_agent,
                        routing_strategy=RoutingStrategy.STICKY_SESSION,
                        decision_factors={
                            "session_key": session_key,
                            "session_age": (datetime.utcnow() - session.created_at).total_seconds(),
                            "access_count": session.access_count
                        },
                        confidence_score=0.9  # High confidence for sticky sessions
                    )
                else:
                    # Agent no longer available, clean up session
                    del self.sticky_sessions[session_key]
        
        return None
    
    def _generate_session_key(self, context: RoutingContext, affinity_type: SessionAffinityType) -> Optional[str]:
        """Generate session key based on affinity type"""
        if affinity_type == SessionAffinityType.USER_ID and context.user_id:
            return f"user:{context.user_id}:{context.capability}"
        elif affinity_type == SessionAffinityType.SESSION_ID and context.session_id:
            return f"session:{context.session_id}:{context.capability}"
        elif affinity_type == SessionAffinityType.IP_ADDRESS and context.client_ip:
            return f"ip:{context.client_ip}:{context.capability}"
        elif affinity_type == SessionAffinityType.CUSTOM_KEY:
            custom_key = context.custom_attributes.get("session_key")
            if custom_key:
                return f"custom:{custom_key}:{context.capability}"
        elif affinity_type == SessionAffinityType.CONTEXT_BASED:
            # Create hash from context attributes
            context_str = f"{context.user_id}:{context.session_id}:{context.capability}"
            return f"context:{hashlib.md5(context_str.encode()).hexdigest()[:16]}"
        
        return None
    
    def _is_session_expired(self, session: StickySession, ttl: timedelta) -> bool:
        """Check if session is expired"""
        if session.ttl:
            return datetime.utcnow() > session.created_at + session.ttl
        return datetime.utcnow() > session.last_accessed + ttl
    
    async def _apply_routing_strategy(
        self,
        context: RoutingContext,
        strategy: RoutingStrategy,
        available_agents: List[AgentEndpoint],
        policy: RoutingPolicy
    ) -> Optional[RoutingDecision]:
        """Apply specific routing strategy"""
        if not available_agents:
            return None
        
        try:
            if strategy == RoutingStrategy.ROUND_ROBIN:
                return await self._round_robin_routing(context, available_agents)
            elif strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
                return await self._weighted_round_robin_routing(context, available_agents)
            elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
                return await self._least_connections_routing(context, available_agents)
            elif strategy == RoutingStrategy.LEAST_RESPONSE_TIME:
                return await self._least_response_time_routing(context, available_agents)
            elif strategy == RoutingStrategy.PERFORMANCE_BASED:
                return await self._performance_based_routing(context, available_agents, policy)
            elif strategy == RoutingStrategy.CONSISTENT_HASH:
                return await self._consistent_hash_routing(context, available_agents)
            elif strategy == RoutingStrategy.ADAPTIVE_WEIGHTED:
                return await self._adaptive_weighted_routing(context, available_agents)
            elif strategy == RoutingStrategy.RESOURCE_BASED:
                return await self._resource_based_routing(context, available_agents)
            else:
                # Default to round robin
                return await self._round_robin_routing(context, available_agents)
                
        except Exception as e:
            logger.error(f"Error applying routing strategy {strategy}: {e}")
            return None
    
    async def _round_robin_routing(
        self, 
        context: RoutingContext, 
        available_agents: List[AgentEndpoint]
    ) -> RoutingDecision:
        """Round robin agent selection"""
        with self._lock:
            counter_key = f"rr:{context.capability}"
            counter = self.round_robin_counters[counter_key]
            selected_agent = available_agents[counter % len(available_agents)]
            self.round_robin_counters[counter_key] = counter + 1
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.ROUND_ROBIN,
            decision_factors={
                "counter": counter,
                "total_agents": len(available_agents)
            },
            confidence_score=0.7,
            alternative_agents=available_agents[:3]  # Top 3 alternatives
        )
    
    async def _weighted_round_robin_routing(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> RoutingDecision:
        """Weighted round robin based on agent weights and performance"""
        # Calculate weights based on performance and capacity
        weighted_agents = []
        
        with self._lock:
            for agent in available_agents:
                profile = self.agent_profiles.get(agent.agent_id, AgentPerformanceProfile(agent.agent_id))
                
                # Base weight from agent configuration
                base_weight = getattr(agent, 'weight', 1.0)
                
                # Adjust weight based on performance
                performance_weight = profile.success_rate * profile.recent_performance_trend
                
                # Adjust weight based on current load
                load_factor = 1.0 / (1.0 + profile.current_load * 0.1)
                
                total_weight = base_weight * performance_weight * load_factor
                
                for _ in range(int(total_weight * 10)):  # Scale weights
                    weighted_agents.append(agent)
        
        if not weighted_agents:
            weighted_agents = available_agents
        
        selected_agent = random.choice(weighted_agents)
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.WEIGHTED_ROUND_ROBIN,
            decision_factors={
                "total_weighted_agents": len(weighted_agents),
                "base_agents": len(available_agents)
            },
            confidence_score=0.8
        )
    
    async def _least_connections_routing(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> RoutingDecision:
        """Select agent with least active connections"""
        selected_agent = min(available_agents, key=lambda a: a.current_connections)
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.LEAST_CONNECTIONS,
            decision_factors={
                "selected_connections": selected_agent.current_connections,
                "avg_connections": statistics.mean(a.current_connections for a in available_agents)
            },
            confidence_score=0.85
        )
    
    async def _least_response_time_routing(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> RoutingDecision:
        """Select agent with lowest average response time"""
        with self._lock:
            agent_times = []
            for agent in available_agents:
                profile = self.agent_profiles.get(agent.agent_id, AgentPerformanceProfile(agent.agent_id))
                response_time = profile.avg_response_time or float('inf')
                agent_times.append((agent, response_time))
            
            selected_agent, response_time = min(agent_times, key=lambda x: x[1])
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.LEAST_RESPONSE_TIME,
            decision_factors={
                "selected_response_time": response_time,
                "avg_response_time": statistics.mean(t for _, t in agent_times if t != float('inf'))
            },
            confidence_score=0.85
        )
    
    async def _performance_based_routing(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint],
        policy: RoutingPolicy
    ) -> RoutingDecision:
        """Advanced performance-based agent selection"""
        with self._lock:
            agent_scores = []
            
            # Get weight factors from policy
            weights = {
                'response_time': policy.weight_factors.get('response_time', 0.3),
                'success_rate': policy.weight_factors.get('success_rate', 0.3),
                'current_load': policy.weight_factors.get('current_load', 0.2),
                'trend': policy.weight_factors.get('trend', 0.2)
            }
            
            for agent in available_agents:
                profile = self.agent_profiles.get(agent.agent_id, AgentPerformanceProfile(agent.agent_id))
                
                # Normalize metrics to 0-1 scale
                response_time_score = self._normalize_response_time(profile.avg_response_time, available_agents)
                success_rate_score = profile.success_rate
                load_score = self._normalize_load(profile.current_load, available_agents)
                trend_score = min(profile.recent_performance_trend, 2.0) / 2.0
                
                # Calculate weighted score
                total_score = (
                    weights['response_time'] * response_time_score +
                    weights['success_rate'] * success_rate_score +
                    weights['current_load'] * load_score +
                    weights['trend'] * trend_score
                )
                
                agent_scores.append((agent, total_score, {
                    'response_time_score': response_time_score,
                    'success_rate_score': success_rate_score,
                    'load_score': load_score,
                    'trend_score': trend_score
                }))
            
            # Select agent with highest score
            selected_agent, score, score_breakdown = max(agent_scores, key=lambda x: x[1])
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.PERFORMANCE_BASED,
            decision_factors={
                'total_score': score,
                'score_breakdown': score_breakdown,
                'weight_factors': weights
            },
            confidence_score=min(score, 1.0),
            alternative_agents=[agent for agent, _, _ in sorted(agent_scores, key=lambda x: x[1], reverse=True)[1:4]]
        )
    
    def _normalize_response_time(self, response_time: float, agents: List[AgentEndpoint]) -> float:
        """Normalize response time to 0-1 scale (lower is better)"""
        with self._lock:
            all_times = []
            for agent in agents:
                profile = self.agent_profiles.get(agent.agent_id, AgentPerformanceProfile(agent.agent_id))
                if profile.avg_response_time > 0:
                    all_times.append(profile.avg_response_time)
            
            if not all_times or response_time <= 0:
                return 1.0
            
            min_time = min(all_times)
            max_time = max(all_times)
            
            if max_time == min_time:
                return 1.0
            
            # Invert so lower response time = higher score
            return 1.0 - ((response_time - min_time) / (max_time - min_time))
    
    def _normalize_load(self, current_load: int, agents: List[AgentEndpoint]) -> float:
        """Normalize current load to 0-1 scale (lower is better)"""
        all_loads = [agent.current_connections for agent in agents]
        
        if not all_loads:
            return 1.0
        
        min_load = min(all_loads)
        max_load = max(all_loads)
        
        if max_load == min_load:
            return 1.0
        
        # Invert so lower load = higher score
        return 1.0 - ((current_load - min_load) / (max_load - min_load))
    
    async def _consistent_hash_routing(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> RoutingDecision:
        """Consistent hash-based routing"""
        capability = context.capability or "*"
        
        # Ensure hash ring exists and is populated
        if capability not in self.consistent_hash_rings:
            self.consistent_hash_rings[capability] = ConsistentHashRing()
        
        hash_ring = self.consistent_hash_rings[capability]
        
        # Add available agents to ring
        for agent in available_agents:
            hash_ring.add_agent(agent.agent_id)
        
        # Generate hash key from context
        hash_key = f"{context.user_id or context.session_id or context.client_ip or context.request_id}"
        
        selected_agent_id = hash_ring.get_agent(hash_key)
        
        if selected_agent_id:
            selected_agent = next(
                (a for a in available_agents if a.agent_id == selected_agent_id),
                available_agents[0]  # Fallback
            )
        else:
            selected_agent = available_agents[0]  # Fallback
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.CONSISTENT_HASH,
            decision_factors={
                'hash_key': hash_key,
                'ring_size': len(hash_ring.ring)
            },
            confidence_score=0.9  # Very deterministic
        )
    
    async def _adaptive_weighted_routing(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> RoutingDecision:
        """Adaptive weighted routing that learns from performance"""
        with self._lock:
            # Calculate adaptive weights based on recent performance
            agent_weights = {}
            
            for agent in available_agents:
                profile = self.agent_profiles.get(agent.agent_id, AgentPerformanceProfile(agent.agent_id))
                
                # Base weight starts at 1.0
                weight = 1.0
                
                # Adjust based on success rate
                weight *= profile.success_rate ** 2  # Square to emphasize high success rates
                
                # Adjust based on response time (lower is better)
                if profile.avg_response_time > 0:
                    avg_response_time = statistics.mean(
                        p.avg_response_time for p in self.agent_profiles.values() 
                        if p.avg_response_time > 0
                    ) or 1.0
                    
                    weight *= avg_response_time / max(profile.avg_response_time, 0.1)
                
                # Adjust based on recent trend
                weight *= profile.recent_performance_trend
                
                # Adjust based on current load (penalize high load)
                load_penalty = 1.0 / (1.0 + profile.current_load * 0.1)
                weight *= load_penalty
                
                agent_weights[agent.agent_id] = max(weight, 0.1)  # Minimum weight
            
            # Select agent using weighted random selection
            total_weight = sum(agent_weights.values())
            if total_weight == 0:
                selected_agent = random.choice(available_agents)
            else:
                rand_val = random.uniform(0, total_weight)
                current_weight = 0
                selected_agent = available_agents[0]  # Fallback
                
                for agent in available_agents:
                    current_weight += agent_weights[agent.agent_id]
                    if rand_val <= current_weight:
                        selected_agent = agent
                        break
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.ADAPTIVE_WEIGHTED,
            decision_factors={
                'agent_weights': agent_weights,
                'selected_weight': agent_weights.get(selected_agent.agent_id, 0)
            },
            confidence_score=0.8
        )
    
    async def _resource_based_routing(
        self,
        context: RoutingContext,
        available_agents: List[AgentEndpoint]
    ) -> RoutingDecision:
        """Resource-based routing considering CPU, memory, etc."""
        # This would integrate with system metrics from performance monitor
        # For now, use a simplified version based on current connections
        
        agent_scores = []
        
        for agent in available_agents:
            # Get system metrics (simplified)
            profile = self.agent_profiles.get(agent.agent_id, AgentPerformanceProfile(agent.agent_id))
            
            # Calculate resource score (0-1, higher is better)
            connection_score = max(0, 1.0 - (agent.current_connections / 100.0))
            utilization_score = max(0, 1.0 - profile.capacity_utilization)
            efficiency_score = profile.resource_efficiency
            
            total_score = (connection_score + utilization_score + efficiency_score) / 3
            agent_scores.append((agent, total_score))
        
        selected_agent, score = max(agent_scores, key=lambda x: x[1])
        
        return RoutingDecision(
            selected_agent=selected_agent,
            routing_strategy=RoutingStrategy.RESOURCE_BASED,
            decision_factors={
                'resource_score': score,
                'current_connections': selected_agent.current_connections
            },
            confidence_score=score
        )
    
    async def _update_sticky_session(
        self,
        context: RoutingContext,
        decision: RoutingDecision,
        policy: RoutingPolicy
    ):
        """Update sticky session after routing decision"""
        if not policy.session_affinity:
            return
        
        session_key = self._generate_session_key(context, policy.session_affinity)
        if not session_key:
            return
        
        with self._lock:
            self.sticky_sessions[session_key] = StickySession(
                session_key=session_key,
                agent_id=decision.selected_agent.agent_id,
                affinity_type=policy.session_affinity,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl=policy.session_ttl,
                access_count=1
            )
    
    async def _record_routing_decision(self, decision: RoutingDecision, response_time: float):
        """Record routing decision for analytics"""
        decision.routing_metadata['response_time'] = response_time
        decision.routing_metadata['timestamp'] = datetime.utcnow().isoformat()
        
        with self._lock:
            self.routing_decisions.append(decision)
            self.strategy_performance[decision.routing_strategy].append({
                'confidence': decision.confidence_score,
                'response_time': response_time,
                'timestamp': datetime.utcnow()
            })
    
    async def _cleanup_loop(self):
        """Background cleanup of expired sessions and old data"""
        while self._running:
            try:
                await self._cleanup_expired_sessions()
                await self._cleanup_old_decisions()
                await asyncio.sleep(self.session_cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sticky sessions"""
        current_time = datetime.utcnow()
        
        with self._lock:
            expired_sessions = []
            
            for session_key, session in self.sticky_sessions.items():
                if self._is_session_expired(session, timedelta(hours=1)):  # Default TTL
                    expired_sessions.append(session_key)
            
            for session_key in expired_sessions:
                del self.sticky_sessions[session_key]
            
            if expired_sessions:
                logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _cleanup_old_decisions(self):
        """Clean up old routing decisions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self._lock:
            # Clean routing decisions
            while (self.routing_decisions and 
                   'timestamp' in self.routing_decisions[0].routing_metadata and
                   datetime.fromisoformat(self.routing_decisions[0].routing_metadata['timestamp']) < cutoff_time):
                self.routing_decisions.popleft()
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        with self._lock:
            total_decisions = len(self.routing_decisions)
            
            # Strategy distribution
            strategy_counts = defaultdict(int)
            for decision in self.routing_decisions:
                strategy_counts[decision.routing_strategy.value] += 1
            
            # Average confidence by strategy
            strategy_confidence = {}
            for strategy, performances in self.strategy_performance.items():
                if performances:
                    avg_confidence = statistics.mean(p['confidence'] for p in performances)
                    strategy_confidence[strategy.value] = avg_confidence
            
            return {
                "total_decisions": total_decisions,
                "active_sessions": len(self.sticky_sessions),
                "strategy_distribution": dict(strategy_counts),
                "strategy_confidence": strategy_confidence,
                "routing_policies": len(self.routing_policies)
            }
    
    def add_routing_callback(self, callback: Callable):
        """Add callback for routing decisions"""
        self.routing_callbacks.append(callback)


# Global routing engine instance
routing_engine = AdvancedRoutingEngine()