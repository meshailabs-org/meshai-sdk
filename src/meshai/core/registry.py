"""
MeshAI Agent Registry Module

Central repository for agent discovery, registration, and health monitoring.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog
from pydantic import BaseModel, Field

from ..core.agent import MeshAgent
from ..core.schemas import AgentStatus, AgentMetadata

logger = structlog.get_logger()


class RegistryEntry(BaseModel):
    """Registry entry for an agent"""
    agent_id: str
    metadata: AgentMetadata
    status: AgentStatus
    last_heartbeat: datetime
    registration_time: datetime
    capabilities: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class MeshRegistry:
    """
    Central agent registry for MeshAI platform.
    
    Manages agent discovery, registration, health monitoring,
    and capability indexing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the registry"""
        self.config = config or {}
        self.agents: Dict[str, RegistryEntry] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self.heartbeat_timeout = self.config.get('heartbeat_timeout', 60)
        
    async def start(self):
        """Start the registry service"""
        if self._running:
            return
            
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("registry_started", config=self.config)
        
    async def stop(self):
        """Stop the registry service"""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("registry_stopped")
        
    async def register_agent(self, agent: MeshAgent) -> bool:
        """
        Register a new agent with the registry.
        
        Args:
            agent: The MeshAgent instance to register
            
        Returns:
            bool: True if registration successful
        """
        async with self._lock:
            agent_id = agent.agent_id
            
            if agent_id in self.agents:
                logger.warning("agent_already_registered", agent_id=agent_id)
                return False
                
            entry = RegistryEntry(
                agent_id=agent_id,
                metadata=AgentMetadata(
                    agent_id=agent_id,
                    name=agent.name,
                    framework=getattr(agent, 'framework', 'unknown'),
                    model=getattr(agent, 'model', 'unknown'),
                    capabilities=agent.capabilities
                ),
                status=AgentStatus.HEALTHY,
                last_heartbeat=datetime.utcnow(),
                registration_time=datetime.utcnow(),
                capabilities=agent.capabilities,
                performance_metrics={}
            )
            
            self.agents[agent_id] = entry
            logger.info("agent_registered", agent_id=agent_id, capabilities=agent.capabilities)
            return True
            
    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: ID of the agent to remove
            
        Returns:
            bool: True if deregistration successful
        """
        async with self._lock:
            if agent_id not in self.agents:
                logger.warning("agent_not_found", agent_id=agent_id)
                return False
                
            del self.agents[agent_id]
            logger.info("agent_deregistered", agent_id=agent_id)
            return True
            
    async def get_agent(self, agent_id: str) -> Optional[RegistryEntry]:
        """
        Get agent information from the registry.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            RegistryEntry or None if not found
        """
        async with self._lock:
            return self.agents.get(agent_id)
            
    async def list_agents(self, 
                         status: Optional[AgentStatus] = None,
                         capability: Optional[str] = None) -> List[RegistryEntry]:
        """
        List all agents with optional filtering.
        
        Args:
            status: Filter by agent status
            capability: Filter by capability
            
        Returns:
            List of matching registry entries
        """
        async with self._lock:
            agents = list(self.agents.values())
            
            if status:
                agents = [a for a in agents if a.status == status]
                
            if capability:
                agents = [a for a in agents if capability in a.capabilities]
                
            return agents
            
    async def update_heartbeat(self, agent_id: str) -> bool:
        """
        Update the last heartbeat time for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if update successful
        """
        async with self._lock:
            if agent_id not in self.agents:
                return False
                
            self.agents[agent_id].last_heartbeat = datetime.utcnow()
            self.agents[agent_id].status = AgentStatus.HEALTHY
            return True
            
    async def update_performance_metrics(self, 
                                        agent_id: str, 
                                        metrics: Dict[str, float]) -> bool:
        """
        Update performance metrics for an agent.
        
        Args:
            agent_id: ID of the agent
            metrics: Dictionary of metric name to value
            
        Returns:
            bool: True if update successful
        """
        async with self._lock:
            if agent_id not in self.agents:
                return False
                
            self.agents[agent_id].performance_metrics.update(metrics)
            return True
            
    async def find_agents_by_capability(self, capability: str) -> List[RegistryEntry]:
        """
        Find all agents that have a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of agents with the capability
        """
        return await self.list_agents(capability=capability)
        
    async def get_healthy_agents(self) -> List[RegistryEntry]:
        """
        Get all healthy agents.
        
        Returns:
            List of healthy agents
        """
        return await self.list_agents(status=AgentStatus.HEALTHY)
        
    async def _health_check_loop(self):
        """Background task to check agent health"""
        while self._running:
            try:
                await self._check_agent_health()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error("health_check_error", error=str(e))
                
    async def _check_agent_health(self):
        """Check health of all registered agents"""
        current_time = datetime.utcnow()
        timeout_threshold = timedelta(seconds=self.heartbeat_timeout)
        
        async with self._lock:
            for agent_id, entry in self.agents.items():
                time_since_heartbeat = current_time - entry.last_heartbeat
                
                if time_since_heartbeat > timeout_threshold:
                    if entry.status != AgentStatus.UNHEALTHY:
                        entry.status = AgentStatus.UNHEALTHY
                        logger.warning("agent_unhealthy", 
                                     agent_id=agent_id,
                                     last_heartbeat=entry.last_heartbeat)
                                     
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        healthy_count = sum(1 for a in self.agents.values() 
                          if a.status == AgentStatus.HEALTHY)
        unhealthy_count = sum(1 for a in self.agents.values() 
                            if a.status == AgentStatus.UNHEALTHY)
        
        capability_counts = {}
        for agent in self.agents.values():
            for cap in agent.capabilities:
                capability_counts[cap] = capability_counts.get(cap, 0) + 1
                
        return {
            'total_agents': len(self.agents),
            'healthy_agents': healthy_count,
            'unhealthy_agents': unhealthy_count,
            'capabilities': capability_counts,
            'agents': [a.agent_id for a in self.agents.values()]
        }