"""
MeshAI Agent Registry Service

FastAPI server that provides agent registration, discovery, and management capabilities.
Integrates with existing health monitoring, performance tracking, and failover systems.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn
import structlog

from ..core.config import MeshConfig, get_config
from ..core.schemas import AgentInfo, AgentRegistration, DiscoveryQuery
from ..core.health_monitor import health_monitor
from ..core.performance_monitor import performance_monitor
from ..core.failover_manager import failover_manager
from ..exceptions.base import (
    AgentNotFoundError,
    RegistrationError,
    ValidationError
)
from ..utils.logging import setup_logging
from ..utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)

# In-memory storage for MVP (would use database in production)
_registered_agents: Dict[str, AgentInfo] = {}
_agent_heartbeats: Dict[str, datetime] = {}


class RegistryHealth(BaseModel):
    """Registry service health status"""
    status: str = "healthy"
    timestamp: datetime
    uptime_seconds: float
    total_agents: int
    healthy_agents: int
    unhealthy_agents: int


class RegistryMetrics(BaseModel):
    """Registry service metrics"""
    total_registrations: int
    total_discoveries: int
    active_agents: int
    avg_response_time_ms: float


class RegistryService:
    """
    MeshAI Agent Registry Service
    
    Provides centralized agent registration, discovery, and health monitoring.
    Integrates with existing MeshAI health monitoring and performance systems.
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or get_config()
        self.start_time = datetime.utcnow()
        self.metrics = MetricsCollector("registry-service")
        
        # Setup logging
        setup_logging(self.config.log_level, "registry-service")
        
        # Initialize monitoring systems
        self._monitoring_initialized = False
        
        logger.info("Registry service initialized")
    
    async def initialize_monitoring(self):
        """Initialize health monitoring and performance tracking"""
        if self._monitoring_initialized:
            return
        
        try:
            await health_monitor.start_monitoring()
            await performance_monitor.start_monitoring()
            await failover_manager.start_monitoring()
            
            # Register health checks
            health_monitor.register_health_check(
                name="registry_storage",
                check_function=self._check_storage_health,
                interval=timedelta(seconds=30),
                timeout=timedelta(seconds=5),
                critical=True
            )
            
            self._monitoring_initialized = True
            logger.info("Registry monitoring systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
    
    async def _check_storage_health(self) -> bool:
        """Check storage system health"""
        try:
            # For MVP with in-memory storage, always healthy
            # In production, this would check database connectivity
            return len(_registered_agents) >= 0
        except Exception:
            return False
    
    async def register_agent(self, registration: AgentRegistration) -> AgentInfo:
        """Register a new agent"""
        try:
            # Check if agent already exists
            if registration.id in _registered_agents:
                raise RegistrationError(
                    f"Agent {registration.id} already exists",
                    registration.id
                )
            
            # Create agent info
            agent_info = AgentInfo(
                id=registration.id,
                name=registration.name,
                framework=registration.framework,
                capabilities=registration.capabilities,
                endpoint=registration.endpoint,
                health_endpoint=registration.health_endpoint,
                status="active",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=registration.tags,
                metadata=registration.metadata
            )
            
            # Store agent
            _registered_agents[registration.id] = agent_info
            _agent_heartbeats[registration.id] = datetime.utcnow()
            
            # Register with failover manager
            if registration.endpoint:
                failover_manager.register_agent(
                    agent_id=registration.id,
                    endpoint_url=registration.endpoint,
                    capabilities=registration.capabilities,
                    weight=1.0,
                    metadata=registration.metadata
                )
            
            # Update metrics
            self.metrics.total_registrations.inc()
            
            logger.info(f"Agent {registration.id} registered successfully")
            return agent_info
            
        except RegistrationError:
            raise
        except Exception as e:
            logger.error(f"Failed to register agent {registration.id}: {e}")
            raise RegistrationError(f"Registration failed: {e}", registration.id)
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in _registered_agents:
            raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
        
        try:
            # Remove from storage
            del _registered_agents[agent_id]
            if agent_id in _agent_heartbeats:
                del _agent_heartbeats[agent_id]
            
            # Unregister from failover manager
            # Note: failover_manager doesn't have unregister method in current implementation
            # This would be added in production
            
            logger.info(f"Agent {agent_id} unregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            raise
    
    async def get_agent(self, agent_id: str) -> AgentInfo:
        """Get agent by ID"""
        if agent_id not in _registered_agents:
            raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
        
        return _registered_agents[agent_id]
    
    async def list_agents(
        self, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[AgentInfo]:
        """List registered agents with pagination"""
        agents = list(_registered_agents.values())
        
        # Filter by status if provided
        if status:
            agents = [agent for agent in agents if agent.status == status]
        
        # Apply pagination
        return agents[skip:skip + limit]
    
    async def discover_agents(self, query: DiscoveryQuery) -> List[AgentInfo]:
        """Discover agents based on query criteria"""
        agents = list(_registered_agents.values())
        
        # Filter by capabilities
        if query.capabilities:
            agents = [
                agent for agent in agents
                if any(cap in agent.capabilities for cap in query.capabilities)
            ]
        
        # Filter by framework
        if query.framework:
            agents = [agent for agent in agents if agent.framework == query.framework]
        
        # Filter by tags
        if query.tags:
            agents = [
                agent for agent in agents
                if any(tag in agent.tags for tag in query.tags)
            ]
        
        # Filter by performance criteria
        if query.min_success_rate is not None:
            agents = [
                agent for agent in agents
                if agent.success_rate is None or agent.success_rate >= query.min_success_rate
            ]
        
        if query.max_response_time_ms is not None:
            agents = [
                agent for agent in agents
                if agent.avg_response_time_ms is None or agent.avg_response_time_ms <= query.max_response_time_ms
            ]
        
        if query.max_load is not None:
            agents = [
                agent for agent in agents
                if agent.current_load is None or agent.current_load <= query.max_load
            ]
        
        # Exclude specified agents
        if query.exclude_agents:
            agents = [agent for agent in agents if agent.id not in query.exclude_agents]
        
        # Include/exclude offline agents
        if not query.include_offline:
            agents = [agent for agent in agents if agent.status != "offline"]
        
        # Apply limit
        if query.limit:
            agents = agents[:query.limit]
        
        self.metrics.total_discoveries.inc()
        return agents
    
    async def heartbeat(self, agent_id: str) -> Dict[str, Any]:
        """Process agent heartbeat"""
        if agent_id not in _registered_agents:
            raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
        
        # Update heartbeat timestamp
        _agent_heartbeats[agent_id] = datetime.utcnow()
        
        # Update agent status
        agent = _registered_agents[agent_id]
        agent.updated_at = datetime.utcnow()
        agent.status = "active"
        
        return {
            "agent_id": agent_id,
            "status": "acknowledged",
            "timestamp": datetime.utcnow(),
            "next_heartbeat_in": self.config.health_check_interval
        }
    
    async def update_agent(
        self, 
        agent_id: str, 
        updates: Dict[str, Any]
    ) -> AgentInfo:
        """Update agent information"""
        if agent_id not in _registered_agents:
            raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
        
        agent = _registered_agents[agent_id]
        
        # Update allowed fields
        allowed_fields = {
            'name', 'capabilities', 'tags', 'metadata', 
            'health_score', 'avg_response_time_ms', 'success_rate', 'current_load'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(agent, field, value)
        
        agent.updated_at = datetime.utcnow()
        return agent
    
    async def get_health(self) -> RegistryHealth:
        """Get registry service health"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Count healthy vs unhealthy agents
        healthy_agents = 0
        unhealthy_agents = 0
        
        current_time = datetime.utcnow()
        heartbeat_timeout = timedelta(seconds=self.config.health_check_interval * 3)
        
        for agent_id, agent in _registered_agents.items():
            last_heartbeat = _agent_heartbeats.get(agent_id)
            if (last_heartbeat and 
                current_time - last_heartbeat < heartbeat_timeout and 
                agent.status == "active"):
                healthy_agents += 1
            else:
                unhealthy_agents += 1
        
        return RegistryHealth(
            status="healthy",
            timestamp=current_time,
            uptime_seconds=uptime,
            total_agents=len(_registered_agents),
            healthy_agents=healthy_agents,
            unhealthy_agents=unhealthy_agents
        )
    
    async def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics.get_prometheus_metrics()
    
    async def cleanup_stale_agents(self):
        """Clean up agents that haven't sent heartbeats"""
        current_time = datetime.utcnow()
        stale_timeout = timedelta(seconds=self.config.health_check_interval * 5)
        
        stale_agents = []
        for agent_id, last_heartbeat in _agent_heartbeats.items():
            if current_time - last_heartbeat > stale_timeout:
                stale_agents.append(agent_id)
        
        for agent_id in stale_agents:
            if agent_id in _registered_agents:
                _registered_agents[agent_id].status = "offline"
                logger.warning(f"Marked agent {agent_id} as offline due to stale heartbeat")


# Global service instance
registry_service = RegistryService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await registry_service.initialize_monitoring()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_loop())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    if registry_service._monitoring_initialized:
        await health_monitor.stop_monitoring()
        await performance_monitor.stop_monitoring()
        await failover_manager.stop_monitoring()


async def cleanup_loop():
    """Periodic cleanup of stale agents"""
    while True:
        try:
            await asyncio.sleep(60)  # Run every minute
            await registry_service.cleanup_stale_agents()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")


def create_app(config: Optional[MeshConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    app_config = config or get_config()
    
    app = FastAPI(
        title="MeshAI Agent Registry",
        description="Agent registration and discovery service for MeshAI",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API Routes
    
    @app.post("/api/v1/agents", response_model=AgentInfo, status_code=status.HTTP_201_CREATED)
    async def register_agent_endpoint(registration: AgentRegistration):
        """Register a new agent"""
        try:
            return await registry_service.register_agent(registration)
        except RegistrationError as e:
            if "already exists" in str(e):
                raise HTTPException(status_code=409, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
    
    @app.delete("/api/v1/agents/{agent_id}")
    async def unregister_agent_endpoint(agent_id: str):
        """Unregister an agent"""
        try:
            await registry_service.unregister_agent(agent_id)
            return {"status": "success", "message": f"Agent {agent_id} unregistered"}
        except AgentNotFoundError:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    @app.get("/api/v1/agents/{agent_id}", response_model=AgentInfo)
    async def get_agent_endpoint(agent_id: str):
        """Get agent by ID"""
        try:
            return await registry_service.get_agent(agent_id)
        except AgentNotFoundError:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    @app.get("/api/v1/agents", response_model=List[AgentInfo])
    async def list_agents_endpoint(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        status: Optional[str] = Query(None)
    ):
        """List registered agents"""
        return await registry_service.list_agents(skip, limit, status)
    
    @app.post("/api/v1/agents/discover", response_model=List[AgentInfo])
    async def discover_agents_endpoint(query: DiscoveryQuery):
        """Discover agents based on criteria"""
        return await registry_service.discover_agents(query)
    
    @app.post("/api/v1/agents/{agent_id}/heartbeat")
    async def heartbeat_endpoint(agent_id: str):
        """Process agent heartbeat"""
        try:
            return await registry_service.heartbeat(agent_id)
        except AgentNotFoundError:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    @app.put("/api/v1/agents/{agent_id}", response_model=AgentInfo)
    async def update_agent_endpoint(agent_id: str, updates: Dict[str, Any]):
        """Update agent information"""
        try:
            return await registry_service.update_agent(agent_id, updates)
        except AgentNotFoundError:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    @app.get("/health", response_model=RegistryHealth)
    async def health_check():
        """Registry service health check"""
        return await registry_service.get_health()
    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        """Prometheus metrics endpoint"""
        return await registry_service.get_metrics()
    
    @app.get("/")
    async def root():
        """Registry service info"""
        return {
            "service": "MeshAI Agent Registry",
            "version": "1.0.0",
            "status": "running",
            "total_agents": len(_registered_agents)
        }
    
    return app


async def run_server(config: Optional[MeshConfig] = None, host: str = "0.0.0.0", port: int = 8001):
    """Run the registry service"""
    app = create_app(config)
    
    config_obj = config or get_config()
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=config_obj.log_level.lower(),
        access_log=config_obj.debug_mode
    )
    
    server = uvicorn.Server(uvicorn_config)
    logger.info(f"Starting Registry Service on {host}:{port}")
    
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run_server())