"""
Database-Integrated MeshAI Agent Registry Service

FastAPI server with PostgreSQL/SQLite database persistence.
Replaces in-memory storage with proper database operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn
import structlog

from ..core.config import MeshConfig, get_config
from ..core.schemas import AgentInfo, AgentRegistration, DiscoveryQuery
from ..core.health_monitor import health_monitor
from ..core.performance_monitor import performance_monitor  
from ..core.failover_manager import failover_manager
from ..database.models import Agent, AgentHeartbeat
from ..database.session import get_db_session, DatabaseManager, get_database_manager
from ..database.migrations import init_database
from ..exceptions.base import (
    AgentNotFoundError,
    RegistrationError, 
    ValidationError
)
from ..utils.logging import setup_logging
from ..utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


class RegistryHealth(BaseModel):
    """Registry service health status"""
    status: str = "healthy"
    timestamp: datetime
    uptime_seconds: float
    total_agents: int
    healthy_agents: int
    unhealthy_agents: int


class DatabaseRegistryService:
    """
    Database-integrated Agent Registry Service
    
    Provides persistent agent registration, discovery, and health monitoring
    using PostgreSQL or SQLite database storage.
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or get_config()
        self.start_time = datetime.utcnow()
        self.metrics = MetricsCollector("registry-service")
        
        # Database manager
        self.db_manager = get_database_manager()
        
        # Setup logging
        setup_logging(self.config.log_level, "registry-service")
        
        # Initialize monitoring
        self._monitoring_initialized = False
        
        logger.info("Database Registry service initialized")
    
    async def initialize(self):
        """Initialize database and monitoring systems"""
        if self._monitoring_initialized:
            return
        
        try:
            # Test database connectivity
            if not await self.db_manager.test_connection():
                logger.error("Database connection test failed")
                raise Exception("Database not accessible")
            
            # Initialize monitoring systems
            await health_monitor.start_monitoring()
            await performance_monitor.start_monitoring()
            await failover_manager.start_monitoring()
            
            # Register health checks
            health_monitor.register_health_check(
                name="registry_database",
                check_function=self._check_database_health,
                interval=timedelta(seconds=30),
                timeout=timedelta(seconds=5),
                critical=True
            )
            
            self._monitoring_initialized = True
            logger.info("Registry monitoring systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize registry service: {e}")
            raise
    
    async def _check_database_health(self) -> bool:
        """Check database health"""
        try:
            return await self.db_manager.test_connection()
        except Exception:
            return False
    
    async def register_agent(self, registration: AgentRegistration) -> AgentInfo:
        """Register a new agent in database"""
        async with self.db_manager.get_async_session() as session:
            try:
                # Check if agent already exists
                existing = await session.get(Agent, registration.id)
                if existing:
                    raise RegistrationError(
                        f"Agent {registration.id} already exists",
                        registration.id
                    )
                
                # Create new agent record
                now = datetime.utcnow()
                agent = Agent(
                    id=registration.id,
                    name=registration.name,
                    framework=registration.framework,
                    capabilities=registration.capabilities,
                    endpoint=registration.endpoint,
                    health_endpoint=registration.health_endpoint,
                    max_concurrent_tasks=registration.max_concurrent_tasks,
                    input_schema=registration.input_schema,
                    output_schema=registration.output_schema,
                    description=registration.description,
                    version=registration.version,
                    tags=registration.tags,
                    agent_metadata=registration.metadata,
                    status="active",
                    created_at=now,
                    updated_at=now,
                    last_seen_at=now
                )
                
                session.add(agent)
                await session.flush()  # Get the ID
                
                # Create initial heartbeat
                heartbeat = AgentHeartbeat(
                    agent_id=agent.id,
                    timestamp=now,
                    status="active",
                    health_score=1.0
                )
                session.add(heartbeat)
                
                await session.commit()
                
                # Register with failover manager
                if registration.endpoint:
                    failover_manager.register_agent(
                        agent_id=registration.id,
                        endpoint_url=registration.endpoint,
                        capabilities=registration.capabilities,
                        weight=1.0,
                        metadata=registration.metadata  # For failover manager, keep the name
                    )
                
                # Update metrics
                self.metrics.record_registration("registry-service")
                
                # Convert to AgentInfo
                agent_info = AgentInfo(**agent.to_dict())
                
                logger.info(f"Agent {registration.id} registered in database")
                return agent_info
                
            except RegistrationError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to register agent {registration.id}: {e}")
                raise RegistrationError(f"Registration failed: {e}", registration.id)
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from database"""
        async with self.db_manager.get_async_session() as session:
            try:
                # Check if agent exists
                agent = await session.get(Agent, agent_id)
                if not agent:
                    raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
                
                # Delete agent (cascade will handle heartbeats and tasks)
                await session.delete(agent)
                await session.commit()
                
                logger.info(f"Agent {agent_id} unregistered from database")
                return True
                
            except AgentNotFoundError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to unregister agent {agent_id}: {e}")
                raise
    
    async def get_agent(self, agent_id: str) -> AgentInfo:
        """Get agent by ID from database"""
        async with self.db_manager.get_async_session() as session:
            agent = await session.get(Agent, agent_id)
            if not agent:
                raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
            
            return AgentInfo(**agent.to_dict())
    
    async def list_agents(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[AgentInfo]:
        """List agents from database with pagination"""
        async with self.db_manager.get_async_session() as session:
            query = select(Agent)
            
            # Filter by status if provided
            if status:
                query = query.where(Agent.status == status)
            
            # Order by creation time (newest first)
            query = query.order_by(Agent.created_at.desc())
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = await session.execute(query)
            agents = result.scalars().all()
            
            return [AgentInfo(**agent.to_dict()) for agent in agents]
    
    async def discover_agents(self, query: DiscoveryQuery) -> List[AgentInfo]:
        """Discover agents based on query criteria"""
        async with self.db_manager.get_async_session() as session:
            stmt = select(Agent)
            conditions = []
            
            # Filter by capabilities (using JSON contains)
            if query.capabilities:
                for capability in query.capabilities:
                    conditions.append(func.json_extract(Agent.capabilities, '$').like(f'%{capability}%'))
            
            # Filter by framework
            if query.framework:
                conditions.append(Agent.framework == query.framework)
            
            # Filter by tags
            if query.tags:
                for tag in query.tags:
                    conditions.append(func.json_extract(Agent.tags, '$').like(f'%{tag}%'))
            
            # Filter by performance criteria
            if query.min_success_rate is not None:
                conditions.append(
                    or_(Agent.success_rate.is_(None), Agent.success_rate >= query.min_success_rate)
                )
            
            if query.max_response_time_ms is not None:
                conditions.append(
                    or_(Agent.avg_response_time_ms.is_(None), Agent.avg_response_time_ms <= query.max_response_time_ms)
                )
            
            if query.max_load is not None:
                conditions.append(
                    or_(Agent.current_load.is_(None), Agent.current_load <= query.max_load)
                )
            
            # Exclude specified agents
            if query.exclude_agents:
                conditions.append(~Agent.id.in_(query.exclude_agents))
            
            # Include/exclude offline agents
            if not query.include_offline:
                conditions.append(Agent.status != "offline")
            
            # Apply all conditions
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            # Order by health score and creation time
            stmt = stmt.order_by(
                Agent.health_score.desc().nullslast(),
                Agent.created_at.desc()
            )
            
            # Apply limit
            if query.limit:
                stmt = stmt.limit(query.limit)
            
            result = await session.execute(stmt)
            agents = result.scalars().all()
            
            self.metrics.record_discovery("registry-service")
            return [AgentInfo(**agent.to_dict()) for agent in agents]
    
    async def heartbeat(self, agent_id: str) -> Dict[str, Any]:
        """Process agent heartbeat in database"""
        async with self.db_manager.get_async_session() as session:
            # Check if agent exists
            agent = await session.get(Agent, agent_id)
            if not agent:
                raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
            
            now = datetime.utcnow()
            
            # Update agent last seen time
            agent.last_seen_at = now
            agent.updated_at = now
            agent.status = "active"
            
            # Create heartbeat record
            heartbeat = AgentHeartbeat(
                agent_id=agent_id,
                timestamp=now,
                status="active",
                health_score=agent.health_score or 1.0
            )
            session.add(heartbeat)
            
            await session.commit()
            
            return {
                "agent_id": agent_id,
                "status": "acknowledged", 
                "timestamp": now,
                "next_heartbeat_in": self.config.health_check_interval
            }
    
    async def update_agent(
        self,
        agent_id: str,
        updates: Dict[str, Any]
    ) -> AgentInfo:
        """Update agent information in database"""
        async with self.db_manager.get_async_session() as session:
            agent = await session.get(Agent, agent_id)
            if not agent:
                raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
            
            # Update allowed fields
            allowed_fields = {
                'name', 'capabilities', 'tags', 'agent_metadata',
                'health_score', 'avg_response_time_ms', 'success_rate', 'current_load',
                'description', 'version'
            }
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(agent, field):
                    setattr(agent, field, value)
            
            agent.updated_at = datetime.utcnow()
            await session.commit()
            
            return AgentInfo(**agent.to_dict())
    
    async def get_health(self) -> RegistryHealth:
        """Get registry service health from database"""
        async with self.db_manager.get_async_session() as session:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Count total agents
            total_count = await session.scalar(select(func.count(Agent.id)))
            
            # Count healthy agents (recent heartbeat)
            heartbeat_timeout = datetime.utcnow() - timedelta(
                seconds=self.config.health_check_interval * 3
            )
            
            healthy_count = await session.scalar(
                select(func.count(Agent.id)).where(
                    and_(
                        Agent.status == "active",
                        Agent.last_seen_at > heartbeat_timeout
                    )
                )
            )
            
            unhealthy_count = total_count - (healthy_count or 0)
            
            return RegistryHealth(
                status="healthy",
                timestamp=datetime.utcnow(),
                uptime_seconds=uptime,
                total_agents=total_count or 0,
                healthy_agents=healthy_count or 0,
                unhealthy_agents=unhealthy_count
            )
    
    async def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics.get_prometheus_metrics()
    
    async def cleanup_stale_agents(self):
        """Clean up agents with stale heartbeats"""
        async with self.db_manager.get_async_session() as session:
            stale_timeout = datetime.utcnow() - timedelta(
                seconds=self.config.health_check_interval * 5
            )
            
            # Mark stale agents as offline
            stmt = (
                update(Agent)
                .where(
                    and_(
                        Agent.status == "active",
                        Agent.last_seen_at < stale_timeout
                    )
                )
                .values(status="offline", updated_at=datetime.utcnow())
            )
            
            result = await session.execute(stmt)
            await session.commit()
            
            if result.rowcount > 0:
                logger.warning(f"Marked {result.rowcount} agents as offline due to stale heartbeats")


# Global service instance
registry_service = DatabaseRegistryService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    cleanup_task = None
    try:
        # Initialize database
        init_database()
        
        # Initialize service
        await registry_service.initialize()
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(cleanup_loop())
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start registry service: {e}")
        raise
    finally:
        # Shutdown
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        if registry_service._monitoring_initialized:
            await health_monitor.stop_monitoring()
            await performance_monitor.stop_monitoring()
            await failover_manager.stop_monitoring()
        
        await registry_service.db_manager.close()


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
    """Create FastAPI application with database integration"""
    app_config = config or get_config()
    
    app = FastAPI(
        title="MeshAI Agent Registry (Database)",
        description="Agent registration and discovery service with database persistence",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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
        health = await registry_service.get_health()
        return {
            "service": "MeshAI Agent Registry (Database)",
            "version": "1.0.0",
            "status": "running",
            "total_agents": health.total_agents,
            "database": "connected"
        }
    
    return app


async def run_server(config: Optional[MeshConfig] = None, host: str = "0.0.0.0", port: int = 8001):
    """Run the database-integrated registry service"""
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
    logger.info(f"Starting Database Registry Service on {host}:{port}")
    
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run_server())