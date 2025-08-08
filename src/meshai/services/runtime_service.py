"""
MeshAI Runtime Service

FastAPI server that handles task execution, routing, and orchestration.
Integrates with routing engine, health monitoring, and performance systems.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import structlog
import httpx

from ..core.config import MeshConfig, get_config
from ..core.schemas import TaskData, TaskResult, TaskStatus, RoutingStrategy
from ..core.routing_engine import routing_engine, RoutingContext
from ..core.health_monitor import health_monitor
from ..core.performance_monitor import performance_monitor
from ..core.failover_manager import failover_manager
from ..core.context import MeshContext
from ..clients.registry import RegistryClient
from ..exceptions.base import (
    TaskExecutionError,
    RoutingError,
    ValidationError,
    AgentNotFoundError
)
from ..utils.logging import setup_logging
from ..utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)

# In-memory task storage for MVP (would use database in production)
_active_tasks: Dict[str, TaskResult] = {}
_task_history: Dict[str, TaskResult] = {}


class TaskSubmissionResponse(BaseModel):
    """Task submission response"""
    task_id: str
    status: str = "submitted"
    message: str = "Task submitted for execution"
    estimated_completion_time: Optional[datetime] = None


class RuntimeHealth(BaseModel):
    """Runtime service health status"""
    status: str = "healthy"
    timestamp: datetime
    uptime_seconds: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int


class RuntimeStats(BaseModel):
    """Runtime service statistics"""
    total_tasks_submitted: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_execution_time_ms: float
    success_rate: float


class RuntimeService:
    """
    MeshAI Runtime Service
    
    Handles task execution orchestration, routing, and lifecycle management.
    Integrates with existing routing engine and monitoring systems.
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or get_config()
        self.start_time = datetime.utcnow()
        self.metrics = MetricsCollector("runtime-service")
        
        # Initialize clients
        self.registry_client = RegistryClient(self.config)
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Statistics
        self.total_submitted = 0
        self.total_completed = 0
        self.total_failed = 0
        self.execution_times: List[float] = []
        
        # Setup logging
        setup_logging(self.config.log_level, "runtime-service")
        
        # Initialize monitoring
        self._monitoring_initialized = False
        
        logger.info("Runtime service initialized")
    
    async def initialize_monitoring(self):
        """Initialize monitoring and routing systems"""
        if self._monitoring_initialized:
            return
        
        try:
            await health_monitor.start_monitoring()
            await performance_monitor.start_monitoring()
            await failover_manager.start_monitoring()
            await routing_engine.start()
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=self.config.default_timeout,
                headers=self.config.get_headers()
            )
            
            # Register health checks
            health_monitor.register_health_check(
                name="runtime_routing",
                check_function=self._check_routing_health,
                interval=timedelta(seconds=30),
                timeout=timedelta(seconds=5),
                critical=True
            )
            
            self._monitoring_initialized = True
            logger.info("Runtime monitoring systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
    
    async def _check_routing_health(self) -> bool:
        """Check routing system health"""
        try:
            # Simple health check - verify routing engine is responsive
            return routing_engine is not None
        except Exception:
            return False
    
    async def submit_task(self, task_data: TaskData) -> TaskSubmissionResponse:
        """Submit a task for execution"""
        # Generate task ID if not provided
        task_id = task_data.task_id or str(uuid.uuid4())
        task_data.task_id = task_id
        task_data.created_at = datetime.utcnow()
        
        # Create initial task result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            started_at=datetime.utcnow()
        )
        
        # Store task
        _active_tasks[task_id] = task_result
        
        # Start task execution in background
        asyncio.create_task(self._execute_task(task_data))
        
        self.total_submitted += 1
        self.metrics.total_tasks_submitted.inc()
        
        logger.info(f"Task {task_id} submitted for execution")
        
        return TaskSubmissionResponse(
            task_id=task_id,
            status="submitted",
            message="Task submitted for execution",
            estimated_completion_time=datetime.utcnow() + timedelta(seconds=task_data.timeout_seconds)
        )
    
    async def _execute_task(self, task_data: TaskData):
        """Execute a task through the routing system"""
        task_id = task_data.task_id
        start_time = datetime.utcnow()
        
        try:
            # Update status to routing
            if task_id in _active_tasks:
                _active_tasks[task_id].status = TaskStatus.ROUTING
            
            # Create routing context
            routing_context = RoutingContext(
                request_id=task_id,
                capability=task_data.required_capabilities[0] if task_data.required_capabilities else "general",
                user_id=task_data.source_agent,
                priority=1,
                routing_strategy=task_data.routing_strategy
            )
            
            # Route the task to an appropriate agent
            routing_decision = await routing_engine.route_request(routing_context)
            
            if not routing_decision or not routing_decision.selected_agent:
                raise RoutingError(f"No suitable agent found for task {task_id}")
            
            selected_agent = routing_decision.selected_agent
            
            # Update status to executing
            if task_id in _active_tasks:
                _active_tasks[task_id].status = TaskStatus.EXECUTING
                _active_tasks[task_id].agent_id = selected_agent.agent_id
                _active_tasks[task_id].routing_strategy_used = routing_decision.routing_strategy
            
            # Execute task on selected agent
            result = await self._call_agent(selected_agent, task_data)
            
            # Update task result with success
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if task_id in _active_tasks:
                task_result = _active_tasks[task_id]
                task_result.status = TaskStatus.COMPLETED
                task_result.result = result
                task_result.execution_time_seconds = execution_time
                task_result.completed_at = datetime.utcnow()
                
                # Move to history
                _task_history[task_id] = task_result
                del _active_tasks[task_id]
            
            # Update statistics
            self.total_completed += 1
            self.execution_times.append(execution_time)
            if len(self.execution_times) > 1000:  # Keep last 1000 execution times
                self.execution_times.pop(0)
            
            # Update metrics
            self.metrics.tasks_completed.inc()
            self.metrics.task_success_rate.inc()
            
            # Record performance for routing optimization
            performance_monitor.record_agent_request(
                selected_agent.agent_id,
                execution_time,
                True
            )
            
            logger.info(f"Task {task_id} completed successfully by agent {selected_agent.agent_id}")
            
        except Exception as e:
            # Handle task failure
            error_message = str(e)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if task_id in _active_tasks:
                task_result = _active_tasks[task_id]
                task_result.status = TaskStatus.FAILED
                task_result.error = error_message
                task_result.execution_time_seconds = execution_time
                task_result.completed_at = datetime.utcnow()
                
                # Move to history
                _task_history[task_id] = task_result
                del _active_tasks[task_id]
            
            # Update statistics
            self.total_failed += 1
            self.metrics.tasks_failed.inc()
            
            # Record failure for routing optimization
            if 'selected_agent' in locals():
                performance_monitor.record_agent_request(
                    selected_agent.agent_id,
                    execution_time,
                    False
                )
            
            logger.error(f"Task {task_id} failed: {error_message}")
    
    async def _call_agent(self, agent_endpoint, task_data: TaskData) -> Dict[str, Any]:
        """Make HTTP call to agent endpoint"""
        if not self.http_client:
            raise TaskExecutionError("HTTP client not initialized")
        
        try:
            # Prepare request data
            request_data = {
                "task_id": task_data.task_id,
                "task_type": task_data.task_type,
                "payload": task_data.input,
                "context": task_data.context,
                "timeout_seconds": task_data.timeout_seconds
            }
            
            # Make request to agent
            response = await self.http_client.post(
                f"{agent_endpoint.endpoint_url}/execute",
                json=request_data,
                timeout=task_data.timeout_seconds
            )
            
            if response.status_code == 408:
                raise TaskExecutionError("Task execution timed out")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise TaskExecutionError("Agent request timed out")
        except httpx.HTTPError as e:
            raise TaskExecutionError(f"HTTP error calling agent: {e}")
        except Exception as e:
            raise TaskExecutionError(f"Unexpected error calling agent: {e}")
    
    async def get_task_status(self, task_id: str) -> TaskResult:
        """Get task status and result"""
        # Check active tasks first
        if task_id in _active_tasks:
            return _active_tasks[task_id]
        
        # Check task history
        if task_id in _task_history:
            return _task_history[task_id]
        
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id not in _active_tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or not active")
        
        # Update task status
        task_result = _active_tasks[task_id]
        task_result.status = TaskStatus.CANCELLED
        task_result.completed_at = datetime.utcnow()
        
        # Move to history
        _task_history[task_id] = task_result
        del _active_tasks[task_id]
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[TaskResult]:
        """List tasks with optional filtering"""
        # Combine active and historical tasks
        all_tasks = {**_active_tasks, **_task_history}
        tasks = list(all_tasks.values())
        
        # Filter by status if provided
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.started_at or datetime.min, reverse=True)
        
        # Apply pagination
        return tasks[skip:skip + limit]
    
    async def get_runtime_stats(self) -> RuntimeStats:
        """Get runtime service statistics"""
        avg_execution_time = (
            sum(self.execution_times) / len(self.execution_times) * 1000
            if self.execution_times else 0
        )
        
        total_tasks = self.total_completed + self.total_failed
        success_rate = (
            self.total_completed / total_tasks
            if total_tasks > 0 else 0
        )
        
        return RuntimeStats(
            total_tasks_submitted=self.total_submitted,
            active_tasks=len(_active_tasks),
            completed_tasks=self.total_completed,
            failed_tasks=self.total_failed,
            avg_execution_time_ms=avg_execution_time,
            success_rate=success_rate
        )
    
    async def get_health(self) -> RuntimeHealth:
        """Get runtime service health"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return RuntimeHealth(
            status="healthy",
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime,
            active_tasks=len(_active_tasks),
            completed_tasks=self.total_completed,
            failed_tasks=self.total_failed
        )
    
    async def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics.get_prometheus_metrics()


# Global service instance
runtime_service = RuntimeService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await runtime_service.initialize_monitoring()
    
    yield
    
    # Shutdown
    if runtime_service.http_client:
        await runtime_service.http_client.aclose()
    
    await runtime_service.registry_client.close()
    
    if runtime_service._monitoring_initialized:
        await routing_engine.stop()
        await health_monitor.stop_monitoring()
        await performance_monitor.stop_monitoring()
        await failover_manager.stop_monitoring()


def create_app(config: Optional[MeshConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    app_config = config or get_config()
    
    app = FastAPI(
        title="MeshAI Runtime Service",
        description="Task execution and orchestration service for MeshAI",
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
    
    @app.post("/api/v1/tasks", response_model=TaskSubmissionResponse, status_code=status.HTTP_201_CREATED)
    async def submit_task_endpoint(task_data: TaskData, background_tasks: BackgroundTasks):
        """Submit a task for execution"""
        try:
            return await runtime_service.submit_task(task_data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RoutingError as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    @app.get("/api/v1/tasks/{task_id}", response_model=TaskResult)
    async def get_task_status_endpoint(task_id: str):
        """Get task status and result"""
        return await runtime_service.get_task_status(task_id)
    
    @app.delete("/api/v1/tasks/{task_id}")
    async def cancel_task_endpoint(task_id: str):
        """Cancel a running task"""
        success = await runtime_service.cancel_task(task_id)
        return {"status": "cancelled" if success else "failed", "task_id": task_id}
    
    @app.get("/api/v1/tasks", response_model=List[TaskResult])
    async def list_tasks_endpoint(
        status: Optional[TaskStatus] = Query(None),
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000)
    ):
        """List tasks with optional filtering"""
        return await runtime_service.list_tasks(status, skip, limit)
    
    @app.get("/api/v1/stats", response_model=RuntimeStats)
    async def get_runtime_stats_endpoint():
        """Get runtime service statistics"""
        return await runtime_service.get_runtime_stats()
    
    @app.get("/health", response_model=RuntimeHealth)
    async def health_check():
        """Runtime service health check"""
        return await runtime_service.get_health()
    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        """Prometheus metrics endpoint"""
        return await runtime_service.get_metrics()
    
    @app.get("/")
    async def root():
        """Runtime service info"""
        return {
            "service": "MeshAI Runtime Service",
            "version": "1.0.0",
            "status": "running",
            "active_tasks": len(_active_tasks)
        }
    
    return app


async def run_server(config: Optional[MeshConfig] = None, host: str = "0.0.0.0", port: int = 8002):
    """Run the runtime service"""
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
    logger.info(f"Starting Runtime Service on {host}:{port}")
    
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run_server())