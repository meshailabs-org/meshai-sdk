"""
Core MeshAgent base class and registration utilities
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Type, Union, Awaitable
from contextlib import asynccontextmanager
import logging

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .config import get_config, MeshConfig
from .context import MeshContext
from .schemas import TaskData, TaskResult, AgentInfo, AgentRegistration, TaskStatus
from ..clients.registry import RegistryClient
from ..clients.runtime import RuntimeClient
from ..exceptions.base import (
    MeshAIError,
    TaskExecutionError,
    RegistrationError,
    ValidationError
)
from ..utils.logging import setup_logging
from ..utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime
    agent_id: str
    uptime_seconds: float
    active_tasks: int = 0
    metrics: Dict[str, Any] = {}


class TaskRequest(BaseModel):
    """Incoming task request model"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    context: Dict[str, Any] = {}
    timeout_seconds: int = 30


class MeshAgent(ABC):
    """
    Base class for all MeshAI agents.
    
    Provides core functionality for:
    - Agent registration with MeshAI registry
    - Task execution handling
    - Context management
    - Health monitoring
    - Metrics collection
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: List[str],
        framework: str = "custom",
        config: Optional[MeshConfig] = None,
        auto_register: bool = True,
        **metadata
    ):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.framework = framework
        self.metadata = metadata
        self.config = config or get_config()
        
        # Initialize components
        self.registry_client = RegistryClient(self.config)
        self.runtime_client = RuntimeClient(self.config)
        self.context = MeshContext(self.config)
        self.metrics = MetricsCollector(agent_id)
        
        # Internal state
        self._start_time = datetime.utcnow()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._health_status = "healthy"
        self._server_task: Optional[asyncio.Task] = None
        self._app: Optional[FastAPI] = None
        
        # Setup logging
        setup_logging(self.config.log_level, agent_id)
        
        # Auto-register if enabled
        if auto_register and self.config.auto_register:
            asyncio.create_task(self._auto_register())
    
    @abstractmethod
    async def handle_task(
        self, 
        task_data: TaskData, 
        context: MeshContext
    ) -> Union[Dict[str, Any], str]:
        """
        Handle incoming task execution.
        
        Args:
            task_data: The task to execute
            context: Shared context for the task
            
        Returns:
            Task result data
            
        Raises:
            TaskExecutionError: If task execution fails
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return self.capabilities.copy()
    
    def get_agent_info(self) -> AgentInfo:
        """Get agent information"""
        return AgentInfo(
            id=self.agent_id,
            name=self.name,
            framework=self.framework,
            capabilities=self.capabilities,
            endpoint=self.config.agent_endpoint,
            health_endpoint=f"{self.config.agent_endpoint}/health",
            status=self._health_status,
            created_at=self._start_time,
            updated_at=datetime.utcnow(),
            metadata=self.metadata
        )
    
    async def register(self) -> bool:
        """Register agent with MeshAI registry"""
        try:
            registration = AgentRegistration(
                id=self.agent_id,
                name=self.name,
                framework=self.framework,
                capabilities=self.capabilities,
                endpoint=f"{self.config.agent_endpoint}/execute",
                health_endpoint=f"{self.config.agent_endpoint}/health",
                max_concurrent_tasks=self.config.max_concurrent_tasks,
                metadata=self.metadata
            )
            
            await self.registry_client.register_agent(registration)
            logger.info(f"Agent {self.agent_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {self.agent_id}: {e}")
            raise RegistrationError(f"Registration failed: {e}", self.agent_id)
    
    async def unregister(self) -> bool:
        """Unregister agent from MeshAI registry"""
        try:
            await self.registry_client.unregister_agent(self.agent_id)
            logger.info(f"Agent {self.agent_id} unregistered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister agent {self.agent_id}: {e}")
            return False
    
    async def invoke_agent(
        self,
        capabilities: List[str],
        task: Union[Dict[str, Any], TaskData],
        routing_strategy: str = "capability_match",
        timeout: int = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """
        Invoke another agent through MeshAI runtime.
        
        Args:
            capabilities: Required capabilities for the task
            task: Task data to execute
            routing_strategy: How to route the task
            timeout: Task timeout in seconds
            context: Additional context to share
            
        Returns:
            Task execution result
        """
        if isinstance(task, dict):
            task_data = TaskData(
                task_type="invoke",
                input=task,
                required_capabilities=capabilities,
                routing_strategy=routing_strategy,
                timeout_seconds=timeout or self.config.default_timeout,
                source_agent=self.agent_id,
                context=context or {}
            )
        else:
            task_data = task
            task_data.required_capabilities = capabilities
            task_data.source_agent = self.agent_id
        
        return await self.runtime_client.submit_task(task_data)
    
    async def heartbeat(self) -> bool:
        """Send heartbeat to registry"""
        try:
            await self.registry_client.heartbeat(self.agent_id)
            return True
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            return False
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application for agent endpoints"""
        app = FastAPI(
            title=f"MeshAI Agent: {self.name}",
            description=f"Agent {self.agent_id} ({self.framework})",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check endpoint
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
            return HealthResponse(
                status=self._health_status,
                timestamp=datetime.utcnow(),
                agent_id=self.agent_id,
                uptime_seconds=uptime,
                active_tasks=len(self._active_tasks),
                metrics=self.metrics.get_metrics()
            )
        
        # Task execution endpoint
        @app.post("/execute", response_model=Dict[str, Any])
        async def execute_task(request: TaskRequest):
            try:
                # Convert to TaskData
                task_data = TaskData(
                    task_id=request.task_id,
                    task_type=request.task_type,
                    input=request.payload,
                    context=request.context,
                    timeout_seconds=request.timeout_seconds
                )
                
                # Create context
                context = MeshContext.from_dict(request.context)
                
                # Execute task with timeout
                start_time = datetime.utcnow()
                
                with self.metrics.task_duration.time():
                    result = await asyncio.wait_for(
                        self.handle_task(task_data, context),
                        timeout=request.timeout_seconds
                    )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Update metrics
                self.metrics.tasks_completed.inc()
                self.metrics.task_success_rate.inc()
                
                return {
                    "status": "completed",
                    "result": result,
                    "execution_time": execution_time,
                    "agent_id": self.agent_id
                }
                
            except asyncio.TimeoutError:
                self.metrics.tasks_failed.inc()
                self.metrics.task_timeout_count.inc()
                raise HTTPException(status_code=408, detail="Task execution timed out")
                
            except TaskExecutionError as e:
                self.metrics.tasks_failed.inc()
                logger.error(f"Task execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
            except Exception as e:
                self.metrics.tasks_failed.inc()
                logger.error(f"Unexpected error in task execution: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
        
        # Agent info endpoint
        @app.get("/info", response_model=AgentInfo)
        async def get_agent_info():
            return self.get_agent_info()
        
        # Metrics endpoint
        @app.get("/metrics")
        async def get_metrics():
            return self.metrics.get_prometheus_metrics()
        
        return app
    
    async def start_server(self) -> None:
        """Start the agent server"""
        if self._server_task is not None:
            logger.warning("Server is already running")
            return
        
        self._app = self._create_app()
        
        config = uvicorn.Config(
            self._app,
            host=self.config.agent_host,
            port=self.config.agent_port,
            log_level=self.config.log_level.lower(),
            access_log=self.config.debug_mode
        )
        
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        
        logger.info(
            f"Agent {self.agent_id} server started on "
            f"{self.config.agent_host}:{self.config.agent_port}"
        )
    
    async def stop_server(self) -> None:
        """Stop the agent server"""
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
            self._server_task = None
            logger.info(f"Agent {self.agent_id} server stopped")
    
    async def run(self) -> None:
        """Run the agent (start server and register)"""
        try:
            # Start server
            await self.start_server()
            
            # Register with MeshAI
            await self.register()
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Wait for server task
            await self._server_task
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            # Cleanup
            await self.unregister()
            await self.stop_server()
            heartbeat_task.cancel()
    
    async def _auto_register(self) -> None:
        """Automatically register agent after a short delay"""
        await asyncio.sleep(1)  # Give server time to start
        try:
            await self.register()
        except Exception as e:
            logger.error(f"Auto-registration failed: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    @asynccontextmanager
    async def lifecycle(self):
        """Context manager for agent lifecycle"""
        try:
            await self.start_server()
            await self.register()
            yield self
        finally:
            await self.unregister()
            await self.stop_server()


# Agent registration decorator
_registered_agents: Dict[str, Type[MeshAgent]] = {}


def register_agent(
    capabilities: List[str],
    name: Optional[str] = None,
    framework: str = "custom",
    agent_id: Optional[str] = None,
    **metadata
) -> Callable[[Type[MeshAgent]], Type[MeshAgent]]:
    """
    Decorator to register an agent class with MeshAI.
    
    Args:
        capabilities: List of agent capabilities
        name: Human-readable agent name
        framework: Agent framework identifier
        agent_id: Unique agent ID (auto-generated if not provided)
        **metadata: Additional agent metadata
    
    Returns:
        Decorated agent class
    
    Example:
        @register_agent(
            capabilities=["text-analysis", "summarization"],
            name="Text Analyzer Agent"
        )
        class MyAgent(MeshAgent):
            async def handle_task(self, task_data, context):
                # Implementation here
                return {"result": "processed"}
    """
    def decorator(cls: Type[MeshAgent]) -> Type[MeshAgent]:
        # Generate agent ID if not provided
        final_agent_id = agent_id or f"{cls.__name__.lower()}_{uuid.uuid4().hex[:8]}"
        final_name = name or cls.__name__
        
        # Create wrapper class that auto-initializes
        class RegisteredAgent(cls):
            def __init__(self, **kwargs):
                # Merge provided kwargs with decorator parameters
                init_kwargs = {
                    "agent_id": final_agent_id,
                    "name": final_name,
                    "capabilities": capabilities,
                    "framework": framework,
                    **metadata,
                    **kwargs  # Allow override of decorator parameters
                }
                super().__init__(**init_kwargs)
        
        # Preserve original class name and metadata
        RegisteredAgent.__name__ = cls.__name__
        RegisteredAgent.__qualname__ = cls.__qualname__
        RegisteredAgent.__module__ = cls.__module__
        RegisteredAgent.__doc__ = cls.__doc__
        
        # Store in global registry
        _registered_agents[final_agent_id] = RegisteredAgent
        
        return RegisteredAgent
    
    return decorator


def get_registered_agents() -> Dict[str, Type[MeshAgent]]:
    """Get all registered agent classes"""
    return _registered_agents.copy()