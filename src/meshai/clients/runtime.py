"""
Client for MeshAI Runtime Service
"""

import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import MeshConfig
from ..core.schemas import TaskData, TaskResult, TaskStatus, RoutingStrategy
from ..exceptions.base import (
    MeshAIError,
    TaskExecutionError,
    RoutingError,
    TimeoutError,
    ValidationError
)

logger = structlog.get_logger(__name__)


class RuntimeClient:
    """
    Client for interacting with MeshAI Runtime Service.
    
    Provides methods for:
    - Task submission and management
    - Task status monitoring
    - Runtime statistics and metrics
    - Task cancellation and cleanup
    """
    
    def __init__(self, config: MeshConfig):
        self.config = config
        self.base_url = config.runtime_base_url
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.config.get_headers(),
                timeout=self.config.default_timeout
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def submit_task(self, task_data: TaskData) -> TaskResult:
        """
        Submit a task for execution.
        
        Args:
            task_data: Task data to execute
            
        Returns:
            Task execution result
            
        Raises:
            ValidationError: If task data is invalid
            RoutingError: If task routing fails
            TaskExecutionError: If task execution fails
        """
        try:
            # Submit task
            response = await self.client.post(
                "/api/v1/tasks",
                json=task_data.model_dump()
            )
            
            if response.status_code == 422:
                raise ValidationError(
                    f"Invalid task data: {response.text}",
                    value=task_data.model_dump()
                )
            
            response.raise_for_status()
            submission_data = response.json()
            task_id = submission_data["task_id"]
            
            logger.info(f"Task {task_id} submitted successfully")
            
            # Wait for completion if this is a synchronous request
            if task_data.timeout_seconds and task_data.timeout_seconds > 0:
                return await self.wait_for_completion(
                    task_id, 
                    timeout=task_data.timeout_seconds
                )
            else:
                # Return immediate result
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.PENDING
                )
                
        except httpx.TimeoutException:
            raise TimeoutError(f"Task submission timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise TaskExecutionError(f"HTTP error during task submission: {e}")
        except Exception as e:
            raise TaskExecutionError(f"Unexpected error during task submission: {e}")
    
    async def get_task_status(self, task_id: str) -> TaskResult:
        """
        Get task status and result.
        
        Args:
            task_id: ID of task to check
            
        Returns:
            Task status and result
            
        Raises:
            TaskExecutionError: If task lookup fails
        """
        try:
            response = await self.client.get(f"/api/v1/tasks/{task_id}")
            
            if response.status_code == 404:
                raise TaskExecutionError(f"Task {task_id} not found", task_id)
            
            response.raise_for_status()
            data = response.json()
            
            return TaskResult(**data)
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Task status check timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise TaskExecutionError(f"HTTP error during task status check: {e}", task_id)
        except Exception as e:
            raise TaskExecutionError(f"Unexpected error during task status check: {e}", task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled successfully
            
        Raises:
            TaskExecutionError: If cancellation fails
        """
        try:
            response = await self.client.delete(f"/api/v1/tasks/{task_id}")
            
            if response.status_code == 404:
                raise TaskExecutionError(f"Task {task_id} not found", task_id)
            elif response.status_code == 400:
                # Task already completed
                return False
            
            response.raise_for_status()
            
            logger.info(f"Task {task_id} cancelled successfully")
            return True
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Task cancellation timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise TaskExecutionError(f"HTTP error during task cancellation: {e}", task_id)
        except Exception as e:
            raise TaskExecutionError(f"Unexpected error during task cancellation: {e}", task_id)
    
    async def wait_for_completion(
        self, 
        task_id: str, 
        timeout: int = 30,
        poll_interval: float = 1.0
    ) -> TaskResult:
        """
        Wait for task completion with polling.
        
        Args:
            task_id: ID of task to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds
            
        Returns:
            Final task result
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
            TaskExecutionError: If task fails
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            try:
                result = await self.get_task_status(task_id)
                
                # Check if task is complete
                if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    if result.status == TaskStatus.FAILED:
                        raise TaskExecutionError(
                            f"Task {task_id} failed: {result.error}",
                            task_id
                        )
                    return result
                
                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    # Try to cancel the task
                    await self.cancel_task(task_id)
                    raise TimeoutError(
                        f"Task {task_id} did not complete within {timeout} seconds",
                        timeout
                    )
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
            except TaskExecutionError:
                # Re-raise task execution errors
                raise
            except Exception as e:
                logger.warning(f"Error polling task {task_id}: {e}")
                await asyncio.sleep(poll_interval)
    
    async def submit_and_wait(
        self,
        task_type: str,
        payload: Dict[str, Any],
        capabilities: List[str],
        routing_strategy: Union[str, RoutingStrategy] = RoutingStrategy.CAPABILITY_MATCH,
        timeout: int = 30,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TaskResult:
        """
        Submit task and wait for completion (convenience method).
        
        Args:
            task_type: Type of task to execute
            payload: Task payload data
            capabilities: Required agent capabilities
            routing_strategy: Task routing strategy
            timeout: Task timeout in seconds
            context: Task context data
            **kwargs: Additional task parameters
            
        Returns:
            Task execution result
        """
        if isinstance(routing_strategy, str):
            routing_strategy = RoutingStrategy(routing_strategy)
        
        task_data = TaskData(
            task_type=task_type,
            input=payload,
            required_capabilities=capabilities,
            routing_strategy=routing_strategy,
            timeout_seconds=timeout,
            context=context or {},
            **kwargs
        )
        
        return await self.submit_task(task_data)
    
    async def get_runtime_stats(self) -> Dict[str, Any]:
        """
        Get runtime statistics and metrics.
        
        Returns:
            Runtime statistics
        """
        try:
            response = await self.client.get("/api/v1/stats")
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Runtime stats request timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during runtime stats request: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during runtime stats request: {e}")
    
    async def get_runtime_health(self) -> Dict[str, Any]:
        """
        Check runtime service health.
        
        Returns:
            Runtime health status
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Runtime health check timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during runtime health check: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during runtime health check: {e}")
    
    async def get_runtime_metrics(self) -> str:
        """
        Get runtime Prometheus metrics.
        
        Returns:
            Prometheus metrics as text
        """
        try:
            response = await self.client.get("/metrics")
            response.raise_for_status()
            return response.text
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Runtime metrics request timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during runtime metrics request: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during runtime metrics request: {e}")
    
    async def batch_submit_tasks(self, tasks: List[TaskData]) -> List[TaskResult]:
        """
        Submit multiple tasks concurrently.
        
        Args:
            tasks: List of tasks to submit
            
        Returns:
            List of task results
        """
        semaphore = asyncio.Semaphore(10)  # Limit concurrent submissions
        
        async def submit_single_task(task_data: TaskData) -> TaskResult:
            async with semaphore:
                return await self.submit_task(task_data)
        
        # Submit all tasks concurrently
        tasks_coroutines = [submit_single_task(task) for task in tasks]
        results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        
        # Process results and handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed result for exception
                task_id = tasks[i].task_id or f"batch_task_{i}"
                final_results.append(TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def stream_task_updates(
        self, 
        task_id: str,
        poll_interval: float = 1.0
    ):
        """
        Stream task status updates (async generator).
        
        Args:
            task_id: ID of task to monitor
            poll_interval: Polling interval in seconds
            
        Yields:
            TaskResult objects with status updates
        """
        last_status = None
        
        while True:
            try:
                result = await self.get_task_status(task_id)
                
                # Only yield if status changed
                if result.status != last_status:
                    yield result
                    last_status = result.status
                
                # Stop if task is complete
                if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    break
                
                await asyncio.sleep(poll_interval)
                
            except TaskExecutionError:
                # Task might have been deleted
                break
            except Exception as e:
                logger.warning(f"Error streaming task {task_id} updates: {e}")
                await asyncio.sleep(poll_interval)