"""
Client for MeshAI Agent Registry Service
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import MeshConfig
from ..core.schemas import AgentInfo, AgentRegistration, DiscoveryQuery
from ..exceptions.base import (
    MeshAIError,
    AgentNotFoundError,
    RegistrationError,
    ValidationError,
    TimeoutError
)

logger = structlog.get_logger(__name__)


class RegistryClient:
    """
    Client for interacting with MeshAI Agent Registry.
    
    Provides methods for:
    - Agent registration and deregistration
    - Agent discovery and lookup
    - Health monitoring and heartbeats
    - Agent status management
    """
    
    def __init__(self, config: MeshConfig):
        self.config = config
        self.base_url = config.registry_base_url
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
    async def register_agent(self, registration: AgentRegistration) -> AgentInfo:
        """
        Register an agent with the registry.
        
        Args:
            registration: Agent registration data
            
        Returns:
            Registered agent information
            
        Raises:
            RegistrationError: If registration fails
            ValidationError: If registration data is invalid
        """
        try:
            response = await self.client.post(
                "/api/v1/agents",
                json=registration.model_dump()
            )
            
            if response.status_code == 409:
                raise RegistrationError(
                    f"Agent {registration.id} already exists",
                    registration.id
                )
            elif response.status_code == 422:
                raise ValidationError(
                    f"Invalid registration data: {response.text}",
                    value=registration.model_dump()
                )
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Agent {registration.id} registered successfully")
            return AgentInfo(**data)
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Agent registration timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise RegistrationError(f"HTTP error during registration: {e}", registration.id)
        except Exception as e:
            raise RegistrationError(f"Unexpected error during registration: {e}", registration.id)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if successful
            
        Raises:
            AgentNotFoundError: If agent doesn't exist
            MeshAIError: If unregistration fails
        """
        try:
            response = await self.client.delete(f"/api/v1/agents/{agent_id}")
            
            if response.status_code == 404:
                raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
            
            response.raise_for_status()
            
            logger.info(f"Agent {agent_id} unregistered successfully")
            return True
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Agent unregistration timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during unregistration: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during unregistration: {e}")
    
    async def get_agent(self, agent_id: str) -> AgentInfo:
        """
        Get agent information by ID.
        
        Args:
            agent_id: ID of agent to retrieve
            
        Returns:
            Agent information
            
        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        try:
            response = await self.client.get(f"/api/v1/agents/{agent_id}")
            
            if response.status_code == 404:
                raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
            
            response.raise_for_status()
            data = response.json()
            
            return AgentInfo(**data)
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Agent lookup timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during agent lookup: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during agent lookup: {e}")
    
    async def list_agents(
        self, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[AgentInfo]:
        """
        List all registered agents.
        
        Args:
            skip: Number of agents to skip
            limit: Maximum number of agents to return
            status: Filter by agent status
            
        Returns:
            List of agent information
        """
        try:
            params = {"skip": skip, "limit": limit}
            if status:
                params["status"] = status
                
            response = await self.client.get("/api/v1/agents", params=params)
            response.raise_for_status()
            
            data = response.json()
            return [AgentInfo(**agent) for agent in data]
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Agent listing timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during agent listing: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during agent listing: {e}")
    
    async def discover_agents(self, query: DiscoveryQuery) -> List[AgentInfo]:
        """
        Discover agents based on capabilities and criteria.
        
        Args:
            query: Discovery query parameters
            
        Returns:
            List of matching agents
        """
        try:
            response = await self.client.post(
                "/api/v1/agents/discover",
                json=query.model_dump()
            )
            response.raise_for_status()
            
            data = response.json()
            agents = [AgentInfo(**agent) for agent in data]
            
            logger.debug(f"Discovered {len(agents)} agents matching query")
            return agents
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Agent discovery timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during agent discovery: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during agent discovery: {e}")
    
    async def discover_by_capabilities(
        self, 
        capabilities: List[str],
        framework: Optional[str] = None,
        exclude_unhealthy: bool = True
    ) -> List[AgentInfo]:
        """
        Discover agents by required capabilities.
        
        Args:
            capabilities: Required capabilities
            framework: Preferred framework
            exclude_unhealthy: Whether to exclude unhealthy agents
            
        Returns:
            List of matching agents
        """
        query = DiscoveryQuery(
            capabilities=capabilities,
            framework=framework,
            include_offline=not exclude_unhealthy
        )
        
        return await self.discover_agents(query)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def heartbeat(self, agent_id: str) -> Dict[str, Any]:
        """
        Send heartbeat for an agent.
        
        Args:
            agent_id: ID of agent sending heartbeat
            
        Returns:
            Heartbeat response
            
        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        try:
            response = await self.client.post(f"/api/v1/agents/{agent_id}/heartbeat")
            
            if response.status_code == 404:
                raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
            
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Heartbeat timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during heartbeat: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during heartbeat: {e}")
    
    async def update_agent(
        self, 
        agent_id: str,
        updates: Dict[str, Any]
    ) -> AgentInfo:
        """
        Update agent information.
        
        Args:
            agent_id: ID of agent to update
            updates: Fields to update
            
        Returns:
            Updated agent information
            
        Raises:
            AgentNotFoundError: If agent doesn't exist
            ValidationError: If update data is invalid
        """
        try:
            response = await self.client.patch(
                f"/api/v1/agents/{agent_id}",
                json=updates
            )
            
            if response.status_code == 404:
                raise AgentNotFoundError(f"Agent {agent_id} not found", agent_id)
            elif response.status_code == 422:
                raise ValidationError(
                    f"Invalid update data: {response.text}",
                    value=updates
                )
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Agent {agent_id} updated successfully")
            return AgentInfo(**data)
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Agent update timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during agent update: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during agent update: {e}")
    
    async def get_registry_health(self) -> Dict[str, Any]:
        """
        Check registry service health.
        
        Returns:
            Registry health status
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Registry health check timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during registry health check: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during registry health check: {e}")
    
    async def get_registry_metrics(self) -> str:
        """
        Get registry Prometheus metrics.
        
        Returns:
            Prometheus metrics as text
        """
        try:
            response = await self.client.get("/metrics")
            response.raise_for_status()
            return response.text
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Registry metrics request timed out after {self.config.default_timeout}s")
        except httpx.HTTPError as e:
            raise MeshAIError(f"HTTP error during registry metrics request: {e}")
        except Exception as e:
            raise MeshAIError(f"Unexpected error during registry metrics request: {e}")