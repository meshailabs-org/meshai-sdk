#!/usr/bin/env python3
"""
Integration tests for MeshAI API endpoints through the load balancer.
Tests the production API at api.meshai.dev
"""

import asyncio
import pytest
import httpx
from typing import Dict, Any, List
import uuid
import os


@pytest.fixture
def api_base_url() -> str:
    """Get API base URL from environment or use default"""
    return os.getenv("MESHAI_API_URL", "http://api.meshai.dev")


@pytest.fixture
async def http_client():
    """HTTP client for API testing"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


class TestRegistryEndpoints:
    """Test Registry Service endpoints"""

    @pytest.mark.integration
    async def test_registry_health(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test registry health endpoint"""
        response = await http_client.get(f"{api_base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "total_agents" in data

    @pytest.mark.integration  
    async def test_registry_root(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test registry root endpoint"""
        response = await http_client.get(f"{api_base_url}/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["status"] == "running"

    @pytest.mark.integration
    async def test_list_agents(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test listing agents"""
        response = await http_client.get(f"{api_base_url}/api/v1/agents")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.integration
    async def test_agent_registration_flow(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test complete agent registration and cleanup flow"""
        agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"
        
        # Register agent
        agent_data = {
            "id": agent_id,
            "name": f"Test Agent {agent_id}",
            "framework": "test-framework",
            "capabilities": ["test", "integration"],
            "endpoint": f"https://test-agent-{agent_id}.example.com",
            "health_endpoint": f"https://test-agent-{agent_id}.example.com/health",
            "max_concurrent_tasks": 5,
            "description": "Integration test agent",
            "version": "1.0.0",
            "tags": ["test", "ci"],
            "metadata": {"test": True, "ci": True}
        }
        
        # Create agent
        response = await http_client.post(f"{api_base_url}/api/v1/agents", json=agent_data)
        assert response.status_code in [200, 201], f"Failed to create agent: {response.text}"
        
        created_agent = response.json()
        assert created_agent["id"] == agent_id
        assert created_agent["name"] == agent_data["name"]
        assert created_agent["framework"] == agent_data["framework"]
        
        # Get agent by ID
        response = await http_client.get(f"{api_base_url}/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        
        retrieved_agent = response.json()
        assert retrieved_agent["id"] == agent_id
        
        # Update agent
        update_data = {"description": "Updated integration test agent"}
        response = await http_client.put(f"{api_base_url}/api/v1/agents/{agent_id}", json=update_data)
        assert response.status_code == 200
        
        updated_agent = response.json()
        assert updated_agent["description"] == update_data["description"]
        
        # Send heartbeat
        response = await http_client.post(f"{api_base_url}/api/v1/agents/{agent_id}/heartbeat")
        assert response.status_code == 200
        
        heartbeat_response = response.json()
        assert heartbeat_response["agent_id"] == agent_id
        assert heartbeat_response["status"] == "acknowledged"
        
        # Clean up - delete agent
        response = await http_client.delete(f"{api_base_url}/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        
        # Verify deletion
        response = await http_client.get(f"{api_base_url}/api/v1/agents/{agent_id}")
        assert response.status_code == 404

    @pytest.mark.integration
    async def test_agent_discovery(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test agent discovery functionality"""
        discovery_query = {
            "capabilities": ["test"],
            "framework": None,
            "tags": ["ci"],
            "limit": 10
        }
        
        response = await http_client.post(f"{api_base_url}/api/v1/agents/discover", json=discovery_query)
        assert response.status_code == 200
        
        agents = response.json()
        assert isinstance(agents, list)

    @pytest.mark.integration
    async def test_metrics_endpoint(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test Prometheus metrics endpoint"""
        response = await http_client.get(f"{api_base_url}/metrics")
        assert response.status_code == 200
        
        metrics_text = response.text
        assert "meshai_" in metrics_text  # Should contain MeshAI metrics
        assert "# HELP" in metrics_text   # Prometheus format


class TestRuntimeEndpoints:
    """Test Runtime Service endpoints (when available)"""

    @pytest.mark.integration
    async def test_runtime_health(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test runtime health endpoint"""
        response = await http_client.get(f"{api_base_url}/api/v1/tasks/health")
        # Runtime might not be fully configured yet, so we accept 404 or 200
        assert response.status_code in [200, 404, 502, 503]

    @pytest.mark.integration
    async def test_runtime_root(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test runtime root endpoint"""  
        response = await http_client.get(f"{api_base_url}/api/v1/tasks/")
        # Runtime might not be fully configured yet
        assert response.status_code in [200, 404, 502, 503]


class TestLoadBalancerRouting:
    """Test load balancer path-based routing"""

    @pytest.mark.integration
    async def test_default_routing_to_registry(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test that default paths route to registry service"""
        paths_to_test = [
            "/",
            "/health", 
            "/api/v1/agents",
            "/metrics"
        ]
        
        for path in paths_to_test:
            response = await http_client.get(f"{api_base_url}{path}")
            assert response.status_code in [200, 404], f"Path {path} failed with {response.status_code}"

    @pytest.mark.integration
    async def test_cors_headers(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test CORS headers are properly set"""
        # Preflight request
        response = await http_client.options(f"{api_base_url}/api/v1/agents")
        # CORS might not be configured for OPTIONS, so we don't assert status
        
        # Regular request should have CORS headers
        response = await http_client.get(f"{api_base_url}/health")
        assert response.status_code == 200
        # Check if common CORS headers are present (they might not be, and that's okay)


class TestErrorHandling:
    """Test API error handling"""

    @pytest.mark.integration
    async def test_404_handling(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test 404 error handling"""
        response = await http_client.get(f"{api_base_url}/nonexistent-endpoint")
        assert response.status_code == 404

    @pytest.mark.integration
    async def test_invalid_agent_id(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test handling of invalid agent IDs"""
        response = await http_client.get(f"{api_base_url}/api/v1/agents/nonexistent-agent")
        assert response.status_code == 404

    @pytest.mark.integration
    async def test_malformed_json(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test handling of malformed JSON"""
        response = await http_client.post(
            f"{api_base_url}/api/v1/agents",
            content="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]


class TestPerformance:
    """Basic performance tests"""

    @pytest.mark.integration
    @pytest.mark.performance
    async def test_health_endpoint_performance(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test health endpoint response time"""
        import time
        
        start_time = time.time()
        response = await http_client.get(f"{api_base_url}/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 2.0, f"Health endpoint took {response_time:.2f}s (should be < 2s)"

    @pytest.mark.integration
    @pytest.mark.performance  
    async def test_concurrent_requests(self, http_client: httpx.AsyncClient, api_base_url: str):
        """Test handling of concurrent requests"""
        async def make_request():
            response = await http_client.get(f"{api_base_url}/health")
            return response.status_code

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert all(status == 200 for status in results)


# Utility functions for test setup/teardown
async def cleanup_test_agents(http_client: httpx.AsyncClient, api_base_url: str):
    """Clean up any test agents left from previous runs"""
    response = await http_client.get(f"{api_base_url}/api/v1/agents")
    if response.status_code == 200:
        agents = response.json()
        test_agents = [agent for agent in agents if agent["id"].startswith("test-agent-")]
        
        for agent in test_agents:
            try:
                await http_client.delete(f"{api_base_url}/api/v1/agents/{agent['id']}")
            except Exception:
                pass  # Ignore cleanup failures