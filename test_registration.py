#!/usr/bin/env python3
"""
Test script to check if the MetricsCollector bug is fixed
"""
import requests
import json

def test_agent_registration():
    """Test agent registration to see if metrics error is resolved"""
    
    registry_url = "https://meshai-registry-zype6jntia-uc.a.run.app"
    
    # Test agent data
    agent_data = {
        "id": "test-agent-metrics-fix",
        "name": "Test Agent for Metrics Fix",
        "framework": "custom",
        "capabilities": ["test", "debug"],
        "endpoint": "http://test-agent.example.com:8000",
        "health_endpoint": "http://test-agent.example.com:8000/health",
        "max_concurrent_tasks": 5,
        "description": "Test agent to verify metrics bug fix",
        "version": "1.0.0",
        "tags": ["test", "debug"],
        "metadata": {"test": True}
    }
    
    try:
        # Try to register the agent
        print("Testing agent registration with metrics fix...")
        response = requests.post(
            f"{registry_url}/api/v1/agents",
            json=agent_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code in [200, 201]:
            print("✅ Agent registration successful! Metrics bug appears to be fixed.")
            
            # Clean up by unregistering the agent
            cleanup_response = requests.delete(
                f"{registry_url}/api/v1/agents/{agent_data['id']}",
                timeout=30
            )
            print(f"Cleanup status: {cleanup_response.status_code}")
            
        else:
            print("❌ Agent registration failed.")
            if "total_registrations" in response.text:
                print("⚠️  The metrics error is still present.")
            else:
                print("ℹ️  Different error - metrics issue may be resolved.")
                
    except Exception as e:
        print(f"❌ Error testing registration: {e}")

if __name__ == "__main__":
    test_agent_registration()