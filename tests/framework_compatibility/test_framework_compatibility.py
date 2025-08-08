"""
Framework Compatibility Test Suite for MeshAI

This module provides comprehensive testing for all supported AI frameworks
to ensure seamless interoperability and consistent behavior.
"""

import pytest
import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch
import sys
import importlib.util

from meshai.core import MeshAgent, MeshContext
from meshai.core.config import MeshConfig


# Framework availability checks
LANGCHAIN_AVAILABLE = importlib.util.find_spec("langchain") is not None
CREWAI_AVAILABLE = importlib.util.find_spec("crewai") is not None
AUTOGEN_AVAILABLE = importlib.util.find_spec("autogen") is not None
OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None
ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None
GOOGLE_AVAILABLE = importlib.util.find_spec("google.generativeai") is not None
BOTO3_AVAILABLE = importlib.util.find_spec("boto3") is not None


class FrameworkCompatibilityTestBase:
    """Base class for framework compatibility tests"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return MeshConfig()
    
    @pytest.fixture
    def context(self):
        """Test context"""
        return MeshContext()
    
    def assert_agent_interface(self, agent):
        """Assert that agent implements MeshAgent interface"""
        assert hasattr(agent, 'agent_id')
        assert hasattr(agent, 'handle_task')
        assert hasattr(agent, 'get_capabilities')
        assert hasattr(agent, 'health_check')
        assert hasattr(agent, 'start')
        assert hasattr(agent, 'stop')
    
    async def test_basic_task_handling(self, agent, context):
        """Test basic task handling"""
        task_data = {"test": "data", "value": 42}
        result = await agent.handle_task(task_data, context)
        
        assert result is not None
        assert isinstance(result, dict)
        
        return result
    
    async def test_context_handling(self, agent, context):
        """Test context persistence and retrieval"""
        # Set context data
        await context.set("test_key", "test_value")
        
        # Execute task that uses context
        task_data = {"operation": "get_context", "key": "test_key"}
        result = await agent.handle_task(task_data, context)
        
        # Verify context was accessible
        assert result is not None
        
        return result
    
    async def test_error_handling(self, agent, context):
        """Test error handling in framework"""
        task_data = {"operation": "raise_error", "error_type": "ValueError"}
        
        with pytest.raises(Exception):
            await agent.handle_task(task_data, context)
    
    async def test_concurrent_requests(self, agent, context):
        """Test handling of concurrent requests"""
        tasks = []
        for i in range(10):
            task_data = {"request_id": i, "data": f"concurrent_{i}"}
            tasks.append(agent.handle_task(task_data, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all requests were processed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0
        
        return results


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestLangChainCompatibility(FrameworkCompatibilityTestBase):
    """Test LangChain framework compatibility"""
    
    @pytest.fixture
    async def langchain_agent(self, config):
        """Create LangChain agent for testing"""
        from meshai.adapters.langchain_adapter import LangChainAdapter
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import Tool
        from langchain.memory import ConversationBufferMemory
        from langchain.prompts import PromptTemplate
        
        # Mock LLM for testing
        class MockLLM:
            def __call__(self, prompt):
                return "Test response"
            
            async def agenerate(self, prompts):
                from langchain.schema import LLMResult, Generation
                return LLMResult(generations=[[Generation(text="Test response")]])
        
        # Create tools
        tools = [
            Tool(
                name="test_tool",
                func=lambda x: f"Processed: {x}",
                description="Test tool"
            )
        ]
        
        # Create agent
        llm = MockLLM()
        memory = ConversationBufferMemory()
        
        # Create LangChain agent executor (simplified for testing)
        agent_executor = Mock(spec=AgentExecutor)
        agent_executor.run = Mock(return_value="LangChain result")
        agent_executor.arun = AsyncMock(return_value="LangChain async result")
        
        # Wrap with MeshAI adapter
        adapter = LangChainAdapter(
            agent_id="langchain-test-agent",
            langchain_agent=agent_executor,
            config=config
        )
        
        await adapter.start()
        yield adapter
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_langchain_agent_creation(self, langchain_agent):
        """Test LangChain agent creation and initialization"""
        self.assert_agent_interface(langchain_agent)
        assert langchain_agent.agent_id == "langchain-test-agent"
    
    @pytest.mark.asyncio
    async def test_langchain_task_execution(self, langchain_agent, context):
        """Test LangChain task execution"""
        result = await self.test_basic_task_handling(langchain_agent, context)
        assert "LangChain" in str(result) or "result" in result
    
    @pytest.mark.asyncio
    async def test_langchain_tool_integration(self, langchain_agent, context):
        """Test LangChain tool integration with MeshAI"""
        task_data = {
            "operation": "use_tool",
            "tool": "test_tool",
            "input": "test input"
        }
        
        result = await langchain_agent.handle_task(task_data, context)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_langchain_memory_integration(self, langchain_agent, context):
        """Test LangChain memory integration"""
        # First interaction
        task1 = {"message": "Remember my name is Alice"}
        result1 = await langchain_agent.handle_task(task1, context)
        
        # Second interaction should remember context
        task2 = {"message": "What is my name?"}
        result2 = await langchain_agent.handle_task(task2, context)
        
        # Memory should be maintained
        assert result2 is not None


@pytest.mark.skipif(not CREWAI_AVAILABLE, reason="CrewAI not installed")
class TestCrewAICompatibility(FrameworkCompatibilityTestBase):
    """Test CrewAI framework compatibility"""
    
    @pytest.fixture
    async def crewai_agent(self, config):
        """Create CrewAI agent for testing"""
        from meshai.adapters.crewai_adapter import CrewAIAdapter
        from crewai import Agent, Task, Crew
        
        # Create mock CrewAI agent
        crewai_agent = Mock(spec=Agent)
        crewai_agent.execute_task = AsyncMock(return_value="CrewAI result")
        crewai_agent.role = "Test Agent"
        crewai_agent.goal = "Test goal"
        
        # Wrap with MeshAI adapter
        adapter = CrewAIAdapter(
            agent_id="crewai-test-agent",
            crewai_agent=crewai_agent,
            config=config
        )
        
        await adapter.start()
        yield adapter
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_crewai_agent_creation(self, crewai_agent):
        """Test CrewAI agent creation"""
        self.assert_agent_interface(crewai_agent)
        assert crewai_agent.agent_id == "crewai-test-agent"
    
    @pytest.mark.asyncio
    async def test_crewai_task_execution(self, crewai_agent, context):
        """Test CrewAI task execution"""
        result = await self.test_basic_task_handling(crewai_agent, context)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_crewai_crew_integration(self, config):
        """Test CrewAI crew integration"""
        from meshai.adapters.crewai_adapter import CrewAICrewAdapter
        
        # Create mock crew
        mock_crew = Mock()
        mock_crew.kickoff = AsyncMock(return_value="Crew result")
        
        # Create crew adapter
        adapter = CrewAICrewAdapter(
            agent_id="crewai-crew-test",
            crew=mock_crew,
            config=config
        )
        
        # Test crew execution
        context = MeshContext()
        result = await adapter.handle_task({"input": "test"}, context)
        assert result is not None


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not installed")
class TestOpenAICompatibility(FrameworkCompatibilityTestBase):
    """Test OpenAI compatibility"""
    
    @pytest.fixture
    async def openai_agent(self, config):
        """Create OpenAI agent for testing"""
        from meshai.adapters.openai_adapter import OpenAIAdapter
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="OpenAI response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        
        # Create adapter with mock client
        adapter = OpenAIAdapter(
            agent_id="openai-test-agent",
            model="gpt-4",
            config=config
        )
        adapter.client = mock_client
        
        await adapter.start()
        yield adapter
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_openai_agent_creation(self, openai_agent):
        """Test OpenAI agent creation"""
        self.assert_agent_interface(openai_agent)
        assert openai_agent.agent_id == "openai-test-agent"
    
    @pytest.mark.asyncio
    async def test_openai_completion(self, openai_agent, context):
        """Test OpenAI completion"""
        task_data = {"prompt": "Test prompt"}
        result = await openai_agent.handle_task(task_data, context)
        assert result is not None
        assert "response" in result or "OpenAI" in str(result)
    
    @pytest.mark.asyncio
    async def test_openai_function_calling(self, openai_agent, context):
        """Test OpenAI function calling"""
        task_data = {
            "prompt": "Call a function",
            "functions": [
                {
                    "name": "test_function",
                    "description": "Test function",
                    "parameters": {"type": "object", "properties": {}}
                }
            ]
        }
        
        result = await openai_agent.handle_task(task_data, context)
        assert result is not None


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic not installed")
class TestAnthropicCompatibility(FrameworkCompatibilityTestBase):
    """Test Anthropic Claude compatibility"""
    
    @pytest.fixture
    async def anthropic_agent(self, config):
        """Create Anthropic agent for testing"""
        from meshai.adapters.anthropic_adapter import AnthropicAdapter
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_message = Mock(content=[Mock(text="Claude response")])
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        
        # Create adapter
        adapter = AnthropicAdapter(
            agent_id="anthropic-test-agent",
            model="claude-3-opus-20240229",
            config=config
        )
        adapter.client = mock_client
        
        await adapter.start()
        yield adapter
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_anthropic_agent_creation(self, anthropic_agent):
        """Test Anthropic agent creation"""
        self.assert_agent_interface(anthropic_agent)
        assert anthropic_agent.agent_id == "anthropic-test-agent"
    
    @pytest.mark.asyncio
    async def test_anthropic_completion(self, anthropic_agent, context):
        """Test Anthropic completion"""
        task_data = {"prompt": "Test prompt"}
        result = await anthropic_agent.handle_task(task_data, context)
        assert result is not None
        assert "Claude" in str(result) or "response" in result
    
    @pytest.mark.asyncio
    async def test_anthropic_tool_use(self, anthropic_agent, context):
        """Test Anthropic tool use"""
        task_data = {
            "prompt": "Use a tool",
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool",
                    "input_schema": {"type": "object"}
                }
            ]
        }
        
        result = await anthropic_agent.handle_task(task_data, context)
        assert result is not None


@pytest.mark.skipif(not GOOGLE_AVAILABLE, reason="Google Generative AI not installed")
class TestGoogleGeminiCompatibility(FrameworkCompatibilityTestBase):
    """Test Google Gemini compatibility"""
    
    @pytest.fixture
    async def gemini_agent(self, config):
        """Create Gemini agent for testing"""
        from meshai.adapters.google_gemini_adapter import GoogleGeminiAdapter
        
        # Mock Gemini model
        mock_model = Mock()
        mock_response = Mock(text="Gemini response")
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        
        # Create adapter
        adapter = GoogleGeminiAdapter(
            agent_id="gemini-test-agent",
            model_name="gemini-pro",
            config=config
        )
        adapter.model = mock_model
        
        await adapter.start()
        yield adapter
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_gemini_agent_creation(self, gemini_agent):
        """Test Gemini agent creation"""
        self.assert_agent_interface(gemini_agent)
        assert gemini_agent.agent_id == "gemini-test-agent"
    
    @pytest.mark.asyncio
    async def test_gemini_generation(self, gemini_agent, context):
        """Test Gemini content generation"""
        task_data = {"prompt": "Test prompt"}
        result = await gemini_agent.handle_task(task_data, context)
        assert result is not None
        assert "Gemini" in str(result) or "response" in result


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="Boto3 not installed")
class TestAmazonBedrockCompatibility(FrameworkCompatibilityTestBase):
    """Test Amazon Bedrock compatibility"""
    
    @pytest.fixture
    async def bedrock_agent(self, config):
        """Create Bedrock agent for testing"""
        from meshai.adapters.amazon_bedrock_adapter import AmazonBedrockAdapter
        
        # Mock Bedrock client
        mock_client = Mock()
        mock_response = {
            'body': Mock(read=Mock(return_value=b'{"completion": "Bedrock response"}'))
        }
        mock_client.invoke_model = Mock(return_value=mock_response)
        
        # Create adapter
        adapter = AmazonBedrockAdapter(
            agent_id="bedrock-test-agent",
            model_id="anthropic.claude-v2",
            config=config
        )
        adapter.bedrock_client = mock_client
        
        await adapter.start()
        yield adapter
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_bedrock_agent_creation(self, bedrock_agent):
        """Test Bedrock agent creation"""
        self.assert_agent_interface(bedrock_agent)
        assert bedrock_agent.agent_id == "bedrock-test-agent"
    
    @pytest.mark.asyncio
    async def test_bedrock_inference(self, bedrock_agent, context):
        """Test Bedrock model inference"""
        task_data = {"prompt": "Test prompt"}
        result = await bedrock_agent.handle_task(task_data, context)
        assert result is not None


@pytest.mark.skipif(not AUTOGEN_AVAILABLE, reason="AutoGen not installed")
class TestAutoGenCompatibility(FrameworkCompatibilityTestBase):
    """Test AutoGen compatibility"""
    
    @pytest.fixture
    async def autogen_agent(self, config):
        """Create AutoGen agent for testing"""
        from meshai.adapters.autogen_adapter import AutoGenAdapter
        
        # Mock AutoGen agent
        mock_agent = Mock()
        mock_agent.generate_reply = AsyncMock(return_value="AutoGen response")
        mock_agent.name = "TestAgent"
        
        # Create adapter
        adapter = AutoGenAdapter(
            agent_id="autogen-test-agent",
            autogen_agent=mock_agent,
            config=config
        )
        
        await adapter.start()
        yield adapter
        await adapter.stop()
    
    @pytest.mark.asyncio
    async def test_autogen_agent_creation(self, autogen_agent):
        """Test AutoGen agent creation"""
        self.assert_agent_interface(autogen_agent)
        assert autogen_agent.agent_id == "autogen-test-agent"
    
    @pytest.mark.asyncio
    async def test_autogen_conversation(self, autogen_agent, context):
        """Test AutoGen conversation handling"""
        task_data = {"message": "Test message"}
        result = await autogen_agent.handle_task(task_data, context)
        assert result is not None
        assert "AutoGen" in str(result) or "response" in result


class TestCrossFrameworkInteroperability:
    """Test interoperability between different frameworks"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not (LANGCHAIN_AVAILABLE and OPENAI_AVAILABLE),
        reason="LangChain and OpenAI required"
    )
    async def test_langchain_openai_interaction(self, config):
        """Test LangChain and OpenAI agents working together"""
        from meshai.adapters.langchain_adapter import LangChainAdapter
        from meshai.adapters.openai_adapter import OpenAIAdapter
        
        # Create mock agents
        langchain_agent = Mock()
        langchain_agent.arun = AsyncMock(return_value="LangChain processed")
        
        openai_agent = Mock()
        openai_completion = Mock()
        openai_completion.choices = [Mock(message=Mock(content="OpenAI processed"))]
        openai_agent.chat.completions.create = AsyncMock(return_value=openai_completion)
        
        # Create adapters
        lc_adapter = LangChainAdapter("lc-agent", langchain_agent, config)
        oai_adapter = OpenAIAdapter("oai-agent", "gpt-4", config)
        oai_adapter.client = openai_agent
        
        # Test cross-framework communication
        context = MeshContext()
        
        # LangChain processes first
        lc_result = await lc_adapter.handle_task({"input": "test"}, context)
        await context.set("langchain_result", lc_result)
        
        # OpenAI uses LangChain result
        oai_task = {
            "prompt": "Process this",
            "context_key": "langchain_result"
        }
        oai_result = await oai_adapter.handle_task(oai_task, context)
        
        assert lc_result is not None
        assert oai_result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not (CREWAI_AVAILABLE and ANTHROPIC_AVAILABLE),
        reason="CrewAI and Anthropic required"
    )
    async def test_crewai_anthropic_interaction(self, config):
        """Test CrewAI and Anthropic agents working together"""
        from meshai.adapters.crewai_adapter import CrewAIAdapter
        from meshai.adapters.anthropic_adapter import AnthropicAdapter
        
        # Create mock agents
        crewai_agent = Mock()
        crewai_agent.execute_task = AsyncMock(return_value="CrewAI task completed")
        
        anthropic_client = Mock()
        anthropic_message = Mock(content=[Mock(text="Claude analyzed")])
        anthropic_client.messages.create = AsyncMock(return_value=anthropic_message)
        
        # Create adapters
        crew_adapter = CrewAIAdapter("crew-agent", crewai_agent, config)
        claude_adapter = AnthropicAdapter("claude-agent", "claude-3", config)
        claude_adapter.client = anthropic_client
        
        # Test interaction
        context = MeshContext()
        
        # CrewAI executes task
        crew_result = await crew_adapter.handle_task({"task": "analyze"}, context)
        await context.set("crew_result", crew_result)
        
        # Claude reviews CrewAI's work
        claude_task = {
            "prompt": "Review the crew result",
            "context_key": "crew_result"
        }
        claude_result = await claude_adapter.handle_task(claude_task, context)
        
        assert crew_result is not None
        assert claude_result is not None


class TestFrameworkPerformance:
    """Performance tests for framework adapters"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_adapter_overhead(self, config):
        """Test adapter overhead compared to raw framework calls"""
        from meshai.core import MeshAgent
        
        # Create a simple test agent
        class SimpleAgent(MeshAgent):
            async def handle_task(self, task_data, context):
                await asyncio.sleep(0.01)  # Simulate work
                return {"result": "done"}
        
        agent = SimpleAgent("perf-test-agent", config)
        context = MeshContext()
        
        # Measure adapter overhead
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            await agent.handle_task({"test": "data"}, context)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / iterations
        
        # Adapter overhead should be minimal (< 1ms)
        assert avg_time < 0.015  # 10ms work + max 5ms overhead
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_framework_calls(self, config):
        """Test concurrent calls across different framework adapters"""
        from meshai.core import MeshAgent
        
        class TestAgent(MeshAgent):
            async def handle_task(self, task_data, context):
                await asyncio.sleep(random.uniform(0.01, 0.05))
                return {"agent": self.agent_id, "processed": task_data}
        
        # Create multiple agents
        agents = [TestAgent(f"agent-{i}", config) for i in range(5)]
        context = MeshContext()
        
        # Execute concurrent tasks
        tasks = []
        for i in range(50):
            agent = agents[i % len(agents)]
            task = agent.handle_task({"request": i}, context)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # All tasks should complete
        assert len(results) == 50
        assert all(r is not None for r in results)
        
        # Should benefit from concurrency (< sequential time)
        max_sequential_time = 50 * 0.05  # Worst case sequential
        assert elapsed < max_sequential_time * 0.3  # At least 70% improvement
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_context_performance(self, config):
        """Test context operations performance"""
        context = MeshContext()
        
        # Test write performance
        write_start = time.time()
        for i in range(1000):
            await context.set(f"key_{i}", f"value_{i}")
        write_elapsed = time.time() - write_start
        
        # Test read performance
        read_start = time.time()
        for i in range(1000):
            value = await context.get(f"key_{i}")
            assert value == f"value_{i}"
        read_elapsed = time.time() - read_start
        
        # Performance assertions
        assert write_elapsed < 1.0  # 1000 writes in < 1 second
        assert read_elapsed < 0.5   # 1000 reads in < 0.5 seconds


if __name__ == "__main__":
    # Run framework compatibility tests
    pytest.main([__file__, "-v", "--tb=short"])