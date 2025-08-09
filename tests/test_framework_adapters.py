"""
Comprehensive tests for all framework adapters
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.exceptions.base import TaskExecutionError, ConfigurationError


class TestLangChainAdapter:
    """Test LangChain adapter functionality"""

    @pytest.fixture
    def mock_langchain_agent(self):
        with patch('meshai.adapters.langchain_adapter.AgentExecutor'):
            mock_agent = Mock()
            mock_agent.invoke = Mock(return_value={"output": "LangChain response"})
            mock_agent.ainvoke = AsyncMock(return_value={"output": "Async LangChain response"})
            yield mock_agent

    @pytest.fixture
    def langchain_adapter(self, mock_langchain_agent):
        from meshai.adapters.langchain_adapter import LangChainMeshAgent
        return LangChainMeshAgent(
            langchain_agent=mock_langchain_agent,
            agent_id="test-langchain",
            name="Test LangChain Agent",
            capabilities=["text-generation", "reasoning"]
        )

    @pytest.mark.asyncio
    async def test_langchain_handle_task(self, langchain_adapter):
        """Test LangChain task handling"""
        context = MeshContext()
        task_data = TaskData(input="Test query", parameters={})
        
        result = await langchain_adapter.handle_task(task_data, context)
        
        assert "result" in result
        assert result["type"] == "langchain_response"

    @pytest.mark.asyncio
    async def test_langchain_tool_integration(self, langchain_adapter):
        """Test MeshAI tool integration"""
        # Mock the invoke_agent method
        langchain_adapter.invoke_agent = AsyncMock(return_value=Mock(
            status="completed", 
            result="Tool response"
        ))
        
        # Test tool execution
        from meshai.adapters.langchain_adapter import MeshAITool
        tool = MeshAITool(langchain_adapter)
        result = await tool._arun("test query", "coding")
        
        assert "Tool response" in result

    def test_langchain_add_tool(self, langchain_adapter):
        """Test adding tools to LangChain agent"""
        def test_function(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        langchain_adapter.add_tool(
            test_function, 
            name="test_tool", 
            description="A test tool"
        )
        
        assert len(langchain_adapter.tools) > 1  # Should have MeshAI tool + test tool


class TestCrewAIAdapter:
    """Test CrewAI adapter functionality"""

    @pytest.fixture
    def mock_crewai_agent(self):
        with patch('meshai.adapters.crewai_adapter.CrewAIAgent'):
            mock_agent = Mock()
            mock_agent.name = "TestAgent"
            mock_agent.role = "Assistant"
            mock_agent.tools = []
            yield mock_agent

    @pytest.fixture
    def crewai_adapter(self, mock_crewai_agent):
        from meshai.adapters.crewai_adapter import CrewAIMeshAgent
        return CrewAIMeshAgent(
            crewai_component=mock_crewai_agent,
            agent_id="test-crewai",
            name="Test CrewAI Agent",
            capabilities=["collaboration", "task-management"]
        )

    @pytest.mark.asyncio
    async def test_crewai_handle_task(self, crewai_adapter):
        """Test CrewAI task handling"""
        context = MeshContext()
        task_data = TaskData(input="Test crew task", parameters={})
        
        with patch.object(crewai_adapter, '_execute_agent') as mock_execute:
            mock_execute.return_value = {
                "result": "CrewAI response",
                "type": "agent_execution"
            }
            
            result = await crewai_adapter.handle_task(task_data, context)
            
            assert "result" in result
            assert result["type"] == "agent_execution"

    def test_crewai_component_info(self, crewai_adapter):
        """Test getting CrewAI component information"""
        info = crewai_adapter.get_crew_info()
        
        assert info["type"] == "agent"
        assert "role" in info


class TestAutoGenAdapter:
    """Test AutoGen adapter functionality"""

    @pytest.fixture
    def mock_autogen_agent(self):
        with patch('meshai.adapters.autogen_adapter.ConversableAgent'):
            mock_agent = Mock()
            mock_agent.name = "TestAutoGenAgent"
            mock_agent.system_message = "You are a helpful assistant"
            yield mock_agent

    @pytest.fixture
    def autogen_adapter(self, mock_autogen_agent):
        from meshai.adapters.autogen_adapter import AutoGenMeshAgent
        return AutoGenMeshAgent(
            autogen_component=mock_autogen_agent,
            agent_id="test-autogen",
            name="Test AutoGen Agent",
            capabilities=["conversation", "multi-agent"]
        )

    @pytest.mark.asyncio
    async def test_autogen_handle_task(self, autogen_adapter):
        """Test AutoGen task handling"""
        context = MeshContext()
        task_data = TaskData(input="Test AutoGen query", parameters={})
        
        with patch.object(autogen_adapter, '_execute_agent') as mock_execute:
            mock_execute.return_value = {
                "result": "AutoGen response",
                "type": "autogen_agent_response"
            }
            
            result = await autogen_adapter.handle_task(task_data, context)
            
            assert "result" in result
            assert result["type"] == "autogen_agent_response"

    @pytest.mark.asyncio
    async def test_autogen_capabilities_detection(self, autogen_adapter):
        """Test dynamic capability detection"""
        capabilities = await autogen_adapter.get_agent_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) >= 2  # Should have at least base capabilities

    def test_autogen_component_info(self, autogen_adapter):
        """Test getting AutoGen component information"""
        info = autogen_adapter.get_component_info()
        
        assert info["type"] == "agent"
        assert "name" in info


class TestOpenAIAdapter:
    """Test OpenAI adapter functionality"""

    @pytest.fixture
    def mock_openai_response(self):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "OpenAI response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25
        return mock_response

    @pytest.fixture
    def openai_adapter(self):
        with patch('meshai.adapters.openai_adapter.AsyncOpenAI') as mock_client:
            from meshai.adapters.openai_adapter import OpenAIMeshAgent
            
            # Mock API key
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                adapter = OpenAIMeshAgent(
                    model="gpt-3.5-turbo",
                    agent_id="test-openai",
                    name="Test OpenAI Agent",
                    capabilities=["text-generation", "coding"]
                )
                return adapter

    @pytest.mark.asyncio
    async def test_openai_handle_task(self, openai_adapter, mock_openai_response):
        """Test OpenAI task handling"""
        context = MeshContext()
        task_data = TaskData(input="Test OpenAI query", parameters={})
        
        # Mock the client response
        openai_adapter.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        
        result = await openai_adapter.handle_task(task_data, context)
        
        assert "result" in result
        assert result["type"] == "openai_response"
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_openai_tool_calling(self, openai_adapter):
        """Test OpenAI function/tool calling"""
        # Mock tool call response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.tool_calls = [Mock()]
        mock_response.choices[0].message.tool_calls[0].function = Mock()
        mock_response.choices[0].message.tool_calls[0].function.name = "invoke_meshai_agent"
        mock_response.choices[0].message.tool_calls[0].function.arguments = '{"capabilities": ["coding"], "task": "Write code"}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25
        
        openai_adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)
        openai_adapter.invoke_agent = AsyncMock(return_value=Mock(
            status="completed",
            result="Code written successfully"
        ))
        
        context = MeshContext()
        task_data = TaskData(input="Write some code", parameters={})
        
        result = await openai_adapter.handle_task(task_data, context)
        
        assert "tools_used" in result
        assert result["tools_used"] is True

    def test_openai_model_info(self, openai_adapter):
        """Test getting OpenAI model information"""
        info = openai_adapter.get_model_info()
        
        assert info["model"] == "gpt-3.5-turbo"
        assert info["provider"] == "openai"
        assert info["supports_tools"] is True


class TestAnthropicAdapter:
    """Test Anthropic Claude adapter functionality"""

    @pytest.fixture
    def mock_anthropic_response(self):
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude response"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 15
        return mock_response

    @pytest.fixture
    def anthropic_adapter(self):
        with patch('meshai.adapters.anthropic_adapter.AsyncAnthropic'):
            from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
            
            # Mock API key
            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
                adapter = AnthropicMeshAgent(
                    model="claude-3-sonnet-20240229",
                    agent_id="test-anthropic",
                    name="Test Claude Agent",
                    capabilities=["reasoning", "analysis"]
                )
                return adapter

    @pytest.mark.asyncio
    async def test_anthropic_handle_task(self, anthropic_adapter, mock_anthropic_response):
        """Test Anthropic task handling"""
        context = MeshContext()
        task_data = TaskData(input="Test Claude query", parameters={})
        
        # Mock the client response
        anthropic_adapter.client.messages.create = AsyncMock(return_value=mock_anthropic_response)
        
        result = await anthropic_adapter.handle_task(task_data, context)
        
        assert "result" in result
        assert result["type"] == "anthropic_response"
        assert "usage" in result

    def test_anthropic_add_tool(self, anthropic_adapter):
        """Test adding tools to Anthropic agent"""
        tool_def = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            }
        }
        
        anthropic_adapter.add_tool(tool_def)
        
        assert len(anthropic_adapter.tools) == 1

    def test_anthropic_model_info(self, anthropic_adapter):
        """Test getting Anthropic model information"""
        info = anthropic_adapter.get_model_info()
        
        assert info["model"] == "claude-3-sonnet-20240229"
        assert info["provider"] == "anthropic"


class TestGoogleAdapter:
    """Test Google Gemini adapter functionality"""

    @pytest.fixture
    def mock_google_response(self):
        mock_response = Mock()
        mock_response.text = "Gemini response"
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Gemini response"
        mock_response.candidates[0].safety_ratings = []
        return mock_response

    @pytest.fixture
    def google_adapter(self):
        with patch('meshai.adapters.google_adapter.genai') as mock_genai:
            from meshai.adapters.google_adapter import GoogleMeshAgent
            
            # Mock API key and model
            with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model
                
                adapter = GoogleMeshAgent(
                    model="gemini-pro",
                    agent_id="test-google",
                    name="Test Gemini Agent",
                    capabilities=["multimodal", "reasoning"]
                )
                return adapter

    @pytest.mark.asyncio
    async def test_google_handle_task(self, google_adapter, mock_google_response):
        """Test Google Gemini task handling"""
        context = MeshContext()
        task_data = TaskData(input="Test Gemini query", parameters={})
        
        # Mock the generate content method
        with patch.object(google_adapter, '_generate_content_async') as mock_generate:
            mock_generate.return_value = mock_google_response
            
            result = await google_adapter.handle_task(task_data, context)
            
            assert "result" in result
            assert result["type"] == "google_response"
            assert "safety_ratings" in result

    def test_google_model_info(self, google_adapter):
        """Test getting Google model information"""
        info = google_adapter.get_model_info()
        
        assert info["model"] == "gemini-pro"
        assert info["provider"] == "google"


class TestBedrockAdapter:
    """Test Amazon Bedrock adapter functionality"""

    @pytest.fixture
    def mock_bedrock_response(self):
        return {
            "content": [{"text": "Bedrock response"}],
            "usage": {"input_tokens": 10, "output_tokens": 15}
        }

    @pytest.fixture
    def bedrock_adapter(self):
        with patch('meshai.adapters.amazon_adapter.boto3'):
            from meshai.adapters.amazon_adapter import BedrockMeshAgent
            
            adapter = BedrockMeshAgent(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                agent_id="test-bedrock",
                name="Test Bedrock Agent",
                capabilities=["text-generation", "reasoning"]
            )
            return adapter

    @pytest.mark.asyncio
    async def test_bedrock_handle_task(self, bedrock_adapter, mock_bedrock_response):
        """Test Bedrock task handling"""
        context = MeshContext()
        task_data = TaskData(input="Test Bedrock query", parameters={})
        
        # Mock the client response
        mock_response = {
            "body": Mock()
        }
        mock_response["body"].read.return_value = mock_bedrock_response
        
        bedrock_adapter.bedrock_client.invoke_model = Mock(return_value=mock_response)
        
        # Mock JSON parsing
        with patch('json.loads', return_value=mock_bedrock_response):
            result = await bedrock_adapter.handle_task(task_data, context)
            
            assert "result" in result
            assert result["type"] == "bedrock_response"

    def test_bedrock_model_info(self, bedrock_adapter):
        """Test getting Bedrock model information"""
        info = bedrock_adapter.get_model_info()
        
        assert info["provider"] == "amazon-bedrock"
        assert "model_family" in info


class TestAdapterIntegration:
    """Test adapter integration features"""

    @pytest.mark.asyncio
    async def test_all_adapters_available(self):
        """Test that all adapters are properly registered"""
        from meshai.adapters import get_available_adapters, is_adapter_available
        
        adapters = get_available_adapters()
        
        # Check that main frameworks are available (when dependencies installed)
        expected_frameworks = ["langchain", "crewai", "autogen", "openai", "anthropic", "google", "amazon"]
        
        for framework in expected_frameworks:
            # Some may not be available due to missing dependencies in test environment
            if is_adapter_available(framework):
                assert framework in adapters

    @pytest.mark.asyncio
    async def test_context_sharing(self):
        """Test context sharing between different adapters"""
        context = MeshContext()
        
        # Simulate conversation history
        await context.set("conversation_history", [
            {"type": "human", "content": "Hello", "timestamp": "2024-01-01T00:00:00"},
            {"type": "ai", "content": "Hi there!", "timestamp": "2024-01-01T00:01:00", "source": "openai"}
        ])
        
        # Test that context is preserved
        history = await context.get("conversation_history", [])
        assert len(history) == 2
        assert history[0]["type"] == "human"
        assert history[1]["source"] == "openai"

    def test_error_handling(self):
        """Test error handling in adapters"""
        # Test configuration errors
        with pytest.raises(ConfigurationError):
            from meshai.adapters.openai_adapter import OpenAIMeshAgent
            OpenAIMeshAgent(api_key=None)  # Should raise error without API key

    @pytest.mark.asyncio
    async def test_adapter_task_execution_failure(self):
        """Test adapter behavior when task execution fails"""
        from meshai.adapters.openai_adapter import OpenAIMeshAgent
        
        with patch('meshai.adapters.openai_adapter.AsyncOpenAI'):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                adapter = OpenAIMeshAgent()
                
                # Mock client to raise exception
                adapter.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
                
                context = MeshContext()
                task_data = TaskData(input="Test query", parameters={})
                
                with pytest.raises(TaskExecutionError):
                    await adapter.handle_task(task_data, context)


# Fixtures and utilities for all tests
@pytest.fixture
def sample_task_data():
    """Sample task data for testing"""
    return TaskData(
        input="Test query for the agent",
        parameters={"temperature": 0.7, "max_tokens": 100}
    )


@pytest.fixture  
def sample_context():
    """Sample context for testing"""
    context = MeshContext()
    asyncio.run(context.set("test_key", "test_value"))
    return context


# Performance tests
class TestAdapterPerformance:
    """Test adapter performance characteristics"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        from meshai.adapters.openai_adapter import OpenAIMeshAgent
        
        with patch('meshai.adapters.openai_adapter.AsyncOpenAI'):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                adapter = OpenAIMeshAgent()
                
                # Mock fast response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "Quick response"
                mock_response.choices[0].message.tool_calls = None
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 5
                mock_response.usage.completion_tokens = 5
                mock_response.usage.total_tokens = 10
                
                adapter.client.chat.completions.create = AsyncMock(return_value=mock_response)
                
                # Run multiple concurrent requests
                contexts = [MeshContext() for _ in range(5)]
                task_datas = [TaskData(input=f"Query {i}", parameters={}) for i in range(5)]
                
                tasks = [
                    adapter.handle_task(task_data, context)
                    for task_data, context in zip(task_datas, contexts)
                ]
                
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 5
                for result in results:
                    assert "result" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])