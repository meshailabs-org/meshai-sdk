#!/usr/bin/env python3
"""
MeshAI Framework Adapter Examples

This file demonstrates how to use MeshAI with different AI frameworks:
- LangChain
- CrewAI  
- AutoGen
- OpenAI
- Anthropic Claude
- Google Gemini
- Amazon Bedrock

Each example shows:
1. Basic setup and initialization
2. Task execution
3. Cross-agent communication via MeshAI tools
4. Context sharing and memory management
"""

import asyncio
import os
from typing import List, Dict, Any

# Import MeshAI core components
from meshai.core.context import MeshContext
from meshai.core.registry import MeshRegistry
from meshai.core.schemas import TaskData

print("🚀 MeshAI Framework Adapter Examples")
print("=" * 50)


async def langchain_example():
    """Example using LangChain with MeshAI"""
    print("\n📚 LangChain Example")
    print("-" * 30)
    
    try:
        from meshai.adapters.langchain_adapter import LangChainMeshAgent
        from langchain_openai import ChatOpenAI
        from langchain.agents import initialize_agent, AgentType
        from langchain.tools import Tool
        
        # Create a simple LangChain tool
        def calculate_tool(expression: str) -> str:
            """Calculate mathematical expressions safely"""
            try:
                # Basic safety check
                if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                    return "Error: Unsafe expression"
                result = eval(expression)
                return f"Result: {result}"
            except:
                return "Error: Invalid expression"
        
        calculator = Tool(
            name="calculator",
            description="Calculate mathematical expressions",
            func=calculate_tool
        )
        
        # Initialize LangChain agent (requires OpenAI API key)
        if os.getenv('OPENAI_API_KEY'):
            llm = ChatOpenAI(temperature=0)
            langchain_agent = initialize_agent(
                tools=[calculator],
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False
            )
            
            # Wrap with MeshAI
            mesh_agent = LangChainMeshAgent(
                langchain_agent=langchain_agent,
                agent_id="langchain-calculator",
                name="LangChain Calculator Agent",
                capabilities=["mathematics", "calculation", "reasoning"]
            )
            
            # Register with MeshAI
            registry = MeshRegistry()
            await registry.register_agent(mesh_agent)
            
            # Execute a task
            context = MeshContext()
            task = TaskData(
                input="What is 25 * 4 + 10?",
                parameters={"temperature": 0.1}
            )
            
            result = await mesh_agent.handle_task(task, context)
            print(f"✅ LangChain Result: {result['result']}")
            
            # Show tool integration
            tools = mesh_agent.get_tools()
            print(f"🔧 Available tools: {[tool.name for tool in tools]}")
            
        else:
            print("⚠️  OpenAI API key not found - skipping LangChain example")
            
    except ImportError as e:
        print(f"⚠️  LangChain not installed: {e}")
    except Exception as e:
        print(f"❌ LangChain example failed: {e}")


async def crewai_example():
    """Example using CrewAI with MeshAI"""
    print("\n🤝 CrewAI Example")
    print("-" * 30)
    
    try:
        from meshai.adapters.crewai_adapter import CrewAIMeshAgent
        from crewai import Agent, Task, Crew
        
        # Create CrewAI agents
        researcher = Agent(
            role="Research Analyst",
            goal="Research and analyze topics thoroughly",
            backstory="You are an expert researcher with deep analytical skills",
            verbose=False
        )
        
        writer = Agent(
            role="Content Writer", 
            goal="Write clear and engaging content",
            backstory="You are a skilled writer who can explain complex topics simply",
            verbose=False
        )
        
        # Create tasks
        research_task = Task(
            description="Research the latest trends in AI agent frameworks",
            agent=researcher,
            expected_output="A comprehensive analysis of AI agent framework trends"
        )
        
        writing_task = Task(
            description="Write a summary based on the research findings",
            agent=writer,
            expected_output="A clear, concise summary for general audience"
        )
        
        # Create crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            verbose=False
        )
        
        # Wrap with MeshAI
        mesh_agent = CrewAIMeshAgent(
            crewai_component=crew,
            agent_id="crewai-research-team",
            name="CrewAI Research Team", 
            capabilities=["research", "writing", "collaboration", "analysis"]
        )
        
        # Register with MeshAI
        registry = MeshRegistry()
        await registry.register_agent(mesh_agent)
        
        # Execute a task
        context = MeshContext()
        task = TaskData(
            input="Research and write about AI agent interoperability",
            parameters={}
        )
        
        result = await mesh_agent.handle_task(task, context)
        print(f"✅ CrewAI Result: {result['result'][:200]}...")
        
        # Show crew information
        crew_info = mesh_agent.get_crew_info()
        print(f"👥 Crew type: {crew_info['type']}")
        print(f"📊 Agent count: {crew_info.get('agent_count', 0)}")
        
    except ImportError as e:
        print(f"⚠️  CrewAI not installed: {e}")
    except Exception as e:
        print(f"❌ CrewAI example failed: {e}")


async def autogen_example():
    """Example using AutoGen with MeshAI"""
    print("\n💬 AutoGen Example")
    print("-" * 30)
    
    try:
        from meshai.adapters.autogen_adapter import AutoGenMeshAgent
        from autogen import ConversableAgent
        
        # Create AutoGen agent
        assistant = ConversableAgent(
            name="helpful_assistant",
            system_message="You are a helpful AI assistant. Be concise and helpful.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        # Wrap with MeshAI
        mesh_agent = AutoGenMeshAgent(
            autogen_component=assistant,
            agent_id="autogen-assistant",
            name="AutoGen Assistant",
            capabilities=["conversation", "assistance", "general"]
        )
        
        # Register with MeshAI  
        registry = MeshRegistry()
        await registry.register_agent(mesh_agent)
        
        # Execute a task
        context = MeshContext()
        task = TaskData(
            input="Explain what makes a good AI agent architecture",
            parameters={}
        )
        
        result = await mesh_agent.handle_task(task, context)
        print(f"✅ AutoGen Result: {result['result'][:200]}...")
        
        # Show dynamic capabilities
        capabilities = await mesh_agent.get_agent_capabilities()
        print(f"🎯 Detected capabilities: {capabilities}")
        
    except ImportError as e:
        print(f"⚠️  AutoGen not installed: {e}")
    except Exception as e:
        print(f"❌ AutoGen example failed: {e}")


async def openai_example():
    """Example using OpenAI with MeshAI"""
    print("\n🤖 OpenAI Example")
    print("-" * 30)
    
    try:
        from meshai.adapters.openai_adapter import OpenAIMeshAgent
        
        if os.getenv('OPENAI_API_KEY'):
            # Create OpenAI agent
            mesh_agent = OpenAIMeshAgent(
                model="gpt-3.5-turbo",
                agent_id="openai-assistant",
                name="OpenAI Assistant",
                capabilities=["text-generation", "reasoning", "coding"],
                temperature=0.7,
                max_tokens=500,
                system_prompt="You are an expert AI assistant specializing in distributed systems and agent architectures."
            )
            
            # Register with MeshAI
            registry = MeshRegistry() 
            await registry.register_agent(mesh_agent)
            
            # Execute a task
            context = MeshContext()
            task = TaskData(
                input="What are the key challenges in building multi-agent systems?",
                parameters={"temperature": 0.8}
            )
            
            result = await mesh_agent.handle_task(task, context)
            print(f"✅ OpenAI Result: {result['result'][:200]}...")
            print(f"📊 Token usage: {result.get('usage', {})}")
            
            # Test tool integration
            print("\n🔧 Testing tool integration...")
            tool_task = TaskData(
                input="Help me find another agent to write some Python code for data analysis",
                parameters={}
            )
            
            # This would use the MeshAI tool to find a coding agent
            tool_result = await mesh_agent.handle_task(tool_task, context)
            print(f"🛠️  Tool result: {tool_result.get('tools_used', False)}")
            
        else:
            print("⚠️  OpenAI API key not found - skipping OpenAI example")
            
    except ImportError as e:
        print(f"⚠️  OpenAI not installed: {e}")
    except Exception as e:
        print(f"❌ OpenAI example failed: {e}")


async def anthropic_example():
    """Example using Anthropic Claude with MeshAI"""
    print("\n🧠 Anthropic Claude Example")  
    print("-" * 30)
    
    try:
        from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
        
        if os.getenv('ANTHROPIC_API_KEY'):
            # Create Anthropic agent
            mesh_agent = AnthropicMeshAgent(
                model="claude-3-sonnet-20240229",
                agent_id="claude-analyst",
                name="Claude Analyst",
                capabilities=["analysis", "reasoning", "research", "writing"],
                max_tokens=1000,
                temperature=0.3,
                system_prompt="You are Claude, an AI assistant created by Anthropic. You excel at analysis and reasoning."
            )
            
            # Add custom tool
            custom_tool = {
                "name": "data_processor",
                "description": "Process and analyze structured data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Data to process"},
                        "operation": {"type": "string", "description": "Operation to perform"}
                    },
                    "required": ["data", "operation"]
                }
            }
            mesh_agent.add_tool(custom_tool)
            
            # Register with MeshAI
            registry = MeshRegistry()
            await registry.register_agent(mesh_agent)
            
            # Execute a task
            context = MeshContext() 
            task = TaskData(
                input="Analyze the benefits and challenges of using AI agents in production systems",
                parameters={"temperature": 0.1}
            )
            
            result = await mesh_agent.handle_task(task, context)
            print(f"✅ Claude Result: {result['result'][:200]}...")
            print(f"📊 Token usage: {result.get('usage', {})}")
            
            # Show model info
            model_info = mesh_agent.get_model_info()
            print(f"🔧 Tools available: {model_info['tools_count']}")
            
        else:
            print("⚠️  Anthropic API key not found - skipping Claude example")
            
    except ImportError as e:
        print(f"⚠️  Anthropic not installed: {e}")
    except Exception as e:
        print(f"❌ Anthropic example failed: {e}")


async def google_example():
    """Example using Google Gemini with MeshAI"""
    print("\n🌟 Google Gemini Example")
    print("-" * 30)
    
    try:
        from meshai.adapters.google_adapter import GoogleMeshAgent
        
        if os.getenv('GOOGLE_API_KEY'):
            # Create Google Gemini agent
            mesh_agent = GoogleMeshAgent(
                model="gemini-pro", 
                agent_id="gemini-multimodal",
                name="Gemini Multimodal Agent",
                capabilities=["multimodal", "reasoning", "analysis", "vision"]
            )
            
            # Register with MeshAI
            registry = MeshRegistry()
            await registry.register_agent(mesh_agent)
            
            # Execute a task
            context = MeshContext()
            task = TaskData(
                input="Explain how multimodal AI can enhance agent-to-agent communication",
                parameters={"temperature": 0.6, "max_output_tokens": 300}
            )
            
            result = await mesh_agent.handle_task(task, context)
            print(f"✅ Gemini Result: {result['result'][:200]}...")
            print(f"🛡️  Safety ratings: {len(result.get('safety_ratings', []))}")
            
            # Show model capabilities
            model_info = mesh_agent.get_model_info()
            print(f"🎥 Multimodal support: {model_info['supports_multimodal']}")
            
        else:
            print("⚠️  Google API key not found - skipping Gemini example")
            
    except ImportError as e:
        print(f"⚠️  Google AI not installed: {e}")
    except Exception as e:
        print(f"❌ Google example failed: {e}")


async def bedrock_example():
    """Example using Amazon Bedrock with MeshAI"""  
    print("\n☁️  Amazon Bedrock Example")
    print("-" * 30)
    
    try:
        from meshai.adapters.amazon_adapter import BedrockMeshAgent
        
        # Create Bedrock agent (requires AWS credentials)
        mesh_agent = BedrockMeshAgent(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            agent_id="bedrock-claude",
            name="Bedrock Claude Agent", 
            capabilities=["enterprise-ai", "analysis", "reasoning"],
            region="us-east-1"
        )
        
        # Register with MeshAI
        registry = MeshRegistry()
        await registry.register_agent(mesh_agent)
        
        print(f"✅ Bedrock agent initialized: {mesh_agent.model_id}")
        
        # Show available models (would require AWS credentials)
        try:
            available_models = mesh_agent.get_available_models()
            print(f"📋 Available models: {len(available_models)}")
        except Exception:
            print("⚠️  AWS credentials not configured - cannot list models")
            
        # Show model info
        model_info = mesh_agent.get_model_info()
        print(f"🏢 Provider: {model_info['provider']}")
        print(f"🌍 Region: {model_info['region']}")
        
    except ImportError as e:
        print(f"⚠️  AWS SDK not installed: {e}")
    except Exception as e:
        print(f"❌ Bedrock example failed: {e}")


async def cross_framework_collaboration():
    """Example showing collaboration between different frameworks"""
    print("\n🤝 Cross-Framework Collaboration Example")
    print("-" * 45)
    
    try:
        # Create agents from different frameworks
        agents = []
        
        # OpenAI agent for general reasoning
        if os.getenv('OPENAI_API_KEY'):
            from meshai.adapters.openai_adapter import OpenAIMeshAgent
            openai_agent = OpenAIMeshAgent(
                model="gpt-3.5-turbo",
                agent_id="openai-reasoner",
                name="OpenAI Reasoning Agent", 
                capabilities=["reasoning", "planning"],
                system_prompt="You are a strategic thinking agent that breaks down complex problems."
            )
            agents.append(openai_agent)
        
        # Anthropic agent for analysis
        if os.getenv('ANTHROPIC_API_KEY'):
            from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
            claude_agent = AnthropicMeshAgent(
                model="claude-3-sonnet-20240229",
                agent_id="claude-analyzer",
                name="Claude Analysis Agent",
                capabilities=["analysis", "critical-thinking"],
                system_prompt="You are an analytical agent that provides deep insights and evaluations."
            )
            agents.append(claude_agent)
        
        # Register all agents
        registry = MeshRegistry()
        for agent in agents:
            await registry.register_agent(agent)
        
        if agents:
            print(f"✅ Registered {len(agents)} agents from different frameworks")
            
            # Create shared context
            context = MeshContext()
            await context.set("project_context", {
                "domain": "AI agent systems",
                "goal": "Design scalable multi-agent architecture",
                "constraints": ["cost-effective", "fault-tolerant", "secure"]
            })
            
            # First agent: Strategic planning
            if len(agents) > 0:
                planning_task = TaskData(
                    input="Create a high-level architecture plan for a distributed AI agent system that can scale to handle millions of requests",
                    parameters={"temperature": 0.7}
                )
                
                planning_result = await agents[0].handle_task(planning_task, context)
                print(f"📋 Planning result: {planning_result['result'][:150]}...")
                
                # Second agent: Analyze the plan
                if len(agents) > 1:
                    analysis_task = TaskData(
                        input="Analyze the proposed architecture and identify potential risks and optimization opportunities",
                        parameters={"temperature": 0.3}
                    )
                    
                    analysis_result = await agents[1].handle_task(analysis_task, context)
                    print(f"🔍 Analysis result: {analysis_result['result'][:150]}...")
            
            # Show shared context evolution
            conversation_history = await context.get("conversation_history", [])
            print(f"💬 Conversation history: {len(conversation_history)} messages")
            
        else:
            print("⚠️  No API keys found - cannot demonstrate cross-framework collaboration")
            
    except Exception as e:
        print(f"❌ Cross-framework collaboration failed: {e}")


async def streaming_examples():
    """Examples of streaming responses from different adapters"""
    print("\n🌊 Streaming Response Examples")
    print("-" * 35)
    
    try:
        context = MeshContext()
        
        # OpenAI streaming
        if os.getenv('OPENAI_API_KEY'):
            from meshai.adapters.openai_adapter import OpenAIMeshAgent
            openai_agent = OpenAIMeshAgent(model="gpt-3.5-turbo")
            
            print("📺 OpenAI Streaming:")
            async for chunk in openai_agent.stream_response("Tell me about AI agents", context):
                print(chunk, end="", flush=True)
            print("\n")
        
        # Anthropic streaming 
        if os.getenv('ANTHROPIC_API_KEY'):
            from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
            claude_agent = AnthropicMeshAgent()
            
            print("📺 Claude Streaming:")
            async for chunk in claude_agent.stream_response("Explain machine learning", context):
                print(chunk, end="", flush=True)
            print("\n")
            
        # Google streaming
        if os.getenv('GOOGLE_API_KEY'):
            from meshai.adapters.google_adapter import GoogleMeshAgent  
            gemini_agent = GoogleMeshAgent()
            
            print("📺 Gemini Streaming:")
            async for chunk in gemini_agent.stream_response("What is the future of AI?", context):
                print(chunk, end="", flush=True)
            print("\n")
        
        if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY'), os.getenv('GOOGLE_API_KEY')]):
            print("⚠️  No API keys found - cannot demonstrate streaming")
            
    except Exception as e:
        print(f"❌ Streaming examples failed: {e}")


async def main():
    """Run all framework adapter examples"""
    print("🎯 Starting MeshAI Framework Adapter Examples...")
    
    # Run individual framework examples
    await langchain_example()
    await crewai_example() 
    await autogen_example()
    await openai_example()
    await anthropic_example()
    await google_example()
    await bedrock_example()
    
    # Advanced examples
    await cross_framework_collaboration()
    await streaming_examples()
    
    print("\n🎉 All examples completed!")
    print("\n📚 Next Steps:")
    print("1. Set up API keys for the frameworks you want to use")
    print("2. Install optional dependencies: pip install meshai-sdk[all]")
    print("3. Check the MeshAI documentation for advanced features")
    print("4. Build your own multi-agent applications!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())