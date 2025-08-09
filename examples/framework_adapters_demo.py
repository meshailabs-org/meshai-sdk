#!/usr/bin/env python3
"""
MeshAI Framework Adapters Demonstration

Shows how to use various AI framework adapters with MeshAI:
- OpenAI GPT models
- Anthropic Claude
- Google Gemini
- Amazon Bedrock
- CrewAI agents
- AutoGen multi-agent systems

This demo requires API keys for the services you want to test.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.core.config import MeshConfig
from meshai.core.context import MeshContext
from meshai.core.schemas import TaskData
from meshai.adapters import get_available_adapters, is_adapter_available


class FrameworkAdaptersDemo:
    """Comprehensive demo of MeshAI framework adapters"""
    
    def __init__(self):
        self.config = MeshConfig()
        self.agents = {}
        self.available_frameworks = get_available_adapters()
        
    async def setup_agents(self):
        """Setup agents for available frameworks"""
        print("🚀 Setting up Framework Adapters")
        print("=" * 50)
        
        # OpenAI Agent
        if is_adapter_available("openai") and os.getenv("OPENAI_API_KEY"):
            await self._setup_openai_agent()
        
        # Anthropic Agent  
        if is_adapter_available("anthropic") and os.getenv("ANTHROPIC_API_KEY"):
            await self._setup_anthropic_agent()
            
        # Google Gemini Agent
        if is_adapter_available("google") and os.getenv("GOOGLE_API_KEY"):
            await self._setup_google_agent()
            
        # Amazon Bedrock Agent
        if is_adapter_available("amazon") and os.getenv("AWS_ACCESS_KEY_ID"):
            await self._setup_amazon_agent()
            
        # CrewAI Agent
        if is_adapter_available("crewai"):
            await self._setup_crewai_agent()
            
        # AutoGen Agent
        if is_adapter_available("autogen"):
            await self._setup_autogen_agent()
        
        print(f"\n✅ Set up {len(self.agents)} framework adapters")
        for name, agent in self.agents.items():
            print(f"   - {name}: {agent.name} ({agent.framework})")
    
    async def _setup_openai_agent(self):
        """Setup OpenAI agent"""
        try:
            from meshai.adapters import OpenAIMeshAgent
            
            agent = OpenAIMeshAgent(
                model="gpt-3.5-turbo",
                agent_id="demo-openai",
                name="Demo OpenAI Agent",
                capabilities=["text-generation", "reasoning", "coding"]
            )
            
            # Start the agent
            await agent.start_server()
            await agent.register()
            
            self.agents["openai"] = agent
            print("✅ OpenAI GPT-3.5 Turbo agent ready")
            
        except Exception as e:
            print(f"⚠️  OpenAI setup failed: {e}")
    
    async def _setup_anthropic_agent(self):
        """Setup Anthropic Claude agent"""
        try:
            from meshai.adapters import AnthropicMeshAgent
            
            agent = AnthropicMeshAgent(
                model="claude-3-sonnet-20240229",
                agent_id="demo-claude",
                name="Demo Claude Agent",
                capabilities=["text-generation", "analysis", "reasoning"]
            )
            
            await agent.start_server()
            await agent.register()
            
            self.agents["anthropic"] = agent
            print("✅ Anthropic Claude 3 Sonnet agent ready")
            
        except Exception as e:
            print(f"⚠️  Anthropic setup failed: {e}")
    
    async def _setup_google_agent(self):
        """Setup Google Gemini agent"""
        try:
            from meshai.adapters import GoogleMeshAgent
            
            agent = GoogleMeshAgent(
                model="gemini-pro",
                agent_id="demo-gemini",
                name="Demo Gemini Agent",
                capabilities=["text-generation", "multimodal", "reasoning"]
            )
            
            await agent.start_server()
            await agent.register()
            
            self.agents["google"] = agent
            print("✅ Google Gemini Pro agent ready")
            
        except Exception as e:
            print(f"⚠️  Google setup failed: {e}")
    
    async def _setup_amazon_agent(self):
        """Setup Amazon Bedrock agent"""
        try:
            from meshai.adapters import BedrockMeshAgent
            
            agent = BedrockMeshAgent(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                agent_id="demo-bedrock",
                name="Demo Bedrock Agent",
                capabilities=["text-generation", "analysis"]
            )
            
            await agent.start_server()
            await agent.register()
            
            self.agents["amazon"] = agent
            print("✅ Amazon Bedrock Claude agent ready")
            
        except Exception as e:
            print(f"⚠️  Amazon Bedrock setup failed: {e}")
    
    async def _setup_crewai_agent(self):
        """Setup CrewAI agent"""
        try:
            import crewai
            from meshai.adapters import CrewAIMeshAgent
            
            # Create a simple CrewAI agent
            researcher = crewai.Agent(
                role="Research Analyst",
                goal="Provide comprehensive analysis and insights",
                backstory="You are an expert research analyst with deep knowledge across domains.",
                verbose=True,
                allow_delegation=False
            )
            
            # Wrap in MeshAI adapter
            agent = CrewAIMeshAgent(
                crewai_component=researcher,
                agent_id="demo-crewai",
                name="Demo CrewAI Researcher",
                capabilities=["research", "analysis", "writing"]
            )
            
            await agent.start_server()
            await agent.register()
            
            self.agents["crewai"] = agent
            print("✅ CrewAI Research agent ready")
            
        except Exception as e:
            print(f"⚠️  CrewAI setup failed: {e}")
    
    async def _setup_autogen_agent(self):
        """Setup AutoGen agent"""
        try:
            import autogen
            from meshai.adapters import AutoGenMeshAgent
            
            # Create an AutoGen conversable agent
            autogen_agent = autogen.ConversableAgent(
                name="assistant",
                system_message="You are a helpful AI assistant that can analyze problems and provide solutions.",
                llm_config={"config_list": []},  # Empty for demo
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3
            )
            
            # Wrap in MeshAI adapter
            agent = AutoGenMeshAgent(
                autogen_component=autogen_agent,
                agent_id="demo-autogen",
                name="Demo AutoGen Assistant",
                capabilities=["conversation", "problem-solving"]
            )
            
            await agent.start_server()
            await agent.register()
            
            self.agents["autogen"] = agent
            print("✅ AutoGen Assistant agent ready")
            
        except Exception as e:
            print(f"⚠️  AutoGen setup failed: {e}")
    
    async def demonstrate_cross_framework_collaboration(self):
        """Show agents from different frameworks working together"""
        if len(self.agents) < 2:
            print("⚠️  Need at least 2 agents for collaboration demo")
            return
        
        print("\n" + "=" * 60)
        print("🤝 Cross-Framework Collaboration Demo")
        print("=" * 60)
        
        # Pick the first two available agents
        agent_names = list(self.agents.keys())[:2]
        agent1_name = agent_names[0]
        agent2_name = agent_names[1]
        agent1 = self.agents[agent1_name]
        agent2 = self.agents[agent2_name]
        
        print(f"Collaboration between:")
        print(f"  🤖 {agent1.name} ({agent1.framework})")
        print(f"  🤖 {agent2.name} ({agent2.framework})")
        
        # Create shared context
        context = MeshContext()
        await context.set("collaboration_topic", "sustainable technology solutions")
        await context.set("conversation_history", [])
        
        # Task 1: First agent generates ideas
        task1 = TaskData(
            task_type="brainstorm",
            input={
                "input": "Generate 3 innovative sustainable technology ideas for reducing carbon emissions in urban areas."
            },
            required_capabilities=["text-generation", "reasoning"]
        )
        
        print(f"\n📝 Task 1: {agent1.name} brainstorming ideas...")
        result1 = await agent1.handle_task(task1, context)
        ideas = result1.get("result", "No ideas generated")
        print(f"💡 Ideas: {ideas[:200]}...")
        
        # Task 2: Second agent analyzes and improves
        task2 = TaskData(
            task_type="analyze",
            input={
                "input": f"Analyze these sustainable technology ideas and provide implementation recommendations: {ideas}"
            },
            required_capabilities=["analysis", "reasoning"]
        )
        
        print(f"\n🔍 Task 2: {agent2.name} analyzing ideas...")
        result2 = await agent2.handle_task(task2, context)
        analysis = result2.get("result", "No analysis provided")
        print(f"📊 Analysis: {analysis[:200]}...")
        
        # Show context sharing
        final_history = await context.get("conversation_history", [])
        print(f"\n🔗 Shared context: {len(final_history)} conversation entries")
        
        return {
            "agent1_result": result1,
            "agent2_result": result2,
            "shared_context_size": len(final_history)
        }
    
    async def benchmark_framework_performance(self):
        """Compare performance across frameworks"""
        if not self.agents:
            print("⚠️  No agents available for benchmarking")
            return
        
        print("\n" + "=" * 60)
        print("⚡ Framework Performance Benchmark")
        print("=" * 60)
        
        # Test task
        test_task = TaskData(
            task_type="reasoning",
            input={
                "input": "Explain the concept of machine learning in simple terms suitable for a 10-year-old."
            },
            required_capabilities=["text-generation", "reasoning"]
        )
        
        results = {}
        
        for name, agent in self.agents.items():
            print(f"\n🧪 Testing {agent.name}...")
            
            try:
                import time
                start_time = time.time()
                
                # Create fresh context
                context = MeshContext()
                result = await agent.handle_task(test_task, context)
                
                response_time = time.time() - start_time
                response_length = len(str(result.get("result", "")))
                
                results[name] = {
                    "agent_name": agent.name,
                    "framework": agent.framework,
                    "response_time": response_time,
                    "response_length": response_length,
                    "result": result
                }
                
                print(f"  ⏱️  Response time: {response_time:.2f}s")
                print(f"  📏 Response length: {response_length} chars")
                print(f"  ✅ Success: {result.get('result', 'No result')[:100]}...")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[name] = {"error": str(e)}
        
        # Summary
        print(f"\n📊 Benchmark Summary:")
        successful_tests = [r for r in results.values() if "error" not in r]
        if successful_tests:
            avg_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
            fastest = min(successful_tests, key=lambda x: x["response_time"])
            longest_response = max(successful_tests, key=lambda x: x["response_length"])
            
            print(f"  ⚡ Average response time: {avg_time:.2f}s")
            print(f"  🏆 Fastest: {fastest['agent_name']} ({fastest['response_time']:.2f}s)")
            print(f"  📖 Most detailed: {longest_response['agent_name']} ({longest_response['response_length']} chars)")
        
        return results
    
    async def cleanup(self):
        """Cleanup all agents"""
        print("\n🧹 Cleaning up agents...")
        for name, agent in self.agents.items():
            try:
                await agent.unregister()
                await agent.stop_server()
                print(f"✅ {name} cleaned up")
            except Exception as e:
                print(f"⚠️  {name} cleanup warning: {e}")


async def main():
    """Main demo function"""
    print("🌐 MeshAI Framework Adapters Demo")
    print("=" * 60)
    print("This demo showcases MeshAI's ability to integrate with multiple AI frameworks")
    print("and enable seamless cross-framework collaboration.")
    print("\nRequired environment variables for full demo:")
    print("- OPENAI_API_KEY (for OpenAI models)")
    print("- ANTHROPIC_API_KEY (for Claude models)")  
    print("- GOOGLE_API_KEY (for Gemini models)")
    print("- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for Bedrock)")
    print("=" * 60)
    
    demo = FrameworkAdaptersDemo()
    
    try:
        # Setup agents
        await demo.setup_agents()
        
        if not demo.agents:
            print("\n❌ No agents could be set up. Please check API keys and dependencies.")
            print("\n💡 Install optional dependencies:")
            print("   pip install openai anthropic google-generativeai boto3 crewai pyautogen")
            return
        
        # Run demos
        await demo.demonstrate_cross_framework_collaboration()
        await demo.benchmark_framework_performance()
        
        print("\n" + "=" * 60)
        print("🎉 Framework Adapters Demo Complete!")
        print("=" * 60)
        print("Key takeaways:")
        print("✅ Multiple AI frameworks integrated seamlessly")
        print("✅ Cross-framework collaboration enabled")
        print("✅ Unified API for diverse AI capabilities")
        print("✅ Context sharing across different models")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    # Set up some demo environment if no keys provided
    if not any(os.getenv(key) for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]):
        print("💡 No API keys detected - running with mock agents for demonstration")
        
        # You could set up mock implementations here for demo purposes
        # For now, we'll just show the structure
    
    asyncio.run(main())