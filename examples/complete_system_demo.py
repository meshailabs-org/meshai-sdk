#!/usr/bin/env python3
"""
Complete MeshAI System Demo

This example demonstrates the complete MeshAI system:
1. Start Registry and Runtime services
2. Create and register agents
3. Submit and execute tasks
4. Show real-time monitoring

Run this after starting the services with: python scripts/start-all.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from meshai.core.agent import MeshAgent
from meshai.core.context import MeshContext
from meshai.core.config import MeshConfig
from meshai.core.schemas import TaskData, TaskStatus
from meshai.clients.registry import RegistryClient
from meshai.clients.runtime import RuntimeClient


class DemoTextProcessor(MeshAgent):
    """Demo text processing agent"""
    
    def __init__(self):
        super().__init__(
            agent_id="demo-text-processor",
            name="Demo Text Processor",
            capabilities=["text-processing", "analysis"],
            framework="custom"
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """Process text analysis tasks"""
        # Extract input text
        if isinstance(task_data.input, dict):
            text = task_data.input.get("text", str(task_data.input))
        else:
            text = str(task_data.input)
        
        # Simulate some processing time
        await asyncio.sleep(1)
        
        # Analyze text
        words = text.split()
        result = {
            "original_text": text,
            "character_count": len(text),
            "word_count": len(words),
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "reversed": text[::-1],
            "first_word": words[0] if words else "",
            "last_word": words[-1] if words else ""
        }
        
        # Store in context for other agents
        await context.set("analysis_result", result)
        
        return {
            "status": "completed",
            "analysis": result,
            "message": f"Analyzed text: {len(words)} words, {len(text)} characters"
        }


class DemoSummarizer(MeshAgent):
    """Demo text summarization agent"""
    
    def __init__(self):
        super().__init__(
            agent_id="demo-summarizer",
            name="Demo Summarizer",
            capabilities=["text-summarization", "nlp"],
            framework="custom"
        )
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """Summarize text"""
        if isinstance(task_data.input, dict):
            text = task_data.input.get("text", str(task_data.input))
        else:
            text = str(task_data.input)
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Simple summarization (just take first and last sentences)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 2:
            summary = text
        else:
            summary = f"{sentences[0]}. {sentences[-1]}."
        
        result = {
            "original_text": text,
            "original_length": len(text),
            "summary": summary,
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if text else 0,
            "total_sentences": len(sentences)
        }
        
        return {
            "status": "completed",
            "summary": result,
            "message": f"Summarized {len(sentences)} sentences to {len(summary)} characters"
        }


async def wait_for_services():
    """Wait for services to be available"""
    config = MeshConfig()
    registry_client = RegistryClient(config)
    runtime_client = RuntimeClient(config)
    
    print("⏳ Waiting for services to be available...")
    
    for attempt in range(30):  # Wait up to 30 seconds
        try:
            # Test registry service
            await registry_client.get_registry_health()
            
            # Test runtime service  
            await runtime_client.get_runtime_health()
            
            print("✅ Services are available!")
            await registry_client.close()
            await runtime_client.close()
            return True
            
        except Exception:
            if attempt == 29:  # Last attempt
                print("❌ Services are not available. Please start them with: python scripts/start-all.py")
                return False
            
            await asyncio.sleep(1)
    
    await registry_client.close()
    await runtime_client.close()
    return False


async def demo_agents():
    """Demonstrate agent creation and registration"""
    print("\n" + "=" * 60)
    print("📋 DEMO: Agent Creation and Registration")
    print("=" * 60)
    
    # Create agents
    text_processor = DemoTextProcessor()
    summarizer = DemoSummarizer()
    
    print(f"✅ Created agent: {text_processor.name} ({text_processor.agent_id})")
    print(f"✅ Created agent: {summarizer.name} ({summarizer.agent_id})")
    
    # Start agent servers
    print("\n🚀 Starting agent servers...")
    
    # Use different ports for each agent
    text_processor.config = text_processor.config.update(agent_port=8100)
    summarizer.config = summarizer.config.update(agent_port=8101)
    
    await text_processor.start_server()
    await summarizer.start_server()
    
    print(f"✅ Text processor running on port 8100")
    print(f"✅ Summarizer running on port 8101")
    
    # Wait a moment for servers to start
    await asyncio.sleep(2)
    
    # Register agents
    print("\n📝 Registering agents...")
    
    try:
        await text_processor.register()
        await summarizer.register()
        print("✅ All agents registered successfully!")
    except Exception as e:
        print(f"❌ Agent registration failed: {e}")
        return None, None
    
    return text_processor, summarizer


async def demo_task_execution():
    """Demonstrate task execution through runtime"""
    print("\n" + "=" * 60)
    print("🎯 DEMO: Task Execution")
    print("=" * 60)
    
    config = MeshConfig()
    runtime_client = RuntimeClient(config)
    
    # Sample texts for processing
    sample_texts = [
        "MeshAI is an interoperability layer for autonomous AI agents. It enables seamless communication between agents built on different frameworks like LangChain, CrewAI, and AutoGen.",
        
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. It's often used for testing fonts and keyboards.",
        
        "Artificial intelligence is transforming how we work and live. Machine learning algorithms can now process vast amounts of data. Natural language processing helps computers understand human language. The future of AI is incredibly exciting."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Task {i}: Text Analysis ---")
        print(f"Input: {text[:50]}...")
        
        # Submit task to text processor
        task_data = TaskData(
            task_type="text_analysis",
            input={"text": text},
            required_capabilities=["text-processing"],
            timeout_seconds=10
        )
        
        try:
            # Submit task
            result = await runtime_client.submit_task(task_data)
            print(f"✅ Task submitted: {result.task_id}")
            
            # Wait for completion
            final_result = await runtime_client.wait_for_completion(
                result.task_id, 
                timeout=15,
                poll_interval=1
            )
            
            if final_result.status == TaskStatus.COMPLETED:
                print(f"✅ Task completed by agent: {final_result.agent_id}")
                if final_result.result:
                    analysis = final_result.result.get('analysis', {})
                    print(f"   📊 Words: {analysis.get('word_count', 'N/A')}")
                    print(f"   📊 Characters: {analysis.get('character_count', 'N/A')}")
                    print(f"   ⏱️  Execution time: {final_result.execution_time_seconds:.2f}s")
            else:
                print(f"❌ Task failed: {final_result.error}")
                
        except Exception as e:
            print(f"❌ Task execution failed: {e}")
        
        # Wait between tasks
        await asyncio.sleep(1)
    
    await runtime_client.close()


async def demo_multi_agent_workflow():
    """Demonstrate multi-agent workflow"""
    print("\n" + "=" * 60)
    print("🔄 DEMO: Multi-Agent Workflow")
    print("=" * 60)
    
    config = MeshConfig()
    runtime_client = RuntimeClient(config)
    
    text = """
    Artificial intelligence represents one of the most significant technological advances of our time. 
    Machine learning algorithms can process and analyze vast amounts of data at unprecedented speeds. 
    Natural language processing enables computers to understand and generate human language with remarkable accuracy. 
    Computer vision allows machines to interpret and analyze visual information from the world around us. 
    These technologies are being applied across industries, from healthcare and finance to transportation and entertainment. 
    The potential applications seem limitless, and we are only beginning to scratch the surface of what AI can accomplish.
    """
    
    print(f"Input text: {len(text)} characters")
    
    # Step 1: Analyze text
    print("\n📊 Step 1: Analyzing text...")
    analysis_task = TaskData(
        task_type="analysis",
        input={"text": text},
        required_capabilities=["text-processing"],
        timeout_seconds=15
    )
    
    try:
        analysis_result = await runtime_client.submit_and_wait(analysis_task, timeout=20)
        
        if analysis_result.status == TaskStatus.COMPLETED:
            print(f"✅ Analysis completed by {analysis_result.agent_id}")
            analysis_data = analysis_result.result.get('analysis', {})
            print(f"   Words: {analysis_data.get('word_count')}")
            print(f"   Characters: {analysis_data.get('character_count')}")
        else:
            print(f"❌ Analysis failed: {analysis_result.error}")
            return
            
    except Exception as e:
        print(f"❌ Analysis task failed: {e}")
        return
    
    # Step 2: Summarize text
    print("\n📝 Step 2: Summarizing text...")
    summary_task = TaskData(
        task_type="summarization",
        input={"text": text},
        required_capabilities=["text-summarization"],
        timeout_seconds=15
    )
    
    try:
        summary_result = await runtime_client.submit_and_wait(summary_task, timeout=20)
        
        if summary_result.status == TaskStatus.COMPLETED:
            print(f"✅ Summary completed by {summary_result.agent_id}")
            summary_data = summary_result.result.get('summary', {})
            print(f"   Original: {summary_data.get('original_length')} chars")
            print(f"   Summary: {summary_data.get('summary_length')} chars")
            print(f"   Compression: {summary_data.get('compression_ratio', 0):.1%}")
            print(f"   📄 Summary: {summary_data.get('summary', '')[:100]}...")
        else:
            print(f"❌ Summary failed: {summary_result.error}")
            
    except Exception as e:
        print(f"❌ Summary task failed: {e}")
    
    await runtime_client.close()


async def demo_monitoring():
    """Demonstrate system monitoring"""
    print("\n" + "=" * 60)
    print("📈 DEMO: System Monitoring")
    print("=" * 60)
    
    config = MeshConfig()
    registry_client = RegistryClient(config)
    runtime_client = RuntimeClient(config)
    
    try:
        # Get registry health
        print("🏥 Registry Health:")
        registry_health = await registry_client.get_registry_health()
        print(f"   Status: {registry_health.get('status', 'unknown')}")
        print(f"   Uptime: {registry_health.get('uptime_seconds', 0):.1f}s")
        
        # Get runtime stats
        print("\n⚡ Runtime Statistics:")
        runtime_stats = await runtime_client.get_runtime_stats()
        print(f"   Active tasks: {runtime_stats.get('active_tasks', 0)}")
        print(f"   Completed tasks: {runtime_stats.get('completed_tasks', 0)}")
        print(f"   Success rate: {runtime_stats.get('success_rate', 0):.1%}")
        
        # List registered agents
        print("\n🤖 Registered Agents:")
        agents = await registry_client.list_agents(limit=10)
        for agent in agents:
            print(f"   - {agent.name} ({agent.id})")
            print(f"     Framework: {agent.framework}")
            print(f"     Capabilities: {', '.join(agent.capabilities)}")
            print(f"     Status: {agent.status}")
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
    
    await registry_client.close()
    await runtime_client.close()


async def cleanup_agents(text_processor, summarizer):
    """Clean up agents and services"""
    print("\n" + "=" * 60)
    print("🧹 DEMO: Cleanup")
    print("=" * 60)
    
    if text_processor:
        try:
            await text_processor.unregister()
            await text_processor.stop_server()
            print("✅ Text processor cleaned up")
        except Exception as e:
            print(f"⚠️  Text processor cleanup failed: {e}")
    
    if summarizer:
        try:
            await summarizer.unregister()
            await summarizer.stop_server()
            print("✅ Summarizer cleaned up")
        except Exception as e:
            print(f"⚠️  Summarizer cleanup failed: {e}")


async def main():
    """Main demo function"""
    print("🚀 MeshAI Complete System Demo")
    print("=" * 60)
    print("This demo will showcase:")
    print("- Agent registration and discovery")
    print("- Task submission and execution")
    print("- Multi-agent workflows")
    print("- System monitoring")
    print("=" * 60)
    
    # Check if services are running
    if not await wait_for_services():
        print("\n💡 To start services, run:")
        print("   python scripts/start-all.py")
        print("\nThen run this demo again.")
        return
    
    text_processor = None
    summarizer = None
    
    try:
        # Demo 1: Agent creation and registration
        text_processor, summarizer = await demo_agents()
        
        if not text_processor or not summarizer:
            print("❌ Agent setup failed, skipping remaining demos")
            return
        
        # Wait for registration to complete
        await asyncio.sleep(3)
        
        # Demo 2: Task execution
        await demo_task_execution()
        
        # Demo 3: Multi-agent workflow
        await demo_multi_agent_workflow()
        
        # Demo 4: System monitoring
        await demo_monitoring()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print("Key capabilities demonstrated:")
        print("✅ Agent registration and discovery")
        print("✅ Task routing and execution")
        print("✅ Multi-agent coordination")
        print("✅ Real-time monitoring")
        print("✅ Error handling and recovery")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        await cleanup_agents(text_processor, summarizer)


if __name__ == "__main__":
    asyncio.run(main())