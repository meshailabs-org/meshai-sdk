#!/usr/bin/env python3
"""
Multi-Agent Workflow Example

This example demonstrates multiple agents working together through MeshAI.
"""

import asyncio
import os
from typing import Dict, Any

from meshai import MeshAgent, register_agent, MeshContext
from meshai.core.schemas import TaskData
from meshai.adapters import OpenAIMeshAgent, AnthropicMeshAgent


@register_agent(
    capabilities=["task-coordination", "workflow"],
    name="Workflow Coordinator"
)
class WorkflowAgent(MeshAgent):
    """Coordinates tasks between multiple specialized agents"""
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """Coordinate a multi-step workflow"""
        
        # Extract the workflow request
        if isinstance(task_data.input, dict):
            workflow_type = task_data.input.get("workflow", "analysis")
            content = task_data.input.get("content", "")
        else:
            workflow_type = "analysis"
            content = str(task_data.input)
        
        results = {"workflow": workflow_type, "steps": []}
        
        try:
            if workflow_type == "content_analysis":
                # Step 1: Analyze with Claude
                claude_result = await self.invoke_agent(
                    capabilities=["reasoning", "analysis"],
                    task={
                        "input": f"Analyze this content for themes, sentiment, and key insights: {content}",
                        "analysis_type": "comprehensive"
                    }
                )
                
                results["steps"].append({
                    "step": 1,
                    "agent": "claude",
                    "task": "content_analysis",
                    "result": claude_result.result
                })
                
                # Step 2: Generate summary with GPT
                gpt_result = await self.invoke_agent(
                    capabilities=["text-generation", "summarization"],
                    task={
                        "input": f"Create a concise executive summary based on this analysis: {claude_result.result}",
                        "format": "executive_summary"
                    }
                )
                
                results["steps"].append({
                    "step": 2,
                    "agent": "gpt",
                    "task": "summarization",
                    "result": gpt_result.result
                })
                
                # Step 3: Extract action items
                action_result = await self.invoke_agent(
                    capabilities=["text-processing", "extraction"],
                    task={
                        "input": f"Extract actionable items and recommendations from: {claude_result.result}",
                        "format": "action_items"
                    }
                )
                
                results["steps"].append({
                    "step": 3,
                    "agent": "processor",
                    "task": "action_extraction",
                    "result": action_result.result
                })
                
            elif workflow_type == "research":
                # Research workflow
                research_result = await self.invoke_agent(
                    capabilities=["reasoning", "research"],
                    task={
                        "input": f"Research this topic and provide comprehensive information: {content}",
                        "depth": "comprehensive"
                    }
                )
                
                results["steps"].append({
                    "step": 1,
                    "agent": "researcher",
                    "task": "research",
                    "result": research_result.result
                })
                
            # Store workflow results in context
            await context.set("workflow_results", results)
            
            return {
                "result": results,
                "summary": f"Completed {workflow_type} workflow with {len(results['steps'])} steps"
            }
            
        except Exception as e:
            return {
                "result": {"error": str(e)},
                "summary": f"Workflow failed: {e}"
            }


@register_agent(
    capabilities=["text-processing", "extraction", "formatting"],
    name="Text Processor"
)
class TextProcessorAgent(MeshAgent):
    """Specialized agent for text processing tasks"""
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """Process text with various operations"""
        
        if isinstance(task_data.input, dict):
            text = task_data.input.get("input", "")
            format_type = task_data.input.get("format", "basic")
        else:
            text = str(task_data.input)
            format_type = "basic"
        
        if format_type == "action_items":
            # Extract action items
            lines = text.split('\n')
            actions = []
            
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['should', 'must', 'need', 'recommend', 'suggest']):
                    actions.append(line)
            
            return {
                "result": {
                    "action_items": actions,
                    "count": len(actions),
                    "formatted": "\n".join(f"• {action}" for action in actions)
                }
            }
        
        else:
            # Basic text analysis
            return {
                "result": {
                    "length": len(text),
                    "words": len(text.split()),
                    "sentences": text.count('.') + text.count('!') + text.count('?'),
                    "paragraphs": len([p for p in text.split('\n\n') if p.strip()])
                }
            }


async def create_ai_agents():
    """Create AI service agents if API keys are available"""
    agents = []
    
    # Create OpenAI agent if API key available
    if os.getenv("OPENAI_API_KEY"):
        gpt_agent = OpenAIMeshAgent(
            model="gpt-3.5-turbo",
            agent_id="gpt-summarizer", 
            name="GPT Summarizer",
            capabilities=["text-generation", "summarization"],
            system_prompt="You are a skilled summarizer. Create concise, executive-level summaries."
        )
        agents.append(gpt_agent)
    
    # Create Anthropic agent if API key available
    if os.getenv("ANTHROPIC_API_KEY"):
        claude_agent = AnthropicMeshAgent(
            model="claude-3-sonnet-20240229",
            agent_id="claude-analyzer",
            name="Claude Analyzer", 
            capabilities=["reasoning", "analysis", "research"],
            system_prompt="You are an expert analyst. Provide thorough, insightful analysis."
        )
        agents.append(claude_agent)
    
    return agents


async def main():
    """Run the multi-agent workflow example"""
    
    # Create specialized agents
    coordinator = WorkflowAgent()
    processor = TextProcessorAgent()
    
    # Create AI service agents
    ai_agents = await create_ai_agents()
    
    all_agents = [coordinator, processor] + ai_agents
    
    print(f"Created {len(all_agents)} agents:")
    for agent in all_agents:
        print(f"  - {agent.name} ({agent.agent_id}): {agent.capabilities}")
    
    # Start all agents
    print("\nStarting agent network...")
    
    async def run_agent(agent):
        async with agent.lifecycle():
            print(f"✓ {agent.name} running on port {agent.config.agent_port}")
            try:
                await asyncio.sleep(3600)  # Run for 1 hour
            except asyncio.CancelledError:
                pass
    
    # Start all agents concurrently
    tasks = [asyncio.create_task(run_agent(agent)) for agent in all_agents]
    
    print(f"\n🚀 Multi-agent network is running!")
    print(f"Coordinator endpoint: {coordinator.config.agent_endpoint}")
    print("\nTry a workflow request:")
    print(f'curl -X POST {coordinator.config.agent_endpoint}/execute \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"task_id": "workflow1", "task_type": "workflow", "payload": {"workflow": "content_analysis", "content": "Your text here..."}}\'')
    print("\nPress Ctrl+C to stop all agents...")
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\nStopping all agents...")
        for task in tasks:
            task.cancel()
        
        # Wait for clean shutdown
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())