#!/usr/bin/env python3
"""
Basic MeshAI Agent Example

This example shows how to create a simple MeshAI agent from scratch.
"""

import asyncio
from typing import Dict, Any

from meshai import MeshAgent, register_agent, MeshContext
from meshai.core.schemas import TaskData


@register_agent(
    capabilities=["text-processing", "example"],
    name="Basic Example Agent"
)
class BasicAgent(MeshAgent):
    """A simple example agent that processes text"""
    
    async def handle_task(self, task_data: TaskData, context: MeshContext) -> Dict[str, Any]:
        """Handle incoming tasks"""
        # Extract input
        if isinstance(task_data.input, dict):
            text = task_data.input.get("text", str(task_data.input))
        else:
            text = str(task_data.input)
        
        # Simple text processing
        result = {
            "original": text,
            "length": len(text),
            "words": len(text.split()),
            "uppercase": text.upper(),
            "reversed": text[::-1]
        }
        
        # Store result in context for other agents
        await context.set("last_processed_text", result)
        
        return {
            "result": result,
            "message": f"Processed text with {result['words']} words"
        }


async def main():
    """Run the basic agent example"""
    # Create agent instance
    agent = BasicAgent()
    
    print(f"Created agent: {agent.name} ({agent.agent_id})")
    print(f"Capabilities: {agent.capabilities}")
    
    # Start the agent server
    async with agent.lifecycle():
        print(f"Agent server running on {agent.config.agent_endpoint}")
        print("Press Ctrl+C to stop...")
        
        try:
            # Keep server running
            await asyncio.sleep(3600)  # Run for 1 hour
        except KeyboardInterrupt:
            print("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())