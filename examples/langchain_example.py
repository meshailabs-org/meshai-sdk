#!/usr/bin/env python3
"""
LangChain Integration Example

This example shows how to wrap existing LangChain agents with MeshAI.
"""

import asyncio
import os
from typing import Dict, Any

# LangChain imports (install with: pip install langchain openai)
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
except ImportError:
    print("LangChain not installed. Install with: pip install meshai-sdk[langchain]")
    exit(1)

from meshai.adapters import LangChainMeshAgent


def create_calculator_tool():
    """Create a simple calculator tool for the agent"""
    
    def calculate(expression: str) -> str:
        """Calculate mathematical expressions safely"""
        try:
            # Simple evaluation (in production, use a proper math parser)
            allowed_chars = set("0123456789+-*/().")
            if not all(c in allowed_chars or c.isspace() for c in expression):
                return "Invalid characters in expression"
            
            result = eval(expression)  # Note: Use ast.literal_eval in production
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return Tool(
        name="Calculator",
        description="Calculate mathematical expressions. Input should be a valid mathematical expression.",
        func=calculate
    )


def create_text_tool():
    """Create a text processing tool"""
    
    def process_text(text: str) -> str:
        """Process text with various operations"""
        return f"Text analysis: {len(text)} characters, {len(text.split())} words, {text.count(' ')} spaces"
    
    return Tool(
        name="TextProcessor",
        description="Process and analyze text. Input should be the text to analyze.",
        func=process_text
    )


async def main():
    """Run the LangChain integration example"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create LangChain components
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    tools = [
        create_calculator_tool(),
        create_text_tool()
    ]
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create LangChain agent
    langchain_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    # Wrap with MeshAI
    mesh_agent = LangChainMeshAgent(
        langchain_agent=langchain_agent,
        agent_id="langchain-calculator",
        name="LangChain Calculator Agent", 
        capabilities=["math", "text-processing", "conversation"],
        llm=llm,
        tools=tools,
        memory=memory
    )
    
    print(f"Created MeshAI-wrapped LangChain agent: {mesh_agent.name}")
    print(f"Agent ID: {mesh_agent.agent_id}")
    print(f"Capabilities: {mesh_agent.capabilities}")
    print(f"Tools: {[tool.name for tool in mesh_agent.get_tools()]}")
    
    # Start the agent
    async with mesh_agent.lifecycle():
        print(f"Agent running on {mesh_agent.config.agent_endpoint}")
        print("\nTry making requests to:")
        print(f"  POST {mesh_agent.config.agent_endpoint}/execute")
        print("  With JSON body: {\"task_id\": \"test\", \"task_type\": \"query\", \"payload\": {\"input\": \"Calculate 15 + 27 * 3\"}}")
        print("\nPress Ctrl+C to stop...")
        
        try:
            await asyncio.sleep(3600)
        except KeyboardInterrupt:
            print("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())