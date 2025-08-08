#!/usr/bin/env python3
"""
Advanced Context Management Example

This example demonstrates the sophisticated context management features
of MeshAI, including:
- Agent-specific memory isolation
- Conflict resolution strategies
- Context versioning and rollback
- Access control and sharing policies
- Real-time context synchronization
"""

import asyncio
import logging
from typing import Any, Dict

from meshai.core import MeshAgent, MeshContext
from meshai.core.config import MeshConfig
from meshai.core.context_manager import (
    ContextPolicy, ConflictResolution, ContextScope, AccessLevel
)
from meshai.adapters.langchain_adapter import LangChainAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalystAgent(MeshAgent):
    """Agent that analyzes data and stores findings"""
    
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Analyze data and store results in context"""
        data = task_data.get("dataset", [])
        
        # Perform analysis
        analysis_results = {
            "total_records": len(data),
            "avg_value": sum(data) / len(data) if data else 0,
            "max_value": max(data) if data else 0,
            "min_value": min(data) if data else 0,
        }
        
        # Store results in agent-specific memory
        await context.set("analysis_results", analysis_results, agent_scope=True, agent_id=self.agent_id)
        await context.set("analyst_notes", "Initial analysis completed", agent_scope=True, agent_id=self.agent_id)
        
        # Also store summary in shared memory for other agents
        await context.set("data_summary", {
            "records": analysis_results["total_records"],
            "processed_by": self.agent_id,
            "status": "analyzed"
        })
        
        return {"status": "success", "analysis": analysis_results}


class ReportGeneratorAgent(MeshAgent):
    """Agent that generates reports from analyzed data"""
    
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Generate report from analysis results"""
        
        # Try to get analysis from shared context
        data_summary = await context.get("data_summary")
        if not data_summary:
            return {"status": "error", "message": "No analysis data found"}
        
        # Generate report
        report = {
            "title": "Data Analysis Report",
            "summary": f"Processed {data_summary['records']} records",
            "processed_by": data_summary.get("processed_by", "unknown"),
            "generated_by": self.agent_id,
            "timestamp": "2025-01-08T10:00:00Z"
        }
        
        # Store report in agent memory
        await context.set("generated_report", report, agent_scope=True, agent_id=self.agent_id)
        
        # Update shared status
        await context.set("report_status", "completed")
        
        return {"status": "success", "report": report}


class QualityControlAgent(MeshAgent):
    """Agent that reviews and validates reports"""
    
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Review and validate the generated report"""
        
        # Check if report is ready
        report_status = await context.get("report_status")
        if report_status != "completed":
            return {"status": "waiting", "message": "Report not ready for review"}
        
        # Simulate quality check
        quality_score = 85  # Would be calculated based on actual criteria
        
        # Store QA results in agent memory
        qa_results = {
            "quality_score": quality_score,
            "reviewed_by": self.agent_id,
            "status": "approved" if quality_score >= 80 else "needs_revision",
            "comments": "Report meets quality standards" if quality_score >= 80 else "Requires improvements"
        }
        
        await context.set("qa_results", qa_results, agent_scope=True, agent_id=self.agent_id)
        
        # Update shared context with final status
        await context.set("final_status", qa_results["status"])
        await context.set("quality_score", quality_score)
        
        return {"status": "success", "qa": qa_results}


async def demonstrate_basic_context_flow():
    """Demonstrate basic context sharing between agents"""
    print("\n=== Basic Context Flow Demo ===")
    
    config = MeshConfig()
    context = MeshContext(config=config)
    
    # Create agents
    analyst = DataAnalystAgent("analyst-001", config=config)
    reporter = ReportGeneratorAgent("reporter-001", config=config)
    qa_agent = QualityControlAgent("qa-001", config=config)
    
    # Sample data
    dataset = [10, 20, 30, 40, 50, 25, 35, 45]
    
    # Step 1: Analyst processes data
    print("1. Data Analyst processing dataset...")
    analyst_result = await analyst.handle_task({"dataset": dataset}, context)
    print(f"Analyst result: {analyst_result['status']}")
    
    # Step 2: Reporter generates report
    print("2. Report Generator creating report...")
    reporter_result = await reporter.handle_task({}, context)
    print(f"Reporter result: {reporter_result['status']}")
    
    # Step 3: QA reviews report
    print("3. Quality Control reviewing report...")
    qa_result = await qa_agent.handle_task({}, context)
    print(f"QA result: {qa_result['status']}")
    
    # Show final context state
    print("\nFinal Context State:")
    print(f"Shared memory keys: {await context.keys()}")
    print(f"Final status: {await context.get('final_status')}")
    print(f"Quality score: {await context.get('quality_score')}")


async def demonstrate_advanced_context_features():
    """Demonstrate advanced context management features"""
    print("\n=== Advanced Context Features Demo ===")
    
    config = MeshConfig()
    config.redis_url = "redis://localhost:6379"  # Enable Redis if available
    
    # Create context with advanced features
    context = MeshContext(config=config, use_advanced_manager=True)
    
    analyst = DataAnalystAgent("analyst-002", config=config)
    
    # Create advanced context with policy
    policy = ContextPolicy(
        context_id=context.context_id,
        owner_agent=analyst.agent_id,
        scope=ContextScope.SHARED,
        conflict_resolution=ConflictResolution.MERGE_STRATEGY,
        max_versions=20,
        ttl_seconds=3600
    )
    
    # Initialize advanced context
    success = await context.create_advanced_context(analyst.agent_id, policy)
    print(f"Advanced context created: {success}")
    
    if success:
        # Demonstrate versioning
        print("\nTesting context versioning...")
        await context.set("version_test", "version_1", agent_id=analyst.agent_id)
        await context.set("version_test", "version_2", agent_id=analyst.agent_id)
        
        # Get history
        history = await context.get_context_history(analyst.agent_id, limit=5)
        print(f"Context versions: {len(history)}")
        
        # Demonstrate sharing
        print("\nTesting context sharing...")
        await context.share_with_agent("reporter-002", ["version_test"], analyst.agent_id)
        print("Context shared with reporter-002")
    
    # Cleanup
    await context.close()


async def demonstrate_conflict_resolution():
    """Demonstrate different conflict resolution strategies"""
    print("\n=== Conflict Resolution Demo ===")
    
    config = MeshConfig()
    context = MeshContext(config=config, use_advanced_manager=True)
    
    agent1 = DataAnalystAgent("agent-001", config=config)
    agent2 = ReportGeneratorAgent("agent-002", config=config)
    
    # Create context with merge strategy
    policy = ContextPolicy(
        context_id=context.context_id,
        owner_agent=agent1.agent_id,
        conflict_resolution=ConflictResolution.MERGE_STRATEGY
    )
    
    await context.create_advanced_context(agent1.agent_id, policy)
    
    # Simulate concurrent updates
    print("Simulating conflicting updates...")
    
    # Agent 1 updates
    await context.set("shared_data", {"field1": "value1", "field2": "value2"}, agent_id=agent1.agent_id)
    
    # Agent 2 updates (potential conflict)
    await context.set("shared_data", {"field2": "updated_value2", "field3": "value3"}, agent_id=agent2.agent_id)
    
    # Check merged result
    merged_data = await context.get("shared_data")
    print(f"Merged result: {merged_data}")
    
    await context.close()


async def main():
    """Run all context management demonstrations"""
    try:
        await demonstrate_basic_context_flow()
        await demonstrate_advanced_context_features()
        await demonstrate_conflict_resolution()
        
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())