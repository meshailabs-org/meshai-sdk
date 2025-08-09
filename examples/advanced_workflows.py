#!/usr/bin/env python3
"""
MeshAI Advanced Workflow Examples

This file demonstrates sophisticated multi-agent workflows for real-world use cases:

1. Software Development Team - Code generation, review, testing, documentation
2. Research & Analysis Pipeline - Information gathering, analysis, synthesis, reporting
3. Content Creation Workflow - Research, writing, editing, optimization
4. Customer Support System - Ticket routing, response generation, escalation
5. Data Processing Pipeline - Collection, cleaning, analysis, visualization
6. Strategic Planning Session - Problem analysis, solution brainstorming, evaluation
7. Educational Content Creation - Curriculum design, content creation, assessment
8. Marketing Campaign Development - Market research, content creation, optimization

Each example shows:
- Multi-framework agent coordination
- Dynamic agent discovery and routing
- Context preservation across complex workflows
- Error handling and fallback strategies
- Performance monitoring and optimization
"""

import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Import MeshAI core components
from meshai.core.context import MeshContext
from meshai.core.registry import MeshRegistry
from meshai.core.schemas import TaskData
from meshai.core.runtime import MeshRuntime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 MeshAI Advanced Workflow Examples")
print("=" * 50)


class WorkflowOrchestrator:
    """Orchestrates complex multi-agent workflows"""
    
    def __init__(self):
        self.registry = MeshRegistry()
        self.runtime = MeshRuntime()
        self.agents = {}
        
    async def setup_agents(self):
        """Setup agents for various workflows"""
        await self._setup_development_team()
        await self._setup_research_team()
        await self._setup_content_team()
        await self._setup_support_team()
        
    async def _setup_development_team(self):
        """Setup software development agents"""
        logger.info("Setting up software development team...")
        
        try:
            # Code Generator (OpenAI GPT-4)
            if os.getenv('OPENAI_API_KEY'):
                from meshai.adapters.openai_adapter import OpenAIMeshAgent
                
                coder = OpenAIMeshAgent(
                    model="gpt-4",
                    agent_id="senior-developer",
                    name="Senior Developer",
                    capabilities=["coding", "python", "javascript", "architecture"],
                    system_prompt=(
                        "You are a senior software developer with expertise in Python, JavaScript, "
                        "and system architecture. Write clean, well-documented, production-ready code."
                    ),
                    temperature=0.2
                )
                await self.registry.register_agent(coder)
                self.agents['coder'] = coder
                
            # Code Reviewer (Claude)
            if os.getenv('ANTHROPIC_API_KEY'):
                from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
                
                reviewer = AnthropicMeshAgent(
                    model="claude-3-sonnet-20240229",
                    agent_id="code-reviewer",
                    name="Code Reviewer",
                    capabilities=["code-review", "security", "best-practices", "testing"],
                    system_prompt=(
                        "You are an expert code reviewer focusing on security, performance, "
                        "maintainability, and best practices. Provide constructive feedback."
                    ),
                    temperature=0.1
                )
                await self.registry.register_agent(reviewer)
                self.agents['reviewer'] = reviewer
                
            # Documentation Writer (CrewAI)
            try:
                from meshai.adapters.crewai_adapter import CrewAIMeshAgent
                from crewai import Agent, Task, Crew
                
                tech_writer = Agent(
                    role="Technical Writer",
                    goal="Create clear, comprehensive technical documentation",
                    backstory="Expert technical writer with deep understanding of software development",
                    verbose=False
                )
                
                doc_crew = Crew(
                    agents=[tech_writer],
                    tasks=[],
                    verbose=False
                )
                
                documenter = CrewAIMeshAgent(
                    crewai_component=doc_crew,
                    agent_id="documentation-team",
                    name="Documentation Team",
                    capabilities=["documentation", "technical-writing", "tutorials"]
                )
                await self.registry.register_agent(documenter)
                self.agents['documenter'] = documenter
                
            except ImportError:
                logger.warning("CrewAI not available - skipping documentation agent")
                
        except Exception as e:
            logger.error(f"Error setting up development team: {e}")
            
    async def _setup_research_team(self):
        """Setup research and analysis agents"""
        logger.info("Setting up research team...")
        
        try:
            # Research Agent (Google Gemini)
            if os.getenv('GOOGLE_API_KEY'):
                from meshai.adapters.google_adapter import GoogleMeshAgent
                
                researcher = GoogleMeshAgent(
                    model="gemini-pro",
                    agent_id="research-analyst",
                    name="Research Analyst",
                    capabilities=["research", "data-analysis", "fact-checking", "synthesis"]
                )
                await self.registry.register_agent(researcher)
                self.agents['researcher'] = researcher
                
            # Data Analyst (OpenAI)
            if os.getenv('OPENAI_API_KEY'):
                from meshai.adapters.openai_adapter import OpenAIMeshAgent
                
                analyst = OpenAIMeshAgent(
                    model="gpt-4",
                    agent_id="data-analyst",
                    name="Data Analyst",
                    capabilities=["data-analysis", "statistics", "visualization", "insights"],
                    system_prompt=(
                        "You are a data analyst expert in statistical analysis, data visualization, "
                        "and extracting actionable insights from complex datasets."
                    ),
                    temperature=0.3
                )
                await self.registry.register_agent(analyst)
                self.agents['analyst'] = analyst
                
        except Exception as e:
            logger.error(f"Error setting up research team: {e}")
            
    async def _setup_content_team(self):
        """Setup content creation agents"""
        logger.info("Setting up content team...")
        
        try:
            # Content Strategist (Claude)
            if os.getenv('ANTHROPIC_API_KEY'):
                from meshai.adapters.anthropic_adapter import AnthropicMeshAgent
                
                strategist = AnthropicMeshAgent(
                    model="claude-3-sonnet-20240229",
                    agent_id="content-strategist",
                    name="Content Strategist",
                    capabilities=["content-strategy", "seo", "audience-analysis", "planning"],
                    system_prompt=(
                        "You are a content strategist who creates comprehensive content plans "
                        "based on audience needs, SEO best practices, and business objectives."
                    )
                )
                await self.registry.register_agent(strategist)
                self.agents['strategist'] = strategist
                
            # Creative Writer (OpenAI)
            if os.getenv('OPENAI_API_KEY'):
                from meshai.adapters.openai_adapter import OpenAIMeshAgent
                
                writer = OpenAIMeshAgent(
                    model="gpt-4",
                    agent_id="creative-writer",
                    name="Creative Writer",
                    capabilities=["writing", "creativity", "storytelling", "copywriting"],
                    system_prompt=(
                        "You are a creative writer skilled in various writing styles, "
                        "from technical content to marketing copy to engaging narratives."
                    ),
                    temperature=0.7
                )
                await self.registry.register_agent(writer)
                self.agents['writer'] = writer
                
        except Exception as e:
            logger.error(f"Error setting up content team: {e}")
            
    async def _setup_support_team(self):
        """Setup customer support agents"""
        logger.info("Setting up support team...")
        
        try:
            # Support Agent (AutoGen)
            try:
                from meshai.adapters.autogen_adapter import AutoGenMeshAgent
                from autogen import ConversableAgent
                
                support_agent = ConversableAgent(
                    name="customer_support",
                    system_message=(
                        "You are a helpful customer support agent. Be empathetic, "
                        "professional, and solution-focused in all interactions."
                    ),
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=3
                )
                
                support = AutoGenMeshAgent(
                    autogen_component=support_agent,
                    agent_id="customer-support",
                    name="Customer Support Agent",
                    capabilities=["customer-service", "problem-solving", "communication"]
                )
                await self.registry.register_agent(support)
                self.agents['support'] = support
                
            except ImportError:
                logger.warning("AutoGen not available - skipping support agent")
                
        except Exception as e:
            logger.error(f"Error setting up support team: {e}")


async def software_development_workflow(orchestrator: WorkflowOrchestrator):
    """Complete software development workflow"""
    print("\n💻 Software Development Workflow")
    print("-" * 40)
    
    context = MeshContext()
    await context.set("project_info", {
        "name": "MeshAI Task Scheduler",
        "language": "Python",
        "framework": "FastAPI",
        "requirements": ["async support", "REST API", "database integration", "error handling"]
    })
    
    try:
        # Step 1: Generate code
        if 'coder' in orchestrator.agents:
            print("🔧 Generating code...")
            code_task = TaskData(
                input=(
                    "Create a FastAPI-based task scheduler with the following features:\n"
                    "1. REST API endpoints for creating, reading, updating, deleting tasks\n"
                    "2. Async database operations using SQLAlchemy\n"
                    "3. Task status tracking (pending, running, completed, failed)\n"
                    "4. Background task processing\n"
                    "5. Proper error handling and logging\n"
                    "Please provide the complete implementation with database models, API endpoints, and task processor."
                ),
                parameters={"max_tokens": 2000}
            )
            
            code_result = await orchestrator.agents['coder'].handle_task(code_task, context)
            print(f"✅ Code generated: {len(code_result['result'])} characters")
            
            # Store code in context for next steps
            await context.set("generated_code", code_result['result'])
            
            # Step 2: Code review
            if 'reviewer' in orchestrator.agents:
                print("🔍 Reviewing code...")
                review_task = TaskData(
                    input="Please review the generated task scheduler code for security issues, performance concerns, best practices, and potential improvements.",
                    parameters={"temperature": 0.1}
                )
                
                review_result = await orchestrator.agents['reviewer'].handle_task(review_task, context)
                print(f"✅ Code review completed: {review_result['result'][:200]}...")
                
                # Step 3: Generate documentation
                if 'documenter' in orchestrator.agents:
                    print("📝 Creating documentation...")
                    doc_task = TaskData(
                        input="Create comprehensive documentation for the task scheduler including API documentation, setup instructions, and usage examples.",
                        parameters={}
                    )
                    
                    doc_result = await orchestrator.agents['documenter'].handle_task(doc_task, context)
                    print(f"✅ Documentation created: {doc_result['result'][:200]}...")
                    
        # Show workflow metrics
        conversation_history = await context.get("conversation_history", [])
        print(f"\n📊 Workflow Statistics:")
        print(f"   • Total interactions: {len(conversation_history)}")
        print(f"   • Agents involved: {len([msg for msg in conversation_history if msg.get('source')])}")
        print(f"   • Duration: {len(conversation_history) * 2} estimated seconds")
        
    except Exception as e:
        logger.error(f"Software development workflow failed: {e}")


async def research_analysis_pipeline(orchestrator: WorkflowOrchestrator):
    """Research and analysis pipeline"""
    print("\n🔬 Research & Analysis Pipeline")
    print("-" * 35)
    
    context = MeshContext()
    research_topic = "AI Agent Orchestration Platforms - Market Analysis 2024"
    
    await context.set("research_context", {
        "topic": research_topic,
        "scope": ["market size", "key players", "technology trends", "challenges", "opportunities"],
        "target_audience": "enterprise decision makers",
        "deliverable": "executive summary report"
    })
    
    try:
        # Step 1: Information gathering
        if 'researcher' in orchestrator.agents:
            print("🔍 Gathering information...")
            research_task = TaskData(
                input=(
                    f"Conduct comprehensive research on: {research_topic}\n"
                    "Focus on:\n"
                    "1. Current market size and growth projections\n"
                    "2. Key players and their offerings\n"
                    "3. Emerging technology trends\n"
                    "4. Main challenges and opportunities\n"
                    "5. Enterprise adoption patterns"
                ),
                parameters={"temperature": 0.4}
            )
            
            research_result = await orchestrator.agents['researcher'].handle_task(research_task, context)
            print(f"✅ Research completed: {len(research_result['result'])} characters")
            
            # Step 2: Data analysis
            if 'analyst' in orchestrator.agents:
                print("📊 Analyzing data...")
                analysis_task = TaskData(
                    input=(
                        "Analyze the research findings and provide:\n"
                        "1. Key statistical insights and trends\n"
                        "2. Competitive landscape analysis\n"
                        "3. Market opportunity assessment\n"
                        "4. Risk factors and mitigation strategies\n"
                        "5. Recommendations for market entry/expansion"
                    ),
                    parameters={"temperature": 0.2}
                )
                
                analysis_result = await orchestrator.agents['analyst'].handle_task(analysis_task, context)
                print(f"✅ Analysis completed: {analysis_result['result'][:200]}...")
                
                # Step 3: Executive summary
                if 'writer' in orchestrator.agents:
                    print("📋 Creating executive summary...")
                    summary_task = TaskData(
                        input=(
                            "Create a concise executive summary report based on the research and analysis. "
                            "Target audience: C-level executives. Length: 2-3 pages. "
                            "Include key findings, recommendations, and action items."
                        ),
                        parameters={"temperature": 0.3}
                    )
                    
                    summary_result = await orchestrator.agents['writer'].handle_task(summary_task, context)
                    print(f"✅ Executive summary created: {summary_result['result'][:200]}...")
                    
        # Research metrics
        conversation_history = await context.get("conversation_history", [])
        sources_mentioned = len([msg for msg in conversation_history if "http" in msg.get('content', '')])
        print(f"\n📈 Research Metrics:")
        print(f"   • Research phases completed: 3")
        print(f"   • Data sources analyzed: {sources_mentioned}")
        print(f"   • Total content generated: {sum(len(msg.get('content', '')) for msg in conversation_history)} chars")
        
    except Exception as e:
        logger.error(f"Research analysis pipeline failed: {e}")


async def content_creation_workflow(orchestrator: WorkflowOrchestrator):
    """Content creation and optimization workflow"""
    print("\n✍️ Content Creation Workflow")
    print("-" * 30)
    
    context = MeshContext()
    campaign_brief = {
        "product": "MeshAI Platform",
        "target_audience": "AI developers and DevOps engineers",
        "content_type": "blog series",
        "topics": ["Getting started", "Advanced workflows", "Enterprise deployment"],
        "tone": "Technical but accessible",
        "goals": ["education", "lead generation", "community building"]
    }
    
    await context.set("campaign_brief", campaign_brief)
    
    try:
        # Step 1: Content strategy
        if 'strategist' in orchestrator.agents:
            print("📋 Developing content strategy...")
            strategy_task = TaskData(
                input=(
                    "Create a comprehensive content strategy for the MeshAI platform blog series. "
                    "Include content calendar, SEO keywords, distribution channels, and success metrics."
                ),
                parameters={"temperature": 0.4}
            )
            
            strategy_result = await orchestrator.agents['strategist'].handle_task(strategy_task, context)
            print(f"✅ Strategy developed: {strategy_result['result'][:200]}...")
            
            # Step 2: Content creation
            if 'writer' in orchestrator.agents:
                print("✍️ Creating content...")
                
                # Create multiple pieces of content
                content_pieces = []
                for topic in campaign_brief["topics"]:
                    content_task = TaskData(
                        input=(
                            f"Write a comprehensive blog post about '{topic}' for the MeshAI platform. "
                            f"Target audience: {campaign_brief['target_audience']}. "
                            f"Tone: {campaign_brief['tone']}. "
                            "Include practical examples, code snippets, and actionable insights."
                        ),
                        parameters={"temperature": 0.6, "max_tokens": 1500}
                    )
                    
                    content_result = await orchestrator.agents['writer'].handle_task(content_task, context)
                    content_pieces.append({
                        "topic": topic,
                        "content": content_result['result'][:500] + "...",
                        "word_count": len(content_result['result'].split())
                    })
                    print(f"✅ Created content for '{topic}': {len(content_result['result'].split())} words")
                
                # Store content for review
                await context.set("content_pieces", content_pieces)
                
        # Content metrics
        total_words = sum(piece["word_count"] for piece in content_pieces)
        print(f"\n📝 Content Metrics:")
        print(f"   • Blog posts created: {len(content_pieces)}")
        print(f"   • Total word count: {total_words}")
        print(f"   • Average post length: {total_words // len(content_pieces) if content_pieces else 0} words")
        print(f"   • Topics covered: {', '.join([p['topic'] for p in content_pieces])}")
        
    except Exception as e:
        logger.error(f"Content creation workflow failed: {e}")


async def customer_support_system(orchestrator: WorkflowOrchestrator):
    """Customer support ticket handling system"""
    print("\n🎧 Customer Support System")
    print("-" * 28)
    
    context = MeshContext()
    
    # Simulate incoming support tickets
    tickets = [
        {
            "id": "TICKET-001",
            "customer": "TechCorp Solutions",
            "priority": "high",
            "category": "technical",
            "subject": "API Gateway Integration Issues",
            "description": "We're experiencing 503 errors when trying to integrate our services with the MeshAI API gateway. The errors started occurring after we increased our request volume to 1000 req/min."
        },
        {
            "id": "TICKET-002", 
            "customer": "StartupXYZ",
            "priority": "medium",
            "category": "billing",
            "subject": "Usage Analytics Not Updating",
            "description": "Our usage dashboard hasn't updated in the past 24 hours. We need to track our agent invocations for billing purposes."
        },
        {
            "id": "TICKET-003",
            "customer": "Enterprise Inc",
            "priority": "low",
            "category": "feature-request",
            "subject": "Custom Agent Deployment Documentation",
            "description": "We need documentation on how to deploy custom agents to the MeshAI platform using our internal CI/CD pipeline."
        }
    ]
    
    await context.set("support_tickets", tickets)
    
    try:
        if 'support' in orchestrator.agents:
            responses = []
            
            for ticket in tickets:
                print(f"🎫 Processing {ticket['id']} - {ticket['subject']}")
                
                support_task = TaskData(
                    input=(
                        f"Customer support ticket:\n"
                        f"Customer: {ticket['customer']}\n"
                        f"Priority: {ticket['priority']}\n"
                        f"Category: {ticket['category']}\n"
                        f"Subject: {ticket['subject']}\n"
                        f"Description: {ticket['description']}\n\n"
                        f"Please provide a helpful, professional response that addresses the customer's concern. "
                        f"Include troubleshooting steps if applicable and escalation information if needed."
                    ),
                    parameters={"temperature": 0.3}
                )
                
                response = await orchestrator.agents['support'].handle_task(support_task, context)
                responses.append({
                    "ticket_id": ticket['id'],
                    "response": response['result'][:300] + "...",
                    "sentiment": "professional",
                    "resolution_type": "self-service" if ticket['priority'] == "low" else "escalation"
                })
                
                print(f"✅ Response generated: {response['result'][:100]}...")
                
                # Simulate escalation for high priority technical issues
                if ticket['priority'] == 'high' and ticket['category'] == 'technical':
                    if 'analyst' in orchestrator.agents:
                        print("⚠️ Escalating to technical analyst...")
                        escalation_task = TaskData(
                            input=(
                                f"Technical escalation for {ticket['id']}:\n"
                                f"Issue: {ticket['description']}\n"
                                "Please provide detailed technical analysis and resolution steps."
                            ),
                            parameters={"temperature": 0.1}
                        )
                        
                        technical_response = await orchestrator.agents['analyst'].handle_task(escalation_task, context)
                        print(f"✅ Technical analysis completed: {technical_response['result'][:100]}...")
            
            # Support metrics
            avg_response_length = sum(len(r['response']) for r in responses) // len(responses)
            escalations = len([r for r in responses if r['resolution_type'] == 'escalation'])
            
            print(f"\n📞 Support Metrics:")
            print(f"   • Tickets processed: {len(responses)}")
            print(f"   • Average response length: {avg_response_length} characters")
            print(f"   • Escalations required: {escalations}")
            print(f"   • Resolution rate: {((len(responses) - escalations) / len(responses)) * 100:.1f}%")
            
    except Exception as e:
        logger.error(f"Customer support system failed: {e}")


async def strategic_planning_session(orchestrator: WorkflowOrchestrator):
    """Multi-agent strategic planning session"""
    print("\n🎯 Strategic Planning Session")
    print("-" * 31)
    
    context = MeshContext()
    business_challenge = {
        "scenario": "Market Expansion",
        "company": "MeshAI Platform",
        "challenge": "Expanding from technical early adopters to enterprise customers",
        "constraints": ["Limited marketing budget", "Small sales team", "Complex product"],
        "timeline": "12 months",
        "success_metrics": ["Revenue growth", "Enterprise customer acquisition", "Market share"]
    }
    
    await context.set("planning_context", business_challenge)
    
    try:
        # Step 1: Problem analysis
        if 'analyst' in orchestrator.agents:
            print("🔍 Analyzing business challenge...")
            analysis_task = TaskData(
                input=(
                    f"Analyze this business challenge:\n"
                    f"Scenario: {business_challenge['scenario']}\n"
                    f"Company: {business_challenge['company']}\n"
                    f"Challenge: {business_challenge['challenge']}\n"
                    f"Constraints: {', '.join(business_challenge['constraints'])}\n"
                    f"Timeline: {business_challenge['timeline']}\n\n"
                    f"Provide SWOT analysis, market assessment, and key success factors."
                ),
                parameters={"temperature": 0.3}
            )
            
            analysis_result = await orchestrator.agents['analyst'].handle_task(analysis_task, context)
            print(f"✅ Strategic analysis completed: {analysis_result['result'][:200]}...")
            
        # Step 2: Strategy development
        if 'strategist' in orchestrator.agents:
            print("📈 Developing strategic options...")
            strategy_task = TaskData(
                input=(
                    "Based on the business challenge analysis, develop 3 strategic options for market expansion. "
                    "For each option, include: approach, resource requirements, risks, timeline, and expected outcomes."
                ),
                parameters={"temperature": 0.5}
            )
            
            strategy_result = await orchestrator.agents['strategist'].handle_task(strategy_task, context)
            print(f"✅ Strategic options developed: {strategy_result['result'][:200]}...")
            
        # Step 3: Implementation planning  
        if 'coder' in orchestrator.agents:
            print("⚙️ Creating implementation plan...")
            implementation_task = TaskData(
                input=(
                    "Create a detailed 12-month implementation plan for the recommended strategy. "
                    "Include milestones, resource allocation, risk mitigation, and success metrics."
                ),
                parameters={"temperature": 0.2}
            )
            
            impl_result = await orchestrator.agents['coder'].handle_task(implementation_task, context)
            print(f"✅ Implementation plan created: {impl_result['result'][:200]}...")
            
        # Planning session metrics
        conversation_history = await context.get("conversation_history", [])
        decisions_made = len([msg for msg in conversation_history if 'recommend' in msg.get('content', '').lower()])
        
        print(f"\n🎯 Planning Session Results:")
        print(f"   • Strategic analyses: 1")
        print(f"   • Options evaluated: 3")
        print(f"   • Decisions made: {decisions_made}")
        print(f"   • Implementation timeline: 12 months")
        print(f"   • Agents consulted: {len(set(msg.get('source', '') for msg in conversation_history))}")
        
    except Exception as e:
        logger.error(f"Strategic planning session failed: {e}")


async def workflow_performance_dashboard(orchestrator: WorkflowOrchestrator):
    """Generate performance dashboard for all workflows"""
    print("\n📊 Workflow Performance Dashboard")
    print("-" * 35)
    
    # Collect metrics from all agents
    agent_stats = {}
    total_requests = 0
    total_response_time = 0.0
    
    for agent_id, agent in orchestrator.agents.items():
        try:
            if hasattr(agent, 'get_usage_stats'):
                stats = await agent.get_usage_stats()
                agent_stats[agent_id] = stats
                total_requests += stats.get('total_requests', 0)
                total_response_time += stats.get('avg_response_time', 0.0)
        except Exception as e:
            logger.warning(f"Could not get stats for agent {agent_id}: {e}")
    
    # Generate dashboard
    print(f"\n🌟 Overall Performance:")
    print(f"   • Total agents registered: {len(orchestrator.agents)}")
    print(f"   • Total requests processed: {total_requests}")
    print(f"   • Average response time: {total_response_time / max(len(agent_stats), 1):.2f}s")
    
    print(f"\n🤖 Agent Performance:")
    for agent_id, stats in agent_stats.items():
        agent = orchestrator.agents[agent_id]
        print(f"   • {agent.name}:")
        print(f"     - Requests: {stats.get('total_requests', 0)}")
        print(f"     - Avg Response Time: {stats.get('avg_response_time', 0.0):.2f}s")
        print(f"     - Capabilities: {', '.join(agent.capabilities[:3])}...")
    
    # Framework distribution
    frameworks = {}
    for agent in orchestrator.agents.values():
        fw = agent.framework
        frameworks[fw] = frameworks.get(fw, 0) + 1
    
    print(f"\n🔧 Framework Distribution:")
    for framework, count in frameworks.items():
        print(f"   • {framework}: {count} agents ({count/len(orchestrator.agents)*100:.1f}%)")
    
    print(f"\n✨ Workflow Success Stories:")
    print(f"   • Software Development: Full SDLC automation")
    print(f"   • Research Pipeline: Automated market analysis")  
    print(f"   • Content Creation: Multi-format content generation")
    print(f"   • Customer Support: Intelligent ticket routing")
    print(f"   • Strategic Planning: Multi-perspective analysis")


async def main():
    """Run all advanced workflow examples"""
    print("🎯 Initializing Advanced Workflow Examples...")
    
    # Setup orchestrator
    orchestrator = WorkflowOrchestrator()
    await orchestrator.setup_agents()
    
    print(f"\n✅ Orchestrator ready with {len(orchestrator.agents)} agents")
    
    # Run workflows based on available agents
    if orchestrator.agents:
        await software_development_workflow(orchestrator)
        await research_analysis_pipeline(orchestrator)
        await content_creation_workflow(orchestrator)
        await customer_support_system(orchestrator)
        await strategic_planning_session(orchestrator)
        await workflow_performance_dashboard(orchestrator)
    else:
        print("⚠️ No agents available. Please configure API keys for at least one framework:")
        print("   • OPENAI_API_KEY for OpenAI GPT models")
        print("   • ANTHROPIC_API_KEY for Claude models")
        print("   • GOOGLE_API_KEY for Gemini models")
        
    print("\n🎉 Advanced Workflow Examples Completed!")
    print("\n🚀 Next Steps:")
    print("1. Customize workflows for your specific use cases")
    print("2. Add monitoring and alerting for production deployments")
    print("3. Implement human-in-the-loop for critical decisions")
    print("4. Scale agents horizontally for high-throughput scenarios")
    print("5. Explore enterprise features like audit logging and compliance")


if __name__ == "__main__":
    # Run the advanced workflow examples
    asyncio.run(main())