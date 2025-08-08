#!/usr/bin/env python3
"""
Advanced Routing Algorithms Example

This example demonstrates the sophisticated routing capabilities of MeshAI:
- Performance-based agent selection
- Sticky sessions for user affinity
- Advanced load balancing strategies
- ML-powered routing decisions
- Real-time routing analytics
"""

import asyncio
import logging
import random
import time
from datetime import timedelta
from typing import Any, Dict, List

from meshai.core import MeshAgent, MeshContext
from meshai.core.config import MeshConfig
from meshai.core.routing_engine import (
    AdvancedRoutingEngine, RoutingContext, RoutingStrategy,
    RoutingPolicy, SessionAffinityType, LoadBalancingPolicy,
    routing_engine
)
from meshai.core.routing_ml import MLRoutingEngine, ml_routing_engine
from meshai.core.routing_metrics import (
    RoutingMetricsCollector, MetricPeriod, routing_metrics
)
from meshai.core.failover_manager import failover_manager, AgentEndpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkerAgent(MeshAgent):
    """Generic worker agent with variable performance"""
    
    def __init__(self, agent_id: str, performance_profile: str = "normal"):
        super().__init__(agent_id)
        self.performance_profile = performance_profile
        self.request_count = 0
        
        # Set performance characteristics
        if performance_profile == "fast":
            self.base_response_time = 0.1
            self.variability = 0.05
            self.error_rate = 0.01
        elif performance_profile == "slow":
            self.base_response_time = 2.0
            self.variability = 0.5
            self.error_rate = 0.05
        elif performance_profile == "unreliable":
            self.base_response_time = 0.5
            self.variability = 0.3
            self.error_rate = 0.2
        else:  # normal
            self.base_response_time = 0.5
            self.variability = 0.1
            self.error_rate = 0.02
    
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Process task with simulated performance characteristics"""
        self.request_count += 1
        
        # Simulate variable response time
        response_time = self.base_response_time + random.uniform(-self.variability, self.variability)
        
        # Simulate degraded performance under load
        if self.request_count > 50:
            response_time *= 1.2  # 20% slower under load
        
        await asyncio.sleep(response_time)
        
        # Simulate errors
        if random.random() < self.error_rate:
            raise Exception(f"Agent {self.agent_id} processing error")
        
        return {
            "status": "success",
            "agent_id": self.agent_id,
            "response_time": response_time,
            "request_count": self.request_count,
            "data": task_data.get("data", "processed")
        }


async def setup_routing_system():
    """Set up the advanced routing system"""
    
    print("=== Setting Up Advanced Routing System ===\n")
    
    # 1. Start routing engine
    print("1. Starting routing engine...")
    await routing_engine.start()
    
    # 2. Start ML routing engine
    print("2. Starting ML routing engine...")
    await ml_routing_engine.start()
    
    # 3. Start metrics collection
    print("3. Starting metrics collection...")
    await routing_metrics.start()
    
    # 4. Configure routing policies
    print("4. Configuring routing policies...")
    
    # Performance-based routing for data processing
    data_processing_policy = RoutingPolicy(
        name="data_processing",
        capability_patterns=["data-processing", "analytics"],
        primary_strategy=RoutingStrategy.PERFORMANCE_BASED,
        fallback_strategies=[RoutingStrategy.LEAST_CONNECTIONS, RoutingStrategy.ROUND_ROBIN],
        load_balancing_policy=LoadBalancingPolicy.POWER_OF_TWO,
        weight_factors={
            'response_time': 0.4,
            'success_rate': 0.3,
            'current_load': 0.2,
            'trend': 0.1
        }
    )
    routing_engine.add_routing_policy(data_processing_policy)
    
    # Sticky session routing for user requests
    user_session_policy = RoutingPolicy(
        name="user_sessions",
        capability_patterns=["user-service", "session-handler"],
        primary_strategy=RoutingStrategy.STICKY_SESSION,
        fallback_strategies=[RoutingStrategy.CONSISTENT_HASH],
        session_affinity=SessionAffinityType.USER_ID,
        session_ttl=timedelta(minutes=30)
    )
    routing_engine.add_routing_policy(user_session_policy)
    
    # Adaptive routing for general tasks
    adaptive_policy = RoutingPolicy(
        name="adaptive",
        capability_patterns=["general", "task-processor"],
        primary_strategy=RoutingStrategy.ADAPTIVE_WEIGHTED,
        load_balancing_policy=LoadBalancingPolicy.LEAST_LOADED
    )
    routing_engine.add_routing_policy(adaptive_policy)
    
    print("✅ Routing system setup complete!\n")


async def register_agents():
    """Register worker agents with different performance profiles"""
    
    print("=== Registering Worker Agents ===\n")
    
    agents = []
    
    # Fast agents (2)
    for i in range(2):
        agent = WorkerAgent(f"fast-agent-{i+1}", "fast")
        agents.append(agent)
        
        # Register with failover manager for routing
        failover_manager.register_agent(
            agent_id=agent.agent_id,
            endpoint_url=f"http://{agent.agent_id}:8080",
            capabilities=["data-processing", "general"],
            weight=2.0,  # Higher weight for fast agents
            metadata={"performance": "fast"}
        )
    
    # Normal agents (3)
    for i in range(3):
        agent = WorkerAgent(f"normal-agent-{i+1}", "normal")
        agents.append(agent)
        
        failover_manager.register_agent(
            agent_id=agent.agent_id,
            endpoint_url=f"http://{agent.agent_id}:8080",
            capabilities=["data-processing", "general", "user-service"],
            weight=1.0,
            metadata={"performance": "normal"}
        )
    
    # Slow agent (1)
    agent = WorkerAgent("slow-agent-1", "slow")
    agents.append(agent)
    
    failover_manager.register_agent(
        agent_id=agent.agent_id,
        endpoint_url=f"http://{agent.agent_id}:8080",
        capabilities=["data-processing", "general"],
        weight=0.5,  # Lower weight for slow agent
        metadata={"performance": "slow"}
    )
    
    # Unreliable agent (1)
    agent = WorkerAgent("unreliable-agent-1", "unreliable")
    agents.append(agent)
    
    failover_manager.register_agent(
        agent_id=agent.agent_id,
        endpoint_url=f"http://{agent.agent_id}:8080",
        capabilities=["general"],
        weight=0.3,  # Very low weight for unreliable agent
        metadata={"performance": "unreliable"}
    )
    
    print(f"✅ Registered {len(agents)} worker agents\n")
    print("Agent Performance Profiles:")
    print("- Fast agents: 2 (low latency, high reliability)")
    print("- Normal agents: 3 (balanced performance)")
    print("- Slow agent: 1 (high latency, reliable)")
    print("- Unreliable agent: 1 (moderate latency, high error rate)\n")
    
    return agents


async def demonstrate_routing_strategies(agents: List[WorkerAgent]):
    """Demonstrate different routing strategies"""
    
    print("=== Demonstrating Routing Strategies ===\n")
    
    # 1. Round Robin
    print("1. Round Robin Routing:")
    for i in range(6):
        context = RoutingContext(
            request_id=f"rr-{i}",
            capability="general"
        )
        
        decision = await routing_engine.route_request(context)
        if decision:
            print(f"   Request {i}: -> {decision.selected_agent.agent_id}")
    
    print()
    
    # 2. Performance-based routing
    print("2. Performance-Based Routing:")
    for i in range(5):
        context = RoutingContext(
            request_id=f"perf-{i}",
            capability="data-processing",
            priority=2
        )
        
        decision = await routing_engine.route_request(context)
        if decision:
            factors = decision.decision_factors
            print(f"   Request {i}: -> {decision.selected_agent.agent_id}")
            print(f"      Score: {factors.get('total_score', 0):.3f}, "
                  f"Confidence: {decision.confidence_score:.3f}")
    
    print()
    
    # 3. Sticky Sessions
    print("3. Sticky Session Routing:")
    user_ids = ["user-123", "user-456", "user-789"]
    
    for user_id in user_ids:
        print(f"   User {user_id}:")
        for i in range(3):
            context = RoutingContext(
                request_id=f"sticky-{user_id}-{i}",
                user_id=user_id,
                capability="user-service"
            )
            
            decision = await routing_engine.route_request(context)
            if decision:
                print(f"      Request {i}: -> {decision.selected_agent.agent_id}")
        print()
    
    # 4. ML-Optimized routing
    print("4. ML-Optimized Routing (after training):")
    
    # Train the ML model with some sample data
    for _ in range(20):
        context = RoutingContext(
            request_id=f"train-{_}",
            capability="data-processing"
        )
        
        # Simulate routing and record performance
        for agent in agents[:3]:  # Use first 3 agents for training
            await ml_routing_engine.record_performance(
                context,
                agent.agent_id,
                agent.base_response_time + random.uniform(-0.1, 0.1),
                random.random() > agent.error_rate
            )
    
    # Now use ML for routing
    for i in range(5):
        context = RoutingContext(
            request_id=f"ml-{i}",
            capability="data-processing"
        )
        
        ml_decision = await ml_routing_engine.select_optimal_agent(
            context,
            [AgentEndpoint(
                agent_id=agent.agent_id,
                endpoint_url=f"http://{agent.agent_id}:8080",
                capabilities=["data-processing"]
            ) for agent in agents[:5]]
        )
        
        if ml_decision:
            factors = ml_decision.decision_factors
            print(f"   Request {i}: -> {ml_decision.selected_agent.agent_id}")
            print(f"      Predicted RT: {factors.get('predicted_response_time', 0):.3f}s, "
                  f"Success Prob: {factors.get('predicted_success_probability', 0):.3f}")
    
    print()


async def simulate_load_and_collect_metrics(agents: List[WorkerAgent], duration_seconds: int = 60):
    """Simulate load and collect routing metrics"""
    
    print(f"=== Simulating Load for {duration_seconds} seconds ===\n")
    
    start_time = time.time()
    request_count = 0
    success_count = 0
    
    capabilities = ["data-processing", "general", "user-service"]
    user_pool = [f"user-{i}" for i in range(10)]
    
    while time.time() - start_time < duration_seconds:
        # Create request context
        capability = random.choice(capabilities)
        user_id = random.choice(user_pool) if capability == "user-service" else None
        
        context = RoutingContext(
            request_id=f"load-{request_count}",
            user_id=user_id,
            capability=capability,
            priority=random.randint(1, 3)
        )
        
        # Route request
        routing_start = time.time()
        decision = await routing_engine.route_request(context)
        routing_duration = time.time() - routing_start
        
        if decision:
            # Find the actual agent
            agent = next(
                (a for a in agents if a.agent_id == decision.selected_agent.agent_id),
                None
            )
            
            if agent:
                try:
                    # Execute task
                    result = await agent.handle_task(
                        {"data": f"request-{request_count}"},
                        MeshContext()
                    )
                    
                    # Record success
                    routing_metrics.record_routing_decision(
                        context,
                        decision,
                        routing_duration,
                        success=True,
                        response_time=result.get("response_time", 1.0)
                    )
                    
                    # Update ML model
                    await ml_routing_engine.record_performance(
                        context,
                        agent.agent_id,
                        result.get("response_time", 1.0),
                        True
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    # Record failure
                    routing_metrics.record_routing_decision(
                        context,
                        decision,
                        routing_duration,
                        success=False
                    )
                    
                    # Update ML model
                    await ml_routing_engine.record_performance(
                        context,
                        agent.agent_id,
                        10.0,  # High penalty for failure
                        False
                    )
        
        request_count += 1
        
        # Progress update
        if request_count % 50 == 0:
            elapsed = time.time() - start_time
            rate = request_count / elapsed
            print(f"Processed {request_count} requests ({success_count} successful) - "
                  f"{rate:.1f} req/s")
        
        # Variable load
        await asyncio.sleep(random.uniform(0.01, 0.1))
    
    print(f"\n✅ Load simulation complete!")
    print(f"Total requests: {request_count}")
    print(f"Successful: {success_count}")
    print(f"Success rate: {success_count/request_count*100:.1f}%\n")


async def display_routing_analytics():
    """Display comprehensive routing analytics"""
    
    print("=== Routing Analytics Dashboard ===\n")
    
    # Get analytics for different periods
    for period in [MetricPeriod.MINUTE, MetricPeriod.FIVE_MINUTES]:
        analytics = routing_metrics.get_routing_analytics(period)
        
        if analytics.get('latest_snapshot'):
            snapshot = analytics['latest_snapshot']
            print(f"Period: {period.value}")
            print(f"  Total Requests: {snapshot['total_requests']}")
            print(f"  Success Rate: {snapshot['success_rate']*100:.1f}%")
            print(f"  Avg Routing Time: {snapshot['avg_routing_time']*1000:.2f}ms")
            print(f"  Avg Confidence: {snapshot['avg_confidence']:.3f}")
            print(f"  Error Rate: {snapshot['error_rate']*100:.1f}%")
            print()
    
    # Strategy performance
    print("Top Routing Strategies:")
    analytics = routing_metrics.get_routing_analytics()
    for strategy in analytics.get('top_strategies', [])[:5]:
        print(f"  {strategy['strategy']}:")
        print(f"    Uses: {strategy['total_uses']}, "
              f"Success: {strategy['success_rate']*100:.1f}%, "
              f"Avg RT: {strategy['avg_response_time']*1000:.1f}ms")
    print()
    
    # Agent utilization
    print("Agent Utilization:")
    agent_util = analytics.get('agent_utilization', {})
    for agent_id, stats in list(agent_util.items())[:10]:
        print(f"  {agent_id}:")
        print(f"    Routed: {stats['total_routed']}, "
              f"Success: {stats['success_rate']*100:.1f}%, "
              f"Avg RT: {stats['avg_response_time']*1000:.1f}ms")
    print()
    
    # Routing engine stats
    routing_stats = routing_engine.get_routing_stats()
    print("Routing Engine Statistics:")
    print(f"  Total Decisions: {routing_stats['total_decisions']}")
    print(f"  Active Sticky Sessions: {routing_stats['active_sessions']}")
    print(f"  Active Policies: {routing_stats['routing_policies']}")
    
    strategy_dist = routing_stats.get('strategy_distribution', {})
    if strategy_dist:
        print("  Strategy Distribution:")
        for strategy, count in strategy_dist.items():
            print(f"    {strategy}: {count}")
    print()
    
    # ML engine stats
    ml_stats = ml_routing_engine.get_ml_stats()
    print("ML Routing Statistics:")
    print(f"  Total Training Examples: {ml_stats['total_decisions']}")
    print(f"  Models Trained: {ml_stats['models_trained']}")
    print(f"  Capabilities: {', '.join(ml_stats['capabilities'])}")
    
    capability_stats = ml_stats.get('capability_stats', {})
    for cap, stats in capability_stats.items():
        print(f"  {cap}:")
        print(f"    Total Selections: {stats['total_selections']}")
        print(f"    Exploration Rate: {stats['exploration_rate']:.3f}")
    print()
    
    # Agent recommendations
    print("Agent Recommendations (based on performance):")
    recommendations = routing_metrics.get_agent_recommendation("data-processing")
    for agent_id, score in recommendations[:5]:
        print(f"  {agent_id}: Score = {score:.3f}")


async def cleanup():
    """Clean up routing systems"""
    
    print("\n=== Cleaning Up ===")
    
    await routing_engine.stop()
    await ml_routing_engine.stop()
    await routing_metrics.stop()
    
    print("✅ Cleanup complete")


async def main():
    """Run the advanced routing demonstration"""
    
    print("🚀 MeshAI Advanced Routing Algorithms Demo")
    print("=" * 50)
    
    try:
        # Setup
        await setup_routing_system()
        agents = await register_agents()
        
        # Let systems initialize
        print("⏱️  Allowing systems to initialize...")
        await asyncio.sleep(2)
        
        # Demonstrate routing strategies
        await demonstrate_routing_strategies(agents)
        
        # Simulate load
        await simulate_load_and_collect_metrics(agents, duration_seconds=30)
        
        # Display analytics
        await display_routing_analytics()
        
        print("\n🎉 Advanced routing demonstration complete!")
        print("\nKey Capabilities Demonstrated:")
        print("✅ Performance-based agent selection with real-time metrics")
        print("✅ Sticky sessions for user affinity and session continuity")
        print("✅ Multiple load balancing strategies (Round Robin, Least Connections, etc.)")
        print("✅ ML-powered routing that learns from historical performance")
        print("✅ Consistent hashing for distributed routing")
        print("✅ Comprehensive routing analytics and metrics")
        print("✅ Adaptive weighted routing based on agent performance")
        print("✅ Policy-based routing with fallback strategies")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())