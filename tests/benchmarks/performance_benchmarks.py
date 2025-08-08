"""
Performance Benchmarking Suite for MeshAI

This module provides comprehensive performance benchmarks to measure
system throughput, latency, and resource utilization.
"""

import asyncio
import time
import psutil
import statistics
import gc
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from meshai.core import MeshAgent, MeshContext
from meshai.core.config import MeshConfig
from meshai.core.routing_engine import routing_engine, RoutingContext
from meshai.core.performance_monitor import performance_monitor
from meshai.core.failover_manager import failover_manager
from meshai.core.circuit_breaker import circuit_breaker_manager


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    benchmark_name: str
    duration_seconds: float
    total_operations: int
    operations_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    success_rate: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResourceUsage:
    """System resource usage during benchmark"""
    start_time: datetime
    end_time: datetime
    cpu_samples: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    
    @property
    def avg_cpu_usage(self) -> float:
        return statistics.mean(self.cpu_samples) if self.cpu_samples else 0
    
    @property
    def max_cpu_usage(self) -> float:
        return max(self.cpu_samples) if self.cpu_samples else 0
    
    @property
    def avg_memory_usage_mb(self) -> float:
        return statistics.mean(self.memory_samples) if self.memory_samples else 0
    
    @property
    def max_memory_usage_mb(self) -> float:
        return max(self.memory_samples) if self.memory_samples else 0


class ResourceMonitor:
    """Monitor system resources during benchmarks"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.usage = SystemResourceUsage(datetime.now(), datetime.now())
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        self._monitoring = True
        self.usage = SystemResourceUsage(datetime.now(), datetime.now())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> SystemResourceUsage:
        """Stop resource monitoring and return results"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.usage.end_time = datetime.now()
        return self.usage
    
    async def _monitor_loop(self):
        """Resource monitoring loop"""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.usage.cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = (memory.total - memory.available) / (1024 * 1024)
                self.usage.memory_samples.append(memory_mb)
                
                await asyncio.sleep(self.sample_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if individual samples fail
                pass


class BenchmarkAgent(MeshAgent):
    """High-performance test agent for benchmarking"""
    
    def __init__(self, agent_id: str, config: MeshConfig = None):
        super().__init__(agent_id, config)
        self.request_count = 0
        self.total_processing_time = 0
    
    async def handle_task(self, task_data: Dict[str, Any], context: MeshContext) -> Dict[str, Any]:
        """Process benchmark task"""
        start_time = time.time()
        self.request_count += 1
        
        # Simulate different types of work
        work_type = task_data.get("work_type", "light")
        
        if work_type == "light":
            # Light computational work
            await asyncio.sleep(0.001)
            result = {"computed": sum(range(100))}
        elif work_type == "medium":
            # Medium computational work
            await asyncio.sleep(0.01)
            result = {"computed": sum(range(1000))}
        elif work_type == "heavy":
            # Heavy computational work
            await asyncio.sleep(0.1)
            result = {"computed": sum(range(10000))}
        else:
            result = {"status": "unknown_work_type"}
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        return {
            "agent_id": self.agent_id,
            "request_count": self.request_count,
            "processing_time": processing_time,
            "work_type": work_type,
            "result": result
        }


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite.
    
    Includes benchmarks for:
    - Agent task processing throughput
    - Routing system performance
    - Context management performance
    - Circuit breaker overhead
    - Failover system performance
    - Concurrent operation scaling
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or MeshConfig()
        self.results: List[BenchmarkResult] = []
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        print("🚀 Starting MeshAI Performance Benchmark Suite")
        print("=" * 60)
        
        benchmarks = [
            self.benchmark_agent_throughput,
            self.benchmark_routing_performance,
            self.benchmark_context_performance,
            self.benchmark_circuit_breaker_overhead,
            self.benchmark_failover_performance,
            self.benchmark_concurrent_scaling,
            self.benchmark_memory_efficiency,
            self.benchmark_large_payload_handling,
        ]
        
        for benchmark in benchmarks:
            print(f"\n📊 Running {benchmark.__name__}...")
            try:
                result = await benchmark()
                self.results.append(result)
                self._print_benchmark_result(result)
            except Exception as e:
                print(f"❌ Benchmark {benchmark.__name__} failed: {e}")
        
        self._print_summary()
        return self.results
    
    async def benchmark_agent_throughput(self) -> BenchmarkResult:
        """Benchmark agent task processing throughput"""
        agent = BenchmarkAgent("throughput-agent", self.config)
        await agent.start()
        
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        # Benchmark parameters
        duration = 10.0  # 10 seconds
        concurrent_requests = 100
        
        start_time = time.time()
        tasks_completed = 0
        errors = 0
        latencies = []
        
        async def process_batch():
            nonlocal tasks_completed, errors
            batch_tasks = []
            
            for i in range(concurrent_requests):
                context = MeshContext()
                task = self._timed_agent_call(agent, {"work_type": "light"}, context)
                batch_tasks.append(task)
            
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                else:
                    tasks_completed += 1
                    latencies.append(result[1])  # latency from _timed_agent_call
        
        # Run batches until duration elapsed
        while time.time() - start_time < duration:
            await process_batch()
        
        total_duration = time.time() - start_time
        resource_usage = await monitor.stop_monitoring()
        
        await agent.stop()
        
        return BenchmarkResult(
            benchmark_name="Agent Throughput",
            duration_seconds=total_duration,
            total_operations=tasks_completed,
            operations_per_second=tasks_completed / total_duration,
            avg_latency_ms=statistics.mean(latencies) * 1000 if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 0.95) * 1000 if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 0.99) * 1000 if latencies else 0,
            error_rate=errors / (tasks_completed + errors) if (tasks_completed + errors) > 0 else 0,
            cpu_usage_percent=resource_usage.avg_cpu_usage,
            memory_usage_mb=resource_usage.avg_memory_usage_mb,
            success_rate=tasks_completed / (tasks_completed + errors) if (tasks_completed + errors) > 0 else 0,
            additional_metrics={
                "concurrent_requests": concurrent_requests,
                "max_cpu_usage": resource_usage.max_cpu_usage,
                "max_memory_mb": resource_usage.max_memory_usage_mb
            }
        )
    
    async def benchmark_routing_performance(self) -> BenchmarkResult:
        """Benchmark routing system performance"""
        await routing_engine.start()
        
        # Register test agents
        for i in range(10):
            failover_manager.register_agent(
                agent_id=f"routing-bench-agent-{i}",
                endpoint_url=f"http://agent-{i}:8080",
                capabilities=["benchmark"],
                weight=1.0
            )
        
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        duration = 5.0
        start_time = time.time()
        decisions_made = 0
        errors = 0
        latencies = []
        
        while time.time() - start_time < duration:
            # Create routing context
            context = RoutingContext(
                request_id=f"bench-{decisions_made}",
                capability="benchmark",
                user_id=f"user-{decisions_made % 10}"  # For sticky session testing
            )
            
            # Time routing decision
            decision_start = time.time()
            try:
                decision = await routing_engine.route_request(context)
                decision_time = time.time() - decision_start
                
                if decision:
                    decisions_made += 1
                    latencies.append(decision_time)
                else:
                    errors += 1
            except Exception:
                errors += 1
        
        total_duration = time.time() - start_time
        resource_usage = await monitor.stop_monitoring()
        
        await routing_engine.stop()
        
        return BenchmarkResult(
            benchmark_name="Routing Performance",
            duration_seconds=total_duration,
            total_operations=decisions_made,
            operations_per_second=decisions_made / total_duration,
            avg_latency_ms=statistics.mean(latencies) * 1000 if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 0.95) * 1000 if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 0.99) * 1000 if latencies else 0,
            error_rate=errors / (decisions_made + errors) if (decisions_made + errors) > 0 else 0,
            cpu_usage_percent=resource_usage.avg_cpu_usage,
            memory_usage_mb=resource_usage.avg_memory_usage_mb,
            success_rate=decisions_made / (decisions_made + errors) if (decisions_made + errors) > 0 else 0
        )
    
    async def benchmark_context_performance(self) -> BenchmarkResult:
        """Benchmark context management performance"""
        context = MeshContext(self.config)
        
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        operations = 10000
        start_time = time.time()
        
        # Write performance
        for i in range(operations):
            await context.set(f"key_{i}", {"value": i, "data": f"test_data_{i}"})
        
        write_time = time.time() - start_time
        
        # Read performance
        read_start = time.time()
        for i in range(operations):
            value = await context.get(f"key_{i}")
            assert value is not None
        
        read_time = time.time() - read_start
        
        # Update performance
        update_start = time.time()
        for i in range(operations // 2):
            await context.set(f"key_{i}", {"updated": True, "value": i * 2})
        
        update_time = time.time() - update_start
        
        total_duration = time.time() - start_time
        resource_usage = await monitor.stop_monitoring()
        
        total_ops = operations * 2 + (operations // 2)  # reads + writes + updates
        
        return BenchmarkResult(
            benchmark_name="Context Performance",
            duration_seconds=total_duration,
            total_operations=total_ops,
            operations_per_second=total_ops / total_duration,
            avg_latency_ms=(total_duration / total_ops) * 1000,
            p95_latency_ms=0,  # Not applicable for this benchmark
            p99_latency_ms=0,
            error_rate=0,
            cpu_usage_percent=resource_usage.avg_cpu_usage,
            memory_usage_mb=resource_usage.avg_memory_usage_mb,
            success_rate=1.0,
            additional_metrics={
                "write_ops_per_sec": operations / write_time,
                "read_ops_per_sec": operations / read_time,
                "update_ops_per_sec": (operations // 2) / update_time
            }
        )
    
    async def benchmark_circuit_breaker_overhead(self) -> BenchmarkResult:
        """Benchmark circuit breaker overhead"""
        from meshai.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        breaker = CircuitBreaker(
            "benchmark-breaker",
            CircuitBreakerConfig(failure_threshold=100, timeout=1.0)
        )
        
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        operations = 50000
        
        # Benchmark without circuit breaker
        start_time = time.time()
        for _ in range(operations):
            await self._simple_operation()
        no_breaker_time = time.time() - start_time
        
        # Benchmark with circuit breaker
        start_time = time.time()
        for _ in range(operations):
            await breaker.call(self._simple_operation)
        breaker_time = time.time() - start_time
        
        resource_usage = await monitor.stop_monitoring()
        
        overhead_ms = ((breaker_time - no_breaker_time) / operations) * 1000
        
        return BenchmarkResult(
            benchmark_name="Circuit Breaker Overhead",
            duration_seconds=breaker_time,
            total_operations=operations,
            operations_per_second=operations / breaker_time,
            avg_latency_ms=overhead_ms,
            p95_latency_ms=0,
            p99_latency_ms=0,
            error_rate=0,
            cpu_usage_percent=resource_usage.avg_cpu_usage,
            memory_usage_mb=resource_usage.avg_memory_usage_mb,
            success_rate=1.0,
            additional_metrics={
                "overhead_per_call_ms": overhead_ms,
                "no_breaker_ops_per_sec": operations / no_breaker_time,
                "with_breaker_ops_per_sec": operations / breaker_time
            }
        )
    
    async def benchmark_failover_performance(self) -> BenchmarkResult:
        """Benchmark failover system performance"""
        await failover_manager.start_monitoring()
        
        # Register agents
        for i in range(20):
            failover_manager.register_agent(
                agent_id=f"failover-agent-{i}",
                endpoint_url=f"http://agent-{i}:8080",
                capabilities=["failover-test"],
                weight=1.0
            )
        
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        operations = 10000
        start_time = time.time()
        selections = 0
        errors = 0
        
        for _ in range(operations):
            try:
                agent = await failover_manager.get_healthy_agent("failover-test")
                if agent:
                    selections += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
        
        total_duration = time.time() - start_time
        resource_usage = await monitor.stop_monitoring()
        
        await failover_manager.stop_monitoring()
        
        return BenchmarkResult(
            benchmark_name="Failover Performance",
            duration_seconds=total_duration,
            total_operations=selections,
            operations_per_second=selections / total_duration,
            avg_latency_ms=(total_duration / operations) * 1000,
            p95_latency_ms=0,
            p99_latency_ms=0,
            error_rate=errors / operations,
            cpu_usage_percent=resource_usage.avg_cpu_usage,
            memory_usage_mb=resource_usage.avg_memory_usage_mb,
            success_rate=selections / operations
        )
    
    async def benchmark_concurrent_scaling(self) -> BenchmarkResult:
        """Benchmark performance scaling with concurrent operations"""
        agents = []
        for i in range(5):
            agent = BenchmarkAgent(f"scaling-agent-{i}", self.config)
            await agent.start()
            agents.append(agent)
        
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        # Test different concurrency levels
        concurrency_levels = [1, 10, 50, 100, 500]
        results_by_concurrency = {}
        
        for concurrency in concurrency_levels:
            operations = 1000
            
            start_time = time.time()
            
            # Create batches of concurrent operations
            all_tasks = []
            for i in range(operations):
                agent = agents[i % len(agents)]
                context = MeshContext()
                task = self._timed_agent_call(
                    agent,
                    {"work_type": "light"},
                    context
                )
                all_tasks.append(task)
            
            # Execute in batches of specified concurrency
            batch_results = []
            for i in range(0, len(all_tasks), concurrency):
                batch = all_tasks[i:i + concurrency]
                batch_result = await asyncio.gather(*batch, return_exceptions=True)
                batch_results.extend(batch_result)
            
            duration = time.time() - start_time
            successful = len([r for r in batch_results if not isinstance(r, Exception)])
            
            results_by_concurrency[concurrency] = {
                "ops_per_sec": successful / duration,
                "success_rate": successful / len(batch_results)
            }
        
        resource_usage = await monitor.stop_monitoring()
        
        for agent in agents:
            await agent.stop()
        
        # Use results from highest concurrency level
        best_concurrency = max(concurrency_levels)
        best_result = results_by_concurrency[best_concurrency]
        
        return BenchmarkResult(
            benchmark_name="Concurrent Scaling",
            duration_seconds=0,  # Variable across tests
            total_operations=1000 * len(concurrency_levels),
            operations_per_second=best_result["ops_per_sec"],
            avg_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            error_rate=1 - best_result["success_rate"],
            cpu_usage_percent=resource_usage.avg_cpu_usage,
            memory_usage_mb=resource_usage.avg_memory_usage_mb,
            success_rate=best_result["success_rate"],
            additional_metrics={
                "scaling_results": results_by_concurrency,
                "max_concurrency": best_concurrency
            }
        )
    
    async def benchmark_memory_efficiency(self) -> BenchmarkResult:
        """Benchmark memory efficiency and garbage collection"""
        import gc
        
        gc.collect()  # Start with clean state
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # Create many agents and contexts
        agents = []
        contexts = []
        
        start_time = time.time()
        
        for i in range(1000):
            agent = BenchmarkAgent(f"memory-agent-{i}", self.config)
            context = MeshContext()
            
            # Use them briefly
            await context.set(f"key_{i}", {"data": f"value_{i}"})
            
            agents.append(agent)
            contexts.append(context)
            
            # Simulate cleanup of old objects
            if i > 100:
                old_agent = agents.pop(0)
                old_context = contexts.pop(0)
                del old_agent
                del old_context
        
        # Force garbage collection
        gc.collect()
        
        peak_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # Clean up remaining objects
        agents.clear()
        contexts.clear()
        gc.collect()
        
        final_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        duration = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="Memory Efficiency",
            duration_seconds=duration,
            total_operations=1000,
            operations_per_second=1000 / duration,
            avg_latency_ms=duration / 1000 * 1000,
            p95_latency_ms=0,
            p99_latency_ms=0,
            error_rate=0,
            cpu_usage_percent=0,
            memory_usage_mb=peak_memory - initial_memory,
            success_rate=1.0,
            additional_metrics={
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": peak_memory,
                "final_memory_mb": final_memory,
                "memory_growth_mb": final_memory - initial_memory,
                "peak_increase_mb": peak_memory - initial_memory
            }
        )
    
    async def benchmark_large_payload_handling(self) -> BenchmarkResult:
        """Benchmark handling of large payloads"""
        agent = BenchmarkAgent("payload-agent", self.config)
        await agent.start()
        
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        # Test different payload sizes
        payload_sizes = [1, 10, 100, 1000]  # KB
        operations_per_size = 100
        
        total_operations = 0
        total_time = 0
        latencies = []
        
        for size_kb in payload_sizes:
            # Create payload of specified size
            payload_data = "x" * (size_kb * 1024)
            
            batch_start = time.time()
            batch_latencies = []
            
            for i in range(operations_per_size):
                context = MeshContext()
                task_data = {
                    "work_type": "light",
                    "payload": payload_data,
                    "size_kb": size_kb
                }
                
                result, latency = await self._timed_agent_call(agent, task_data, context)
                batch_latencies.append(latency)
            
            batch_time = time.time() - batch_start
            total_time += batch_time
            total_operations += operations_per_size
            latencies.extend(batch_latencies)
        
        resource_usage = await monitor.stop_monitoring()
        await agent.stop()
        
        return BenchmarkResult(
            benchmark_name="Large Payload Handling",
            duration_seconds=total_time,
            total_operations=total_operations,
            operations_per_second=total_operations / total_time,
            avg_latency_ms=statistics.mean(latencies) * 1000,
            p95_latency_ms=self._percentile(latencies, 0.95) * 1000,
            p99_latency_ms=self._percentile(latencies, 0.99) * 1000,
            error_rate=0,
            cpu_usage_percent=resource_usage.avg_cpu_usage,
            memory_usage_mb=resource_usage.avg_memory_usage_mb,
            success_rate=1.0,
            additional_metrics={
                "payload_sizes_kb": payload_sizes,
                "max_payload_kb": max(payload_sizes)
            }
        )
    
    async def _timed_agent_call(self, agent, task_data, context) -> Tuple[Any, float]:
        """Make a timed agent call"""
        start_time = time.time()
        result = await agent.handle_task(task_data, context)
        latency = time.time() - start_time
        return result, latency
    
    async def _simple_operation(self):
        """Simple operation for overhead testing"""
        return sum(range(10))
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(p * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _print_benchmark_result(self, result: BenchmarkResult):
        """Print formatted benchmark result"""
        print(f"✅ {result.benchmark_name}")
        print(f"   Operations/sec: {result.operations_per_second:.1f}")
        print(f"   Avg Latency: {result.avg_latency_ms:.2f}ms")
        print(f"   P95 Latency: {result.p95_latency_ms:.2f}ms")
        print(f"   Success Rate: {result.success_rate*100:.1f}%")
        print(f"   CPU Usage: {result.cpu_usage_percent:.1f}%")
        print(f"   Memory: {result.memory_usage_mb:.1f}MB")
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            print(f"{result.benchmark_name:.<30} {result.operations_per_second:>10.1f} ops/sec")
        
        print("\n🏆 Performance Highlights:")
        if self.results:
            fastest = max(self.results, key=lambda r: r.operations_per_second)
            lowest_latency = min(self.results, key=lambda r: r.avg_latency_ms)
            
            print(f"   Highest Throughput: {fastest.benchmark_name} ({fastest.operations_per_second:.1f} ops/sec)")
            print(f"   Lowest Latency: {lowest_latency.benchmark_name} ({lowest_latency.avg_latency_ms:.2f}ms)")
    
    def export_results(self, filename: str = "benchmark_results.json"):
        """Export results to JSON file"""
        import json
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": []
        }
        
        for result in self.results:
            data["benchmarks"].append({
                "name": result.benchmark_name,
                "duration_seconds": result.duration_seconds,
                "total_operations": result.total_operations,
                "operations_per_second": result.operations_per_second,
                "avg_latency_ms": result.avg_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "p99_latency_ms": result.p99_latency_ms,
                "error_rate": result.error_rate,
                "cpu_usage_percent": result.cpu_usage_percent,
                "memory_usage_mb": result.memory_usage_mb,
                "success_rate": result.success_rate,
                "additional_metrics": result.additional_metrics
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results exported to {filename}")


async def main():
    """Run performance benchmark suite"""
    config = MeshConfig()
    benchmark_suite = PerformanceBenchmarkSuite(config)
    
    results = await benchmark_suite.run_all_benchmarks()
    benchmark_suite.export_results()
    
    return results


if __name__ == "__main__":
    asyncio.run(main())