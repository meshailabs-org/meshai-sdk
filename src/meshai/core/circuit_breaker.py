"""
Circuit Breaker Implementation for MeshAI Agents

This module provides a robust circuit breaker pattern implementation
for protecting against cascading failures in distributed agent systems.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import threading
from collections import deque, defaultdict

import structlog
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5                    # Failures before opening
    recovery_timeout: float = 30.0               # Seconds before trying half-open
    success_threshold: int = 3                   # Successes to close from half-open
    timeout: float = 10.0                        # Request timeout in seconds
    expected_exception: tuple = (Exception,)      # Exceptions that count as failures
    
    # Advanced configuration
    sliding_window_size: int = 100               # Size of sliding window for metrics
    minimum_throughput: int = 10                 # Min requests before evaluating
    error_rate_threshold: float = 0.5            # Error rate threshold (50%)
    slow_call_duration_threshold: float = 5.0    # Slow call threshold in seconds
    slow_call_rate_threshold: float = 0.3        # Slow call rate threshold (30%)


@dataclass
class CallResult:
    """Result of a circuit breaker protected call"""
    success: bool
    duration: float
    timestamp: datetime
    exception: Optional[Exception] = None
    was_slow_call: bool = False


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    last_failure_time: Optional[datetime] = None
    state_changed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Sliding window metrics
    call_results: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rate: float = 0.0
    slow_call_rate: float = 0.0
    avg_response_time: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker implementation with advanced failure detection.
    
    Provides protection against cascading failures by monitoring:
    - Error rates
    - Response times 
    - Slow calls
    - Success rates in recovery
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        # State management
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        self.metrics = CircuitBreakerMetrics(state=self._state)
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        logger.info(f"Circuit breaker '{name}' initialized", 
                   config=self.config.__dict__)
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.call_counter = Counter(
            'circuit_breaker_calls_total',
            'Total number of calls through circuit breaker',
            ['circuit_name', 'result']
        )
        
        self.call_duration = Histogram(
            'circuit_breaker_call_duration_seconds',
            'Duration of calls through circuit breaker',
            ['circuit_name']
        )
        
        self.state_gauge = Gauge(
            'circuit_breaker_state',
            'Current state of circuit breaker (0=closed, 1=open, 2=half_open)',
            ['circuit_name']
        )
        
        self.failure_rate_gauge = Gauge(
            'circuit_breaker_failure_rate',
            'Current failure rate of circuit breaker',
            ['circuit_name']
        )
    
    @property
    def state(self) -> CircuitState:
        """Current circuit breaker state"""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Original exceptions from the function
        """
        # Check if circuit allows calls
        if not self._can_execute():
            self.call_counter.labels(circuit_name=self.name, result='rejected').inc()
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is {self._state.value}"
            )
        
        # Execute the call with timing
        start_time = time.time()
        call_result = None
        
        try:
            # Set timeout for the call
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)
            
            # Record successful call
            duration = time.time() - start_time
            call_result = CallResult(
                success=True,
                duration=duration,
                timestamp=datetime.utcnow(),
                was_slow_call=duration > self.config.slow_call_duration_threshold
            )
            
            self._record_success(call_result)
            self.call_counter.labels(circuit_name=self.name, result='success').inc()
            self.call_duration.labels(circuit_name=self.name).observe(duration)
            
            return result
            
        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            call_result = CallResult(
                success=False,
                duration=duration,
                timestamp=datetime.utcnow(),
                exception=e,
                was_slow_call=True
            )
            
            self._record_failure(call_result)
            self.call_counter.labels(circuit_name=self.name, result='timeout').inc()
            raise
            
        except self.config.expected_exception as e:
            duration = time.time() - start_time
            call_result = CallResult(
                success=False,
                duration=duration,
                timestamp=datetime.utcnow(),
                exception=e
            )
            
            self._record_failure(call_result)
            self.call_counter.labels(circuit_name=self.name, result='failure').inc()
            raise
    
    def _can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                # Check if we should try half-open
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                    return True
                return False
            elif self._state == CircuitState.HALF_OPEN:
                return True
            
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.metrics.last_failure_time:
            return True
            
        time_since_failure = datetime.utcnow() - self.metrics.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _record_success(self, result: CallResult):
        """Record a successful call"""
        with self._lock:
            self.metrics.success_count += 1
            self.metrics.total_calls += 1
            self.metrics.call_results.append(result)
            
            self._update_metrics()
            
            # Handle state transitions
            if self._state == CircuitState.HALF_OPEN:
                if self.metrics.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def _record_failure(self, result: CallResult):
        """Record a failed call"""
        with self._lock:
            self.metrics.failure_count += 1
            self.metrics.total_calls += 1
            self.metrics.last_failure_time = result.timestamp
            self.metrics.call_results.append(result)
            
            self._update_metrics()
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()
    
    def _update_metrics(self):
        """Update sliding window metrics"""
        if not self.metrics.call_results:
            return
        
        recent_calls = list(self.metrics.call_results)
        total_calls = len(recent_calls)
        
        if total_calls < self.config.minimum_throughput:
            return
        
        # Calculate error rate
        failed_calls = sum(1 for call in recent_calls if not call.success)
        self.metrics.error_rate = failed_calls / total_calls
        
        # Calculate slow call rate
        slow_calls = sum(1 for call in recent_calls if call.was_slow_call)
        self.metrics.slow_call_rate = slow_calls / total_calls
        
        # Calculate average response time
        total_duration = sum(call.duration for call in recent_calls)
        self.metrics.avg_response_time = total_duration / total_calls
        
        # Update Prometheus metrics
        self.failure_rate_gauge.labels(circuit_name=self.name).set(self.metrics.error_rate)
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        if self.metrics.total_calls < self.config.minimum_throughput:
            return False
        
        # Check failure count threshold
        if self.metrics.failure_count >= self.config.failure_threshold:
            return True
        
        # Check error rate threshold
        if self.metrics.error_rate >= self.config.error_rate_threshold:
            return True
        
        # Check slow call rate threshold
        if self.metrics.slow_call_rate >= self.config.slow_call_rate_threshold:
            return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit to open state"""
        old_state = self._state
        self._state = CircuitState.OPEN
        self.metrics.state = self._state
        self.metrics.state_changed_at = datetime.utcnow()
        
        self.state_gauge.labels(circuit_name=self.name).set(1)
        
        logger.warning(
            f"Circuit breaker '{self.name}' opened",
            old_state=old_state.value,
            new_state=self._state.value,
            failure_count=self.metrics.failure_count,
            error_rate=self.metrics.error_rate
        )
        
        if self.on_state_change:
            self.on_state_change(self.name, old_state, self._state)
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self.metrics.state = self._state
        self.metrics.state_changed_at = datetime.utcnow()
        self.metrics.success_count = 0  # Reset success count for half-open test
        
        self.state_gauge.labels(circuit_name=self.name).set(2)
        
        logger.info(
            f"Circuit breaker '{self.name}' half-opened",
            old_state=old_state.value,
            new_state=self._state.value
        )
        
        if self.on_state_change:
            self.on_state_change(self.name, old_state, self._state)
    
    def _transition_to_closed(self):
        """Transition circuit to closed state"""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self.metrics.state = self._state
        self.metrics.state_changed_at = datetime.utcnow()
        self.metrics.failure_count = 0  # Reset failure count
        
        self.state_gauge.labels(circuit_name=self.name).set(0)
        
        logger.info(
            f"Circuit breaker '{self.name}' closed",
            old_state=old_state.value,
            new_state=self._state.value,
            success_count=self.metrics.success_count
        )
        
        if self.on_state_change:
            self.on_state_change(self.name, old_state, self._state)
    
    def force_open(self):
        """Manually force circuit breaker open"""
        with self._lock:
            self._transition_to_open()
        logger.warning(f"Circuit breaker '{self.name}' manually forced open")
    
    def force_close(self):
        """Manually force circuit breaker closed"""
        with self._lock:
            self._transition_to_closed()
        logger.info(f"Circuit breaker '{self.name}' manually forced closed")
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self.metrics = CircuitBreakerMetrics(state=self._state)
        logger.info(f"Circuit breaker '{self.name}' reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self.metrics.failure_count,
                "success_count": self.metrics.success_count,
                "total_calls": self.metrics.total_calls,
                "error_rate": self.metrics.error_rate,
                "slow_call_rate": self.metrics.slow_call_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                "state_changed_at": self.metrics.state_changed_at.isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.metrics.state_changed_at).total_seconds()
            }


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different agents/services.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
        logger.info("Circuit breaker manager initialized")
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config,
                    on_state_change=on_state_change
                )
                logger.info(f"Created circuit breaker: {name}")
            
            return self._breakers[name]
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }
    
    def get_open_breakers(self) -> List[str]:
        """Get names of all open circuit breakers"""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.is_open
            ]
    
    def force_close_all(self):
        """Force close all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.force_close()
        logger.info("All circuit breakers forced closed")
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
        logger.info("All circuit breakers reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
    on_state_change: Optional[Callable] = None
):
    """
    Decorator to apply circuit breaker pattern to functions.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        config: Circuit breaker configuration
        on_state_change: Callback for state changes
    """
    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = circuit_breaker_manager.get_or_create(
            breaker_name, config, on_state_change
        )
        
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator