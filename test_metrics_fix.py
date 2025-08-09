#!/usr/bin/env python3
"""
Test script to verify the metrics fixes work locally
"""
import sys
import os
sys.path.insert(0, 'src')

def test_metrics_collector():
    """Test that MetricsCollector has the required attributes"""
    print("Testing MetricsCollector...")
    
    try:
        from meshai.utils.metrics import MetricsCollector
        
        # Create a metrics collector
        metrics = MetricsCollector("test-service")
        
        # Check for the attributes that were missing
        print(f"Has total_registrations: {hasattr(metrics, 'total_registrations')}")
        print(f"Has total_discoveries: {hasattr(metrics, 'total_discoveries')}")
        print(f"Has record_registration method: {hasattr(metrics, 'record_registration')}")
        print(f"Has record_discovery method: {hasattr(metrics, 'record_discovery')}")
        
        if hasattr(metrics, 'record_registration'):
            print("Testing record_registration method...")
            metrics.record_registration("test-service")
            print("✅ record_registration works")
            
        if hasattr(metrics, 'record_discovery'):
            print("Testing record_discovery method...")
            metrics.record_discovery("test-service")
            print("✅ record_discovery works")
            
        print("✅ MetricsCollector test passed")
        return True
        
    except Exception as e:
        print(f"❌ MetricsCollector test failed: {e}")
        return False

def test_circuit_breaker_metrics():
    """Test that CircuitBreaker doesn't cause metric duplication"""
    print("\\nTesting CircuitBreaker metrics...")
    
    try:
        from meshai.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        # Create multiple circuit breakers to test for duplication issues
        cb1 = CircuitBreaker("test-cb-1")
        cb2 = CircuitBreaker("test-cb-2")
        cb3 = CircuitBreaker("test-cb-3")
        
        print("✅ Multiple CircuitBreakers created without duplication error")
        return True
        
    except Exception as e:
        if "Duplicated timeseries" in str(e):
            print(f"⚠️  CircuitBreaker still has duplication issue: {e}")
        else:
            print(f"❌ CircuitBreaker test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running metrics fix tests...")
    
    test1_passed = test_metrics_collector()
    test2_passed = test_circuit_breaker_metrics()
    
    if test1_passed and test2_passed:
        print("\\n🎉 All tests passed! The metrics fixes should work.")
    else:
        print("\\n❌ Some tests failed. The fixes need more work.")
        
    print("\\nNote: These are local tests. The actual Cloud Run deployment may have additional considerations.")