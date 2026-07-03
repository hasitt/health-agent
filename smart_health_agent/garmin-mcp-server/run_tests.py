#!/usr/bin/env python3
"""
Test runner for unit tests without pytest dependency.
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, '/app/src')

# Set environment
os.environ['GARMIN_EMAIL'] = 'test@example.com'
os.environ['GARMIN_PASSWORD'] = 'testpass123'

def run_validation_tests():
    """Run validation tests."""
    print("🔍 Testing Validation Module")
    print("-" * 40)
    
    try:
        from garmin_mcp.validation import DateParser, validate_tool_parameters, ValidationError
        from datetime import date, timedelta
        
        # Test date parsing
        test_dates = [
            ("today", date.today()),
            ("yesterday", date.today() - timedelta(days=1)), 
            ("2024-08-06", date(2024, 8, 6)),
            ("3 days ago", date.today() - timedelta(days=3))
        ]
        
        for date_str, expected in test_dates:
            result = DateParser.parse_date(date_str)
            assert result == expected, f"Date parsing failed: {date_str} -> {result} != {expected}"
            print(f"✅ Date parsing: '{date_str}' → {result}")
        
        # Test parameter validation
        valid_params = validate_tool_parameters('get_daily_summary', {'date': 'today'})
        assert 'date' in valid_params
        print(f"✅ Parameter validation: {valid_params}")
        
        # Test invalid parameters
        try:
            validate_tool_parameters('get_daily_summary', {'date': 'invalid_date'})
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✅ Invalid parameter rejection works")
        
        print("🎉 Validation tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Validation tests FAILED: {e}")
        traceback.print_exc()
        return False

def run_ai_optimization_tests():
    """Run AI optimization tests."""
    print("\n🔍 Testing AI Optimization Module")
    print("-" * 40)
    
    try:
        from garmin_mcp.ai_optimization import HealthDataFormatter, enhance_data_for_ai
        
        # Test data formatting
        test_data = {
            "status": "success",
            "summary": {
                "date": "2024-08-06",
                "steps": 12500,
                "distance_km": 8.2,
                "active_calories": 456,
                "goal_steps": 10000
            }
        }
        
        formatted = HealthDataFormatter.format_for_conversation(test_data, "daily_summary")
        assert isinstance(formatted, str)
        assert "12,500 steps" in formatted
        print(f"✅ Data formatting: {formatted[:60]}...")
        
        # Test data enhancement
        enhanced = enhance_data_for_ai(test_data, "daily_summary")
        assert isinstance(enhanced, dict)
        assert "ai_context" in enhanced
        assert enhanced["ai_context"]["conversation_ready"] is True
        print("✅ Data enhancement works")
        
        # Test sleep data formatting
        sleep_data = {
            "status": "success",
            "sleep_data": {
                "sleep_duration_hours": 7.5,
                "sleep_score": 82
            }
        }
        
        sleep_formatted = HealthDataFormatter.format_for_conversation(sleep_data, "sleep_analysis")
        assert "7.5 hours" in sleep_formatted
        assert "82/100" in sleep_formatted
        print(f"✅ Sleep formatting: {sleep_formatted[:60]}...")
        
        print("🎉 AI Optimization tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ AI Optimization tests FAILED: {e}")
        traceback.print_exc()
        return False

def run_monitoring_tests():
    """Run monitoring tests."""
    print("\n🔍 Testing Monitoring Module") 
    print("-" * 40)
    
    try:
        from garmin_mcp.monitoring import PerformanceMonitor, CircuitBreaker, performance_monitor
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        initial_metrics = len(monitor.metrics)
        
        monitor.record_metric("test_metric", 100.5, "ms")
        assert len(monitor.metrics) == initial_metrics + 1
        print("✅ Performance metric recording works")
        
        # Test request tracking
        request = monitor.start_request("test_123", "test_tool")
        request.complete(success=True)
        assert request.success is True
        assert request.duration_ms is not None
        print("✅ Request tracking works")
        
        # Test cache operations
        initial_hits = monitor.cache_stats["hits"]
        monitor.record_cache_hit("test_tool")
        assert monitor.cache_stats["hits"] == initial_hits + 1
        print("✅ Cache tracking works")
        
        # Test circuit breaker
        circuit = CircuitBreaker(failure_threshold=2, timeout_seconds=1)
        assert circuit.state == "closed"
        print("✅ Circuit breaker initialization works")
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        required_fields = ["uptime_seconds", "cache", "requests", "tools"]
        for field in required_fields:
            assert field in summary, f"Missing field: {field}"
        print("✅ Performance summary generation works")
        
        print("🎉 Monitoring tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring tests FAILED: {e}")
        traceback.print_exc()
        return False

async def run_server_integration_tests():
    """Run server integration tests."""
    print("\n🔍 Testing Server Integration")
    print("-" * 40)
    
    try:
        from garmin_mcp.config import GarminMCPConfig
        from garmin_mcp.server import GarminMCPServer
        
        # Test server creation
        config = GarminMCPConfig(
            garmin_email='test@example.com',
            garmin_password='testpass123'
        )
        server = GarminMCPServer(config)
        print("✅ Server creation works")
        
        # Test tool listing
        tools = await server._list_tools()
        assert len(tools.tools) == 15
        print(f"✅ Tool listing works: {len(tools.tools)} tools")
        
        # Test resource listing
        resources = await server._list_resources()
        assert len(resources.resources) == 4
        print(f"✅ Resource listing works: {len(resources.resources)} resources")
        
        # Test tool parameter validation integration
        from garmin_mcp.server import GarminMCPServer
        
        # This should work without throwing errors
        validated_args = server.config  # Just test config access
        print("✅ Server configuration access works")
        
        print("🎉 Server Integration tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Server Integration tests FAILED: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print(f"Unit Test Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    test_functions = [
        ("Validation Module", run_validation_tests),
        ("AI Optimization Module", run_ai_optimization_tests), 
        ("Monitoring Module", run_monitoring_tests),
        ("Server Integration", run_server_integration_tests)
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_name, test_func in test_functions:
        print(f"\n🚀 Running {test_name} Tests")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                passed += 1
                print(f"✅ {test_name} Tests PASSED")
            else:
                print(f"❌ {test_name} Tests FAILED")
        except Exception as e:
            print(f"❌ {test_name} Tests FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 TEST RESULTS: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL UNIT TESTS PASSED!")
        return True
    else:
        print("⚠️  Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)