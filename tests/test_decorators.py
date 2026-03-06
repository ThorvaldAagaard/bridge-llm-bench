"""
Unit tests for retry and backoff decorators.

Tests cover exponential backoff, retry on empty, and rate limiting functionality.
"""

import unittest
import time
from unittest.mock import Mock, patch

from bridge_llm_bench.utils.decorators import (
    exponential_backoff,
    retry_on_empty,
    rate_limit
)


class TestExponentialBackoff(unittest.TestCase):
    """Test suite for exponential backoff decorator."""
    
    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        mock_func = Mock(return_value="success")
        
        @exponential_backoff(max_retries=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 1)
    
    def test_retry_on_exception(self):
        """Test that exceptions trigger retries."""
        mock_func = Mock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            "success"
        ])
        
        @exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 3)
    
    def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        @exponential_backoff(max_retries=2, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with self.assertRaises(Exception) as context:
            test_func()
        
        self.assertEqual(str(context.exception), "Always fails")
        self.assertEqual(mock_func.call_count, 3)  # Initial + 2 retries
    
    def test_specific_exception_types(self):
        """Test retry only on specific exception types."""
        mock_func = Mock(side_effect=[
            ValueError("Retry this"),
            RuntimeError("Don't retry this")
        ])
        
        @exponential_backoff(
            max_retries=3,
            initial_delay=0.01,
            exceptions=(ValueError,)
        )
        def test_func():
            return mock_func()
        
        with self.assertRaises(RuntimeError):
            test_func()
        
        self.assertEqual(mock_func.call_count, 2)
    
    def test_exponential_delay_calculation(self):
        """Test that delays increase exponentially."""
        delays = []
        
        def capture_delay(delay):
            delays.append(delay)
        
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        with patch('time.sleep', side_effect=capture_delay):
            @exponential_backoff(
                max_retries=3,
                initial_delay=1.0,
                exponential_base=2.0,
                jitter=False
            )
            def test_func():
                return mock_func()
            
            try:
                test_func()
            except Exception:
                pass
        
        # Check delays: 1, 2, 4 (approximately, without jitter)
        self.assertEqual(len(delays), 3)
        self.assertAlmostEqual(delays[0], 1.0, places=1)
        self.assertAlmostEqual(delays[1], 2.0, places=1)
        self.assertAlmostEqual(delays[2], 4.0, places=1)
    
    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        delays = []
        
        def capture_delay(delay):
            delays.append(delay)
        
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        with patch('time.sleep', side_effect=capture_delay):
            @exponential_backoff(
                max_retries=5,
                initial_delay=1.0,
                max_delay=3.0,
                exponential_base=2.0,
                jitter=False
            )
            def test_func():
                return mock_func()
            
            try:
                test_func()
            except Exception:
                pass
        
        # Check that no delay exceeds max_delay
        self.assertTrue(all(d <= 3.0 for d in delays))
        # Later delays should be capped at 3.0
        self.assertAlmostEqual(delays[-1], 3.0, places=1)
        self.assertAlmostEqual(delays[-2], 3.0, places=1)
    
    def test_function_with_arguments(self):
        """Test decorator with functions that take arguments."""
        mock_func = Mock(side_effect=[
            Exception("First failure"),
            "success"
        ])
        
        @exponential_backoff(max_retries=2, initial_delay=0.01)
        def test_func(arg1, arg2, kwarg=None):
            return mock_func(arg1, arg2, kwarg)
        
        result = test_func("a", "b", kwarg="c")
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)
        # Check arguments were passed correctly
        mock_func.assert_called_with("a", "b", "c")


class TestRetryOnEmpty(unittest.TestCase):
    """Test suite for retry on empty decorator."""
    
    def test_successful_non_empty_response(self):
        """Test that non-empty responses don't trigger retries."""
        mock_func = Mock(return_value="valid response")
        
        @retry_on_empty(max_retries=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "valid response")
        self.assertEqual(mock_func.call_count, 1)
    
    def test_retry_on_empty_string(self):
        """Test retry on empty string responses."""
        mock_func = Mock(side_effect=["", "  ", "valid"])
        
        @retry_on_empty(max_retries=3, delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "valid")
        self.assertEqual(mock_func.call_count, 3)
    
    def test_retry_on_none(self):
        """Test retry on None responses."""
        mock_func = Mock(side_effect=[None, None, "valid"])
        
        @retry_on_empty(max_retries=3, delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "valid")
        self.assertEqual(mock_func.call_count, 3)
    
    def test_fallback_value(self):
        """Test fallback value when all retries fail."""
        mock_func = Mock(return_value="")
        
        @retry_on_empty(max_retries=2, fallback_value="default", delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "default")
        self.assertEqual(mock_func.call_count, 3)  # Initial + 2 retries
    
    def test_non_string_values(self):
        """Test handling of non-string return values."""
        # Non-empty list should not retry
        mock_func1 = Mock(return_value=[1, 2, 3])
        
        @retry_on_empty(max_retries=3)
        def test_func1():
            return mock_func1()
        
        result1 = test_func1()
        self.assertEqual(result1, [1, 2, 3])
        self.assertEqual(mock_func1.call_count, 1)
        
        # Empty list should retry
        mock_func2 = Mock(side_effect=[[], [1, 2, 3]])
        
        @retry_on_empty(max_retries=3, delay=0.01)
        def test_func2():
            return mock_func2()
        
        result2 = test_func2()
        self.assertEqual(result2, [1, 2, 3])
        self.assertEqual(mock_func2.call_count, 2)
    
    def test_zero_as_valid_value(self):
        """Test that 0 is treated as valid (non-empty)."""
        mock_func = Mock(return_value=0)
        
        @retry_on_empty(max_retries=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, 0)
        self.assertEqual(mock_func.call_count, 1)


class TestRateLimit(unittest.TestCase):
    """Test suite for rate limit decorator."""
    
    def test_rate_limiting(self):
        """Test that rate limiting enforces call frequency."""
        call_times = []
        
        @rate_limit(calls_per_second=10.0)
        def test_func():
            call_times.append(time.time())
            return "done"
        
        # Make several rapid calls
        for _ in range(5):
            test_func()
        
        # Check intervals between calls
        intervals = []
        for i in range(1, len(call_times)):
            intervals.append(call_times[i] - call_times[i-1])
        
        # Each interval should be at least 0.1 seconds (1/10)
        min_interval = 1.0 / 10.0
        for interval in intervals:
            self.assertGreaterEqual(interval, min_interval * 0.9)  # Allow small variance
    
    def test_first_call_immediate(self):
        """Test that first call is not delayed."""
        start_time = time.time()
        
        @rate_limit(calls_per_second=1.0)
        def test_func():
            return time.time()
        
        first_call_time = test_func()
        
        # First call should be immediate (within 50ms)
        self.assertLess(first_call_time - start_time, 0.05)
    
    def test_rate_limit_with_slow_function(self):
        """Test rate limiting with function that takes time to execute."""
        call_times = []
        
        @rate_limit(calls_per_second=10.0)
        def test_func():
            call_times.append(time.time())
            time.sleep(0.05)  # Function takes 50ms
            return "done"
        
        # Make several calls
        for _ in range(3):
            test_func()
        
        # Even though function takes 50ms, rate limit is 100ms
        # So intervals should be at least 100ms
        intervals = []
        for i in range(1, len(call_times)):
            intervals.append(call_times[i] - call_times[i-1])
        
        min_interval = 1.0 / 10.0
        for interval in intervals:
            self.assertGreaterEqual(interval, min_interval * 0.9)
    
    def test_function_return_values(self):
        """Test that rate limiting preserves return values."""
        counter = 0
        
        @rate_limit(calls_per_second=100.0)
        def test_func():
            nonlocal counter
            counter += 1
            return counter
        
        results = [test_func() for _ in range(5)]
        
        self.assertEqual(results, [1, 2, 3, 4, 5])
    
    def test_function_with_arguments(self):
        """Test rate limiting with functions that take arguments."""
        @rate_limit(calls_per_second=100.0)
        def test_func(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result1 = test_func(1, 2, c=3)
        result2 = test_func("a", "b", c="c")
        
        self.assertEqual(result1, "1-2-3")
        self.assertEqual(result2, "a-b-c")


class TestDecoratorStacking(unittest.TestCase):
    """Test stacking multiple decorators."""
    
    def test_exponential_backoff_with_rate_limit(self):
        """Test combining exponential backoff with rate limiting."""
        mock_func = Mock(side_effect=[
            Exception("First failure"),
            "success"
        ])
        
        @rate_limit(calls_per_second=100.0)
        @exponential_backoff(max_retries=2, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)
    
    def test_all_decorators_combined(self):
        """Test combining all decorators."""
        mock_func = Mock(side_effect=[
            "",  # Empty response
            Exception("Error"),  # Exception
            "valid response"  # Success
        ])
        
        @rate_limit(calls_per_second=100.0)
        @retry_on_empty(max_retries=3, delay=0.01)
        @exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        self.assertEqual(result, "valid response")
        self.assertEqual(mock_func.call_count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)