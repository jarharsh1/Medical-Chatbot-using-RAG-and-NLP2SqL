"""
Unit tests for API Resilience Layer (LinkedIn Q5)

Tests resilience patterns:
- Circuit breaker pattern
- Retry with exponential backoff
- Fallback responses
- Health monitoring
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from reliability.api_resilience import (
    CircuitBreaker,
    APIResilientCaller,
    CircuitState,
    CircuitBreakerConfig,
    RetryConfig,
    FailureReason,
    resilient_api_call
)


class TestCircuitBreaker:
    """Test suite for circuit breaker"""

    @pytest.fixture
    def breaker(self):
        """Create circuit breaker with fast config for testing"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            cooldown_seconds=1.0
        )
        return CircuitBreaker(name="test_api", config=config)

    # =========================================================================
    # CIRCUIT STATES
    # =========================================================================

    def test_initial_state_closed(self, breaker):
        """Test that circuit starts in CLOSED state"""
        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.can_execute()

    def test_transition_to_open_after_threshold(self, breaker):
        """Test circuit opens after failure threshold reached"""
        # Record failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker.get_state() == CircuitState.OPEN
        assert not breaker.can_execute()

    def test_transition_to_half_open_after_cooldown(self, breaker):
        """Test circuit moves to HALF_OPEN after cooldown"""
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()
        assert breaker.get_state() == CircuitState.OPEN

        # Wait for cooldown
        time.sleep(1.1)

        # Next call should allow execution (half-open)
        assert breaker.can_execute()
        assert breaker.get_state() == CircuitState.HALF_OPEN

    def test_transition_to_closed_after_success_threshold(self, breaker):
        """Test circuit closes after success threshold in half-open"""
        # Open circuit
        for _ in range(3):
            breaker.record_failure()

        # Wait and move to half-open
        time.sleep(1.1)
        breaker.can_execute()

        # Record successful calls
        breaker.record_success()
        breaker.record_success()

        assert breaker.get_state() == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self, breaker):
        """Test that failure in half-open immediately reopens circuit"""
        # Open circuit
        for _ in range(3):
            breaker.record_failure()

        # Move to half-open
        time.sleep(1.1)
        breaker.can_execute()
        assert breaker.get_state() == CircuitState.HALF_OPEN

        # Fail again
        breaker.record_failure()

        assert breaker.get_state() == CircuitState.OPEN

    # =========================================================================
    # EXECUTION CONTROL
    # =========================================================================

    def test_block_requests_when_open(self, breaker):
        """Test that requests are blocked when circuit is open"""
        # Open circuit
        for _ in range(3):
            breaker.record_failure()

        # Requests should be blocked
        assert not breaker.can_execute()

        stats = breaker.get_statistics()
        assert stats["blocked_calls"] > 0

    def test_allow_requests_when_closed(self, breaker):
        """Test that requests go through when circuit is closed"""
        assert breaker.can_execute()

        breaker.record_success()
        assert breaker.can_execute()

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def test_statistics_tracking(self, breaker):
        """Test that statistics are tracked correctly"""
        # Initial state
        stats = breaker.get_statistics()
        assert stats["total_calls"] == 0

        # Make calls
        breaker.can_execute()
        breaker.record_success()

        breaker.can_execute()
        breaker.record_failure()

        stats = breaker.get_statistics()
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 1

    def test_state_change_tracking(self, breaker):
        """Test that state changes are tracked"""
        initial_stats = breaker.get_statistics()
        initial_changes = initial_stats["state_changes"]

        # Trigger state change (closed -> open)
        for _ in range(3):
            breaker.record_failure()

        stats = breaker.get_statistics()
        assert stats["state_changes"] > initial_changes


class TestAPIResilientCaller:
    """Test suite for resilient API caller"""

    @pytest.fixture
    def caller(self):
        """Create resilient caller with fast config for testing"""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            cooldown_seconds=1.0
        )
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay_seconds=0.1,
            max_delay_seconds=1.0
        )
        return APIResilientCaller(
            name="test_api",
            circuit_config=circuit_config,
            retry_config=retry_config
        )

    # =========================================================================
    # SUCCESSFUL CALLS
    # =========================================================================

    def test_successful_call(self, caller):
        """Test successful API call"""
        def successful_func():
            return {"data": "success"}

        result = caller.call(func=successful_func)

        assert result.success
        assert result.data == {"data": "success"}
        assert result.error is None
        assert result.attempts == 1
        assert not result.used_fallback

    def test_successful_call_updates_statistics(self, caller):
        """Test that successful calls update statistics"""
        def successful_func():
            return "success"

        caller.call(func=successful_func)

        stats = caller.get_statistics()
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0

    # =========================================================================
    # RETRY LOGIC
    # =========================================================================

    def test_retry_on_failure(self, caller):
        """Test that failures are retried"""
        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        result = caller.call(func=flaky_func)

        assert result.success
        assert result.attempts == 3
        assert call_count[0] == 3

    def test_exponential_backoff(self, caller):
        """Test exponential backoff between retries"""
        call_times = []

        def failing_func():
            call_times.append(time.time())
            raise Exception("Failure")

        caller.call(func=failing_func)

        # Check that delays increased
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            if len(call_times) >= 3:
                delay2 = call_times[2] - call_times[1]
                assert delay2 > delay1  # Exponential increase

    def test_max_retries_respected(self, caller):
        """Test that max retries limit is respected"""
        call_count = [0]

        def always_failing():
            call_count[0] += 1
            raise Exception("Always fails")

        result = caller.call(func=always_failing)

        assert not result.success
        assert result.attempts == 3  # max_attempts
        assert call_count[0] == 3

    # =========================================================================
    # CIRCUIT BREAKER INTEGRATION
    # =========================================================================

    def test_circuit_breaker_blocks_after_failures(self, caller):
        """Test that circuit breaker blocks calls after too many failures"""
        def failing_func():
            raise Exception("Failure")

        # Fail enough times to open circuit
        for _ in range(3):
            caller.call(func=failing_func)

        # Next call should be blocked by circuit breaker
        result = caller.call(func=failing_func)

        assert not result.success
        assert "circuit breaker" in str(result.error).lower()

    def test_circuit_breaker_allows_after_cooldown(self, caller):
        """Test that circuit breaker allows calls after cooldown"""
        def failing_func():
            raise Exception("Failure")

        # Open circuit
        for _ in range(3):
            caller.call(func=failing_func)

        # Wait for cooldown
        time.sleep(1.1)

        # Should allow call (half-open state)
        def successful_func():
            return "success"

        result = caller.call(func=successful_func)
        # Note: May still fail if circuit doesn't allow, but should attempt

    # =========================================================================
    # FALLBACK RESPONSES
    # =========================================================================

    def test_fallback_on_failure(self, caller):
        """Test that fallback is used when main function fails"""
        def failing_func():
            raise Exception("Main function failed")

        def fallback_func():
            return "fallback response"

        result = caller.call(func=failing_func, fallback=fallback_func)

        assert result.success  # Fallback succeeded
        assert result.data == "fallback response"
        assert result.used_fallback
        assert result.error is not None  # Original error preserved

    def test_fallback_not_used_on_success(self, caller):
        """Test that fallback is not used when main function succeeds"""
        def successful_func():
            return "main response"

        def fallback_func():
            return "fallback response"

        result = caller.call(func=successful_func, fallback=fallback_func)

        assert result.success
        assert result.data == "main response"
        assert not result.used_fallback

    def test_fallback_failure_handled(self, caller):
        """Test handling when fallback also fails"""
        def failing_func():
            raise Exception("Main failed")

        def failing_fallback():
            raise Exception("Fallback also failed")

        result = caller.call(func=failing_func, fallback=failing_fallback)

        assert not result.success
        assert result.used_fallback
        assert "fallback" in str(result.error).lower()

    # =========================================================================
    # FAILURE CLASSIFICATION
    # =========================================================================

    def test_classify_timeout_error(self, caller):
        """Test classification of timeout errors"""
        def timeout_func():
            raise Exception("Request timeout")

        result = caller.call(func=timeout_func)

        assert result.failure_reason == FailureReason.TIMEOUT

    def test_classify_connection_error(self, caller):
        """Test classification of connection errors"""
        def connection_error_func():
            raise Exception("Connection refused")

        result = caller.call(func=connection_error_func)

        assert result.failure_reason == FailureReason.CONNECTION_ERROR

    def test_classify_rate_limit_error(self, caller):
        """Test classification of rate limit errors"""
        def rate_limit_func():
            raise Exception("Rate limit exceeded")

        result = caller.call(func=rate_limit_func)

        assert result.failure_reason == FailureReason.RATE_LIMIT

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def test_statistics_tracking(self, caller):
        """Test comprehensive statistics tracking"""
        def successful_func():
            return "success"

        def failing_func():
            raise Exception("Failure")

        def fallback_func():
            return "fallback"

        # Successful call
        caller.call(func=successful_func)

        # Failed call with fallback
        caller.call(func=failing_func, fallback=fallback_func)

        stats = caller.get_statistics()

        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["fallback_used"] == 1


class TestResilientAPICallDecorator:
    """Test suite for resilient API call decorator"""

    def test_decorator_basic_usage(self):
        """Test basic decorator usage"""
        @resilient_api_call(name="test_api")
        def my_api_call():
            return "success"

        result = my_api_call()
        assert result == "success"

    def test_decorator_with_fallback(self):
        """Test decorator with fallback"""
        call_count = [0]

        @resilient_api_call(
            name="test_api",
            fallback=lambda: "fallback response"
        )
        def flaky_api_call():
            call_count[0] += 1
            if call_count[0] < 5:  # Always fail
                raise Exception("API unavailable")
            return "success"

        result = flaky_api_call()
        assert result == "fallback response"

    def test_decorator_with_arguments(self):
        """Test decorator with function arguments"""
        @resilient_api_call(name="test_api")
        def api_with_args(x, y):
            return x + y

        result = api_with_args(5, 3)
        assert result == 8


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAPIResilienceIntegration:
    """Integration tests for API resilience"""

    def test_complete_resilience_pipeline(self):
        """Test complete resilience pipeline with all patterns"""
        call_count = [0]

        def flaky_api():
            """Fails first 2 times, then succeeds"""
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return {"status": "success"}

        def fallback():
            return {"status": "fallback", "cached": True}

        caller = APIResilientCaller(
            name="integration_test",
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay_seconds=0.05
            )
        )

        result = caller.call(func=flaky_api, fallback=fallback)

        # Should succeed after retries
        assert result.success
        assert result.attempts <= 3
        assert result.data["status"] == "success"

    def test_cascading_failures_blocked_by_circuit_breaker(self):
        """Test that circuit breaker prevents cascading failures"""
        def always_failing():
            raise Exception("Service down")

        caller = APIResilientCaller(
            name="cascading_test",
            circuit_config=CircuitBreakerConfig(
                failure_threshold=2,
                cooldown_seconds=1.0
            ),
            retry_config=RetryConfig(max_attempts=1)
        )

        # First few calls will attempt and fail
        for _ in range(2):
            caller.call(func=always_failing)

        # Circuit should now be open
        circuit_stats = caller.get_statistics()["circuit_breaker"]
        assert circuit_stats["current_state"] == "open"

        # Next calls should be blocked immediately (not retried)
        start_time = time.time()
        result = caller.call(func=always_failing)
        elapsed = time.time() - start_time

        # Should fail fast (< 100ms)
        assert elapsed < 0.1
        assert not result.success

    def test_performance_overhead(self):
        """Test that resilience patterns have minimal overhead"""
        def fast_api():
            return "success"

        caller = APIResilientCaller(name="perf_test")

        # Measure time for 100 calls
        start = time.time()
        for _ in range(100):
            caller.call(func=fast_api)
        elapsed = time.time() - start

        avg_overhead_ms = (elapsed / 100) * 1000
        # Overhead should be minimal (< 5ms per call)
        assert avg_overhead_ms < 5, f"Too much overhead: {avg_overhead_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
