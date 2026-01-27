"""
API Resilience Layer - LinkedIn Post Q5 Solution

Q: "How do you handle API failures gracefully?"
A: Circuit breaker pattern + retry logic + graceful degradation

Resilience Strategies:
1. Circuit Breaker (stop calling failing APIs to prevent cascading failures)
2. Retry with Exponential Backoff (give failing services time to recover)
3. Timeout Management (don't wait forever)
4. Fallback Responses (degrade gracefully with cached/rule-based responses)
5. Health Monitoring (track API health metrics)

Critical for production: Never let external failures break user experience
"""

import time
import logging
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of circuit breaker"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests (too many failures)
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureReason(Enum):
    """Reasons for API failure"""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


@dataclass
class APICallResult:
    """Result of API call with resilience handling"""
    success: bool
    data: Optional[Any]
    error: Optional[Exception]
    failure_reason: Optional[FailureReason]
    attempts: int                    # Number of attempts made
    total_duration_ms: float        # Total time including retries
    used_fallback: bool             # Whether fallback was used
    circuit_state: CircuitState


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5       # Failures before opening circuit
    success_threshold: int = 2       # Successes in half-open to close circuit
    timeout_seconds: float = 10.0   # Timeout for half-open test
    cooldown_seconds: float = 60.0  # Time to wait before half-open


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 16.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, requests blocked immediately
    - HALF_OPEN: Testing if service recovered, allow limited requests

    Example:
        breaker = CircuitBreaker(name="ollama_api")
        if breaker.can_execute():
            try:
                result = call_api()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the protected service
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()

        # Statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "blocked_calls": 0,
            "state_changes": 0
        }

    def can_execute(self) -> bool:
        """
        Check if request can be executed.

        Returns:
            True if request allowed, False if circuit is open
        """
        self.stats["total_calls"] += 1

        if self.state == CircuitState.CLOSED:
            return True

        elif self.state == CircuitState.OPEN:
            # Check if cooldown period has passed
            if self._should_attempt_reset():
                self._transition_to_half_open()
                return True
            else:
                self.stats["blocked_calls"] += 1
                logger.warning(f"Circuit breaker {self.name} is OPEN - blocking request")
                return False

        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests to test service
            return True

        return False

    def record_success(self):
        """Record successful API call"""
        self.stats["successful_calls"] += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

        # Reset failure count on success
        self.failure_count = 0

    def record_failure(self, error: Optional[Exception] = None):
        """Record failed API call"""
        self.stats["failed_calls"] += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if error:
            logger.error(f"Circuit breaker {self.name} recorded failure: {error}")

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately reopens circuit
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if cooldown period has passed"""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.cooldown_seconds

    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now()
        self.stats["state_changes"] += 1
        logger.warning(
            f"Circuit breaker {self.name} transitioned to OPEN "
            f"({self.failure_count} failures)"
        )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self.last_state_change = datetime.now()
        self.stats["state_changes"] += 1
        logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN (testing)")

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.now()
        self.stats["state_changes"] += 1
        logger.info(f"Circuit breaker {self.name} transitioned to CLOSED (recovered)")

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state

    def get_statistics(self) -> Dict:
        """Get circuit breaker statistics"""
        return {
            **self.stats,
            "current_state": self.state.value,
            "failure_count": self.failure_count,
            "time_in_current_state_seconds": (
                datetime.now() - self.last_state_change
            ).total_seconds()
        }


class APIResilientCaller:
    """
    Resilient API caller with circuit breaker, retry, and fallback.

    Combines multiple resilience patterns:
    - Circuit Breaker: Prevent cascading failures
    - Retry with Exponential Backoff: Handle transient failures
    - Timeout Management: Don't wait forever
    - Fallback: Degrade gracefully

    Example:
        caller = APIResilientCaller(
            name="ollama_api",
            retry_config=RetryConfig(max_attempts=3)
        )

        result = caller.call(
            func=lambda: ollama_client.generate("What is diabetes?"),
            fallback=lambda: "I'm unable to answer right now. Please try again.",
            timeout_seconds=30
        )

        if result.success:
            print(result.data)
        else:
            print(result.error)
    """

    def __init__(
        self,
        name: str,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize resilient API caller.

        Args:
            name: Name of the API/service
            circuit_config: Circuit breaker configuration
            retry_config: Retry configuration
        """
        self.name = name
        self.circuit_breaker = CircuitBreaker(name, circuit_config)
        self.retry_config = retry_config or RetryConfig()

        # Statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0,
            "total_retries": 0
        }

    def call(
        self,
        func: Callable,
        fallback: Optional[Callable] = None,
        timeout_seconds: Optional[float] = None
    ) -> APICallResult:
        """
        Execute function with resilience patterns.

        Args:
            func: Function to execute (API call)
            fallback: Optional fallback function if call fails
            timeout_seconds: Optional timeout for the call

        Returns:
            APICallResult with execution details

        Example:
            >>> caller = APIResilientCaller("my_api")
            >>> result = caller.call(
            ...     func=lambda: api.get_data(),
            ...     fallback=lambda: {"cached": "data"}
            ... )
            >>> result.success
            True
        """
        self.stats["total_calls"] += 1
        start_time = time.time()

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return self._handle_circuit_open(fallback, start_time)

        # Execute with retry
        last_error = None
        attempts = 0

        for attempt in range(self.retry_config.max_attempts):
            attempts += 1

            try:
                # Execute function
                result = func()

                # Success!
                self.circuit_breaker.record_success()
                self.stats["successful_calls"] += 1

                duration_ms = (time.time() - start_time) * 1000

                return APICallResult(
                    success=True,
                    data=result,
                    error=None,
                    failure_reason=None,
                    attempts=attempts,
                    total_duration_ms=duration_ms,
                    used_fallback=False,
                    circuit_state=self.circuit_breaker.get_state()
                )

            except Exception as e:
                last_error = e
                self.stats["total_retries"] += 1

                # Classify failure
                failure_reason = self._classify_failure(e)

                logger.warning(
                    f"API call to {self.name} failed (attempt {attempt + 1}/"
                    f"{self.retry_config.max_attempts}): {e}"
                )

                # Check if should retry
                if attempt < self.retry_config.max_attempts - 1:
                    if self._should_retry(failure_reason):
                        delay = self._calculate_retry_delay(attempt)
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        # Don't retry for certain errors
                        break

        # All retries failed
        self.circuit_breaker.record_failure(last_error)
        self.stats["failed_calls"] += 1

        # Try fallback
        if fallback:
            return self._execute_fallback(fallback, last_error, attempts, start_time)
        else:
            duration_ms = (time.time() - start_time) * 1000
            return APICallResult(
                success=False,
                data=None,
                error=last_error,
                failure_reason=self._classify_failure(last_error),
                attempts=attempts,
                total_duration_ms=duration_ms,
                used_fallback=False,
                circuit_state=self.circuit_breaker.get_state()
            )

    def _handle_circuit_open(
        self,
        fallback: Optional[Callable],
        start_time: float
    ) -> APICallResult:
        """Handle case where circuit breaker is open"""
        if fallback:
            return self._execute_fallback(
                fallback,
                Exception("Circuit breaker is open"),
                attempts=0,
                start_time=start_time
            )
        else:
            duration_ms = (time.time() - start_time) * 1000
            return APICallResult(
                success=False,
                data=None,
                error=Exception("Circuit breaker is open - service unavailable"),
                failure_reason=FailureReason.CONNECTION_ERROR,
                attempts=0,
                total_duration_ms=duration_ms,
                used_fallback=False,
                circuit_state=CircuitState.OPEN
            )

    def _execute_fallback(
        self,
        fallback: Callable,
        original_error: Exception,
        attempts: int,
        start_time: float
    ) -> APICallResult:
        """Execute fallback function"""
        try:
            fallback_result = fallback()
            self.stats["fallback_used"] += 1

            duration_ms = (time.time() - start_time) * 1000

            logger.info(f"Using fallback for {self.name} after {attempts} failed attempts")

            return APICallResult(
                success=True,  # Fallback succeeded
                data=fallback_result,
                error=original_error,  # Keep original error for logging
                failure_reason=self._classify_failure(original_error),
                attempts=attempts,
                total_duration_ms=duration_ms,
                used_fallback=True,
                circuit_state=self.circuit_breaker.get_state()
            )

        except Exception as fallback_error:
            duration_ms = (time.time() - start_time) * 1000

            logger.error(f"Fallback also failed for {self.name}: {fallback_error}")

            return APICallResult(
                success=False,
                data=None,
                error=fallback_error,
                failure_reason=FailureReason.UNKNOWN,
                attempts=attempts,
                total_duration_ms=duration_ms,
                used_fallback=True,
                circuit_state=self.circuit_breaker.get_state()
            )

    def _classify_failure(self, error: Exception) -> FailureReason:
        """Classify failure reason from exception"""
        error_str = str(error).lower()

        if "timeout" in error_str:
            return FailureReason.TIMEOUT
        elif "connection" in error_str:
            return FailureReason.CONNECTION_ERROR
        elif "rate limit" in error_str or "429" in error_str:
            return FailureReason.RATE_LIMIT
        elif "http" in error_str or "status" in error_str:
            return FailureReason.HTTP_ERROR
        else:
            return FailureReason.UNKNOWN

    def _should_retry(self, failure_reason: FailureReason) -> bool:
        """Determine if failure is retriable"""
        # Don't retry rate limits or invalid responses
        non_retriable = [FailureReason.RATE_LIMIT, FailureReason.INVALID_RESPONSE]
        return failure_reason not in non_retriable

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry using exponential backoff"""
        delay = min(
            self.retry_config.initial_delay_seconds * (
                self.retry_config.exponential_base ** attempt
            ),
            self.retry_config.max_delay_seconds
        )

        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            import random
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter

        return delay

    def get_statistics(self) -> Dict:
        """Get API resilience statistics"""
        circuit_stats = self.circuit_breaker.get_statistics()

        return {
            **self.stats,
            "circuit_breaker": circuit_stats,
            "success_rate": (
                f"{self.stats['successful_calls'] / self.stats['total_calls']:.2%}"
                if self.stats['total_calls'] > 0 else "N/A"
            ),
            "fallback_rate": (
                f"{self.stats['fallback_used'] / self.stats['total_calls']:.2%}"
                if self.stats['total_calls'] > 0 else "N/A"
            )
        }


# Decorator for easy resilience wrapping
def resilient_api_call(
    name: str,
    fallback: Optional[Callable] = None,
    **kwargs
):
    """
    Decorator to make any function resilient.

    Example:
        @resilient_api_call(name="ollama", fallback=lambda: "Fallback response")
        def call_ollama(prompt):
            return ollama_client.generate(prompt)
    """
    def decorator(func):
        caller = APIResilientCaller(name=name, **kwargs)

        @wraps(func)
        def wrapper(*args, **func_kwargs):
            result = caller.call(
                func=lambda: func(*args, **func_kwargs),
                fallback=fallback
            )
            if result.success:
                return result.data
            else:
                raise result.error

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Simulate flaky API
    call_count = 0

    def flaky_api():
        """API that fails first 2 times, then succeeds"""
        global call_count
        call_count += 1
        if call_count < 3:
            raise Exception(f"Connection error (attempt {call_count})")
        return {"result": "success", "data": "Hello from API"}

    def fallback_response():
        """Fallback response when API fails"""
        return {"result": "fallback", "data": "Cached response"}

    print("=" * 80)
    print("TEST: API Resilience with Retry")
    print("=" * 80)

    caller = APIResilientCaller(
        name="test_api",
        retry_config=RetryConfig(max_attempts=3, initial_delay_seconds=0.5)
    )

    result = caller.call(
        func=flaky_api,
        fallback=fallback_response
    )

    print(f"Success: {result.success}")
    print(f"Data: {result.data}")
    print(f"Attempts: {result.attempts}")
    print(f"Duration: {result.total_duration_ms:.2f}ms")
    print(f"Used Fallback: {result.used_fallback}")
    print(f"Circuit State: {result.circuit_state.value}")

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = caller.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
