"""
Circuit Breaker Pattern Implementation

Provides fault tolerance and resilience for external API calls.
Implements the circuit breaker pattern with three states: CLOSED, OPEN, HALF_OPEN.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""

    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API resilience.

    Prevents cascade failures by failing fast when a service is down.
    Automatically attempts recovery after a timeout period.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)

        def risky_operation():
            return api_call()

        result = breaker.call(risky_operation)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "default",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting reset
            expected_exception: Exception type that triggers circuit
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

        logger.info(
            f"Circuit breaker '{name}' initialized "
            f"(threshold={failure_threshold}, timeout={timeout}s)"
        )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Callable to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func execution

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception if circuit is closed
        """
        # Check if circuit should attempt reset
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service is unavailable. Try again in "
                    f"{self._time_until_reset():.0f} seconds."
                )

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(
                f"Circuit breaker '{self.name}' closed after successful test "
                f"(success_count={self.success_count})"
            )

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        logger.warning(
            f"Circuit breaker '{self.name}' failure {self.failure_count}/"
            f"{self.failure_threshold}"
        )

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker '{self.name}' OPENED after "
                f"{self.failure_count} consecutive failures. "
                f"Requests will be rejected for {self.timeout} seconds."
            )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False

        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure >= timedelta(seconds=self.timeout)

    def _time_until_reset(self) -> float:
        """Calculate seconds until reset attempt"""
        if self.last_failure_time is None:
            return 0.0

        time_since_failure = datetime.utcnow() - self.last_failure_time
        time_until_reset = timedelta(seconds=self.timeout) - time_since_failure
        return max(0.0, time_until_reset.total_seconds())

    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED state")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    def get_state(self) -> dict[str, Any]:
        """
        Get current circuit breaker state.

        Returns:
            Dictionary with state information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "timeout_seconds": self.timeout,
            "time_until_reset": (
                self._time_until_reset() if self.state == CircuitState.OPEN else 0.0
            ),
            "last_failure_time": (
                self.last_failure_time.isoformat()
                if self.last_failure_time
                else None
            ),
        }


# Convenience decorator for circuit breaker
def circuit_breaker(
    failure_threshold: int = 5, timeout: int = 60, name: str = "default"
):
    """
    Decorator for circuit breaker protection.

    Usage:
        @circuit_breaker(failure_threshold=3, timeout=30, name="api")
        def api_call():
            return requests.get("https://api.example.com")
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold, timeout=timeout, name=name
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        wrapper.circuit_breaker = breaker  # Attach breaker for inspection
        return wrapper

    return decorator


__all__ = ["CircuitBreaker", "CircuitBreakerError", "CircuitState", "circuit_breaker"]
