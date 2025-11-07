"""
Tests for Circuit Breaker Pattern

Validates circuit breaker functionality including state transitions,
failure thresholds, and timeout recovery.
"""

import pytest
import time
from datetime import datetime, timedelta

from src.clients.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
)


class TestException(Exception):
    """Test exception for circuit breaker"""

    pass


def test_circuit_breaker_closed_on_success():
    """Test circuit breaker remains closed on successful calls"""
    breaker = CircuitBreaker(
        failure_threshold=3, timeout=60, expected_exception=TestException
    )

    def successful_operation():
        return "success"

    # Multiple successful calls should keep circuit closed
    for _ in range(5):
        result = breaker.call(successful_operation)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_opens_after_threshold():
    """Test circuit breaker opens after reaching failure threshold"""
    breaker = CircuitBreaker(
        failure_threshold=3, timeout=60, expected_exception=TestException
    )

    def failing_operation():
        raise TestException("Test failure")

    # Trigger failures up to threshold
    for i in range(3):
        with pytest.raises(TestException):
            breaker.call(failing_operation)

        # Circuit should stay closed until threshold
        if i < 2:
            assert breaker.state == CircuitState.CLOSED
        else:
            # Circuit opens on 3rd failure
            assert breaker.state == CircuitState.OPEN


def test_circuit_breaker_rejects_when_open():
    """Test circuit breaker rejects calls when open"""
    breaker = CircuitBreaker(
        failure_threshold=2, timeout=60, expected_exception=TestException
    )

    def failing_operation():
        raise TestException("Test failure")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN

    # Next call should be rejected immediately
    with pytest.raises(CircuitBreakerError, match="Circuit breaker .* is OPEN"):
        breaker.call(failing_operation)


def test_circuit_breaker_half_open_after_timeout():
    """Test circuit breaker enters half-open state after timeout"""
    breaker = CircuitBreaker(
        failure_threshold=2, timeout=1, expected_exception=TestException  # 1 second timeout
    )

    def failing_operation():
        raise TestException("Test failure")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN

    # Wait for timeout
    time.sleep(1.1)

    # Next call should enter half-open state
    def successful_operation():
        return "success"

    result = breaker.call(successful_operation)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_closes_on_half_open_success():
    """Test circuit breaker closes after successful call in half-open state"""
    breaker = CircuitBreaker(
        failure_threshold=2, timeout=1, expected_exception=TestException
    )

    def failing_operation():
        raise TestException("Test failure")

    def successful_operation():
        return "success"

    # Open the circuit
    for _ in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN

    # Wait for timeout and make successful call
    time.sleep(1.1)
    result = breaker.call(successful_operation)

    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_reopens_on_half_open_failure():
    """Test circuit breaker reopens if call fails in half-open state"""
    breaker = CircuitBreaker(
        failure_threshold=2, timeout=1, expected_exception=TestException
    )

    def failing_operation():
        raise TestException("Test failure")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN

    # Wait for timeout and make failing call
    time.sleep(1.1)

    with pytest.raises(TestException):
        breaker.call(failing_operation)

    # Circuit should reopen
    assert breaker.state == CircuitState.OPEN


def test_circuit_breaker_get_state():
    """Test circuit breaker state information"""
    breaker = CircuitBreaker(
        failure_threshold=3, timeout=60, expected_exception=TestException, name="test"
    )

    state = breaker.get_state()

    assert state["name"] == "test"
    assert state["state"] == CircuitState.CLOSED.value
    assert state["failure_count"] == 0
    assert state["failure_threshold"] == 3
    assert state["timeout_seconds"] == 60


def test_circuit_breaker_reset():
    """Test manual circuit breaker reset"""
    breaker = CircuitBreaker(
        failure_threshold=2, timeout=60, expected_exception=TestException
    )

    def failing_operation():
        raise TestException("Test failure")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_operation)

    assert breaker.state == CircuitState.OPEN

    # Manually reset
    breaker.reset()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_decorator():
    """Test circuit breaker decorator"""

    @circuit_breaker(failure_threshold=2, timeout=1, name="decorator_test")
    def failing_operation():
        raise TestException("Test failure")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(TestException):
            failing_operation()

    # Next call should be rejected
    with pytest.raises(CircuitBreakerError):
        failing_operation()

    # Check breaker is attached
    assert hasattr(failing_operation, "circuit_breaker")
    assert failing_operation.circuit_breaker.state == CircuitState.OPEN


def test_circuit_breaker_only_catches_expected_exception():
    """Test circuit breaker only catches specified exception type"""
    breaker = CircuitBreaker(
        failure_threshold=3, timeout=60, expected_exception=TestException
    )

    class OtherException(Exception):
        pass

    def operation_with_other_exception():
        raise OtherException("Different exception")

    # OtherException should not be caught by circuit breaker
    with pytest.raises(OtherException):
        breaker.call(operation_with_other_exception)

    # Circuit should still be closed
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_success_count():
    """Test circuit breaker tracks success count"""
    breaker = CircuitBreaker(
        failure_threshold=3, timeout=60, expected_exception=TestException
    )

    def successful_operation():
        return "success"

    # Make multiple successful calls
    for _ in range(5):
        breaker.call(successful_operation)

    assert breaker.success_count == 5


def test_circuit_breaker_time_until_reset():
    """Test circuit breaker calculates time until reset"""
    breaker = CircuitBreaker(
        failure_threshold=2, timeout=5, expected_exception=TestException
    )

    def failing_operation():
        raise TestException("Test failure")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(TestException):
            breaker.call(failing_operation)

    # Check time until reset is approximately 5 seconds
    time_until_reset = breaker._time_until_reset()
    assert 4.5 <= time_until_reset <= 5.5

    # Wait a bit
    time.sleep(2)

    # Time should decrease
    time_until_reset = breaker._time_until_reset()
    assert 2.5 <= time_until_reset <= 3.5
