# Async Testing Guide

Comprehensive guide for testing asynchronous code in RSS Analyzer using pytest-asyncio.

## Overview

The RSS Analyzer uses pytest-asyncio to enable testing of async/await code patterns. This guide covers configuration, fixtures, and best practices for async testing.

## Installation

pytest-asyncio is included in dev dependencies:

```bash
# Already installed via uv
uv sync

# Or manually
uv add --dev pytest-asyncio>=0.23.0
```

## Configuration

### pytest Configuration

Async testing is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # Automatic async test detection
testpaths = ["tests"]
markers = [
    "asyncio: mark test as an asyncio test",
    "slow: mark test as slow running",
    "integration: mark test as integration test",
    "unit: mark test as unit test",
    "benchmark: mark test as a performance benchmark",
]
```

The `asyncio_mode = "auto"` setting automatically detects and runs coroutine tests without requiring explicit event loop management.

## Writing Async Tests

### Basic Async Test

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    """Test an async operation"""
    result = await some_async_function()
    assert result == expected_value
```

The `@pytest.mark.asyncio` decorator marks a test as async and ensures pytest-asyncio handles the event loop.

### Async Fixtures

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
async def mock_api_client():
    """Mock async API client"""
    client = AsyncMock()
    client.analyze_article = AsyncMock(return_value={
        "methodology": "detailed",
        "findings": "key insights"
    })
    return client

@pytest.mark.asyncio
async def test_with_async_fixture(mock_api_client):
    """Test using async fixture"""
    result = await mock_api_client.analyze_article("content")
    assert result["methodology"] == "detailed"
```

### Event Loop Fixture

For manual event loop control (usually not needed with `asyncio_mode = "auto"`):

```python
@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

## Available Fixtures

The test suite includes several pre-built fixtures in `tests/conftest.py`:

### Core Fixtures

- **event_loop**: Manages event loop lifecycle
- **mock_async_api_client**: Mocked async AI client with full interface
- **mock_async_scraper**: Mocked async web scraper

### Existing Fixtures

Standard sync fixtures for configuration and mocking:
- temp_db, temp_dir
- mock_config, app_config
- mock_api_client, mock_db_manager
- mock_rss_parser, mock_scraper
- mock_report_generator
- sample_analysis, sample_article_data

## Testing Patterns

### Concurrent Operations

Test multiple concurrent operations:

```python
import asyncio

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent processing"""
    tasks = [
        fetch_article_async(url)
        for url in article_urls
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == len(article_urls)
```

### Error Handling

Test error conditions in async code:

```python
@pytest.mark.asyncio
async def test_api_error_handling():
    """Test handling of API errors"""
    client = AsyncMock()
    client.analyze_article = AsyncMock(side_effect=APIError("Rate limited"))

    with pytest.raises(APIError):
        await client.analyze_article("content")
```

### Timeout Testing

Test timeout behavior:

```python
@pytest.mark.asyncio
async def test_request_timeout():
    """Test timeout handling"""
    async def slow_request():
        await asyncio.sleep(5)
        return "result"

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_request(), timeout=0.1)
```

### Rate Limiting

Test rate limiting and semaphore behavior:

```python
@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Test concurrent request limiting"""
    semaphore = asyncio.Semaphore(3)
    max_concurrent = 0

    async def limited_task():
        nonlocal max_concurrent
        async with semaphore:
            max_concurrent += 1
            await asyncio.sleep(0.1)

    await asyncio.gather(*[limited_task() for _ in range(10)])
    assert max_concurrent <= 3
```

### Performance Testing

Test performance characteristics of async operations:

```python
import time

@pytest.mark.asyncio
async def test_concurrent_faster_than_sequential():
    """Verify concurrent is faster than sequential"""
    async def task(n):
        await asyncio.sleep(0.1)
        return n

    # Sequential
    start = time.time()
    for i in range(5):
        await task(i)
    sequential = time.time() - start

    # Concurrent
    start = time.time()
    await asyncio.gather(*[task(i) for i in range(5)])
    concurrent = time.time() - start

    assert concurrent < sequential
```

## Mock Usage

### AsyncMock

Use `AsyncMock` for async functions:

```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_mock():
    """Test with AsyncMock"""
    mock = AsyncMock(return_value="result")

    result = await mock("arg1", "arg2")
    assert result == "result"
    mock.assert_called_once_with("arg1", "arg2")
```

### AsyncMock with Side Effects

```python
@pytest.mark.asyncio
async def test_async_mock_side_effects():
    """Test AsyncMock with side effects"""
    mock = AsyncMock(side_effect=[
        "first",
        "second",
        Exception("error")
    ])

    assert await mock() == "first"
    assert await mock() == "second"
    with pytest.raises(Exception):
        await mock()
```

## Running Tests

### Run All Async Tests

```bash
# All tests
uv run pytest tests/test_async_*.py tests/test_rate_limiting.py -v

# Specific test file
uv run pytest tests/test_async_article_processor.py -v

# Specific test
uv run pytest tests/test_async_article_processor.py::test_concurrent_scraping -v
```

### Run with Markers

```bash
# Only asyncio tests
uv run pytest -m asyncio

# Slow tests
uv run pytest -m slow

# Exclude integration tests
uv run pytest -m "not integration"
```

### Coverage

```bash
# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Organization

### Rate Limiting Tests (`tests/test_rate_limiting.py`)

- Rate limiter initialization
- Rate limiting enforcement
- Environment variable configuration
- Concurrent rate limiting

### Async Scraper Tests (`tests/test_async_scraper.py`)

- Content scraping (25 tests)
- Link extraction and following
- Concurrent scraping limits
- Performance comparison

### Async Client Tests (`tests/test_async_clients.py`)

- Client initialization (21 tests)
- Async analysis operations
- Batch processing
- Error handling and rate limiting

### Article Processor Tests (`tests/test_async_article_processor.py`)

- Processor initialization and configuration
- Concurrent scraping and analysis
- Semaphore-based concurrency limiting
- Error handling with mixed failures
- Retry logic and timeouts
- Performance benchmarks
- Result aggregation
- Full pipeline integration

## Best Practices

### 1. Use Appropriate Markers

```python
@pytest.mark.asyncio
@pytest.mark.slow
async def test_slow_operation():
    """Slow test marked appropriately"""
    pass
```

### 2. Clean Up Resources

```python
@pytest.fixture
async def client():
    """Fixture with cleanup"""
    client = AsyncClient()
    yield client
    await client.close()  # Cleanup
```

### 3. Avoid Blocking Calls

```python
# ✓ Correct - use async/await
async def test_correct():
    result = await async_operation()

# ✗ Wrong - blocking in async test
async def test_wrong():
    result = time.sleep(1)  # Blocks event loop!
```

### 4. Test Error Conditions

```python
@pytest.mark.asyncio
async def test_error_handling():
    """Test both success and failure paths"""
    # Success case
    result = await operation()
    assert result.success

    # Failure case
    with pytest.raises(SpecificError):
        await failing_operation()
```

### 5. Use Fixtures for Setup

```python
@pytest.fixture
async def initialized_processor():
    """Setup complex test environment"""
    processor = AsyncArticleProcessor(config)
    await processor.initialize()
    yield processor
    await processor.cleanup()
```

## Debugging

### Enable Verbose Output

```bash
uv run pytest tests/test_async_article_processor.py -v -s
```

### Show Full Tracebacks

```bash
uv run pytest tests/ -vv --tb=long
```

### Run Single Test

```bash
uv run pytest tests/test_async_article_processor.py::test_concurrent_scraping -vv -s
```

### Debug with print()

```python
@pytest.mark.asyncio
async def test_with_debug():
    """Test with debug output"""
    result = await operation()
    print(f"Debug: {result}")  # Use -s flag to see output
    assert result
```

## Common Issues

### Issue: "event loop is closed"

**Cause**: Event loop not properly managed

**Solution**: Ensure event_loop fixture is defined and used

```python
@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

### Issue: "coroutine was never awaited"

**Cause**: Forgot to await async function

**Solution**: Always use await with async functions

```python
# ✓ Correct
result = await async_func()

# ✗ Wrong
result = async_func()  # Missing await!
```

### Issue: Timeout during test

**Cause**: Deadlock or hung coroutine

**Solution**: Add timeout to test or review async logic

```python
@pytest.mark.timeout(10)  # Requires pytest-timeout
async def test_no_hang():
    result = await operation()
```

## Performance Benchmarks

The test suite includes performance benchmarks:

```bash
# Run benchmark tests
uv run pytest tests/test_async_article_processor.py -m benchmark -v
```

Key performance metrics:
- **Concurrent vs Sequential**: 3-5x faster
- **Throughput**: 4.2x improvement
- **Semaphore Limiting**: Correctly limits concurrency

See `/home/mess/dev/rss-analyzer/docs/OPTIMIZATION_RESULTS.md` for detailed benchmarks.

## Advanced Topics

### Custom Async Fixtures

```python
@pytest.fixture
async def complex_setup():
    """Complex async setup"""
    client = AsyncClient()
    await client.connect()
    db = AsyncDatabase()
    await db.setup()

    yield client, db

    await db.teardown()
    await client.disconnect()
```

### Parametrized Async Tests

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("url,expected", [
    ("https://example.com/1", "Article 1"),
    ("https://example.com/2", "Article 2"),
])
async def test_scrape_multiple(url, expected):
    """Test multiple URLs"""
    result = await scraper.scrape(url)
    assert result.title == expected
```

### Async Context Managers in Tests

```python
@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager"""
    async with AsyncResource() as resource:
        result = await resource.do_something()
        assert result
```

## References

- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [RSS Analyzer Async Architecture](./ASYNC_IMPLEMENTATION_COMPLETE.md)
- [Performance Optimization Guide](./docs/optimization/PERFORMANCE_ANALYSIS.md)

## Quick Reference

```bash
# Install and setup
uv sync

# Run all async tests
uv run pytest tests/test_async_*.py tests/test_rate_limiting.py -v

# Run specific test
uv run pytest tests/test_async_article_processor.py::test_name -vv

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Show pytest markers
uv run pytest --markers

# Run only asyncio tests
uv run pytest -m asyncio
```
