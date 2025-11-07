"""
Test Rate Limiting

Verifies that rate limiting is properly configured and works as expected.
"""

import asyncio
import time
import pytest
from src.core.async_scraper import AsyncWebScraper


@pytest.mark.asyncio
async def test_rate_limiter_initialization():
    """Test that rate limiter is initialized correctly"""
    scraper = AsyncWebScraper(
        rate_limit_rps=5.0,
        rate_limit_burst=10
    )

    assert scraper.rate_limiter is not None
    assert scraper.rate_limiter.max_rate == 5.0


@pytest.mark.asyncio
async def test_rate_limiting_enforced():
    """Test that rate limiting actually slows down requests"""
    # Create scraper with strict rate limit (2 req/s)
    scraper = AsyncWebScraper(
        rate_limit_rps=2.0,
        rate_limit_burst=2,
        delay_between_requests=0.0  # Disable additional delay to test only rate limiter
    )

    # Measure time for 5 requests
    start_time = time.time()

    async def mock_request():
        """Mock request that uses rate limiter"""
        async with scraper.rate_limiter:
            await asyncio.sleep(0.01)  # Minimal work

    # Execute 5 requests
    tasks = [mock_request() for _ in range(5)]
    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # With 2 req/s, 5 requests should take at least 1.5 seconds
    # (allowing for burst of 2 initial requests, then 3 more at 0.5s intervals)
    # Burst: 2 requests at t=0, then 3 more requests at t=0.5, t=1.0, t=1.5
    assert elapsed >= 1.4, f"Rate limiting not enforced: completed in {elapsed}s"
    assert elapsed < 4.0, f"Rate limiting too strict: took {elapsed}s"


@pytest.mark.asyncio
async def test_rate_limiting_with_environment_variables(monkeypatch):
    """Test rate limit configuration from environment variables"""
    monkeypatch.setenv("RATE_LIMIT_RPS", "15.0")
    monkeypatch.setenv("RATE_LIMIT_BURST", "30")

    from src.config.settings import AppConfig
    config = AppConfig.from_env()

    assert config.scraping.RATE_LIMIT_RPS == 15.0
    assert config.scraping.RATE_LIMIT_BURST == 30


def test_rate_limit_config_defaults():
    """Test default rate limit configuration"""
    from src.config.settings import CONFIG

    assert CONFIG.scraping.RATE_LIMIT_RPS == 10.0
    assert CONFIG.scraping.RATE_LIMIT_BURST == 20


@pytest.mark.asyncio
async def test_concurrent_rate_limiting():
    """Test rate limiting with concurrent requests"""
    scraper = AsyncWebScraper(
        rate_limit_rps=5.0,
        rate_limit_burst=5,
        max_concurrent=3
    )

    request_times = []

    async def timed_request(request_id):
        """Mock request that records timestamp"""
        async with scraper.rate_limiter:
            request_times.append((request_id, time.time()))
            await asyncio.sleep(0.01)

    # Execute 10 concurrent requests
    tasks = [timed_request(i) for i in range(10)]
    start = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # With 5 req/s and burst of 5, 10 requests should take at least 1 second
    # Burst: 5 requests at t=0, then 5 more at t=0.2, t=0.4, t=0.6, t=0.8, t=1.0
    assert elapsed >= 0.9, f"Rate limiting not enforced for concurrent requests: {elapsed}s"

    # Verify requests are spaced out
    request_times.sort(key=lambda x: x[1])
    for i in range(1, len(request_times)):
        time_diff = request_times[i][1] - request_times[i-1][1]
        # Should have at least ~200ms between requests (1/5 = 0.2s)
        # Allow some tolerance for test execution variance
        if time_diff > 0.1:  # Only check if there's meaningful spacing
            assert time_diff >= 0.15, f"Requests too close: {time_diff}s"


if __name__ == "__main__":
    # Run basic test
    async def main():
        print("Testing rate limiter initialization...")
        await test_rate_limiter_initialization()
        print("✓ Initialization test passed")

        print("\nTesting rate limiting enforcement...")
        await test_rate_limiting_enforced()
        print("✓ Rate limiting enforcement test passed")

        print("\nTesting concurrent rate limiting...")
        await test_concurrent_rate_limiting()
        print("✓ Concurrent rate limiting test passed")

        print("\nAll tests passed!")

    asyncio.run(main())
