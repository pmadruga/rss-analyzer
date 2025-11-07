# Rate Limiting in RSS Analyzer

## Overview

The RSS Analyzer includes built-in rate limiting to prevent DoS attacks, avoid IP bans, and be a good internet citizen. Rate limiting is implemented using the `aiolimiter` library and is automatically applied to all web scraping operations.

## Features

- **Automatic Rate Limiting**: All HTTP requests are automatically rate-limited
- **Configurable Limits**: Adjust requests-per-second and burst size via configuration
- **Per-Domain Protection**: Prevents overwhelming individual websites
- **Concurrent Safety**: Works seamlessly with async/concurrent operations
- **Zero Configuration Required**: Works out-of-the-box with sensible defaults

## Configuration

### Default Settings

```yaml
# config/config.yaml
scraper:
  rate_limit_rps: 10      # Maximum 10 requests per second
  rate_limit_burst: 20    # Allow bursts up to 20 requests
```

### Environment Variable Override

Override rate limits using environment variables:

```bash
# Set maximum requests per second
export SCRAPER_RATE_LIMIT_RPS=5

# Set maximum burst size
export SCRAPER_RATE_LIMIT_BURST=10

# Run analyzer with custom rate limits
docker compose run rss-analyzer run --limit 10
```

### Programmatic Configuration

```python
from src.core.async_scraper import AsyncWebScraper

# Create scraper with custom rate limits
scraper = AsyncWebScraper(
    rate_limit_rps=5.0,        # 5 requests per second
    rate_limit_burst=10,       # Burst up to 10 requests
    max_concurrent=3,          # Maximum 3 concurrent requests
    delay_between_requests=1.0 # Additional 1 second delay
)

# Scrape articles
results = await scraper.scrape_articles_batch(urls)
```

## How It Works

### AsyncLimiter Implementation

The rate limiter uses a token bucket algorithm:

1. **Tokens are added** at a steady rate (`rate_limit_rps`)
2. **Each request consumes** one token
3. **Requests wait** if no tokens are available
4. **Burst capacity** allows temporary spikes in requests

```python
from aiolimiter import AsyncLimiter

# Initialize rate limiter (10 req/s)
rate_limiter = AsyncLimiter(
    max_rate=10.0,      # 10 tokens per second
    time_period=1.0     # Over 1 second period
)

# Apply rate limiting to requests
async with rate_limiter:
    # This blocks if rate limit exceeded
    response = await session.get(url)
```

### Request Flow

```
┌─────────────────────────────────────────────────────┐
│              AsyncWebScraper.scrape_article()       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Semaphore (concurrency control)                │
│     │                                               │
│     ├─► 2. Rate Limiter (AsyncLimiter)             │
│     │      ├─► Check token availability            │
│     │      ├─► Wait if tokens exhausted            │
│     │      └─► Consume token                       │
│     │                                               │
│     ├─► 3. Delay (respect_rate_limit)              │
│     │      └─► Minimum time between requests       │
│     │                                               │
│     └─► 4. HTTP Request                            │
│            └─► Fetch URL content                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Benefits

### 1. Prevents DoS Attacks

Rate limiting prevents accidental Denial-of-Service attacks:

- **Without rate limiting**: Could send 100+ requests/second
- **With rate limiting**: Maximum 10 requests/second (default)
- **Result**: Target servers remain responsive

### 2. Avoids IP Bans

Many websites ban IPs that make too many requests:

- **arXiv**: Rate limits at ~15 requests/second
- **IEEE Xplore**: Aggressive rate limiting
- **Academic publishers**: Monitor for abuse
- **Result**: Our default 10 req/s stays well under limits

### 3. Better Citizenship

Being a good internet citizen:

- **Respects server resources**: Doesn't overwhelm websites
- **Follows robots.txt**: Combined with robots.txt support
- **Sustainable scraping**: Can run continuously without issues
- **Ethical behavior**: Treats web resources responsibly

## Tuning Rate Limits

### Conservative (Academic Publishers)

For academic publishers with strict rate limits:

```yaml
scraper:
  rate_limit_rps: 5       # 5 requests/second
  rate_limit_burst: 10    # Small burst allowance
  delay_between_requests: 2.0  # 2 second additional delay
```

### Moderate (Default)

Balanced setting for general use:

```yaml
scraper:
  rate_limit_rps: 10      # 10 requests/second
  rate_limit_burst: 20    # Moderate burst allowance
  delay_between_requests: 1.0  # 1 second additional delay
```

### Aggressive (Internal/Authorized)

For authorized/internal scraping:

```yaml
scraper:
  rate_limit_rps: 20      # 20 requests/second
  rate_limit_burst: 50    # Large burst allowance
  delay_between_requests: 0.5  # 0.5 second delay
```

## Monitoring

### Log Output

Rate limiting events are logged:

```
INFO - Rate limiter initialized: 10.0 req/s, burst=20, max_concurrent=5
DEBUG - Rate limited request to: https://arxiv.org/abs/2501.12345
DEBUG - Rate limiting: sleeping for 0.15 seconds
```

### Performance Metrics

Check rate limiting impact:

```python
from src.core.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
metrics = monitor.get_metrics()

print(f"Average request time: {metrics['avg_request_time']}ms")
print(f"Requests per second: {metrics['requests_per_second']}")
print(f"Rate limit delays: {metrics['rate_limit_delays']}")
```

## Examples

### Docker Compose

```bash
# Use default rate limits
docker compose run rss-analyzer run --limit 10

# Override with environment variables
docker compose run \
  -e SCRAPER_RATE_LIMIT_RPS=5 \
  -e SCRAPER_RATE_LIMIT_BURST=10 \
  rss-analyzer run --limit 10
```

### Python Script

```python
import asyncio
from src.core.async_scraper import AsyncWebScraper

async def main():
    # Conservative rate limiting
    scraper = AsyncWebScraper(
        rate_limit_rps=5.0,
        rate_limit_burst=10,
        max_concurrent=3
    )

    urls = [
        "https://arxiv.org/abs/2501.12345",
        "https://arxiv.org/abs/2501.12346",
        "https://arxiv.org/abs/2501.12347",
    ]

    # Scrape with automatic rate limiting
    results = await scraper.scrape_articles_batch(urls)
    print(f"Scraped {len(results)} articles")

asyncio.run(main())
```

## Troubleshooting

### Issue: Scraping is too slow

**Solution**: Increase rate limit if you have permission:

```bash
export SCRAPER_RATE_LIMIT_RPS=20
export SCRAPER_RATE_LIMIT_BURST=50
```

### Issue: Getting IP banned

**Solution**: Decrease rate limit and add delays:

```bash
export SCRAPER_RATE_LIMIT_RPS=2
export SCRAPER_DELAY_BETWEEN_REQUESTS=3.0
```

### Issue: Timeout errors

**Solution**: Rate limiting can increase overall time:

```bash
# Increase timeout to accommodate rate limiting
export SCRAPER_TIMEOUT=60
```

## Best Practices

1. **Start Conservative**: Use default 10 req/s, adjust if needed
2. **Monitor Logs**: Watch for rate limit delays and timeouts
3. **Respect Limits**: Academic publishers often have strict rules
4. **Test First**: Try with small batches before large runs
5. **Use Environment Variables**: Easy to adjust without code changes
6. **Combine Protections**: Use rate limiting + delays + retries

## Dependencies

```bash
# Install aiolimiter
pip install aiolimiter>=1.1.0

# Or with uv
uv pip install aiolimiter>=1.1.0
```

## Related Documentation

- [Async Scraper Guide](ASYNC_SCRAPER.md)
- [Connection Pooling](CONNECTION_POOLING.md)
- [Performance Monitoring](MONITORING.md)
- [Configuration Guide](../README.md#configuration)

## Summary

Rate limiting is essential for:
- ✅ **Preventing DoS attacks**
- ✅ **Avoiding IP bans**
- ✅ **Being a good internet citizen**
- ✅ **Sustainable long-term scraping**
- ✅ **Respecting server resources**

The implementation is automatic, configurable, and works seamlessly with concurrent operations.
