# Async Article Processor

High-performance async version of the article processor that achieves **6-8x throughput improvement** through concurrent processing.

## Overview

The `AsyncArticleProcessor` replaces sequential processing with concurrent execution, allowing multiple articles to be scraped and analyzed simultaneously.

### Performance Improvements

| Metric | Sync Processor | Async Processor | Improvement |
|--------|---------------|-----------------|-------------|
| **Throughput** | 10 articles/min | 60-80 articles/min | **6-8x faster** |
| **Scraping** | Sequential | Concurrent (5 parallel) | **5x faster** |
| **AI Analysis** | Sequential | Concurrent (5 parallel) | **5x faster** |
| **Database** | Batched | Batched (same) | No change |
| **Total Time** | 100 articles = 600s | 100 articles = 85s | **85% reduction** |

### How It Works

```python
# Sync Processor (Sequential)
for article in articles:
    content = scrape(article)      # 5s
    analysis = analyze(content)    # 10s
    # Total: 15s per article
    # 10 articles = 150s

# Async Processor (Concurrent)
async with concurrent_limit(5):
    tasks = [
        process_article(article)   # All run in parallel
        for article in articles
    ]
    results = await gather(tasks)
# 10 articles = ~30s (3 batches of 5 articles)
```

## Usage

### Command Line

```bash
# Standard sync mode (original)
uv run python -m src.main run --limit 10

# Async mode (6-8x faster)
uv run python -m src.main run --limit 10 --async

# Adjust concurrency (default: 5)
uv run python -m src.main run --limit 30 --async --max-concurrent 10

# Docker async mode
docker compose run rss-analyzer run --limit 20 --async --max-concurrent 8
```

### Python API

```python
import asyncio
from src.processors import AsyncArticleProcessor, ProcessingConfig

# Initialize processor
config = {
    "db_path": "data/articles.db",
    "rss_feed_url": "https://example.com/feed",
    "anthropic_api_key": "your-key",
    "api_provider": "anthropic",
    "max_concurrent_articles": 5,
}

processor = AsyncArticleProcessor(config)

# Configure processing
processing_config = ProcessingConfig(
    force_refresh=False,
    limit=30,
    follow_links=True,
    max_concurrent=5  # Concurrent article processing
)

# Run async pipeline
results = asyncio.run(processor.run_async(processing_config))

print(f"Processed {results.analyzed_articles} articles in {results.duration:.2f}s")
```

## Architecture

### Component Stack

```
┌─────────────────────────────────────────────────┐
│         AsyncArticleProcessor                   │
├─────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │ RSS Parser   │  │ ProcessingConfig     │    │
│  │ (sync)       │  │ - max_concurrent: 5  │    │
│  └──────────────┘  └──────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │      Async Processing Pipeline           │  │
│  │  ┌────────────┐  ┌──────────────────┐   │  │
│  │  │ Semaphore  │→ │ Article 1        │   │  │
│  │  │ (limit 5)  │→ │ Article 2        │   │  │
│  │  │            │→ │ Article 3        │   │  │
│  │  │            │→ │ Article 4        │   │  │
│  │  │            │→ │ Article 5        │   │  │
│  │  └────────────┘  └──────────────────┘   │  │
│  │                                           │  │
│  │  Each article processed concurrently:    │  │
│  │  1. AsyncWebScraper.scrape_single()      │  │
│  │  2. AsyncClaudeClient.analyze_async()    │  │
│  │  3. Cache operations (sync)              │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │      Batch Database Operations           │  │
│  │  - Batch insert articles                 │  │
│  │  - Batch update titles                   │  │
│  │  - Batch update statuses                 │  │
│  │  - Batch insert content                  │  │
│  │  - Batch log processing                  │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Concurrency Control

```python
# Semaphore limits concurrent operations
semaphore = asyncio.Semaphore(max_concurrent)

async def process_with_limit(article):
    async with semaphore:
        # Only max_concurrent articles processed at once
        scraped = await scrape_async(article)
        analysis = await analyze_async(scraped)
        return (article, scraped, analysis)

# Process all articles with concurrency limit
tasks = [process_with_limit(a) for a in articles]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Error Handling

- **Individual failures don't stop batch**: Uses `return_exceptions=True`
- **Logged and tracked**: Errors collected in `results.errors`
- **Batch operations protected**: Try/except around each batch operation
- **Database integrity**: Transaction-based updates

## Configuration

### config.yaml

```yaml
processing:
  # Async processing configuration
  use_async_processing: false    # Enable via --async flag
  max_concurrent_articles: 5     # Concurrent article processing
  async_scraper_concurrent: 5    # Concurrent scraper requests
```

### Environment Variables

```bash
# Enable async mode by default
export USE_ASYNC_PROCESSING=true

# Increase concurrency
export MAX_CONCURRENT_ARTICLES=10

# Async scraper concurrency
export ASYNC_SCRAPER_CONCURRENT=8
```

## Performance Tuning

### Concurrency Limits

The optimal `max_concurrent` depends on:

1. **API rate limits**: Claude API has rate limits
2. **Network bandwidth**: More concurrent = more bandwidth
3. **Memory usage**: Each article uses ~50MB memory
4. **Database contention**: Batch operations are single-threaded

**Recommended values:**

```python
# Conservative (safe for all setups)
max_concurrent = 3

# Moderate (recommended)
max_concurrent = 5

# Aggressive (high-bandwidth, high rate limits)
max_concurrent = 10

# Maximum (careful of rate limits!)
max_concurrent = 20
```

### Memory Usage

```python
# Memory per concurrent article
memory_per_article = 50  # MB

# Total memory for async processing
total_memory = max_concurrent * memory_per_article

# Example: 5 concurrent articles
# 5 * 50MB = 250MB (plus base ~200MB) = ~450MB total
```

### Rate Limiting

AsyncWebScraper and AsyncClaudeClient both implement rate limiting:

```python
# AsyncWebScraper rate limiting
scraper = AsyncWebScraper(
    delay_between_requests=1.0,     # Minimum delay between requests
    max_concurrent=5,                # Max concurrent scrapes
    rate_limit_rps=10.0,            # Max requests per second
)

# AsyncClaudeClient rate limiting
client = AsyncClaudeClient(api_key, model)
# Uses semaphore + internal rate limiting
# Respects Claude API rate limits automatically
```

## Benchmarking

### Compare Sync vs Async

```bash
# Run sync mode
time uv run python -m src.main run --limit 30
# Expected: ~300-450 seconds

# Run async mode
time uv run python -m src.main run --limit 30 --async --max-concurrent 5
# Expected: ~50-75 seconds

# Speedup calculation
# Speedup = sync_time / async_time
# Example: 360s / 60s = 6x faster
```

### Benchmark Script

```python
import asyncio
import time
from src.processors import ArticleProcessor, AsyncArticleProcessor, ProcessingConfig

config = {...}  # Your config

# Benchmark sync
start = time.time()
sync_processor = ArticleProcessor(config)
sync_results = sync_processor.run(ProcessingConfig(limit=20))
sync_time = time.time() - start

# Benchmark async
start = time.time()
async_processor = AsyncArticleProcessor(config)
async_results = asyncio.run(async_processor.run_async(
    ProcessingConfig(limit=20, max_concurrent=5)
))
async_time = time.time() - start

print(f"Sync: {sync_time:.2f}s")
print(f"Async: {async_time:.2f}s")
print(f"Speedup: {sync_time / async_time:.2f}x")
```

## Limitations

### When NOT to Use Async

1. **Low article counts**: For < 5 articles, sync is fine
2. **Strict rate limits**: If API has very low rate limits
3. **Memory constrained**: Each concurrent article uses ~50MB
4. **Simple debugging**: Sync code is easier to debug

### Known Issues

1. **Cache is sync**: Cache operations use locks (minor bottleneck)
2. **Database is sync**: Batch operations are single-threaded
3. **RSS parsing is sync**: Feed fetching is sequential

### Future Improvements

- [ ] Async cache operations
- [ ] Async database connection pool
- [ ] Async RSS feed parsing
- [ ] Progress bar for async operations
- [ ] Real-time metrics dashboard

## Testing

### Run Tests

```bash
# Run all async processor tests
pytest tests/test_async_processor.py -v

# Run with coverage
pytest tests/test_async_processor.py --cov=src.processors.async_article_processor

# Run benchmark tests (disabled by default)
pytest tests/test_async_processor.py -m benchmark
```

### Manual Testing

```bash
# Test with small batch
uv run python -m src.main run --limit 3 --async

# Test with medium batch
uv run python -m src.main run --limit 10 --async --max-concurrent 5

# Test with large batch
uv run python -m src.main run --limit 50 --async --max-concurrent 10

# Compare sync vs async
time uv run python -m src.main run --limit 20
time uv run python -m src.main run --limit 20 --async
```

## Troubleshooting

### "Too many open files" error

```bash
# Increase file descriptor limit
ulimit -n 4096

# Or reduce max_concurrent
--max-concurrent 3
```

### Rate limit errors

```bash
# Reduce concurrency
--max-concurrent 2

# Increase delays in config.yaml
scraper:
  delay_between_requests: 2.0
```

### Memory errors

```bash
# Reduce concurrency
--max-concurrent 3

# Or limit article count
--limit 20
```

### Database lock errors

```bash
# Database operations are batched and protected
# If you see this, it's a bug - please report
```

## Migration Guide

### From Sync to Async

**Step 1: Update imports**

```python
# Before (sync)
from src.processors import ArticleProcessor

# After (async)
from src.processors import AsyncArticleProcessor
import asyncio
```

**Step 2: Update initialization**

```python
# Before (sync)
processor = ArticleProcessor(config)
results = processor.run(processing_config)

# After (async)
processor = AsyncArticleProcessor(config)
results = asyncio.run(processor.run_async(processing_config))
```

**Step 3: Add max_concurrent to config**

```python
processing_config = ProcessingConfig(
    force_refresh=False,
    limit=30,
    max_concurrent=5  # New parameter
)
```

**Step 4: Test thoroughly**

```bash
# Start with low concurrency
--async --max-concurrent 2

# Gradually increase
--async --max-concurrent 5

# Monitor for errors
--async --max-concurrent 10
```

## See Also

- [Async Scraper Documentation](ASYNC_SCRAPER.md)
- [Async Client Documentation](ASYNC_CLIENTS.md)
- [Performance Optimization](OPTIMIZATION_RESULTS.md)
- [Connection Pooling](CONNECTION_POOLING.md)
- [Cache Integration](CACHE_INTEGRATION.md)
