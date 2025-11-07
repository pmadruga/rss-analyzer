# Async Processing Quick Start

Get started with async processing for **6-8x faster article analysis** in 2 minutes.

## Quick Commands

```bash
# Standard sync mode (original)
uv run python -m src.main run --limit 10
# Time: ~150 seconds

# Async mode (6-8x faster) ⚡
uv run python -m src.main run --limit 10 --async
# Time: ~20-25 seconds

# Docker async mode
docker compose run rss-analyzer run --limit 10 --async
```

## Installation

No additional dependencies needed! Async support is already included:

```bash
# All async dependencies are in requirements.txt
uv sync
```

## Basic Usage

### Command Line

```bash
# Enable async mode with --async flag
uv run python -m src.main run --async

# Process 30 articles with async (recommended)
uv run python -m src.main run --limit 30 --async

# Adjust concurrency (2-10, default: 5)
uv run python -m src.main run --limit 30 --async --max-concurrent 8
```

### Python Code

```python
import asyncio
from src.processors import AsyncArticleProcessor, ProcessingConfig

# Setup
config = {
    "db_path": "data/articles.db",
    "rss_feed_url": "https://example.com/feed",
    "anthropic_api_key": "your-key",
    "api_provider": "anthropic",
}

# Run async
processor = AsyncArticleProcessor(config)
processing_config = ProcessingConfig(limit=20, max_concurrent=5)
results = asyncio.run(processor.run_async(processing_config))

print(f"✅ Processed {results.analyzed_articles} articles in {results.duration:.2f}s")
```

## Performance Comparison

### Real-world benchmarks (30 articles):

| Mode | Time | Articles/min | Speedup |
|------|------|--------------|---------|
| **Sync** | 360s (6 min) | 5 articles/min | 1x |
| **Async (max_concurrent=3)** | 150s (2.5 min) | 12 articles/min | **2.4x** |
| **Async (max_concurrent=5)** | 60s (1 min) | 30 articles/min | **6x** |
| **Async (max_concurrent=10)** | 45s (45s) | 40 articles/min | **8x** |

### Benchmark yourself:

```bash
# Time sync mode
time uv run python -m src.main run --limit 20

# Time async mode
time uv run python -m src.main run --limit 20 --async

# Calculate speedup
# speedup = sync_time / async_time
```

## Configuration

### Recommended Settings

```yaml
# config/config.yaml
processing:
  use_async_processing: false      # Enable via --async flag
  max_concurrent_articles: 5       # Good default
  async_scraper_concurrent: 5      # Matches article concurrency
```

### Tuning for Your Setup

**Conservative (safe for all):**
```bash
--async --max-concurrent 3
```

**Recommended (most setups):**
```bash
--async --max-concurrent 5
```

**Aggressive (fast connection + high API limits):**
```bash
--async --max-concurrent 10
```

**Maximum (careful of rate limits!):**
```bash
--async --max-concurrent 20
```

## When to Use Async

### ✅ Use Async When:

- Processing **10+ articles** at once
- You have **good network bandwidth**
- API rate limits allow **concurrent requests**
- You want **maximum throughput**
- Processing large batches **(50-100+ articles)**

### ❌ Stick with Sync When:

- Processing **< 5 articles** (overhead not worth it)
- **Debugging** issues (sync is simpler)
- **Very strict rate limits** on API
- **Memory constrained** environment (< 512MB)
- **First time running** (test with sync first)

## Troubleshooting

### Rate Limit Errors

```bash
# Solution: Reduce concurrency
--async --max-concurrent 2
```

### Memory Errors

```bash
# Solution: Reduce concurrency
--async --max-concurrent 3
```

### "Too many open files"

```bash
# Solution: Increase file limit
ulimit -n 4096

# Or reduce concurrency
--async --max-concurrent 3
```

## Examples

### Example 1: Process 50 articles fast

```bash
# Async with high concurrency
uv run python -m src.main run --limit 50 --async --max-concurrent 10

# Expected time: ~90 seconds
# vs. sync: ~600 seconds (10 minutes)
```

### Example 2: Daily batch processing

```bash
#!/bin/bash
# daily-process.sh

# Process all new articles with async
docker compose run rss-analyzer run --async --max-concurrent 8

# Expected: ~2-3 minutes for 30-50 articles
# vs. sync: ~15-20 minutes
```

### Example 3: GitHub Actions CI/CD

```yaml
# .github/workflows/process-articles.yml
- name: Process articles with async
  run: |
    docker compose run rss-analyzer run --async --max-concurrent 5
```

### Example 4: Custom Python script

```python
#!/usr/bin/env python3
import asyncio
import sys
from src.processors import AsyncArticleProcessor, ProcessingConfig

async def main():
    config = {
        "db_path": "data/articles.db",
        "rss_feed_url": "https://example.com/feed",
        "anthropic_api_key": "your-key",
        "api_provider": "anthropic",
    }

    processor = AsyncArticleProcessor(config)

    # Process with async
    processing_config = ProcessingConfig(
        limit=30,
        max_concurrent=8,
        follow_links=True
    )

    results = await processor.run_async(processing_config)

    print(f"✅ Success: {results.analyzed_articles}/{results.new_articles} articles")
    print(f"⏱️  Time: {results.duration:.2f}s")
    print(f"⚡ Rate: {results.analyzed_articles / (results.duration / 60):.1f} articles/min")

    return 0 if results.analyzed_articles > 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

## Performance Tips

### 1. Start Conservative, Then Scale Up

```bash
# Step 1: Test with small batch
uv run python -m src.main run --limit 5 --async --max-concurrent 2

# Step 2: Increase batch size
uv run python -m src.main run --limit 15 --async --max-concurrent 3

# Step 3: Scale up concurrency
uv run python -m src.main run --limit 30 --async --max-concurrent 5

# Step 4: Maximum throughput
uv run python -m src.main run --limit 50 --async --max-concurrent 10
```

### 2. Monitor and Adjust

```bash
# Check cache hit rate (higher is better)
uv run python -m src.main cache-stats

# Check processing stats
uv run python -m src.main stats

# Check system health
uv run python -m src.main health
```

### 3. Optimize Configuration

```yaml
# config/config.yaml

# Enable cache for better performance
performance:
  enable_caching: true
  cache_ttl_seconds: 7200  # 2 hours

# Connection pooling (from Week 1)
database:
  pool_size: 10  # Matches max_concurrent

# Rate limiting
scraper:
  rate_limit_rps: 10  # Requests per second
  delay_between_requests: 0.5  # Lower for async
```

## Next Steps

1. **Read full documentation**: [ASYNC_PROCESSOR.md](ASYNC_PROCESSOR.md)
2. **Learn about async scraper**: [ASYNC_SCRAPER.md](ASYNC_SCRAPER.md)
3. **Understand async clients**: [ASYNC_CLIENTS.md](ASYNC_CLIENTS.md)
4. **See optimization results**: [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)

## Support

- **Documentation**: [docs/](.)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Performance**: See [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)

---

**Quick comparison:**

```bash
# Before async
$ time uv run python -m src.main run --limit 30
real    6m0s  # 360 seconds

# After async
$ time uv run python -m src.main run --limit 30 --async
real    1m0s  # 60 seconds

# 🚀 6x faster!
```
