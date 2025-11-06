# Cache Integration Guide

## Quick Start

### 1. Add Cache to WebScraper

```python
# src/scraper.py
from src.core.cache import ContentCache

class WebScraper:
    def __init__(self, config: dict):
        self.config = config
        self.cache = ContentCache()  # Add this line
        # ... rest of initialization

    def scrape_url(self, url: str) -> dict:
        """Scrape URL with caching."""
        # Generate cache key
        cache_key = ContentCache.generate_key(url, "scraped")

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Using cached content for {url}")
            return cached

        # Fetch if not cached
        logger.info(f"Fetching fresh content from {url}")
        content = self._fetch_content(url)  # Original scraping logic

        # Store in cache
        if content:
            self.cache.set(
                cache_key,
                content,
                ttl=ContentCache.TTL_SCRAPED_CONTENT,
                content_type="scraped"
            )

        return content
```

### 2. Add Cache to AI Clients

```python
# src/claude_client.py
import hashlib
from src.core.cache import ContentCache

class ClaudeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = ContentCache()  # Add this line
        # ... rest of initialization

    def analyze_content(self, content: str, prompt: str = None) -> dict:
        """Analyze content with caching."""
        # Generate cache key from content hash
        content_hash = hashlib.md5(
            f"{content}:{prompt}".encode()
        ).hexdigest()
        cache_key = f"analysis_{content_hash}"

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("Using cached analysis")
            return cached

        # Generate new analysis
        logger.info("Generating new AI analysis")
        analysis = self._call_api(content, prompt)  # Original API call

        # Cache successful results
        if analysis and 'error' not in analysis:
            self.cache.set(
                cache_key,
                analysis,
                ttl=ContentCache.TTL_API_RESPONSE,
                content_type="api"
            )

        return analysis
```

### 3. Add Cache to RSS Parser

```python
# src/rss_parser.py
from src.core.cache import ContentCache

class RSSParser:
    def __init__(self, feed_url: str):
        self.feed_url = feed_url
        self.cache = ContentCache()  # Add this line
        # ... rest of initialization

    def parse_feed(self) -> list:
        """Parse RSS feed with caching."""
        # Generate cache key
        cache_key = ContentCache.generate_key(self.feed_url, "rss_feed")

        # Check cache (shorter TTL for feeds)
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Using cached feed: {self.feed_url}")
            return cached

        # Parse fresh feed
        logger.info(f"Parsing fresh feed: {self.feed_url}")
        entries = self._parse_feed()  # Original parsing logic

        # Cache for 1 hour (feeds update frequently)
        if entries:
            self.cache.set(
                cache_key,
                entries,
                ttl=3600,  # 1 hour
                content_type="rss_feed"
            )

        return entries
```

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Cache settings
CACHE_DB_PATH=data/cache.db
CACHE_L1_SIZE_MB=256
CACHE_ENABLE=true
```

### Config File

Add to `config/config.yaml`:

```yaml
cache:
  enabled: true
  db_path: data/cache.db
  l1_size_mb: 256
  ttl:
    scraped_content: 604800  # 7 days
    api_response: 2592000    # 30 days
    rss_feed: 3600           # 1 hour
```

## Performance Monitoring

### Add to ArticleProcessor

```python
# src/main.py
class ArticleProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = ContentCache()
        # ... rest of initialization

    def process_articles(self, limit: int = None):
        """Process articles with cache monitoring."""
        # Get initial stats
        initial_stats = self.cache.get_stats()

        # Process articles
        results = self._process_articles_internal(limit)

        # Get final stats
        final_stats = self.cache.get_stats()

        # Log performance
        logger.info(
            f"Cache Performance: "
            f"{final_stats['hit_rate']:.1f}% hit rate, "
            f"{final_stats['total_hits']} hits, "
            f"{final_stats['total_misses']} misses"
        )

        # Display savings
        api_calls_saved = (
            final_stats['l1_hits'] + final_stats['l2_hits']
            - initial_stats['l1_hits'] - initial_stats['l2_hits']
        )
        logger.info(f"API calls saved: {api_calls_saved}")

        return results
```

## Maintenance Tasks

### Periodic Cleanup

Add to `src/main.py`:

```python
from apscheduler.schedulers.background import BackgroundScheduler

def setup_cache_maintenance(cache: ContentCache):
    """Setup periodic cache maintenance."""
    scheduler = BackgroundScheduler()

    # Cleanup expired entries every 6 hours
    scheduler.add_job(
        lambda: cache.cleanup_expired(),
        'interval',
        hours=6,
        id='cache_cleanup'
    )

    # Log statistics every hour
    scheduler.add_job(
        lambda: logger.info(f"Cache stats: {cache.get_stats()}"),
        'interval',
        hours=1,
        id='cache_stats'
    )

    scheduler.start()
    return scheduler
```

### CLI Commands

Add to `src/main.py`:

```python
@click.group()
def cli():
    """RSS Analyzer CLI."""
    pass

@cli.command()
def cache_stats():
    """Display cache statistics."""
    cache = ContentCache()
    stats = cache.get_stats()

    click.echo("Cache Statistics:")
    click.echo(f"  Hit Rate: {stats['hit_rate']}%")
    click.echo(f"  L1 Hits: {stats['l1_hits']}")
    click.echo(f"  L2 Hits: {stats['l2_hits']}")
    click.echo(f"  Misses: {stats['total_misses']}")
    click.echo(f"  L1 Size: {stats['l1_size_mb']} MB")
    click.echo(f"  L2 Size: {stats['l2_size_mb']} MB")
    click.echo(f"  L1 Entries: {stats['l1_entries']}")
    click.echo(f"  L2 Entries: {stats['l2_entries']}")

@cli.command()
def cache_clear():
    """Clear all cache entries."""
    cache = ContentCache()
    cache.clear()
    click.echo("Cache cleared successfully")

@cli.command()
def cache_cleanup():
    """Remove expired cache entries."""
    cache = ContentCache()
    removed = cache.cleanup_expired()
    click.echo(f"Removed {removed} expired entries")
```

### Docker Integration

Update `docker-compose.yml`:

```yaml
services:
  rss-analyzer:
    # ... existing config
    volumes:
      - ./data:/app/data  # Persist cache database
    environment:
      - CACHE_ENABLE=true
      - CACHE_DB_PATH=/app/data/cache.db
```

## Cost Savings Analysis

### Calculate Savings

```python
def calculate_cache_savings(cache: ContentCache) -> dict:
    """Calculate cost savings from caching."""
    stats = cache.get_stats()

    # Cost per API call (example: $0.01 per call)
    cost_per_call = 0.01

    # API calls prevented
    calls_prevented = stats['l1_hits'] + stats['l2_hits']

    # Money saved
    savings = calls_prevented * cost_per_call

    # Time saved (assume 500ms per API call)
    time_saved_seconds = calls_prevented * 0.5

    return {
        'calls_prevented': calls_prevented,
        'money_saved': savings,
        'time_saved_minutes': time_saved_seconds / 60,
        'hit_rate': stats['hit_rate']
    }
```

### Example Output

```
Cache Savings Report:
  API Calls Prevented: 1,250
  Money Saved: $12.50
  Time Saved: 10.4 minutes
  Hit Rate: 62.5%
```

## Testing Integration

### Unit Test Example

```python
# tests/test_integration_cache.py
import pytest
from src.scraper import WebScraper
from src.core.cache import ContentCache

def test_scraper_with_cache():
    """Test scraper uses cache."""
    scraper = WebScraper({})

    # First call should miss
    content1 = scraper.scrape_url("https://example.com")
    assert scraper.cache.stats.l1_misses > 0

    # Second call should hit
    content2 = scraper.scrape_url("https://example.com")
    assert scraper.cache.stats.l1_hits > 0
    assert content1 == content2
```

## Migration Guide

### Enable Caching in Existing Project

1. **Install cache module** (already done)
2. **Update imports** in your components
3. **Add cache initialization** to constructors
4. **Wrap fetch operations** with cache checks
5. **Test with small dataset** first
6. **Monitor performance** and adjust TTLs
7. **Deploy to production**

### Rollback Plan

If caching causes issues:

```python
# Disable cache with environment variable
CACHE_ENABLE=false

# Or bypass in code
class WebScraper:
    def __init__(self, config):
        self.cache_enabled = config.get('cache_enabled', True)

    def scrape_url(self, url):
        if self.cache_enabled:
            # Use cache
            pass
        else:
            # Direct fetch
            return self._fetch_content(url)
```

## Best Practices Summary

1. ✅ **Always check cache before expensive operations**
2. ✅ **Use appropriate TTLs for different content types**
3. ✅ **Generate consistent cache keys**
4. ✅ **Monitor cache statistics regularly**
5. ✅ **Run cleanup periodically**
6. ✅ **Test cache integration thoroughly**
7. ✅ **Document cache behavior for team**
8. ✅ **Plan for cache invalidation when needed**

## Troubleshooting

### Low Hit Rate

```python
# Check if keys are being generated consistently
cache_key1 = ContentCache.generate_key(url, "scraped")
cache_key2 = ContentCache.generate_key(url, "scraped")
assert cache_key1 == cache_key2

# Check TTL settings
print(f"Scraped TTL: {ContentCache.TTL_SCRAPED_CONTENT} seconds")
```

### High Memory Usage

```python
# Monitor L1 cache size
stats = cache.get_stats()
if stats['l1_size_mb'] > 200:  # Near 256MB limit
    logger.warning("L1 cache near capacity")
    # L1 will auto-evict, but consider reviewing data sizes
```

### Database Growth

```bash
# Check L2 database size
du -h data/cache.db

# Run cleanup
uv run python -c "
from src.core.cache import ContentCache
cache = ContentCache()
removed = cache.cleanup_expired()
print(f'Removed {removed} entries')
"

# Compact database
sqlite3 data/cache.db "VACUUM;"
```

## Next Steps

1. Monitor cache performance in production
2. Adjust TTLs based on actual usage patterns
3. Consider adding cache warming for common requests
4. Implement cache invalidation strategies
5. Add cache metrics to monitoring dashboard
