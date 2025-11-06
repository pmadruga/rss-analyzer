# Content Cache Usage Guide

## Overview

The RSS Analyzer includes a high-performance two-tier caching system that significantly reduces API calls and improves processing speed.

## Architecture

### Two-Tier Design

1. **L1 Cache (In-Memory)**
   - Fast LRU cache with 256MB limit
   - Stores hot data for immediate access
   - Automatically evicts least recently used entries
   - Lost on restart (ephemeral)

2. **L2 Cache (SQLite)**
   - Persistent storage with compression
   - Survives restarts
   - Indexed for fast lookups
   - Automatic expiration cleanup

### Cache Flow

```
Request → L1 Check → L2 Check → Fetch Data → Store in L1 & L2
   ↓         ↓          ↓
  Hit      Miss       Miss
   ↓         ↓          ↓
Return   Promote    Return
         to L1      New Data
```

## Basic Usage

### Initialization

```python
from src.core.cache import ContentCache

# Initialize cache with default database location
cache = ContentCache()

# Or specify custom database path
cache = ContentCache(db_path="custom/path/cache.db")
```

### Storing Data

```python
# Cache scraped content (7-day TTL by default)
cache.set(
    key="article_123",
    value={"title": "Example", "content": "..."},
    ttl=7 * 24 * 60 * 60,  # 7 days in seconds
    content_type="scraped"
)

# Cache API responses (30-day TTL by default)
cache.set(
    key="api_response_xyz",
    value={"analysis": "..."},
    ttl=30 * 24 * 60 * 60,  # 30 days in seconds
    content_type="api"
)

# Use predefined TTL constants
cache.set(
    key="test",
    value=data,
    ttl=ContentCache.TTL_SCRAPED_CONTENT,
    content_type="scraped"
)
```

### Retrieving Data

```python
# Get from cache
data = cache.get("article_123")

if data is not None:
    print("Cache hit!")
    process_data(data)
else:
    print("Cache miss - fetch from source")
    data = fetch_from_source()
    cache.set("article_123", data)
```

### Cache Key Generation

```python
# Generate consistent cache keys from URLs
key = ContentCache.generate_key(
    url="https://example.com/article",
    content_type="scraped"
)

cache.set(key, content)
data = cache.get(key)
```

### Deleting Entries

```python
# Delete specific entry
cache.delete("article_123")

# Clear all entries
cache.clear()

# Cleanup expired entries
removed_count = cache.cleanup_expired()
print(f"Removed {removed_count} expired entries")
```

### Monitoring Statistics

```python
# Get comprehensive cache statistics
stats = cache.get_stats()

print(f"Hit Rate: {stats['hit_rate']}%")
print(f"L1 Hits: {stats['l1_hits']}")
print(f"L2 Hits: {stats['l2_hits']}")
print(f"Total Misses: {stats['total_misses']}")
print(f"L1 Size: {stats['l1_size_mb']} MB")
print(f"L2 Size: {stats['l2_size_mb']} MB")
print(f"L1 Entries: {stats['l1_entries']}")
print(f"L2 Entries: {stats['l2_entries']}")
```

## Integration Examples

### WebScraper Integration

```python
from src.scraper import WebScraper
from src.core.cache import ContentCache

class CachedWebScraper(WebScraper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = ContentCache()

    def scrape_url(self, url: str) -> dict:
        # Generate cache key
        cache_key = ContentCache.generate_key(url, "scraped")

        # Check cache first
        cached_content = self.cache.get(cache_key)
        if cached_content:
            logger.info(f"Cache hit for {url}")
            return cached_content

        # Fetch if not cached
        logger.info(f"Cache miss for {url} - scraping")
        content = super().scrape_url(url)

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

### AI Client Integration

```python
from src.claude_client import ClaudeClient
from src.core.cache import ContentCache
import hashlib

class CachedClaudeClient(ClaudeClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = ContentCache()

    def analyze_content(self, content: str) -> dict:
        # Generate cache key from content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"analysis_{content_hash}"

        # Check cache
        cached_analysis = self.cache.get(cache_key)
        if cached_analysis:
            logger.info("Using cached analysis")
            return cached_analysis

        # Generate new analysis
        logger.info("Generating new analysis")
        analysis = super().analyze_content(content)

        # Cache the result
        if analysis:
            self.cache.set(
                cache_key,
                analysis,
                ttl=ContentCache.TTL_API_RESPONSE,
                content_type="api"
            )

        return analysis
```

### RSS Parser Integration

```python
from src.rss_parser import RSSParser
from src.core.cache import ContentCache

class CachedRSSParser(RSSParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = ContentCache()

    def parse_feed(self, feed_url: str) -> list:
        # Generate cache key
        cache_key = ContentCache.generate_key(feed_url, "rss_feed")

        # Check cache
        cached_entries = self.cache.get(cache_key)
        if cached_entries:
            logger.info(f"Using cached feed entries for {feed_url}")
            return cached_entries

        # Parse feed
        logger.info(f"Parsing fresh feed: {feed_url}")
        entries = super().parse_feed(feed_url)

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

## Performance Considerations

### Cache Hit Rates

| Content Type | Expected Hit Rate | TTL |
|--------------|-------------------|-----|
| Scraped content | 60-80% | 7 days |
| API responses | 70-90% | 30 days |
| RSS feeds | 40-60% | 1 hour |

### Size Management

```python
# Monitor cache sizes
stats = cache.get_stats()

# L1 cache auto-evicts when full (256MB limit)
# L2 cache grows until cleanup

# Periodic cleanup recommended
if stats['l2_size_mb'] > 1000:  # 1GB threshold
    cache.cleanup_expired()
```

### Concurrency

The cache is thread-safe with internal locking:

```python
import threading

def worker(cache, item_id):
    key = f"item_{item_id}"
    data = cache.get(key)
    if not data:
        data = fetch_data(item_id)
        cache.set(key, data)
    process_data(data)

# Safe to use from multiple threads
threads = [
    threading.Thread(target=worker, args=(cache, i))
    for i in range(10)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Maintenance

### Scheduled Cleanup

```python
import schedule
import time

def cleanup_task():
    removed = cache.cleanup_expired()
    logger.info(f"Cleanup removed {removed} expired entries")

    stats = cache.get_stats()
    logger.info(f"Cache stats: {stats}")

# Run cleanup every 6 hours
schedule.every(6).hours.do(cleanup_task)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Database Maintenance

```bash
# Compact SQLite database
sqlite3 data/cache.db "VACUUM;"

# Check database size
du -h data/cache.db

# Analyze query performance
sqlite3 data/cache.db "ANALYZE;"
```

### Monitoring Queries

```sql
-- View cache statistics
SELECT
    content_type,
    COUNT(*) as entries,
    SUM(size_bytes) / 1024 / 1024 as total_mb,
    AVG(access_count) as avg_accesses
FROM cache
GROUP BY content_type;

-- Find most accessed entries
SELECT key, content_type, access_count, last_accessed
FROM cache
ORDER BY access_count DESC
LIMIT 10;

-- Check expiration distribution
SELECT
    DATE(expires_at) as expiry_date,
    COUNT(*) as expiring_entries
FROM cache
GROUP BY DATE(expires_at)
ORDER BY expiry_date;

-- Find large entries
SELECT key, content_type, size_bytes / 1024 / 1024 as size_mb
FROM cache
ORDER BY size_bytes DESC
LIMIT 10;
```

## Best Practices

### 1. Choose Appropriate TTLs

```python
# Short TTL for frequently changing data
cache.set(key, data, ttl=3600)  # 1 hour

# Long TTL for static content
cache.set(key, data, ttl=30 * 24 * 60 * 60)  # 30 days
```

### 2. Use Descriptive Keys

```python
# Good: Descriptive and predictable
key = ContentCache.generate_key(url, "scraped")

# Bad: Opaque and hard to debug
key = hashlib.md5(url.encode()).hexdigest()
```

### 3. Handle Cache Misses Gracefully

```python
data = cache.get(key)
if data is None:
    data = fetch_from_source()
    if data:
        cache.set(key, data)
return data
```

### 4. Monitor Cache Performance

```python
# Log cache statistics periodically
def log_cache_stats():
    stats = cache.get_stats()
    logger.info(
        f"Cache: {stats['hit_rate']:.1f}% hit rate, "
        f"{stats['l1_entries']} L1 entries, "
        f"{stats['l2_entries']} L2 entries"
    )
```

### 5. Regular Maintenance

```python
# Clean up expired entries daily
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(
    lambda: cache.cleanup_expired(),
    'interval',
    hours=24
)
scheduler.start()
```

## Troubleshooting

### High Cache Miss Rate

```python
stats = cache.get_stats()
if stats['hit_rate'] < 30:
    logger.warning("Low cache hit rate - check TTL settings")

    # Increase TTL if appropriate
    # Or check if keys are being generated consistently
```

### L1 Cache Not Being Used

```python
# L1 cache might be too small for your data
stats = cache.get_stats()
if stats['l1_entries'] == 0 and stats['l2_entries'] > 0:
    logger.warning("L1 cache not being used - entries might be too large")
```

### Database Growing Too Large

```python
# Regular cleanup
cache.cleanup_expired()

# Check for entries with very long TTLs
# Consider reducing TTL for less critical data
```

### Stale Data Issues

```python
# Force refresh by deleting cached entry
cache.delete(key)

# Or reduce TTL for frequently changing data
cache.set(key, data, ttl=1800)  # 30 minutes instead of days
```

## Performance Metrics

### Expected Performance

- **L1 Hit**: < 1ms
- **L2 Hit**: < 10ms
- **Cache Miss**: Network latency + processing time

### Benchmarks

```python
import time

def benchmark_cache():
    cache = ContentCache()
    data = {"test": "x" * 1000}  # 1KB data

    # Write performance
    start = time.time()
    for i in range(1000):
        cache.set(f"key_{i}", data)
    write_time = time.time() - start
    print(f"Write: {write_time:.2f}s for 1000 entries")

    # L1 read performance
    start = time.time()
    for i in range(1000):
        cache.get(f"key_{i}")
    l1_read_time = time.time() - start
    print(f"L1 Read: {l1_read_time:.2f}s for 1000 entries")

    # L2 read performance (clear L1 first)
    cache.l1.clear()
    start = time.time()
    for i in range(100):
        cache.get(f"key_{i}")
    l2_read_time = time.time() - start
    print(f"L2 Read: {l2_read_time:.2f}s for 100 entries")
```

## API Reference

See the main `cache.py` module for complete API documentation with type hints and docstrings.

Key classes:
- `ContentCache`: Main cache interface
- `CacheEntry`: Cache entry with metadata
- `CacheStats`: Statistics tracking
- `L1Cache`: In-memory LRU cache
- `L2Cache`: Persistent SQLite cache
