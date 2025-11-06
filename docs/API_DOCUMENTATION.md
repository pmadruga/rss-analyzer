# API Documentation for RSS Analyzer

## Overview

This document provides comprehensive API documentation for the RSS Analyzer, including async methods, caching integration, performance monitoring, and usage examples.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Database API](#database-api)
3. [Cache API](#cache-api)
4. [Monitoring API](#monitoring-api)
5. [AI Clients API](#ai-clients-api)
6. [Async Methods](#async-methods)
7. [Configuration](#configuration)
8. [Examples](#examples)

---

## Core Classes

### ArticleProcessor

Main orchestrator for the RSS analysis pipeline.

```python
from src.processors.article_processor import ArticleProcessor

processor = ArticleProcessor(config: dict)
```

#### Methods

##### `process_articles(limit: int = None) -> List[dict]`

Process articles from RSS feed with full pipeline.

**Parameters:**
- `limit` (int, optional): Maximum number of articles to process

**Returns:**
- `List[dict]`: List of processed article results

**Example:**
```python
processor = ArticleProcessor(config)
results = processor.process_articles(limit=10)

for result in results:
    print(f"Processed: {result['title']}")
    print(f"Status: {result['status']}")
```

---

## Database API

### DatabaseManager

Thread-safe database manager with connection pooling.

```python
from src.core.database import DatabaseManager

db = DatabaseManager(
    db_path: str = "data/articles.db",
    pool_size: int = 5
)
```

#### Methods

##### `insert_article(title: str, url: str, content_hash: str, **kwargs) -> int`

Insert new article into database.

**Parameters:**
- `title` (str): Article title
- `url` (str): Article URL
- `content_hash` (str): MD5 hash of article content
- `**kwargs`: Additional article fields

**Returns:**
- `int`: Article ID

**Example:**
```python
article_id = db.insert_article(
    title="Example Article",
    url="https://example.com/article",
    content_hash="abc123def456",
    description="Article description",
    published_date="2025-10-12"
)
```

##### `get_connection() -> ContextManager[sqlite3.Connection]`

Get database connection from pool.

**Returns:**
- Context manager yielding `sqlite3.Connection`

**Example:**
```python
with db.get_connection() as conn:
    cursor = conn.execute("SELECT * FROM articles")
    articles = cursor.fetchall()
```

##### `get_pool_stats() -> dict`

Get connection pool statistics.

**Returns:**
- `dict`: Pool statistics including size, active, idle connections

**Example:**
```python
stats = db.get_pool_stats()
print(f"Pool size: {stats['pool_size']}")
print(f"Active: {stats['active_connections']}")
print(f"Idle: {stats['idle_connections']}")
print(f"Utilization: {stats['active_connections'] / stats['pool_size'] * 100:.1f}%")
```

##### `close_pool()`

Close all connections in pool.

**Example:**
```python
db.close_pool()
```

#### Performance Tips

- Use context managers for automatic connection return
- Monitor pool utilization with `get_pool_stats()`
- Increase `pool_size` for high concurrency (recommended: 2 × CPU cores)
- Keep connections open only as long as needed

---

## Cache API

### ContentCache

Two-tier caching system with L1 memory and L2 disk storage.

```python
from src.core.cache import ContentCache

cache = ContentCache(
    db_path: str = "data/cache.db",
    l1_max_size_mb: int = 256
)
```

#### Class Attributes

##### TTL Constants

```python
ContentCache.TTL_SCRAPED_CONTENT = 604800    # 7 days
ContentCache.TTL_API_RESPONSE = 2592000      # 30 days
ContentCache.TTL_RSS_FEED = 3600             # 1 hour
```

#### Methods

##### `get(key: str) -> Any | None`

Retrieve value from cache.

**Parameters:**
- `key` (str): Cache key

**Returns:**
- `Any | None`: Cached value or None if not found

**Example:**
```python
# Check cache before expensive operation
cached_content = cache.get("article_12345")
if cached_content:
    print("Cache hit!")
    return cached_content
else:
    print("Cache miss - fetching...")
    content = fetch_expensive_data()
    cache.set("article_12345", content)
    return content
```

##### `set(key: str, value: Any, ttl: int = None, content_type: str = "general")`

Store value in cache.

**Parameters:**
- `key` (str): Cache key
- `value` (Any): Value to cache (must be JSON-serializable)
- `ttl` (int, optional): Time-to-live in seconds
- `content_type` (str, optional): Content type for organization

**Example:**
```python
# Cache with default TTL
cache.set("article_content", article_data)

# Cache with custom TTL
cache.set("temporary_data", data, ttl=3600)  # 1 hour

# Cache with content type
cache.set(
    "api_response",
    analysis_result,
    ttl=ContentCache.TTL_API_RESPONSE,
    content_type="api"
)
```

##### `delete(key: str) -> bool`

Delete entry from cache.

**Parameters:**
- `key` (str): Cache key

**Returns:**
- `bool`: True if deleted, False if not found

**Example:**
```python
if cache.delete("outdated_article"):
    print("Cache entry deleted")
```

##### `clear()`

Clear all cache entries.

**Example:**
```python
cache.clear()
print("Cache cleared")
```

##### `get_stats() -> dict`

Get cache statistics.

**Returns:**
- `dict`: Statistics including hit rate, sizes, entry counts

**Example:**
```python
stats = cache.get_stats()
print(f"""
Cache Statistics:
- Hit Rate: {stats['hit_rate']}%
- L1 Hits: {stats['l1_hits']:,}
- L2 Hits: {stats['l2_hits']:,}
- Misses: {stats['total_misses']:,}
- L1 Size: {stats['l1_size_mb']:.1f} MB
- L2 Size: {stats['l2_size_mb']:.1f} MB
""")
```

##### `cleanup_expired() -> int`

Remove expired entries from L2 cache.

**Returns:**
- `int`: Number of entries removed

**Example:**
```python
removed = cache.cleanup_expired()
print(f"Removed {removed} expired entries")
```

#### Static Methods

##### `generate_key(identifier: str, content_type: str) -> str`

Generate consistent cache key.

**Parameters:**
- `identifier` (str): Unique identifier (URL, ID, etc.)
- `content_type` (str): Content type suffix

**Returns:**
- `str`: MD5-based cache key

**Example:**
```python
# Generate keys
url = "https://example.com/article/123"
scraped_key = ContentCache.generate_key(url, "scraped")
api_key = ContentCache.generate_key(url, "api")

# Use keys
cache.set(scraped_key, scraped_content)
cache.set(api_key, api_response)
```

#### Performance Tips

- Always check cache before expensive operations
- Use appropriate TTLs for different content types
- Monitor cache hit rate with `get_stats()`
- Run `cleanup_expired()` periodically
- Use consistent key generation with `generate_key()`

---

## Monitoring API

### PerformanceMonitor

Track system performance metrics.

```python
from src.core.monitoring import PerformanceMonitor, track_processing

monitor = PerformanceMonitor()
```

#### Methods

##### `record_processing(duration: float)`

Record article processing time.

**Parameters:**
- `duration` (float): Processing duration in seconds

**Example:**
```python
import time

start = time.time()
process_article(article)
duration = time.time() - start
monitor.record_processing(duration)
```

##### `record_api_call(duration: float, success: bool = True)`

Record API call metrics.

**Parameters:**
- `duration` (float): API call duration in seconds
- `success` (bool): Whether call succeeded

**Example:**
```python
start = time.time()
try:
    result = api_client.analyze(content)
    monitor.record_api_call(time.time() - start, success=True)
except Exception as e:
    monitor.record_api_call(time.time() - start, success=False)
    raise
```

##### `get_metrics() -> PerformanceMetrics`

Get current performance metrics.

**Returns:**
- `PerformanceMetrics`: Object with performance data

**Example:**
```python
metrics = monitor.get_metrics()
print(f"Total operations: {metrics.total_operations}")
print(f"Average duration: {metrics.avg_duration}s")
print(f"Success rate: {metrics.success_rate}%")
```

#### Context Managers

##### `track_processing(operation_name: str, monitor: PerformanceMonitor)`

Context manager for automatic metric tracking.

**Example:**
```python
from src.core.monitoring import track_processing

with track_processing("article_analysis", monitor):
    # Your code here - automatically timed
    result = analyze_article(article)
```

### MonitoringManager

Comprehensive monitoring management.

```python
from src.core.monitoring import MonitoringManager

monitoring = MonitoringManager(
    thresholds: AlertThresholds = None
)
```

#### Methods

##### `get_system_health() -> SystemHealth`

Get overall system health.

**Returns:**
- `SystemHealth`: Object with health metrics

**Example:**
```python
health = monitoring.get_system_health()
print(f"Overall health: {health.overall_health}%")
print(f"API health: {health.api_health}%")
print(f"DB health: {health.db_health}%")
print(f"Cache health: {health.cache_health}%")
```

##### `get_performance_report() -> dict`

Get detailed performance report.

**Returns:**
- `dict`: Comprehensive performance metrics

**Example:**
```python
report = monitoring.get_performance_report()
print(f"Processing: {report['processing']}")
print(f"API: {report['api']}")
print(f"Database: {report['database']}")
print(f"Cache: {report['cache']}")
```

### APIHealthMonitor

Monitor API provider health.

```python
from tools.api_health_monitor import APIHealthMonitor

monitor = APIHealthMonitor()
```

#### Async Methods

##### `async run_health_check() -> MonitoringReport`

Run comprehensive health check on all APIs.

**Returns:**
- `MonitoringReport`: Detailed health report

**Example:**
```python
import asyncio

async def check_apis():
    monitor = APIHealthMonitor()
    report = await monitor.run_health_check()

    print(f"Working APIs: {report.working_apis}/{report.total_apis}")
    print(f"Recommended: {report.recommended_provider}")

    for result in report.results:
        print(f"{result.provider}: {result.status}")

asyncio.run(check_apis())
```

---

## AI Clients API

### ClaudeClient

Anthropic Claude API client.

```python
from src.clients.claude import ClaudeClient

client = ClaudeClient(config: dict)
```

#### Methods

##### `analyze_content(content: str, prompt: str = None) -> dict`

Analyze content using Claude.

**Parameters:**
- `content` (str): Content to analyze
- `prompt` (str, optional): Custom analysis prompt

**Returns:**
- `dict`: Analysis result

**Example:**
```python
result = client.analyze_content(
    content="Article text here...",
    prompt="Explain this paper using the Feynman technique"
)

print(result['analysis'])
print(f"Confidence: {result['confidence']}")
```

##### `test_connection() -> bool`

Test API connectivity.

**Returns:**
- `bool`: True if connection successful

**Example:**
```python
if client.test_connection():
    print("API connection OK")
else:
    print("API connection failed")
```

---

## Async Methods

### Async API Clients

For async operations, use `asyncio.to_thread` with sync clients:

```python
import asyncio
from src.clients.claude import ClaudeClient

async def analyze_async(content: str) -> dict:
    """Analyze content asynchronously"""
    client = ClaudeClient(config)
    result = await asyncio.to_thread(
        client.analyze_content,
        content
    )
    return result

# Run async
result = asyncio.run(analyze_async("content"))
```

### Concurrent Processing

Process multiple articles concurrently:

```python
import asyncio

async def process_articles_async(articles: List[dict]) -> List[dict]:
    """Process articles concurrently"""
    tasks = [
        asyncio.to_thread(process_single_article, article)
        for article in articles
    ]
    results = await asyncio.gather(*tasks)
    return results

# Process 10 articles concurrently
articles = get_articles(limit=10)
results = asyncio.run(process_articles_async(articles))
```

---

## Configuration

### Environment Variables

```bash
# API Provider
API_PROVIDER=anthropic|mistral|openai

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
OPENAI_API_KEY=sk-...

# Database
DATABASE_PATH=data/articles.db
DB_POOL_SIZE=5

# Cache
CACHE_ENABLE=true
CACHE_DB_PATH=data/cache.db
CACHE_L1_SIZE_MB=256

# Monitoring
MONITORING_ENABLE=true
ALERT_THRESHOLD_ERROR_RATE=5.0
ALERT_THRESHOLD_RESPONSE_TIME=1000
```

### Config File

```yaml
# config/config.yaml

database:
  path: data/articles.db
  pool_size: 5

cache:
  enabled: true
  db_path: data/cache.db
  l1_size_mb: 256
  ttl:
    scraped_content: 604800  # 7 days
    api_response: 2592000    # 30 days
    rss_feed: 3600           # 1 hour

monitoring:
  enabled: true
  alert_thresholds:
    api_error_rate: 5.0
    api_response_time: 1000
    db_query_time: 100
    cache_hit_rate: 50

api:
  anthropic:
    model: claude-3-5-sonnet-20241022
    max_tokens: 4096
  mistral:
    model: mistral-large-latest
  openai:
    model: gpt-4
```

---

## Examples

### Complete Processing Pipeline

```python
from src.processors.article_processor import ArticleProcessor
from src.core.database import DatabaseManager
from src.core.cache import ContentCache
from src.core.monitoring import MonitoringManager

# Initialize components
db = DatabaseManager(pool_size=5)
cache = ContentCache(l1_max_size_mb=256)
monitoring = MonitoringManager()
processor = ArticleProcessor(config)

# Process articles with monitoring
with track_processing("pipeline", monitoring.get_monitor()):
    results = processor.process_articles(limit=10)

# Get statistics
cache_stats = cache.get_stats()
pool_stats = db.get_pool_stats()
health = monitoring.get_system_health()

print(f"""
Processing Complete:
- Articles processed: {len(results)}
- Cache hit rate: {cache_stats['hit_rate']}%
- Pool utilization: {pool_stats['active_connections']}/{pool_stats['pool_size']}
- System health: {health.overall_health}%
""")

# Cleanup
db.close_pool()
cache.cleanup_expired()
```

### Optimized Article Processing

```python
def process_article_optimized(article: dict) -> dict:
    """Process article with all optimizations"""
    cache = ContentCache()
    db = DatabaseManager()

    # Check cache first
    cache_key = ContentCache.generate_key(article['url'], "scraped")
    cached_content = cache.get(cache_key)

    if cached_content:
        print(f"Cache hit for {article['url']}")
        content = cached_content
    else:
        print(f"Cache miss - scraping {article['url']}")
        content = scrape_article(article['url'])
        cache.set(cache_key, content, ttl=ContentCache.TTL_SCRAPED_CONTENT)

    # Analyze with caching
    analysis_key = ContentCache.generate_key(article['url'], "api")
    cached_analysis = cache.get(analysis_key)

    if cached_analysis:
        print(f"Cached analysis for {article['url']}")
        analysis = cached_analysis
    else:
        print(f"Generating new analysis")
        analysis = api_client.analyze_content(content)
        cache.set(analysis_key, analysis, ttl=ContentCache.TTL_API_RESPONSE)

    # Store with pooled connection
    with db.get_connection() as conn:
        article_id = db.insert_article(
            title=article['title'],
            url=article['url'],
            content_hash=create_content_hash(content)
        )
        db.store_analysis(article_id, analysis)

    return {
        'id': article_id,
        'status': 'success',
        'analysis': analysis
    }
```

### Monitoring and Alerting

```python
from src.core.monitoring import MonitoringManager, AlertThresholds

# Configure alerts
thresholds = AlertThresholds(
    api_error_rate=5.0,
    api_response_time=1000,
    cache_hit_rate=50
)

monitoring = MonitoringManager(thresholds=thresholds)

# Custom alert handler
def send_alert(alert: dict):
    print(f"ALERT: {alert['title']}")
    print(f"Message: {alert['message']}")
    # Send to Slack, email, etc.

monitoring.register_alert_handler(send_alert)

# Monitor during processing
health = monitoring.get_system_health()
if health.overall_health < 80:
    print("System health degraded!")
    report = monitoring.get_performance_report()
    # Take action based on report
```

### Async Health Checks

```python
import asyncio
from tools.api_health_monitor import APIHealthMonitor

async def monitor_apis_continuously():
    """Monitor API health continuously"""
    monitor = APIHealthMonitor()

    while True:
        report = await monitor.run_health_check()

        print(f"[{report.timestamp}] API Health Check:")
        print(f"  Working: {report.working_apis}/{report.total_apis}")

        if report.failed_apis > 0:
            print("  WARNING: Some APIs failing!")
            for result in report.results:
                if result.status == "failed":
                    print(f"  - {result.provider}: {result.error_message}")

        # Check every 5 minutes
        await asyncio.sleep(300)

# Run continuous monitoring
asyncio.run(monitor_apis_continuously())
```

---

## Performance Best Practices

### Database

1. **Use connection pooling**: Set appropriate `pool_size`
2. **Monitor utilization**: Check `get_pool_stats()` regularly
3. **Use context managers**: Ensure connections are returned
4. **Batch operations**: Group related queries together

### Cache

1. **Check before fetch**: Always query cache before expensive operations
2. **Use appropriate TTLs**: Match TTL to content update frequency
3. **Monitor hit rate**: Aim for 60-80% hit rate
4. **Cleanup regularly**: Run `cleanup_expired()` hourly
5. **Generate consistent keys**: Use `ContentCache.generate_key()`

### Monitoring

1. **Track all operations**: Use context managers for automatic tracking
2. **Set realistic thresholds**: Based on baseline measurements
3. **React to alerts**: Don't ignore warning signs
4. **Review reports**: Analyze performance trends weekly

### API Usage

1. **Cache aggressively**: 30-day TTL for stable analyses
2. **Implement retry logic**: Handle transient failures
3. **Monitor health**: Run async health checks periodically
4. **Use recommended provider**: Switch based on performance

---

## Troubleshooting

### Common Issues

#### Low Cache Hit Rate

```python
# Check cache configuration
cache = ContentCache()
stats = cache.get_stats()

if stats['hit_rate'] < 60:
    print("Investigating low hit rate...")
    print(f"L1 hits: {stats['l1_hits']}")
    print(f"L2 hits: {stats['l2_hits']}")
    print(f"Misses: {stats['total_misses']}")

    # Potential fixes:
    # 1. Increase TTLs
    # 2. Check key generation consistency
    # 3. Verify cache is enabled
```

#### Database Pool Exhaustion

```python
# Check pool status
db = DatabaseManager()
stats = db.get_pool_stats()

if stats['idle_connections'] == 0:
    print("Pool exhausted!")
    print(f"Active: {stats['active_connections']}")
    print(f"Pool size: {stats['pool_size']}")

    # Solutions:
    # 1. Increase pool_size
    # 2. Audit long-running queries
    # 3. Ensure connections are released
```

#### High API Costs

```python
# Check API usage
monitoring = MonitoringManager()
report = monitoring.get_performance_report()

api_calls = report['api']['total_calls']
cache_hits = report['cache']['l1_hits'] + report['cache']['l2_hits']
hit_rate = (cache_hits / (cache_hits + api_calls)) * 100

print(f"Cache hit rate: {hit_rate}%")
print(f"API calls: {api_calls}")
print(f"Cost saved: ${report['cache']['cost_savings']}")

if hit_rate < 60:
    print("Consider increasing cache TTLs")
```

---

## API Reference Summary

| Component | Key Methods | Purpose |
|-----------|-------------|---------|
| `DatabaseManager` | `insert_article()`, `get_connection()`, `get_pool_stats()` | Database operations with pooling |
| `ContentCache` | `get()`, `set()`, `get_stats()`, `cleanup_expired()` | Two-tier caching |
| `PerformanceMonitor` | `record_processing()`, `get_metrics()` | Performance tracking |
| `MonitoringManager` | `get_system_health()`, `get_performance_report()` | System monitoring |
| `APIHealthMonitor` | `run_health_check()` | API health checks |
| `ClaudeClient` | `analyze_content()`, `test_connection()` | AI analysis |

---

For more information, see the comprehensive guides:
- [Async Programming Guide](ASYNC_GUIDE.md)
- [Monitoring Guide](MONITORING_GUIDE.md)
- [Cache Integration Guide](CACHE_INTEGRATION.md)
- [Optimization Results](OPTIMIZATION_RESULTS.md)
