# Monitoring and Observability

Comprehensive monitoring system for the RSS Analyzer with minimal performance overhead.

## Overview

The monitoring system provides real-time performance metrics, system health monitoring, and alert capabilities. It uses efficient data structures and thread-safe operations to minimize impact on application performance.

## Features

- **Performance Metrics Tracking**: Articles processed, API calls, scraping performance, database queries
- **System Health Monitoring**: Memory usage, CPU usage, disk space
- **Alert Thresholds**: Configurable alerts for resource usage and errors
- **Trend Tracking**: Historical performance data
- **Multi-format Export**: JSON and CSV export formats
- **Thread-Safe**: Safe for concurrent operations
- **Minimal Overhead**: Optimized for production use

## CLI Commands

### View Current Metrics

```bash
# Show metrics in terminal
python -m src.main metrics

# Export to JSON
python -m src.main metrics --format json --output logs/metrics.json

# Export to CSV
python -m src.main metrics --format csv --output logs/metrics.csv
```

### System Health Check

```bash
# Run comprehensive health check
python -m src.main health

# Save to custom location
python -m src.main health --output reports/health.json
```

### Performance Benchmark

```bash
# Run default benchmark (10 iterations)
python -m src.main benchmark

# Custom iterations
python -m src.main benchmark --iterations 20

# Save results
python -m src.main benchmark --output logs/benchmark.json
```

## Docker Usage

```bash
# View metrics
docker compose run rss-analyzer metrics

# Health check
docker compose run rss-analyzer health

# Run benchmark
docker compose run rss-analyzer benchmark --iterations 20

# Export metrics
docker compose run rss-analyzer metrics --format json --output /app/logs/metrics.json
```

## Metrics Tracked

### Processing Metrics

```python
articles_processed: int          # Total articles processed
total_processing_time: float     # Total time spent processing
average_processing_time: float   # Average time per article
```

### API Metrics

```python
api_calls_made: int              # Total API calls
api_call_time: float             # Total API time
api_tokens_used: int             # Total tokens consumed
api_cost_estimate: float         # Estimated cost in USD
avg_api_call_time: float         # Average time per call
```

### Scraping Metrics

```python
pages_scraped: int               # Successfully scraped pages
scraping_time: float             # Total scraping time
failed_scrapes: int              # Failed scraping attempts
followed_links: int              # Links followed in articles
avg_scraping_time: float         # Average time per page
```

### Database Metrics

```python
db_queries_executed: int         # Total queries
db_query_time: float             # Total query time
db_cache_hits: int               # Cache hits
db_cache_misses: int             # Cache misses
cache_hit_rate: float            # Hit rate percentage
avg_db_query_time: float         # Average query time
```

### System Metrics

```python
memory_usage_mb: float           # Current memory usage
cpu_usage_percent: float         # Current CPU usage
error_count: int                 # Total errors
warning_count: int               # Total warnings
```

## Programmatic Usage

### Context Managers

```python
from src.core import track_processing, track_api_call, track_scraping, track_db_query

# Track article processing
with track_processing():
    process_article(article)

# Track API calls with token estimation
with track_api_call(estimated_tokens=1500):
    result = api_client.analyze(content)

# Track web scraping
with track_scraping():
    content = scraper.scrape(url)

# Track database queries with cache status
with track_db_query(cached=False):
    results = db.query("SELECT * FROM articles")
```

### Direct Monitor Access

```python
from src.core import get_monitor

# Get global monitor instance
monitor = get_monitor()

# Get current metrics
metrics = monitor.get_metrics()
print(f"Articles processed: {metrics.articles_processed}")
print(f"Success rate: {metrics.success_rate:.1f}%")

# Get system health
health = monitor.get_system_health()
print(f"Status: {health.status}")
print(f"Issues: {health.issues}")

# Get dashboard data
dashboard = monitor.get_dashboard_data()
print(dashboard["overview"])

# Record events
monitor.record_error()
monitor.record_warning()
monitor.record_link_followed()

# Export metrics
json_data = monitor.export_metrics("json")
csv_data = monitor.export_metrics("csv")

# Save to file
monitor.save_metrics("logs/metrics.json", format="json")
```

## Alert Configuration

### Default Thresholds

```python
max_processing_time_seconds: 300.0    # 5 minutes
max_memory_usage_mb: 1024.0           # 1 GB
max_cpu_usage_percent: 90.0           # 90%
max_error_count: 10                   # 10 errors
min_cache_hit_rate: 50.0              # 50%
max_failed_scrapes: 5                 # 5 failures
```

### Custom Thresholds

```python
from src.core.monitoring import PerformanceMonitor, AlertThresholds

# Create custom thresholds
thresholds = AlertThresholds(
    max_memory_usage_mb=2048.0,
    max_error_count=20,
    min_cache_hit_rate=70.0
)

# Create monitor with custom thresholds
monitor = PerformanceMonitor(alert_thresholds=thresholds)

# Get alerts
alerts = monitor.get_alerts()
for alert in alerts:
    print(f"{alert['timestamp']}: {alert['message']}")
```

## Integration Examples

### Article Processor Integration

```python
from src.core import get_monitor

class ArticleProcessor:
    def __init__(self, config):
        self.monitor = get_monitor()
        # ... other initialization

    def process_article(self, article):
        with self.monitor.track_processing():
            try:
                # Scrape content
                with self.monitor.track_scraping():
                    content = self.scraper.scrape(article.url)

                # Analyze with AI
                with self.monitor.track_api_call(estimated_tokens=2000):
                    analysis = self.ai_client.analyze(content)

                # Store in database
                with self.monitor.track_db_query():
                    self.db.save_analysis(article.id, analysis)

            except Exception as e:
                self.monitor.record_error()
                raise
```

### Database Manager Integration

```python
from src.core import get_monitor

class DatabaseManager:
    def __init__(self, db_path):
        self.monitor = get_monitor()
        # ... other initialization

    def get_article(self, article_id):
        # Check cache first
        if article_id in self.cache:
            with self.monitor.track_db_query(cached=True):
                return self.cache[article_id]

        # Query database
        with self.monitor.track_db_query(cached=False):
            result = self._query_db(article_id)
            self.cache[article_id] = result
            return result
```

## Performance Impact

The monitoring system is designed for minimal overhead:

- **Context Managers**: ~0.001ms overhead per operation
- **Metric Updates**: Thread-safe atomic operations
- **System Metrics**: Updated on-demand, not continuously
- **Memory Footprint**: ~1-2 MB for metrics storage
- **CPU Impact**: <0.1% additional CPU usage

## Metrics Export Formats

### JSON Format

```json
{
  "articles_processed": 150,
  "total_processing_time": 450.5,
  "average_processing_time": 3.0,
  "api_calls_made": 150,
  "api_call_time": 300.2,
  "api_tokens_used": 225000,
  "api_cost_estimate": 2.25,
  "pages_scraped": 148,
  "scraping_time": 148.0,
  "failed_scrapes": 2,
  "db_queries_executed": 450,
  "db_query_time": 2.25,
  "cache_hit_rate": 65.5,
  "memory_usage_mb": 512.3,
  "cpu_usage_percent": 45.2,
  "error_count": 2,
  "warning_count": 5
}
```

### CSV Format

```csv
metric,value
articles_processed,150
total_processing_time,450.5
average_processing_time,3.0
api_calls_made,150
api_call_time,300.2
...
```

## Dashboard Data Format

```python
{
    "overview": {
        "status": "healthy",
        "articles_processed": 150,
        "success_rate": "98.7%",
        "total_time": "450.5s",
        "avg_processing_time": "3.00s"
    },
    "performance": {
        "api_calls": 150,
        "avg_api_time": "2.00s",
        "api_cost": "$2.2500",
        "pages_scraped": 148,
        "avg_scraping_time": "1.00s"
    },
    "database": {
        "queries_executed": 450,
        "avg_query_time": "0.0050s",
        "cache_hit_rate": "65.5%"
    },
    "system": {
        "memory_usage": "512.3 MB",
        "cpu_usage": "45.2%",
        "errors": 2,
        "warnings": 5
    },
    "health": {
        "status": "healthy",
        "issues": []
    },
    "alerts": []
}
```

## Best Practices

1. **Use Context Managers**: Prefer context managers for automatic tracking
2. **Monitor System Metrics Periodically**: Don't update on every operation
3. **Set Realistic Thresholds**: Based on your system capacity
4. **Export Metrics Regularly**: For long-term trend analysis
5. **Review Alerts**: Check alerts after each run
6. **Track API Costs**: Monitor token usage and costs
7. **Optimize Cache Hit Rate**: Aim for >70% cache hit rate

## Troubleshooting

### High Memory Usage

```bash
# Check system health
python -m src.main health

# Review metrics
python -m src.main metrics | grep memory
```

### High Error Count

```bash
# Get detailed metrics
python -m src.main metrics --format json --output logs/metrics.json

# Check alerts
python -m src.main health
```

### Low Cache Hit Rate

```bash
# Monitor database performance
python -m src.main benchmark --iterations 20

# Check cache metrics
python -m src.main metrics | grep cache
```

## Future Enhancements

- Prometheus metrics export
- Grafana dashboard integration
- Real-time metrics streaming
- Historical trend graphs
- Predictive alerts
- Cost optimization recommendations

## See Also

- [Database Documentation](../CLAUDE.md#database-schema)
- [Configuration Guide](../config/config.yaml)
- [Performance Optimization](../README.md#performance)
