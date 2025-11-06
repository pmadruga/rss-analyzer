# Monitoring System Quick Start

Get up and running with the RSS Analyzer monitoring system in 5 minutes.

## What You Get

- **Real-time Performance Metrics**: Track processing speed, API costs, scraping performance
- **System Health Monitoring**: Memory, CPU, disk space tracking
- **Alert System**: Get notified when thresholds are exceeded
- **Zero Configuration**: Works out of the box with sensible defaults
- **Minimal Overhead**: <0.1% performance impact

## Quick Commands

### View Current Metrics

```bash
# Show all metrics in terminal
python -m src.main metrics

# Export to JSON file
python -m src.main metrics --format json --output logs/metrics.json

# Export to CSV
python -m src.main metrics --format csv --output logs/metrics.csv
```

### System Health Check

```bash
# Run comprehensive health check
python -m src.main health

# Shows:
# - Database connectivity
# - RSS feed accessibility
# - AI provider status
# - System resources (memory, CPU, disk)
# - Active issues and alerts
```

### Performance Benchmark

```bash
# Run default benchmark (10 iterations)
python -m src.main benchmark

# Custom iterations
python -m src.main benchmark --iterations 20

# Save results to file
python -m src.main benchmark --output logs/benchmark.json
```

## Docker Usage

All commands work in Docker:

```bash
# View metrics
docker compose run rss-analyzer metrics

# Health check
docker compose run rss-analyzer health

# Benchmark
docker compose run rss-analyzer benchmark
```

## Key Metrics Explained

### Processing Metrics

- **articles_processed**: Total articles successfully processed
- **average_processing_time**: Average time per article (lower is better)
- **success_rate**: Percentage of successful processing (target: >95%)

### API Metrics

- **api_calls_made**: Total API calls to AI provider
- **api_cost_estimate**: Estimated cost in USD (based on token usage)
- **avg_api_call_time**: Average API response time

### Scraping Metrics

- **pages_scraped**: Successfully scraped web pages
- **failed_scrapes**: Failed scraping attempts
- **followed_links**: Links followed within articles

### Database Metrics

- **db_queries_executed**: Total database queries
- **cache_hit_rate**: Percentage of cached queries (target: >70%)
- **avg_db_query_time**: Average query time (lower is better)

### System Metrics

- **memory_usage_mb**: Current memory usage in MB
- **cpu_usage_percent**: Current CPU usage percentage
- **error_count**: Total errors encountered
- **warning_count**: Total warnings encountered

## What Good Looks Like

### Healthy System

```
📊 Performance Metrics
==================================================

🔄 Processing:
   Articles Processed: 150
   Success Rate: 98.7%
   Total Time: 450.5s
   Average Time: 3.00s

🧠 API Calls:
   Total Calls: 150
   Average Time: 2.00s
   Estimated Cost: $2.25

💾 Database:
   Cache Hit Rate: 75.2%
   Average Time: 0.0050s

💻 System:
   Memory Usage: 512.3 MB
   CPU Usage: 45.2%
   Errors: 0
```

### Warning Signs

- **Success Rate < 90%**: Check scraping errors
- **Cache Hit Rate < 50%**: Database performance issue
- **Average Processing Time > 5s**: Slow API or scraping
- **Error Count > 10**: System issues
- **Memory Usage > 1GB**: Possible memory leak

## Using in Code

### Basic Tracking

```python
from src.core import track_processing

# Track article processing
with track_processing():
    process_article(article)
```

### API Call Tracking

```python
from src.core import track_api_call

# Track API calls with cost estimation
with track_api_call(estimated_tokens=1500):
    analysis = ai_client.analyze(content)
```

### Get Current Metrics

```python
from src.core import get_monitor

monitor = get_monitor()
metrics = monitor.get_metrics()

print(f"Processed: {metrics.articles_processed}")
print(f"Success rate: {metrics.success_rate:.1f}%")
print(f"API cost: ${metrics.api_cost_estimate:.4f}")
```

## Common Use Cases

### 1. Monitor During Processing

```bash
# Terminal 1: Run processing
python -m src.main run --limit 100

# Terminal 2: Watch metrics
watch -n 5 'python -m src.main metrics'
```

### 2. Daily Health Check

```bash
# Run health check and save report
python -m src.main health --output logs/health_$(date +%Y%m%d).json

# Check for issues
python -m src.main metrics | grep -E "(Error|CPU|Memory)"
```

### 3. Performance Testing

```bash
# Baseline benchmark
python -m src.main benchmark --iterations 20 --output baseline.json

# After optimization
python -m src.main benchmark --iterations 20 --output optimized.json

# Compare results
diff baseline.json optimized.json
```

### 4. Cost Tracking

```bash
# Check API costs
python -m src.main metrics | grep "Estimated Cost"

# Export for accounting
python -m src.main metrics --format json --output costs_$(date +%Y%m).json
```

## Alert Thresholds

Default alert thresholds (configurable in code):

- **max_processing_time_seconds**: 300 (5 minutes)
- **max_memory_usage_mb**: 1024 (1 GB)
- **max_cpu_usage_percent**: 90%
- **max_error_count**: 10 errors
- **min_cache_hit_rate**: 50%
- **max_failed_scrapes**: 5 failures

## Performance Impact

The monitoring system is designed for production use:

- **Context managers**: ~0.001ms overhead per operation
- **Memory footprint**: 1-2 MB
- **CPU impact**: <0.1%
- **Thread-safe**: Safe for concurrent operations

## Troubleshooting

### "No metrics to display"

The monitoring system tracks metrics during runtime. Run the main processing pipeline first:

```bash
python -m src.main run --limit 5
python -m src.main metrics
```

### High memory usage

Check system health and review alerts:

```bash
python -m src.main health
```

Look for memory-related issues and consider:
- Reducing batch sizes
- Clearing cache more frequently
- Checking for memory leaks

### Low cache hit rate

Improve database performance:

```bash
# Check database metrics
python -m src.main metrics | grep -A 5 "Database:"

# Run benchmark to identify bottlenecks
python -m src.main benchmark
```

## Examples

See comprehensive examples in:

```bash
# Run all monitoring examples
python examples/monitoring_example.py
```

## Next Steps

- [Full Documentation](MONITORING.md)
- [Integration Examples](../examples/monitoring_example.py)
- [Test Suite](../tests/test_monitoring.py)
- [Configuration Guide](../config/config.yaml)

## Support

For issues or questions:
- Check [MONITORING.md](MONITORING.md) for detailed documentation
- Review examples in `examples/monitoring_example.py`
- See test cases in `tests/test_monitoring.py`
