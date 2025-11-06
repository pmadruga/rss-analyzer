# Monitoring Guide for RSS Analyzer

## Overview

This guide covers the comprehensive monitoring system implemented in the RSS Analyzer, including performance metrics, health checks, alerting, and dashboard setup.

## Table of Contents

1. [Monitoring Architecture](#monitoring-architecture)
2. [Available Metrics](#available-metrics)
3. [API Health Monitoring](#api-health-monitoring)
4. [Performance Tracking](#performance-tracking)
5. [Database Monitoring](#database-monitoring)
6. [Cache Monitoring](#cache-monitoring)
7. [Alerting](#alerting)
8. [Dashboard Setup](#dashboard-setup)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Monitoring Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   Monitoring System                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────┐ │
│  │ API Health    │  │ Performance   │  │ Database   │ │
│  │ Monitor       │  │ Monitor       │  │ Monitor    │ │
│  └───────┬───────┘  └───────┬───────┘  └─────┬──────┘ │
│          │                  │                 │        │
│          └─────────┬────────┴────────┬────────┘        │
│                    │                 │                 │
│            ┌───────▼──────┐  ┌───────▼─────────┐      │
│            │ Metrics      │  │ Alert           │      │
│            │ Collector    │  │ Manager         │      │
│            └───────┬──────┘  └───────┬─────────┘      │
│                    │                 │                 │
│            ┌───────▼──────┐  ┌───────▼─────────┐      │
│            │ Time Series  │  │ Notification    │      │
│            │ Database     │  │ System          │      │
│            └──────────────┘  └─────────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Features

1. **Real-time Monitoring**: Live system health tracking
2. **Historical Data**: Time-series metrics storage
3. **Automated Alerts**: Threshold-based notifications
4. **Health Checks**: Periodic API availability tests
5. **Performance Tracking**: Operation timing and throughput
6. **Cost Analysis**: API usage and savings tracking

---

## Available Metrics

### System Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `system.cpu_usage` | Gauge | CPU utilization | Percentage |
| `system.memory_usage` | Gauge | Memory consumption | MB |
| `system.disk_usage` | Gauge | Disk space used | MB |
| `system.uptime` | Counter | System uptime | Seconds |

### Database Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `db.pool_size` | Gauge | Connection pool size | Connections |
| `db.active_connections` | Gauge | Active connections | Connections |
| `db.idle_connections` | Gauge | Idle connections | Connections |
| `db.query_time` | Histogram | Query execution time | Milliseconds |
| `db.operations_per_sec` | Rate | Database operations rate | Ops/sec |
| `db.connection_errors` | Counter | Connection failures | Count |

### Cache Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `cache.hit_rate` | Gauge | Cache hit rate | Percentage |
| `cache.l1_hits` | Counter | L1 cache hits | Count |
| `cache.l2_hits` | Counter | L2 cache hits | Count |
| `cache.misses` | Counter | Cache misses | Count |
| `cache.l1_size` | Gauge | L1 cache size | MB |
| `cache.l2_size` | Gauge | L2 cache size | MB |
| `cache.evictions` | Counter | Cache evictions | Count |

### API Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `api.requests_total` | Counter | Total API calls | Count |
| `api.requests_failed` | Counter | Failed API calls | Count |
| `api.response_time` | Histogram | API response time | Milliseconds |
| `api.cost_total` | Counter | Total API cost | USD |
| `api.cost_saved` | Counter | Cost saved by caching | USD |
| `api.rate_limits` | Counter | Rate limit hits | Count |

### Processing Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `processing.articles_total` | Counter | Total articles processed | Count |
| `processing.articles_success` | Counter | Successfully processed | Count |
| `processing.articles_failed` | Counter | Failed to process | Count |
| `processing.duration` | Histogram | Processing time per article | Seconds |
| `processing.queue_size` | Gauge | Pending articles | Count |

---

## API Health Monitoring

### Using the API Health Monitor

```python
from tools.api_health_monitor import APIHealthMonitor

# Initialize monitor
monitor = APIHealthMonitor()

# Run health check
report = await monitor.run_health_check()

# Print summary
monitor.print_summary(report)

# Save report
monitor.save_report(report, "logs/api_health.json")
```

### Example Health Report

```json
{
  "timestamp": "2025-10-12T16:30:00Z",
  "total_apis": 3,
  "working_apis": 3,
  "failed_apis": 0,
  "apis_without_keys": 0,
  "recommended_provider": "mistral",
  "results": [
    {
      "provider": "anthropic",
      "status": "success",
      "response_time_ms": 450,
      "model_used": "claude-3-5-sonnet-20241022",
      "credits_info": "Available"
    },
    {
      "provider": "mistral",
      "status": "success",
      "response_time_ms": 380,
      "model_used": "mistral-large-latest",
      "credits_info": "Available"
    },
    {
      "provider": "openai",
      "status": "success",
      "response_time_ms": 520,
      "model_used": "gpt-4",
      "credits_info": "Available"
    }
  ]
}
```

### Automated Health Checks

Set up periodic health checks:

```python
import schedule
import asyncio

def run_health_check():
    """Run health check and send alerts"""
    monitor = APIHealthMonitor()
    report = asyncio.run(monitor.run_health_check())

    # Check for issues
    if report.failed_apis > 0:
        send_alert(f"⚠️ {report.failed_apis} APIs failing!")

    # Save report
    monitor.save_report(report)

# Schedule every 5 minutes
schedule.every(5).minutes.do(run_health_check)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Performance Tracking

### Performance Monitor Class

```python
from src.core.monitoring import PerformanceMonitor, track_processing

# Initialize monitor
monitor = PerformanceMonitor()

# Track operation
with track_processing("article_processing", monitor):
    # Your processing code here
    result = process_article(article)

# Get metrics
metrics = monitor.get_metrics()
print(f"Avg processing time: {metrics.avg_duration}ms")
```

### Custom Metrics Tracking

```python
from src.core.monitoring import get_monitor

def process_with_metrics(article):
    """Process article with metrics tracking"""
    monitor = get_monitor()

    # Track scraping
    start = time.time()
    content = scrape_article(article.url)
    monitor.record_scraping(time.time() - start)

    # Track API call
    start = time.time()
    analysis = api_client.analyze(content)
    monitor.record_api_call(time.time() - start)

    # Track database operation
    start = time.time()
    db.store_analysis(article.id, analysis)
    monitor.record_db_query(time.time() - start)

    return analysis
```

### Performance Metrics Example

```python
from src.core.monitoring import MonitoringManager

# Get monitoring manager
monitoring = MonitoringManager()

# Get performance report
report = monitoring.get_performance_report()

print(f"""
Performance Report:
==================
Processing:
  - Total articles: {report['processing']['total']}
  - Success rate: {report['processing']['success_rate']}%
  - Avg duration: {report['processing']['avg_duration']}s

API Calls:
  - Total calls: {report['api']['total_calls']}
  - Failed calls: {report['api']['failed_calls']}
  - Avg response time: {report['api']['avg_response_time']}ms
  - Cost saved: ${report['api']['cost_saved']}

Database:
  - Queries: {report['database']['total_queries']}
  - Avg query time: {report['database']['avg_query_time']}ms
  - Pool utilization: {report['database']['pool_utilization']}%

Cache:
  - Hit rate: {report['cache']['hit_rate']}%
  - Cost savings: ${report['cache']['cost_savings']}
""")
```

---

## Database Monitoring

### Connection Pool Statistics

```python
from src.core.database import DatabaseManager

# Get pool stats
db = DatabaseManager()
stats = db.get_pool_stats()

print(f"""
Database Pool Statistics:
========================
Pool Size: {stats['pool_size']}
Active Connections: {stats['active_connections']}
Idle Connections: {stats['idle_connections']}
Total Created: {stats['total_connections_created']}
Status: {'Healthy' if not stats['closed'] else 'Closed'}

Utilization: {(stats['active_connections'] / stats['pool_size']) * 100:.1f}%
""")
```

### Query Performance Monitoring

```python
from src.core.monitoring import track_db_query

def monitored_query():
    """Database query with monitoring"""
    with track_db_query("select_articles"):
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM articles
                WHERE processed_date > datetime('now', '-7 days')
            """)
            return cursor.fetchall()
```

### Database Health Checks

```python
def check_database_health():
    """Check database health"""
    db = DatabaseManager()

    # Check connection
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
        connection_status = "✅ Healthy"
    except Exception as e:
        connection_status = f"❌ Failed: {e}"

    # Check pool stats
    stats = db.get_pool_stats()
    pool_status = "✅ Healthy" if stats['idle_connections'] > 0 else "⚠️ Full"

    # Check database size
    size_mb = os.path.getsize("data/articles.db") / (1024 * 1024)
    size_status = "✅ Normal" if size_mb < 1000 else "⚠️ Large"

    return {
        'connection': connection_status,
        'pool': pool_status,
        'size_mb': size_mb,
        'size_status': size_status
    }
```

---

## Cache Monitoring

### Cache Statistics Dashboard

```python
from src.core.cache import ContentCache

def display_cache_stats():
    """Display cache statistics"""
    cache = ContentCache()
    stats = cache.get_stats()

    print(f"""
Cache Statistics:
================
Performance:
  - Hit Rate: {stats['hit_rate']}%
  - L1 Hits: {stats['l1_hits']:,}
  - L2 Hits: {stats['l2_hits']:,}
  - Misses: {stats['total_misses']:,}

Size:
  - L1 Entries: {stats['l1_entries']:,}
  - L2 Entries: {stats['l2_entries']:,}
  - L1 Size: {stats['l1_size_mb']:.1f} MB
  - L2 Size: {stats['l2_size_mb']:.1f} MB

Efficiency:
  - Memory Usage: {(stats['l1_size_mb'] / 256) * 100:.1f}%
  - API Calls Saved: {stats['l1_hits'] + stats['l2_hits']:,}
  - Cost Saved: ${(stats['l1_hits'] + stats['l2_hits']) * 0.01:.2f}
    """)
```

### Cache Performance Monitoring

```python
from src.core.monitoring import get_monitor

def process_with_cache_metrics(url: str):
    """Process with cache metrics"""
    monitor = get_monitor()
    cache = ContentCache()

    # Check cache
    cache_key = ContentCache.generate_key(url, "scraped")
    start = time.time()
    cached = cache.get(cache_key)
    lookup_time = (time.time() - start) * 1000

    if cached:
        monitor.record_cache_hit(lookup_time)
        return cached
    else:
        monitor.record_cache_miss(lookup_time)
        # Fetch and cache
        data = fetch_data(url)
        cache.set(cache_key, data)
        return data
```

### Real-Time Cache Monitoring

```python
import time
from threading import Thread

def monitor_cache_real_time(interval: int = 60):
    """Monitor cache in real-time"""
    cache = ContentCache()

    def monitor_loop():
        while True:
            stats = cache.get_stats()

            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"Hit Rate: {stats['hit_rate']}% | "
                  f"L1: {stats['l1_entries']} | "
                  f"L2: {stats['l2_entries']} | "
                  f"Size: {stats['l1_size_mb']:.0f}MB",
                  end='', flush=True)

            time.sleep(interval)

    thread = Thread(target=monitor_loop, daemon=True)
    thread.start()
```

---

## Alerting

### Alert Configuration

```python
from src.core.monitoring import AlertThresholds, MonitoringManager

# Configure alert thresholds
thresholds = AlertThresholds(
    # API thresholds
    api_error_rate=5.0,          # 5% error rate
    api_response_time=1000.0,     # 1 second
    api_cost_per_hour=10.0,       # $10/hour

    # Database thresholds
    db_query_time=100.0,          # 100ms
    db_pool_utilization=80.0,     # 80% pool usage
    db_connection_errors=5,       # 5 errors/hour

    # Cache thresholds
    cache_hit_rate=50.0,          # 50% minimum
    cache_memory_usage=80.0,      # 80% of limit

    # System thresholds
    memory_usage=80.0,            # 80% RAM
    disk_usage=90.0,              # 90% disk
    cpu_usage=90.0                # 90% CPU
)

# Initialize monitoring with alerts
monitoring = MonitoringManager(thresholds=thresholds)
```

### Custom Alert Handlers

```python
def send_email_alert(alert: dict):
    """Send alert via email"""
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg['Subject'] = f"RSS Analyzer Alert: {alert['title']}"
    msg['From'] = "alerts@example.com"
    msg['To'] = "admin@example.com"
    msg.set_content(alert['message'])

    with smtplib.SMTP('smtp.example.com', 587) as smtp:
        smtp.starttls()
        smtp.login("user", "password")
        smtp.send_message(msg)

def send_slack_alert(alert: dict):
    """Send alert to Slack"""
    import requests

    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    payload = {
        "text": f"⚠️ *{alert['title']}*\n{alert['message']}",
        "username": "RSS Analyzer",
        "icon_emoji": ":warning:"
    }

    requests.post(webhook_url, json=payload)

# Register alert handlers
monitoring.register_alert_handler(send_email_alert)
monitoring.register_alert_handler(send_slack_alert)
```

### Alert Examples

```python
from src.core.monitoring import MonitoringManager

def check_and_alert():
    """Check metrics and send alerts"""
    monitoring = MonitoringManager()

    # Check API health
    health = monitoring.get_system_health()

    if health.api_health < 50:
        monitoring.send_alert({
            'title': 'API Health Low',
            'message': f'API health at {health.api_health}%',
            'severity': 'warning'
        })

    # Check cache performance
    if health.cache_hit_rate < 50:
        monitoring.send_alert({
            'title': 'Cache Performance Low',
            'message': f'Hit rate at {health.cache_hit_rate}%',
            'severity': 'info'
        })

    # Check database
    if health.db_pool_utilization > 80:
        monitoring.send_alert({
            'title': 'Database Pool High',
            'message': f'Pool utilization at {health.db_pool_utilization}%',
            'severity': 'warning'
        })
```

---

## Dashboard Setup

### Grafana Dashboard

#### Install Prometheus Exporter

```python
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time

# Define metrics
api_response_time = Histogram('api_response_time_seconds',
                               'API response time in seconds')
cache_hit_rate = Gauge('cache_hit_rate_percent',
                       'Cache hit rate percentage')
db_query_time = Histogram('db_query_time_seconds',
                          'Database query time in seconds')
articles_processed = Counter('articles_processed_total',
                             'Total articles processed')

# Start Prometheus HTTP server
start_http_server(8000)

# Update metrics
def update_metrics():
    """Update Prometheus metrics"""
    cache = ContentCache()
    db = DatabaseManager()

    while True:
        # Update cache metrics
        stats = cache.get_stats()
        cache_hit_rate.set(stats['hit_rate'])

        # Update database metrics
        pool_stats = db.get_pool_stats()
        # ... record metrics ...

        time.sleep(60)  # Update every minute
```

#### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "RSS Analyzer Monitoring",
    "panels": [
      {
        "title": "API Response Time",
        "targets": [
          {
            "expr": "rate(api_response_time_seconds_sum[5m]) / rate(api_response_time_seconds_count[5m])"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "cache_hit_rate_percent"
          }
        ]
      },
      {
        "title": "Articles Processed",
        "targets": [
          {
            "expr": "rate(articles_processed_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Simple Web Dashboard

```python
from flask import Flask, jsonify, render_template
from src.core.monitoring import MonitoringManager

app = Flask(__name__)
monitoring = MonitoringManager()

@app.route('/metrics')
def get_metrics():
    """Return JSON metrics"""
    return jsonify(monitoring.get_all_metrics())

@app.route('/health')
def health_check():
    """Health check endpoint"""
    health = monitoring.get_system_health()
    return jsonify({
        'status': 'healthy' if health.overall_health > 80 else 'degraded',
        'metrics': {
            'api_health': health.api_health,
            'db_health': health.db_health,
            'cache_health': health.cache_health
        }
    })

@app.route('/dashboard')
def dashboard():
    """Render dashboard HTML"""
    metrics = monitoring.get_all_metrics()
    return render_template('dashboard.html', metrics=metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Performance Tuning

### Identify Bottlenecks

```python
from src.core.monitoring import MonitoringManager

def identify_bottlenecks():
    """Identify performance bottlenecks"""
    monitoring = MonitoringManager()
    metrics = monitoring.get_performance_report()

    bottlenecks = []

    # Check API performance
    if metrics['api']['avg_response_time'] > 1000:
        bottlenecks.append({
            'component': 'API',
            'issue': 'Slow response time',
            'value': f"{metrics['api']['avg_response_time']}ms",
            'recommendation': 'Consider caching or using faster model'
        })

    # Check database performance
    if metrics['database']['avg_query_time'] > 100:
        bottlenecks.append({
            'component': 'Database',
            'issue': 'Slow queries',
            'value': f"{metrics['database']['avg_query_time']}ms",
            'recommendation': 'Add indexes or increase pool size'
        })

    # Check cache performance
    if metrics['cache']['hit_rate'] < 60:
        bottlenecks.append({
            'component': 'Cache',
            'issue': 'Low hit rate',
            'value': f"{metrics['cache']['hit_rate']}%",
            'recommendation': 'Adjust TTLs or increase cache size'
        })

    return bottlenecks
```

### Optimization Recommendations

```python
def get_optimization_recommendations():
    """Get optimization recommendations"""
    monitoring = MonitoringManager()
    metrics = monitoring.get_all_metrics()

    recommendations = []

    # Database optimization
    pool_util = metrics['database']['pool_utilization']
    if pool_util > 80:
        recommendations.append({
            'priority': 'high',
            'component': 'database',
            'action': f'Increase pool size from {metrics["database"]["pool_size"]} to {metrics["database"]["pool_size"] + 5}',
            'expected_impact': '20-30% improvement in concurrent throughput'
        })

    # Cache optimization
    hit_rate = metrics['cache']['hit_rate']
    if hit_rate < 70:
        recommendations.append({
            'priority': 'medium',
            'component': 'cache',
            'action': 'Increase cache TTLs or size',
            'expected_impact': f'Potential {(70 - hit_rate) * 1.5:.0f}% cost reduction'
        })

    # API optimization
    if metrics['api']['error_rate'] > 5:
        recommendations.append({
            'priority': 'high',
            'component': 'api',
            'action': 'Implement retry logic and better error handling',
            'expected_impact': 'Reduce error rate by 50-70%'
        })

    return recommendations
```

---

## Troubleshooting

### Common Issues and Solutions

#### High API Costs

```python
# Check API usage
monitoring = MonitoringManager()
api_metrics = monitoring.get_api_metrics()

if api_metrics['cost_per_hour'] > 10:
    print("High API costs detected!")
    print(f"Current hit rate: {api_metrics['cache_hit_rate']}%")
    print("Recommendations:")
    print("1. Increase cache TTLs")
    print("2. Enable more aggressive caching")
    print("3. Review duplicate article detection")
```

#### Low Cache Hit Rate

```python
# Analyze cache performance
cache = ContentCache()
stats = cache.get_stats()

if stats['hit_rate'] < 60:
    print("Low cache hit rate!")
    print(f"L1 hits: {stats['l1_hits']}")
    print(f"L2 hits: {stats['l2_hits']}")
    print(f"Misses: {stats['total_misses']}")
    print("\nRecommendations:")
    print("1. Check if cache keys are consistent")
    print("2. Increase TTLs for stable content")
    print("3. Review cache size limits")
```

#### Database Contention

```python
# Check database pool
db = DatabaseManager()
stats = db.get_pool_stats()

if stats['active_connections'] >= stats['pool_size']:
    print("Database pool exhausted!")
    print(f"Utilization: 100%")
    print("\nRecommendations:")
    print("1. Increase pool size")
    print("2. Optimize slow queries")
    print("3. Review connection lifecycle")
```

---

## Conclusion

The monitoring system provides:
- **Comprehensive metrics** across all system components
- **Real-time health checks** for proactive issue detection
- **Automated alerting** for critical events
- **Performance insights** for optimization
- **Cost tracking** for budget management

Follow this guide to effectively monitor and optimize your RSS Analyzer deployment.
