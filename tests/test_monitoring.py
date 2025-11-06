"""
Tests for monitoring and observability module
"""

import time
import pytest
from src.core.monitoring import (
    PerformanceMonitor,
    PerformanceMetrics,
    AlertThresholds,
    get_monitor,
)


def test_performance_metrics_creation():
    """Test creating performance metrics"""
    metrics = PerformanceMetrics()
    assert metrics.articles_processed == 0
    assert metrics.api_calls_made == 0
    assert metrics.cache_hit_rate == 0.0


def test_cache_hit_rate_calculation():
    """Test cache hit rate calculation"""
    metrics = PerformanceMetrics(
        db_cache_hits=80,
        db_cache_misses=20
    )
    assert metrics.cache_hit_rate == 80.0


def test_success_rate_calculation():
    """Test success rate calculation"""
    metrics = PerformanceMetrics(
        articles_processed=90,
        failed_scrapes=10
    )
    assert metrics.success_rate == 90.0


def test_track_processing():
    """Test tracking processing time"""
    monitor = PerformanceMonitor()

    with monitor.track_processing():
        time.sleep(0.1)

    metrics = monitor.get_metrics()
    assert metrics.articles_processed == 1
    assert metrics.total_processing_time >= 0.1
    assert metrics.average_processing_time >= 0.1


def test_track_api_call():
    """Test tracking API calls"""
    monitor = PerformanceMonitor()

    with monitor.track_api_call(estimated_tokens=1000):
        time.sleep(0.05)

    metrics = monitor.get_metrics()
    assert metrics.api_calls_made == 1
    assert metrics.api_call_time >= 0.05
    assert metrics.api_tokens_used == 1000
    assert metrics.api_cost_estimate > 0


def test_track_scraping_success():
    """Test tracking successful scraping"""
    monitor = PerformanceMonitor()

    with monitor.track_scraping():
        time.sleep(0.05)

    metrics = monitor.get_metrics()
    assert metrics.pages_scraped == 1
    assert metrics.scraping_time >= 0.05
    assert metrics.failed_scrapes == 0


def test_track_scraping_failure():
    """Test tracking failed scraping"""
    monitor = PerformanceMonitor()

    try:
        with monitor.track_scraping():
            raise Exception("Scraping failed")
    except Exception:
        pass

    metrics = monitor.get_metrics()
    assert metrics.pages_scraped == 0
    assert metrics.failed_scrapes == 1


def test_track_db_query():
    """Test tracking database queries"""
    monitor = PerformanceMonitor()

    # Track cached query
    with monitor.track_db_query(cached=True):
        time.sleep(0.01)

    # Track uncached query
    with monitor.track_db_query(cached=False):
        time.sleep(0.01)

    metrics = monitor.get_metrics()
    assert metrics.db_queries_executed == 2
    assert metrics.db_cache_hits == 1
    assert metrics.db_cache_misses == 1
    assert metrics.cache_hit_rate == 50.0


def test_record_error():
    """Test recording errors"""
    monitor = PerformanceMonitor()

    monitor.record_error()
    monitor.record_error()

    metrics = monitor.get_metrics()
    assert metrics.error_count == 2


def test_record_warning():
    """Test recording warnings"""
    monitor = PerformanceMonitor()

    monitor.record_warning()

    metrics = monitor.get_metrics()
    assert metrics.warning_count == 1


def test_record_link_followed():
    """Test recording followed links"""
    monitor = PerformanceMonitor()

    monitor.record_link_followed()
    monitor.record_link_followed()

    metrics = monitor.get_metrics()
    assert metrics.followed_links == 2


def test_alert_thresholds():
    """Test alert threshold checking"""
    thresholds = AlertThresholds(
        max_error_count=5,
        max_failed_scrapes=3
    )

    monitor = PerformanceMonitor(alert_thresholds=thresholds)

    # Trigger alerts
    for _ in range(6):
        monitor.record_error()

    alerts = monitor.get_alerts()
    assert len(alerts) > 0


def test_system_health():
    """Test system health check"""
    monitor = PerformanceMonitor()
    health = monitor.get_system_health()

    assert health.status in ["healthy", "degraded", "unhealthy", "unknown"]
    assert health.memory_available_mb >= 0
    assert health.cpu_count >= 0
    assert isinstance(health.issues, list)


def test_metrics_export_json():
    """Test exporting metrics as JSON"""
    monitor = PerformanceMonitor()
    monitor.record_error()

    json_output = monitor.export_metrics("json")
    assert "error_count" in json_output
    assert isinstance(json_output, str)


def test_metrics_export_csv():
    """Test exporting metrics as CSV"""
    monitor = PerformanceMonitor()
    monitor.record_error()

    csv_output = monitor.export_metrics("csv")
    assert "error_count" in csv_output
    assert "metric,value" in csv_output


def test_reset_metrics():
    """Test resetting metrics"""
    monitor = PerformanceMonitor()

    # Add some metrics
    monitor.record_error()
    with monitor.track_processing():
        pass

    # Reset
    monitor.reset_metrics()

    metrics = monitor.get_metrics()
    assert metrics.error_count == 0
    assert metrics.articles_processed == 0


def test_dashboard_data():
    """Test getting dashboard data"""
    monitor = PerformanceMonitor()

    with monitor.track_processing():
        time.sleep(0.1)

    monitor.record_error()

    dashboard = monitor.get_dashboard_data()

    assert "overview" in dashboard
    assert "performance" in dashboard
    assert "database" in dashboard
    assert "system" in dashboard
    assert "health" in dashboard
    assert "alerts" in dashboard


def test_global_monitor_singleton():
    """Test global monitor is singleton"""
    monitor1 = get_monitor()
    monitor2 = get_monitor()

    assert monitor1 is monitor2


def test_concurrent_tracking():
    """Test thread-safe concurrent tracking"""
    import threading

    monitor = PerformanceMonitor()

    def process_article():
        with monitor.track_processing():
            time.sleep(0.01)
        monitor.record_error()

    threads = [threading.Thread(target=process_article) for _ in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    metrics = monitor.get_metrics()
    assert metrics.articles_processed == 10
    assert metrics.error_count == 10


def test_metrics_to_dict():
    """Test converting metrics to dictionary"""
    metrics = PerformanceMetrics(
        articles_processed=5,
        api_calls_made=3,
        error_count=1
    )

    data = metrics.to_dict()

    assert isinstance(data, dict)
    assert data["articles_processed"] == 5
    assert data["api_calls_made"] == 3
    assert data["error_count"] == 1
    assert "cache_hit_rate" in data
    assert "success_rate" in data


def test_update_system_metrics():
    """Test updating system resource metrics"""
    monitor = PerformanceMonitor()

    monitor.update_system_metrics()

    metrics = monitor.get_metrics()
    assert metrics.memory_usage_mb > 0
    assert metrics.cpu_usage_percent >= 0


def test_average_calculations():
    """Test average time calculations"""
    monitor = PerformanceMonitor()

    # Process multiple items
    for _ in range(5):
        with monitor.track_api_call(estimated_tokens=100):
            time.sleep(0.02)

    metrics = monitor.get_metrics()

    assert metrics.api_calls_made == 5
    assert metrics.avg_api_call_time > 0
    assert metrics.api_tokens_used == 500
