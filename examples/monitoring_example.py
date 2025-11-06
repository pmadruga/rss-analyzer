"""
Example: Using the monitoring system

This example demonstrates how to use the monitoring and observability features.
"""

import time
from src.core.monitoring import (
    get_monitor,
    track_processing,
    track_api_call,
    track_scraping,
    track_db_query,
    AlertThresholds,
    PerformanceMonitor,
)


def example_basic_tracking():
    """Example: Basic metric tracking"""
    print("=" * 60)
    print("Example 1: Basic Metric Tracking")
    print("=" * 60)

    monitor = get_monitor()

    # Simulate processing articles
    for i in range(5):
        with track_processing():
            # Simulate work
            time.sleep(0.1)
            print(f"Processed article {i + 1}")

    # Get and display metrics
    metrics = monitor.get_metrics()
    print(f"\nMetrics:")
    print(f"  Articles processed: {metrics.articles_processed}")
    print(f"  Total time: {metrics.total_processing_time:.2f}s")
    print(f"  Average time: {metrics.average_processing_time:.2f}s")


def example_api_tracking():
    """Example: API call tracking with cost estimation"""
    print("\n" + "=" * 60)
    print("Example 2: API Call Tracking")
    print("=" * 60)

    monitor = get_monitor()
    monitor.reset_metrics()  # Reset for clean example

    # Simulate API calls
    for i in range(3):
        with track_api_call(estimated_tokens=1500):
            time.sleep(0.2)
            print(f"API call {i + 1} completed")

    metrics = monitor.get_metrics()
    print(f"\nAPI Metrics:")
    print(f"  Total calls: {metrics.api_calls_made}")
    print(f"  Total tokens: {metrics.api_tokens_used}")
    print(f"  Estimated cost: ${metrics.api_cost_estimate:.4f}")
    print(f"  Average time: {metrics.avg_api_call_time:.2f}s")


def example_scraping_tracking():
    """Example: Web scraping tracking with failures"""
    print("\n" + "=" * 60)
    print("Example 3: Web Scraping Tracking")
    print("=" * 60)

    monitor = get_monitor()
    monitor.reset_metrics()

    # Simulate successful scrapes
    for i in range(4):
        with track_scraping():
            time.sleep(0.1)
            print(f"Scraped page {i + 1}")

    # Simulate failed scrape
    try:
        with track_scraping():
            print("Attempting problematic scrape...")
            raise Exception("Connection timeout")
    except Exception as e:
        print(f"Scraping failed: {e}")

    metrics = monitor.get_metrics()
    print(f"\nScraping Metrics:")
    print(f"  Successful scrapes: {metrics.pages_scraped}")
    print(f"  Failed scrapes: {metrics.failed_scrapes}")
    print(f"  Success rate: {metrics.success_rate:.1f}%")
    print(f"  Average time: {metrics.avg_scraping_time:.2f}s")


def example_database_tracking():
    """Example: Database query tracking with caching"""
    print("\n" + "=" * 60)
    print("Example 4: Database Query Tracking")
    print("=" * 60)

    monitor = get_monitor()
    monitor.reset_metrics()

    # Simulate database queries
    cache = {}

    def query_article(article_id):
        if article_id in cache:
            with track_db_query(cached=True):
                time.sleep(0.001)
                return cache[article_id]
        else:
            with track_db_query(cached=False):
                time.sleep(0.01)
                result = f"Article {article_id} data"
                cache[article_id] = result
                return result

    # Query some articles (some cached, some not)
    for article_id in [1, 2, 3, 1, 2, 4, 1, 3]:
        result = query_article(article_id)
        status = "cached" if article_id in cache else "fresh"
        print(f"Queried article {article_id} ({status})")

    metrics = monitor.get_metrics()
    print(f"\nDatabase Metrics:")
    print(f"  Total queries: {metrics.db_queries_executed}")
    print(f"  Cache hits: {metrics.db_cache_hits}")
    print(f"  Cache misses: {metrics.db_cache_misses}")
    print(f"  Cache hit rate: {metrics.cache_hit_rate:.1f}%")
    print(f"  Average query time: {metrics.avg_db_query_time:.4f}s")


def example_system_health():
    """Example: System health monitoring"""
    print("\n" + "=" * 60)
    print("Example 5: System Health Monitoring")
    print("=" * 60)

    monitor = get_monitor()

    # Update system metrics
    monitor.update_system_metrics()

    # Get system health
    health = monitor.get_system_health()

    print(f"\nSystem Health:")
    print(f"  Status: {health.status}")
    print(f"  Platform: {health.platform}")
    print(f"  Python: {health.python_version}")
    print(f"  CPU Count: {health.cpu_count}")
    print(f"  Memory Available: {health.memory_available_mb:.1f} MB")
    print(f"  Disk Space: {health.disk_space_available_mb:.1f} MB")

    if health.issues:
        print("  Issues:")
        for issue in health.issues:
            print(f"    - {issue}")
    else:
        print("  No issues detected")


def example_alert_thresholds():
    """Example: Custom alert thresholds"""
    print("\n" + "=" * 60)
    print("Example 6: Alert Thresholds")
    print("=" * 60)

    # Create monitor with custom thresholds
    thresholds = AlertThresholds(
        max_error_count=3,
        max_failed_scrapes=2,
        min_cache_hit_rate=70.0
    )

    monitor = PerformanceMonitor(alert_thresholds=thresholds)

    # Trigger some alerts
    print("Triggering alerts...")
    for i in range(4):
        monitor.record_error()
        print(f"  Recorded error {i + 1}")

    # Get alerts
    alerts = monitor.get_alerts()
    print(f"\nAlerts triggered: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert['message']}")


def example_export_metrics():
    """Example: Exporting metrics"""
    print("\n" + "=" * 60)
    print("Example 7: Exporting Metrics")
    print("=" * 60)

    monitor = get_monitor()

    # Add some data
    with track_processing():
        time.sleep(0.1)
    monitor.record_error()

    # Export as JSON
    print("\nJSON Export (first 200 chars):")
    json_export = monitor.export_metrics("json")
    print(json_export[:200] + "...")

    # Export as CSV
    print("\nCSV Export (first 10 lines):")
    csv_export = monitor.export_metrics("csv")
    lines = csv_export.split("\n")[:10]
    print("\n".join(lines))


def example_dashboard_data():
    """Example: Getting dashboard data"""
    print("\n" + "=" * 60)
    print("Example 8: Dashboard Data")
    print("=" * 60)

    monitor = get_monitor()
    monitor.reset_metrics()

    # Simulate some activity
    for _ in range(5):
        with track_processing():
            with track_api_call(estimated_tokens=1000):
                with track_scraping():
                    time.sleep(0.05)

    # Get dashboard data
    dashboard = monitor.get_dashboard_data()

    print("\nDashboard Overview:")
    for key, value in dashboard["overview"].items():
        print(f"  {key}: {value}")

    print("\nPerformance:")
    for key, value in dashboard["performance"].items():
        print(f"  {key}: {value}")

    print("\nDatabase:")
    for key, value in dashboard["database"].items():
        print(f"  {key}: {value}")


def example_complete_workflow():
    """Example: Complete monitoring workflow"""
    print("\n" + "=" * 60)
    print("Example 9: Complete Workflow")
    print("=" * 60)

    monitor = get_monitor()
    monitor.reset_metrics()

    print("Simulating complete article processing workflow...\n")

    # Simulate processing multiple articles
    articles = [
        {"id": 1, "title": "AI Research Paper"},
        {"id": 2, "title": "Machine Learning Tutorial"},
        {"id": 3, "title": "Deep Learning News"},
    ]

    for article in articles:
        print(f"Processing: {article['title']}")

        try:
            with track_processing():
                # Scrape content
                with track_scraping():
                    print("  - Scraping content...")
                    time.sleep(0.1)

                # Query database (some cached)
                with track_db_query(cached=(article["id"] > 1)):
                    print("  - Checking database...")
                    time.sleep(0.01 if article["id"] > 1 else 0.05)

                # Analyze with AI
                with track_api_call(estimated_tokens=2000):
                    print("  - Analyzing with AI...")
                    time.sleep(0.15)

                # Save results
                with track_db_query():
                    print("  - Saving results...")
                    time.sleep(0.02)

                monitor.record_link_followed()
                print("  ✓ Complete\n")

        except Exception as e:
            monitor.record_error()
            print(f"  ✗ Failed: {e}\n")

    # Display final metrics
    metrics = monitor.get_metrics()

    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    print(f"\nProcessing:")
    print(f"  Articles: {metrics.articles_processed}")
    print(f"  Success rate: {metrics.success_rate:.1f}%")
    print(f"  Total time: {metrics.total_processing_time:.2f}s")

    print(f"\nAPI:")
    print(f"  Calls: {metrics.api_calls_made}")
    print(f"  Tokens: {metrics.api_tokens_used}")
    print(f"  Cost: ${metrics.api_cost_estimate:.4f}")

    print(f"\nScraping:")
    print(f"  Pages: {metrics.pages_scraped}")
    print(f"  Links followed: {metrics.followed_links}")

    print(f"\nDatabase:")
    print(f"  Queries: {metrics.db_queries_executed}")
    print(f"  Cache hit rate: {metrics.cache_hit_rate:.1f}%")

    # System health
    health = monitor.get_system_health()
    print(f"\nSystem Health: {health.status}")


def main():
    """Run all examples"""
    print("\n")
    print("*" * 60)
    print("MONITORING SYSTEM EXAMPLES")
    print("*" * 60)

    try:
        example_basic_tracking()
        example_api_tracking()
        example_scraping_tracking()
        example_database_tracking()
        example_system_health()
        example_alert_thresholds()
        example_export_metrics()
        example_dashboard_data()
        example_complete_workflow()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
