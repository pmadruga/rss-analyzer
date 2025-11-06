"""
Monitoring and Observability Module

Provides performance metrics tracking, system health monitoring, and real-time statistics
with minimal overhead. Uses efficient data structures and lazy evaluation for optimal performance.
"""

import json
import logging
import os
import platform
import psutil
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""

    # Processing metrics
    articles_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0

    # Database metrics
    db_queries_executed: int = 0
    db_query_time: float = 0.0
    db_cache_hits: int = 0
    db_cache_misses: int = 0

    # API metrics
    api_calls_made: int = 0
    api_call_time: float = 0.0
    api_tokens_used: int = 0
    api_cost_estimate: float = 0.0

    # Scraping metrics
    pages_scraped: int = 0
    scraping_time: float = 0.0
    failed_scrapes: int = 0
    followed_links: int = 0

    # System metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    warning_count: int = 0

    # Timing metrics
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.db_cache_hits + self.db_cache_misses
        if total == 0:
            return 0.0
        return (self.db_cache_hits / total) * 100

    @property
    def success_rate(self) -> float:
        """Calculate article processing success rate"""
        total = self.articles_processed + self.failed_scrapes
        if total == 0:
            return 100.0
        return (self.articles_processed / total) * 100

    @property
    def avg_api_call_time(self) -> float:
        """Calculate average API call time"""
        if self.api_calls_made == 0:
            return 0.0
        return self.api_call_time / self.api_calls_made

    @property
    def avg_scraping_time(self) -> float:
        """Calculate average scraping time"""
        if self.pages_scraped == 0:
            return 0.0
        return self.scraping_time / self.pages_scraped

    @property
    def avg_db_query_time(self) -> float:
        """Calculate average database query time"""
        if self.db_queries_executed == 0:
            return 0.0
        return self.db_query_time / self.db_queries_executed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed properties"""
        data = asdict(self)
        data["cache_hit_rate"] = self.cache_hit_rate
        data["success_rate"] = self.success_rate
        data["avg_api_call_time"] = self.avg_api_call_time
        data["avg_scraping_time"] = self.avg_scraping_time
        data["avg_db_query_time"] = self.avg_db_query_time

        # Convert datetime objects to ISO format
        if self.start_time:
            data["start_time"] = self.start_time.isoformat()
        if self.last_update_time:
            data["last_update_time"] = self.last_update_time.isoformat()

        return data


@dataclass
class AlertThresholds:
    """Alert threshold configuration"""

    max_processing_time_seconds: float = 300.0  # 5 minutes
    max_memory_usage_mb: float = 1024.0  # 1 GB
    max_cpu_usage_percent: float = 90.0
    max_error_count: int = 10
    min_cache_hit_rate: float = 50.0  # 50%
    max_failed_scrapes: int = 5


@dataclass
class SystemHealth:
    """System health status"""

    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    memory_available_mb: float
    disk_space_available_mb: float
    cpu_count: int
    platform: str
    python_version: str
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PerformanceMonitor:
    """
    Thread-safe performance monitoring with minimal overhead

    Uses atomic operations and lazy evaluation to minimize performance impact.
    """

    def __init__(self, alert_thresholds: Optional[AlertThresholds] = None):
        """Initialize performance monitor"""
        self.metrics = PerformanceMetrics(
            start_time=datetime.now(),
            last_update_time=datetime.now()
        )
        self.alert_thresholds = alert_thresholds or AlertThresholds()
        self._lock = threading.RLock()
        self._alerts = []

        # Process handle for system metrics
        self._process = psutil.Process()

        logger.info("PerformanceMonitor initialized")

    @contextmanager
    def track_processing(self):
        """Context manager to track article processing time"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                self.metrics.articles_processed += 1
                self.metrics.total_processing_time += duration
                self.metrics.average_processing_time = (
                    self.metrics.total_processing_time / self.metrics.articles_processed
                )
                self.metrics.last_update_time = datetime.now()

    @contextmanager
    def track_api_call(self, estimated_tokens: int = 0):
        """Context manager to track API call time and cost"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                self.metrics.api_calls_made += 1
                self.metrics.api_call_time += duration
                self.metrics.api_tokens_used += estimated_tokens
                # Rough cost estimate: $0.01 per 1K tokens (adjust as needed)
                self.metrics.api_cost_estimate = (self.metrics.api_tokens_used / 1000) * 0.01
                self.metrics.last_update_time = datetime.now()

    @contextmanager
    def track_scraping(self):
        """Context manager to track web scraping time"""
        start = time.perf_counter()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                if success:
                    self.metrics.pages_scraped += 1
                    self.metrics.scraping_time += duration
                else:
                    self.metrics.failed_scrapes += 1
                self.metrics.last_update_time = datetime.now()

    @contextmanager
    def track_db_query(self, cached: bool = False):
        """Context manager to track database query time"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                self.metrics.db_queries_executed += 1
                self.metrics.db_query_time += duration
                if cached:
                    self.metrics.db_cache_hits += 1
                else:
                    self.metrics.db_cache_misses += 1
                self.metrics.last_update_time = datetime.now()

    def record_error(self):
        """Record an error occurrence"""
        with self._lock:
            self.metrics.error_count += 1
            self.metrics.last_update_time = datetime.now()
            self._check_alert(f"Error count: {self.metrics.error_count}")

    def record_warning(self):
        """Record a warning occurrence"""
        with self._lock:
            self.metrics.warning_count += 1
            self.metrics.last_update_time = datetime.now()

    def record_link_followed(self):
        """Record a followed link"""
        with self._lock:
            self.metrics.followed_links += 1

    def update_system_metrics(self):
        """Update system resource metrics (call periodically, not on every operation)"""
        try:
            with self._lock:
                # Memory usage (in MB)
                mem_info = self._process.memory_info()
                self.metrics.memory_usage_mb = mem_info.rss / (1024 * 1024)

                # CPU usage (percentage)
                self.metrics.cpu_usage_percent = self._process.cpu_percent(interval=0.1)

                self.metrics.last_update_time = datetime.now()

                # Check alerts
                self._check_thresholds()

        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def _check_thresholds(self):
        """Check if any thresholds are exceeded"""
        if self.metrics.memory_usage_mb > self.alert_thresholds.max_memory_usage_mb:
            self._check_alert(
                f"Memory usage high: {self.metrics.memory_usage_mb:.1f} MB"
            )

        if self.metrics.cpu_usage_percent > self.alert_thresholds.max_cpu_usage_percent:
            self._check_alert(
                f"CPU usage high: {self.metrics.cpu_usage_percent:.1f}%"
            )

        if self.metrics.error_count > self.alert_thresholds.max_error_count:
            self._check_alert(
                f"Error count high: {self.metrics.error_count}"
            )

        if self.metrics.cache_hit_rate < self.alert_thresholds.min_cache_hit_rate:
            self._check_alert(
                f"Cache hit rate low: {self.metrics.cache_hit_rate:.1f}%"
            )

        if self.metrics.failed_scrapes > self.alert_thresholds.max_failed_scrapes:
            self._check_alert(
                f"Failed scrapes high: {self.metrics.failed_scrapes}"
            )

    def _check_alert(self, message: str):
        """Record an alert"""
        alert = {"timestamp": datetime.now().isoformat(), "message": message}
        self._alerts.append(alert)
        logger.warning(f"Alert: {message}")

        # Keep only last 100 alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics (thread-safe copy)"""
        with self._lock:
            # Update system metrics before returning
            self.update_system_metrics()
            # Return a copy of metrics
            return PerformanceMetrics(**asdict(self.metrics))

    def get_alerts(self) -> list[dict[str, Any]]:
        """Get recent alerts"""
        with self._lock:
            return self._alerts.copy()

    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics = PerformanceMetrics(
                start_time=datetime.now(),
                last_update_time=datetime.now()
            )
            self._alerts.clear()
            logger.info("Metrics reset")

    def get_system_health(self) -> SystemHealth:
        """Get system health status"""
        try:
            # Memory info
            mem = psutil.virtual_memory()
            mem_available_mb = mem.available / (1024 * 1024)

            # Disk space
            disk = psutil.disk_usage('/')
            disk_available_mb = disk.free / (1024 * 1024)

            # Determine health status
            issues = []
            if self.metrics.memory_usage_mb > self.alert_thresholds.max_memory_usage_mb:
                issues.append(f"High memory usage: {self.metrics.memory_usage_mb:.1f} MB")

            if self.metrics.cpu_usage_percent > self.alert_thresholds.max_cpu_usage_percent:
                issues.append(f"High CPU usage: {self.metrics.cpu_usage_percent:.1f}%")

            if self.metrics.error_count > self.alert_thresholds.max_error_count:
                issues.append(f"High error count: {self.metrics.error_count}")

            if disk_available_mb < 1024:  # Less than 1GB free
                issues.append(f"Low disk space: {disk_available_mb:.1f} MB")

            # Determine overall status
            if len(issues) == 0:
                status = "healthy"
            elif len(issues) <= 2:
                status = "degraded"
            else:
                status = "unhealthy"

            return SystemHealth(
                status=status,
                timestamp=datetime.now(),
                memory_available_mb=mem_available_mb,
                disk_space_available_mb=disk_available_mb,
                cpu_count=psutil.cpu_count(),
                platform=platform.platform(),
                python_version=platform.python_version(),
                issues=issues
            )

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                status="unknown",
                timestamp=datetime.now(),
                memory_available_mb=0.0,
                disk_space_available_mb=0.0,
                cpu_count=0,
                platform="unknown",
                python_version="unknown",
                issues=[f"Health check failed: {str(e)}"]
            )

    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics to JSON or CSV format

        Args:
            format: Export format ("json" or "csv")

        Returns:
            Formatted metrics string
        """
        metrics_dict = self.get_metrics().to_dict()

        if format == "json":
            return json.dumps(metrics_dict, indent=2)
        elif format == "csv":
            # Simple CSV format
            lines = ["metric,value"]
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{key},{value}")
                else:
                    lines.append(f"{key},\"{value}\"")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_metrics(self, output_path: str | Path, format: str = "json"):
        """
        Save metrics to file

        Args:
            output_path: Output file path
            format: Export format ("json" or "csv")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = self.export_metrics(format)

        with open(output_path, "w") as f:
            f.write(metrics_data)

        logger.info(f"Metrics saved to: {output_path}")

    def get_dashboard_data(self) -> dict[str, Any]:
        """
        Get data formatted for dashboard display

        Returns:
            Dictionary with dashboard-ready metrics
        """
        metrics = self.get_metrics()
        health = self.get_system_health()

        return {
            "overview": {
                "status": health.status,
                "articles_processed": metrics.articles_processed,
                "success_rate": f"{metrics.success_rate:.1f}%",
                "total_time": f"{metrics.total_processing_time:.1f}s",
                "avg_processing_time": f"{metrics.average_processing_time:.2f}s"
            },
            "performance": {
                "api_calls": metrics.api_calls_made,
                "avg_api_time": f"{metrics.avg_api_call_time:.2f}s",
                "api_cost": f"${metrics.api_cost_estimate:.4f}",
                "pages_scraped": metrics.pages_scraped,
                "avg_scraping_time": f"{metrics.avg_scraping_time:.2f}s"
            },
            "database": {
                "queries_executed": metrics.db_queries_executed,
                "avg_query_time": f"{metrics.avg_db_query_time:.4f}s",
                "cache_hit_rate": f"{metrics.cache_hit_rate:.1f}%"
            },
            "system": {
                "memory_usage": f"{metrics.memory_usage_mb:.1f} MB",
                "cpu_usage": f"{metrics.cpu_usage_percent:.1f}%",
                "errors": metrics.error_count,
                "warnings": metrics.warning_count
            },
            "health": health.to_dict(),
            "alerts": self.get_alerts()[-10:]  # Last 10 alerts
        }


class MonitoringManager:
    """
    Singleton monitoring manager for application-wide metrics tracking
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.monitor = PerformanceMonitor()
        self._initialized = True
        logger.info("MonitoringManager initialized")

    @classmethod
    def get_instance(cls) -> "MonitoringManager":
        """Get singleton instance"""
        return cls()

    def get_monitor(self) -> PerformanceMonitor:
        """Get the performance monitor"""
        return self.monitor


# Global accessor functions for convenience
def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return MonitoringManager.get_instance().get_monitor()


def track_processing():
    """Context manager for tracking processing operations"""
    return get_monitor().track_processing()


def track_api_call(estimated_tokens: int = 0):
    """Context manager for tracking API calls"""
    return get_monitor().track_api_call(estimated_tokens)


def track_scraping():
    """Context manager for tracking web scraping"""
    return get_monitor().track_scraping()


def track_db_query(cached: bool = False):
    """Context manager for tracking database queries"""
    return get_monitor().track_db_query(cached)
