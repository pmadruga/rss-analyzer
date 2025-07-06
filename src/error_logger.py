"""
Error Logging System for RSS Analyzer

This module provides comprehensive error logging that integrates with the website
to show processing status and errors by date.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ProcessingError:
    """Represents a processing error"""

    timestamp: str
    date: str  # YYYY-MM-DD format for grouping
    error_type: str
    error_message: str
    component: str  # e.g., "api", "scraper", "parser"
    details: dict[str, Any] | None = None


@dataclass
class ProcessingStatus:
    """Represents processing status for a date"""

    date: str
    status: str  # "success", "partial", "failed"
    total_attempted: int
    successful: int
    failed: int
    errors: list[ProcessingError]
    last_attempt: str


class ErrorLogger:
    """Enhanced error logger with website integration"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.error_log_file = self.output_dir / "processing_errors.json"
        self.status_file = self.output_dir / "processing_status.json"

        # Setup logging
        self.logger = logging.getLogger("error_logger")
        self.logger.setLevel(logging.INFO)

        # Create file handler for errors
        error_handler = logging.FileHandler(self.output_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        error_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(error_handler)

    def log_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        details: dict[str, Any] | None = None,
    ):
        """Log an error with comprehensive details"""
        now = datetime.now(UTC)

        error = ProcessingError(
            timestamp=now.isoformat(),
            date=now.strftime("%Y-%m-%d"),
            error_type=error_type,
            error_message=error_message,
            component=component,
            details=details or {},
        )

        # Log to file
        self.logger.error(f"[{component}] {error_type}: {error_message}")
        if details:
            self.logger.error(f"Details: {details}")

        # Add to error log
        self._append_error_log(error)

        # Update status
        self._update_processing_status(error.date, success=False)

    def log_success(
        self, component: str, message: str, details: dict[str, Any] | None = None
    ):
        """Log a successful operation"""
        now = datetime.now(UTC)

        self.logger.info(f"[{component}] SUCCESS: {message}")
        if details:
            self.logger.info(f"Details: {details}")

        # Update status
        self._update_processing_status(now.strftime("%Y-%m-%d"), success=True)

    def _append_error_log(self, error: ProcessingError):
        """Append error to the error log file"""
        try:
            # Load existing errors
            errors = []
            if self.error_log_file.exists():
                with open(self.error_log_file) as f:
                    data = json.load(f)
                    errors = data.get("errors", [])

            # Add new error
            errors.append(asdict(error))

            # Keep only last 1000 errors to prevent file from growing too large
            if len(errors) > 1000:
                errors = errors[-1000:]

            # Save back to file
            error_data = {
                "last_updated": datetime.now(UTC).isoformat(),
                "total_errors": len(errors),
                "errors": errors,
            }

            with open(self.error_log_file, "w") as f:
                json.dump(error_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save error log: {e}")

    def _update_processing_status(self, date: str, success: bool):
        """Update processing status for a date"""
        try:
            # Load existing status
            status_data = {}
            if self.status_file.exists():
                with open(self.status_file) as f:
                    status_data = json.load(f)

            # Get or create status for this date
            daily_status = status_data.get("dates", {}).get(
                date,
                {
                    "date": date,
                    "status": "unknown",
                    "total_attempted": 0,
                    "successful": 0,
                    "failed": 0,
                    "errors": [],
                    "last_attempt": datetime.now(UTC).isoformat(),
                },
            )

            # Update counters
            daily_status["total_attempted"] += 1
            daily_status["last_attempt"] = datetime.now(UTC).isoformat()

            if success:
                daily_status["successful"] += 1
            else:
                daily_status["failed"] += 1

            # Determine overall status
            if daily_status["failed"] == 0:
                daily_status["status"] = "success"
            elif daily_status["successful"] > 0:
                daily_status["status"] = "partial"
            else:
                daily_status["status"] = "failed"

            # Update the status data
            if "dates" not in status_data:
                status_data["dates"] = {}

            status_data["dates"][date] = daily_status
            status_data["last_updated"] = datetime.now(UTC).isoformat()

            # Save back to file
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to update processing status: {e}")

    def get_status_summary(self, days: int = 30) -> dict[str, Any]:
        """Get processing status summary for recent days"""
        try:
            if not self.status_file.exists():
                return {
                    "dates": {},
                    "summary": {
                        "total_days": 0,
                        "successful_days": 0,
                        "failed_days": 0,
                    },
                }

            with open(self.status_file) as f:
                status_data = json.load(f)

            dates = status_data.get("dates", {})

            # Filter to recent days
            recent_dates = {}
            today = datetime.now(UTC)

            for date_str, status in dates.items():
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    days_ago = (today - date_obj.replace(tzinfo=UTC)).days

                    if days_ago <= days:
                        recent_dates[date_str] = status
                except ValueError:
                    continue

            # Generate summary
            total_days = len(recent_dates)
            successful_days = sum(
                1 for s in recent_dates.values() if s.get("status") == "success"
            )
            failed_days = sum(
                1 for s in recent_dates.values() if s.get("status") == "failed"
            )

            return {
                "dates": recent_dates,
                "summary": {
                    "total_days": total_days,
                    "successful_days": successful_days,
                    "failed_days": failed_days,
                    "partial_days": total_days - successful_days - failed_days,
                },
                "last_updated": status_data.get("last_updated"),
            }

        except Exception as e:
            self.logger.error(f"Failed to get status summary: {e}")
            return {
                "dates": {},
                "summary": {"total_days": 0, "successful_days": 0, "failed_days": 0},
            }

    def get_recent_errors(self, days: int = 7) -> list[ProcessingError]:
        """Get recent errors within the specified number of days"""
        try:
            if not self.error_log_file.exists():
                return []

            with open(self.error_log_file) as f:
                data = json.load(f)
                errors = data.get("errors", [])

            # Filter to recent errors
            recent_errors = []
            from datetime import timedelta

            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            for error_dict in errors:
                try:
                    error_date = datetime.fromisoformat(
                        error_dict["timestamp"].replace("Z", "+00:00")
                    )
                    if error_date > cutoff_date:
                        recent_errors.append(ProcessingError(**error_dict))
                except (ValueError, KeyError):
                    continue

            return recent_errors

        except Exception as e:
            self.logger.error(f"Failed to get recent errors: {e}")
            return []

    def generate_website_status(self) -> dict[str, Any]:
        """Generate status data for website integration"""
        status_summary = self.get_status_summary(days=30)
        recent_errors = self.get_recent_errors(days=7)

        # Group errors by date
        errors_by_date = {}
        for error in recent_errors:
            date = error.date
            if date not in errors_by_date:
                errors_by_date[date] = []
            errors_by_date[date].append(
                {
                    "timestamp": error.timestamp,
                    "type": error.error_type,
                    "message": error.error_message,
                    "component": error.component,
                }
            )

        # Determine overall system status
        recent_status = list(status_summary["dates"].values())
        if not recent_status:
            system_status = "unknown"
        else:
            latest_status = max(recent_status, key=lambda x: x["last_attempt"])
            system_status = latest_status["status"]

        return {
            "system_status": system_status,
            "last_updated": status_summary.get("last_updated"),
            "summary": status_summary["summary"],
            "dates": status_summary["dates"],
            "recent_errors_by_date": errors_by_date,
            "health_check": {
                "timestamp": datetime.now(UTC).isoformat(),
                "apis_working": 0,  # This should be updated by API health checks
                "rss_feed_accessible": True,  # This should be updated by RSS checks
                "database_accessible": True,  # This should be updated by DB checks
            },
        }


# Global error logger instance
_error_logger = None


def get_error_logger(output_dir: str = "output") -> ErrorLogger:
    """Get the global error logger instance"""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger(output_dir)
    return _error_logger


def log_api_error(
    error_type: str,
    error_message: str,
    api_provider: str,
    details: dict[str, Any] | None = None,
):
    """Convenience function to log API errors"""
    logger = get_error_logger()
    logger.log_error(
        error_type=error_type,
        error_message=error_message,
        component=f"api_{api_provider}",
        details=details,
    )


def log_scraping_error(
    error_type: str,
    error_message: str,
    url: str,
    details: dict[str, Any] | None = None,
):
    """Convenience function to log scraping errors"""
    logger = get_error_logger()
    logger.log_error(
        error_type=error_type,
        error_message=error_message,
        component="scraper",
        details={"url": url, **(details or {})},
    )


def log_processing_success(
    component: str, message: str, details: dict[str, Any] | None = None
):
    """Convenience function to log successful processing"""
    logger = get_error_logger()
    logger.log_success(component, message, details)
