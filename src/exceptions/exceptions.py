"""
Custom Exceptions Module

Defines specific exceptions for better error handling and debugging.
Replaces generic Exception usage throughout the codebase.
"""

from typing import Any


class RSSAnalyzerError(Exception):
    """Base exception for all RSS analyzer errors"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(RSSAnalyzerError):
    """Configuration related errors"""


class APIClientError(RSSAnalyzerError):
    """Base class for API client errors"""

    def __init__(
        self, message: str, provider: str, details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)
        self.provider = provider


class APIConnectionError(APIClientError):
    """API connection failed"""


class APIRateLimitError(APIClientError):
    """API rate limit exceeded"""

    def __init__(self, message: str, provider: str, retry_after: int | None = None):
        super().__init__(message, provider)
        self.retry_after = retry_after


class APIQuotaExceededError(APIClientError):
    """API quota exceeded"""


class APIResponseError(APIClientError):
    """Invalid API response"""

    def __init__(self, message: str, provider: str, status_code: int | None = None):
        super().__init__(message, provider)
        self.status_code = status_code


class ContentProcessingError(RSSAnalyzerError):
    """Content processing related errors"""


class ScrapingError(RSSAnalyzerError):
    """Web scraping related errors"""

    def __init__(self, message: str, url: str, details: dict[str, Any] | None = None):
        super().__init__(message, details)
        self.url = url


class ScrapingTimeoutError(ScrapingError):
    """Scraping operation timed out"""


class InvalidURLError(ScrapingError):
    """Invalid URL provided"""


class DatabaseError(RSSAnalyzerError):
    """Database operation errors"""


class DatabaseConnectionError(DatabaseError):
    """Database connection failed"""


class DatabaseMigrationError(DatabaseError):
    """Database migration failed"""


class RSSParsingError(RSSAnalyzerError):
    """RSS feed parsing errors"""

    def __init__(
        self, message: str, feed_url: str, details: dict[str, Any] | None = None
    ):
        super().__init__(message, details)
        self.feed_url = feed_url


class ReportGenerationError(RSSAnalyzerError):
    """Report generation errors"""


class ValidationError(RSSAnalyzerError):
    """Data validation errors"""

    def __init__(self, message: str, field: str, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value
