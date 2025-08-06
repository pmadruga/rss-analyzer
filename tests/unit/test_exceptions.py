"""
Unit tests for exceptions module
"""

from src.exceptions import (
    APIClientError,
    APIConnectionError,
    APIQuotaExceededError,
    APIRateLimitError,
    APIResponseError,
    ConfigurationError,
    ContentProcessingError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseMigrationError,
    InvalidURLError,
    ReportGenerationError,
    RSSAnalyzerError,
    RSSParsingError,
    ScrapingError,
    ScrapingTimeoutError,
    ValidationError,
)


class TestRSSAnalyzerError:
    """Test base RSSAnalyzerError"""

    def test_basic_exception(self):
        """Test basic exception creation"""
        error = RSSAnalyzerError("Test error")
        assert str(error) == "Test error"
        assert error.details == {}

    def test_exception_with_details(self):
        """Test exception with details"""
        details = {"code": 500, "context": "test"}
        error = RSSAnalyzerError("Test error", details)
        assert str(error) == "Test error"
        assert error.details == details

    def test_inheritance(self):
        """Test that it inherits from Exception"""
        error = RSSAnalyzerError("Test")
        assert isinstance(error, Exception)


class TestConfigurationError:
    """Test ConfigurationError"""

    def test_configuration_error(self):
        """Test configuration error creation"""
        error = ConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
        assert isinstance(error, RSSAnalyzerError)


class TestAPIClientError:
    """Test APIClientError and subclasses"""

    def test_api_client_error(self):
        """Test basic API client error"""
        error = APIClientError("API failed", "anthropic")
        assert str(error) == "API failed"
        assert error.provider == "anthropic"
        assert isinstance(error, RSSAnalyzerError)

    def test_api_connection_error(self):
        """Test API connection error"""
        error = APIConnectionError("Connection failed", "openai")
        assert str(error) == "Connection failed"
        assert error.provider == "openai"
        assert isinstance(error, APIClientError)

    def test_api_rate_limit_error(self):
        """Test API rate limit error"""
        error = APIRateLimitError("Rate limited", "mistral", retry_after=60)
        assert str(error) == "Rate limited"
        assert error.provider == "mistral"
        assert error.retry_after == 60
        assert isinstance(error, APIClientError)

    def test_api_rate_limit_error_no_retry(self):
        """Test API rate limit error without retry_after"""
        error = APIRateLimitError("Rate limited", "anthropic")
        assert error.retry_after is None

    def test_api_quota_exceeded_error(self):
        """Test API quota exceeded error"""
        error = APIQuotaExceededError("Quota exceeded", "openai")
        assert str(error) == "Quota exceeded"
        assert error.provider == "openai"
        assert isinstance(error, APIClientError)

    def test_api_response_error(self):
        """Test API response error"""
        error = APIResponseError("Bad response", "claude", status_code=400)
        assert str(error) == "Bad response"
        assert error.provider == "claude"
        assert error.status_code == 400
        assert isinstance(error, APIClientError)

    def test_api_response_error_no_status(self):
        """Test API response error without status code"""
        error = APIResponseError("Bad response", "claude")
        assert error.status_code is None


class TestContentProcessingError:
    """Test ContentProcessingError"""

    def test_content_processing_error(self):
        """Test content processing error"""
        error = ContentProcessingError("Processing failed")
        assert str(error) == "Processing failed"
        assert isinstance(error, RSSAnalyzerError)


class TestScrapingError:
    """Test ScrapingError and subclasses"""

    def test_scraping_error(self):
        """Test basic scraping error"""
        url = "https://example.com"
        error = ScrapingError("Scraping failed", url)
        assert str(error) == "Scraping failed"
        assert error.url == url
        assert isinstance(error, RSSAnalyzerError)

    def test_scraping_timeout_error(self):
        """Test scraping timeout error"""
        url = "https://example.com"
        error = ScrapingTimeoutError("Timeout", url)
        assert str(error) == "Timeout"
        assert error.url == url
        assert isinstance(error, ScrapingError)

    def test_invalid_url_error(self):
        """Test invalid URL error"""
        url = "invalid-url"
        error = InvalidURLError("Invalid URL", url)
        assert str(error) == "Invalid URL"
        assert error.url == url
        assert isinstance(error, ScrapingError)


class TestDatabaseError:
    """Test DatabaseError and subclasses"""

    def test_database_error(self):
        """Test basic database error"""
        error = DatabaseError("DB operation failed")
        assert str(error) == "DB operation failed"
        assert isinstance(error, RSSAnalyzerError)

    def test_database_connection_error(self):
        """Test database connection error"""
        error = DatabaseConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, DatabaseError)

    def test_database_migration_error(self):
        """Test database migration error"""
        error = DatabaseMigrationError("Migration failed")
        assert str(error) == "Migration failed"
        assert isinstance(error, DatabaseError)


class TestRSSParsingError:
    """Test RSSParsingError"""

    def test_rss_parsing_error(self):
        """Test RSS parsing error"""
        feed_url = "https://example.com/feed.xml"
        error = RSSParsingError("Parse failed", feed_url)
        assert str(error) == "Parse failed"
        assert error.feed_url == feed_url
        assert isinstance(error, RSSAnalyzerError)


class TestReportGenerationError:
    """Test ReportGenerationError"""

    def test_report_generation_error(self):
        """Test report generation error"""
        error = ReportGenerationError("Report failed")
        assert str(error) == "Report failed"
        assert isinstance(error, RSSAnalyzerError)


class TestValidationError:
    """Test ValidationError"""

    def test_validation_error(self):
        """Test validation error"""
        error = ValidationError("Invalid value", "email", "invalid-email")
        assert str(error) == "Invalid value"
        assert error.field == "email"
        assert error.value == "invalid-email"
        assert isinstance(error, RSSAnalyzerError)

    def test_validation_error_no_value(self):
        """Test validation error without value"""
        error = ValidationError("Required field", "name")
        assert error.field == "name"
        assert error.value is None
