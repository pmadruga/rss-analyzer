"""
Unit tests for configuration module
"""

import os
from unittest.mock import patch

import pytest

from src.config.settings import (
    CONFIG,
    APIConfig,
    AppConfig,
    DatabaseConfig,
    ProcessingConfig,
    ScrapingConfig,
)


class TestAPIConfig:
    """Test APIConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = APIConfig()
        assert config.MAX_TOKENS == 4000
        assert config.TEMPERATURE == 0.3
        assert config.MAX_RETRIES == 3
        assert config.BASE_DELAY == 1.0
        assert config.TIMEOUT == 30
        assert config.RATE_LIMIT_DELAY == 3.0

    def test_custom_values(self):
        """Test custom configuration values"""
        config = APIConfig(MAX_TOKENS=2000, TEMPERATURE=0.5, MAX_RETRIES=5)
        assert config.MAX_TOKENS == 2000
        assert config.TEMPERATURE == 0.5
        assert config.MAX_RETRIES == 5

    def test_immutable(self):
        """Test that config is immutable"""
        config = APIConfig()
        with pytest.raises(AttributeError):
            config.MAX_TOKENS = 5000


class TestProcessingConfig:
    """Test ProcessingConfig dataclass"""

    def test_default_values(self):
        """Test default processing configuration values"""
        config = ProcessingConfig()
        assert config.MAX_CONTENT_LENGTH == 50000
        assert config.MAX_ARTICLES_PER_RUN == 10
        assert config.SCRAPER_DELAY == 1.0
        assert config.REQUEST_TIMEOUT == 30
        assert config.MAX_LINKED_ARTICLES == 3
        assert config.FOLLOW_LINKS is True


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass"""

    def test_default_values(self):
        """Test default database configuration values"""
        config = DatabaseConfig()
        assert config.BATCH_SIZE == 100
        assert config.CONNECTION_TIMEOUT == 30
        assert config.MAX_LOG_AGE_DAYS == 30
        assert config.VACUUM_THRESHOLD == 1000


class TestScrapingConfig:
    """Test ScrapingConfig dataclass"""

    def test_default_values(self):
        """Test default scraping configuration values"""
        config = ScrapingConfig()
        assert config.USER_AGENT == "RSS-Article-Analyzer/2.0"
        assert config.MAX_REDIRECTS == 5
        assert config.RETRY_ATTEMPTS == 3
        assert config.CHUNK_SIZE == 8192
        assert config.MAX_FILE_SIZE == 10 * 1024 * 1024


class TestAppConfig:
    """Test AppConfig dataclass"""

    def test_structure(self):
        """Test AppConfig structure"""
        config = AppConfig(
            api=APIConfig(),
            processing=ProcessingConfig(),
            database=DatabaseConfig(),
            scraping=ScrapingConfig(),
        )

        assert isinstance(config.api, APIConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.scraping, ScrapingConfig)

    @patch.dict(
        os.environ,
        {
            "API_MAX_TOKENS": "2000",
            "API_TEMPERATURE": "0.5",
            "MAX_CONTENT_LENGTH": "25000",
            "USER_AGENT": "Custom-Agent/1.0",
        },
    )
    def test_from_env(self):
        """Test configuration from environment variables"""
        config = AppConfig.from_env()

        assert config.api.MAX_TOKENS == 2000
        assert config.api.TEMPERATURE == 0.5
        assert config.processing.MAX_CONTENT_LENGTH == 25000
        assert config.scraping.USER_AGENT == "Custom-Agent/1.0"

    @patch.dict(os.environ, {"API_MAX_TOKENS": "invalid", "API_TEMPERATURE": "invalid"})
    def test_from_env_invalid_values(self):
        """Test handling of invalid environment variable values"""
        with pytest.raises(ValueError):
            AppConfig.from_env()


class TestGlobalConfig:
    """Test global CONFIG instance"""

    def test_config_exists(self):
        """Test that global CONFIG exists"""
        assert CONFIG is not None
        assert isinstance(CONFIG, AppConfig)

    def test_config_structure(self):
        """Test global CONFIG structure"""
        assert hasattr(CONFIG, "api")
        assert hasattr(CONFIG, "processing")
        assert hasattr(CONFIG, "database")
        assert hasattr(CONFIG, "scraping")

    def test_config_values(self):
        """Test global CONFIG has reasonable values"""
        assert CONFIG.api.MAX_TOKENS > 0
        assert CONFIG.api.TEMPERATURE >= 0
        assert CONFIG.processing.MAX_CONTENT_LENGTH > 0
        assert len(CONFIG.scraping.USER_AGENT) > 0
