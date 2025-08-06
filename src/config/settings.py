"""
Configuration Module

Central configuration management with type safety and validation.
Replaces magic numbers and hardcoded values throughout the codebase.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class APIConfig:
    """Configuration for API clients"""

    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.3
    MAX_RETRIES: int = 3
    BASE_DELAY: float = 1.0
    TIMEOUT: int = 30
    RATE_LIMIT_DELAY: float = 3.0


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for content processing"""

    MAX_CONTENT_LENGTH: int = 50000
    MAX_ARTICLES_PER_RUN: int = 10
    SCRAPER_DELAY: float = 1.0
    REQUEST_TIMEOUT: int = 30
    MAX_LINKED_ARTICLES: int = 3
    FOLLOW_LINKS: bool = True


@dataclass(frozen=True)
class DatabaseConfig:
    """Configuration for database operations"""

    BATCH_SIZE: int = 100
    CONNECTION_TIMEOUT: int = 30
    MAX_LOG_AGE_DAYS: int = 30
    VACUUM_THRESHOLD: int = 1000


@dataclass(frozen=True)
class ScrapingConfig:
    """Configuration for web scraping"""

    USER_AGENT: str = "RSS-Article-Analyzer/2.0"
    MAX_REDIRECTS: int = 5
    RETRY_ATTEMPTS: int = 3
    CHUNK_SIZE: int = 8192
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration"""

    api: APIConfig
    processing: ProcessingConfig
    database: DatabaseConfig
    scraping: ScrapingConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        return cls(
            api=APIConfig(
                MAX_TOKENS=int(os.getenv("API_MAX_TOKENS", "4000")),
                TEMPERATURE=float(os.getenv("API_TEMPERATURE", "0.3")),
                MAX_RETRIES=int(os.getenv("API_MAX_RETRIES", "3")),
                BASE_DELAY=float(os.getenv("API_BASE_DELAY", "1.0")),
                TIMEOUT=int(os.getenv("API_TIMEOUT", "30")),
                RATE_LIMIT_DELAY=float(os.getenv("API_RATE_LIMIT_DELAY", "3.0")),
            ),
            processing=ProcessingConfig(
                MAX_CONTENT_LENGTH=int(os.getenv("MAX_CONTENT_LENGTH", "50000")),
                MAX_ARTICLES_PER_RUN=int(os.getenv("MAX_ARTICLES_PER_RUN", "10")),
                SCRAPER_DELAY=float(os.getenv("SCRAPER_DELAY", "1.0")),
                REQUEST_TIMEOUT=int(os.getenv("REQUEST_TIMEOUT", "30")),
                MAX_LINKED_ARTICLES=int(os.getenv("MAX_LINKED_ARTICLES", "3")),
                FOLLOW_LINKS=os.getenv("FOLLOW_LINKS", "true").lower() == "true",
            ),
            database=DatabaseConfig(
                BATCH_SIZE=int(os.getenv("DB_BATCH_SIZE", "100")),
                CONNECTION_TIMEOUT=int(os.getenv("DB_TIMEOUT", "30")),
                MAX_LOG_AGE_DAYS=int(os.getenv("MAX_LOG_AGE_DAYS", "30")),
                VACUUM_THRESHOLD=int(os.getenv("VACUUM_THRESHOLD", "1000")),
            ),
            scraping=ScrapingConfig(
                USER_AGENT=os.getenv("USER_AGENT", "RSS-Article-Analyzer/2.0"),
                MAX_REDIRECTS=int(os.getenv("MAX_REDIRECTS", "5")),
                RETRY_ATTEMPTS=int(os.getenv("RETRY_ATTEMPTS", "3")),
                CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "8192")),
                MAX_FILE_SIZE=int(os.getenv("MAX_FILE_SIZE", "10485760")),
            ),
        )


# Global configuration instance
CONFIG = AppConfig.from_env()
