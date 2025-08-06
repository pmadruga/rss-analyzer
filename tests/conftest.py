"""
Test configuration and fixtures for RSS Analyzer tests
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.config import (
    APIConfig,
    AppConfig,
    DatabaseConfig,
    ProcessingConfig,
    ScrapingConfig,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "api_provider": "anthropic",
        "anthropic_api_key": "sk-test-key-123",
        "claude_model": "claude-3-5-sonnet-20241022",
        "mistral_api_key": "test-mistral-key",
        "mistral_model": "mistral-large-latest",
        "openai_api_key": "sk-test-openai-key",
        "openai_model": "gpt-4",
        "rss_feed_url": "https://example.com/feed.xml",
        "db_path": "test.db",
        "output_dir": "test_output",
        "user_agent": "Test-Agent/1.0",
        "max_articles_per_run": 5,
        "scraper_delay": 0.1,
        "request_timeout": 10,
    }


@pytest.fixture
def app_config():
    """Test AppConfig instance"""
    return AppConfig(
        api=APIConfig(
            MAX_TOKENS=1000,
            TEMPERATURE=0.3,
            MAX_RETRIES=2,
            BASE_DELAY=0.5,
            TIMEOUT=10,
            RATE_LIMIT_DELAY=1.0,
        ),
        processing=ProcessingConfig(
            MAX_CONTENT_LENGTH=10000,
            MAX_ARTICLES_PER_RUN=5,
            SCRAPER_DELAY=0.1,
            REQUEST_TIMEOUT=10,
            MAX_LINKED_ARTICLES=2,
            FOLLOW_LINKS=True,
        ),
        database=DatabaseConfig(
            BATCH_SIZE=50,
            CONNECTION_TIMEOUT=10,
            MAX_LOG_AGE_DAYS=7,
            VACUUM_THRESHOLD=100,
        ),
        scraping=ScrapingConfig(
            USER_AGENT="Test-Agent/1.0",
            MAX_REDIRECTS=3,
            RETRY_ATTEMPTS=2,
            CHUNK_SIZE=1024,
            MAX_FILE_SIZE=1024 * 1024,
        ),
    )


@pytest.fixture
def mock_api_client():
    """Mock AI client for testing"""
    client = Mock()
    client.provider_name = "test_provider"
    client.model = "test_model"
    client.analyze_article.return_value = {
        "methodology_detailed": "Test methodology",
        "technical_approach": "Test approach",
        "key_findings": "Test findings",
        "research_design": "Test design",
        "metadata": {
            "ai_provider": "test_provider",
            "model": "test_model",
            "processed_at": 1234567890,
        },
    }
    client.test_connection.return_value = True
    client.get_provider_info.return_value = {
        "provider": "test_provider",
        "model": "test_model",
        "max_retries": 3,
        "rate_limit_delay": 1.0,
        "timeout": 30,
    }
    return client


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing"""
    db = Mock()
    db.insert_article.return_value = 1  # article ID
    db.insert_content.return_value = None
    db.update_article_status.return_value = None
    db.log_processing.return_value = None
    db.get_analyzed_content_hashes.return_value = set()
    db.cleanup_old_logs.return_value = None
    db.get_processing_statistics.return_value = {
        "total_articles": 10,
        "by_status": {"completed": 8, "failed": 2},
        "recent_activity": [],
    }
    return db


@pytest.fixture
def mock_rss_parser():
    """Mock RSS parser for testing"""
    parser = Mock()
    parser.fetch_feed.return_value = [
        Mock(
            title="Test Article 1",
            link="https://example.com/article1",
            content_hash="hash1",
            guid="guid1",
            publication_date=None,
        ),
        Mock(
            title="Test Article 2",
            link="https://example.com/article2",
            content_hash="hash2",
            guid="guid2",
            publication_date=None,
        ),
    ]
    parser.filter_new_entries.return_value = parser.fetch_feed.return_value
    parser.get_feed_info.return_value = {
        "title": "Test Feed",
        "description": "Test Description",
        "entry_count": 2,
    }
    return parser


@pytest.fixture
def mock_scraper():
    """Mock web scraper for testing"""
    scraper = Mock()
    content_mock = Mock()
    content_mock.content = "Test article content"
    content_mock.title = "Test Article Title"
    content_mock.url = "https://example.com/article"
    content_mock.metadata = {}
    scraper.scrape_article.return_value = content_mock
    return scraper


@pytest.fixture
def mock_report_generator():
    """Mock report generator for testing"""
    generator = Mock()
    generator.generate_report.return_value = "test_report.md"
    generator.generate_summary_report.return_value = "test_summary.md"
    generator.generate_json_export.return_value = "test_export.json"
    generator.generate_csv_export.return_value = "test_export.csv"
    generator.list_reports.return_value = []
    return generator


@pytest.fixture
def sample_analysis():
    """Sample analysis result for testing"""
    return {
        "methodology_detailed": "This is a detailed methodology explanation...",
        "technical_approach": "The technical approach involves...",
        "key_findings": "The key findings are...",
        "research_design": "The research design follows...",
        "metadata": {
            "ai_provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "processed_at": 1234567890,
        },
    }


@pytest.fixture
def sample_article_data():
    """Sample article data for testing"""
    return {
        "id": 1,
        "title": "Test Article",
        "url": "https://example.com/article",
        "publication_date": "2024-01-01T00:00:00",
        "processed_date": "2024-01-01 12:00:00",
        "methodology_detailed": "Test methodology",
        "technical_approach": "Test approach",
        "key_findings": "Test findings",
        "research_design": "Test design",
        "metadata": {"ai_provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    }


class MockAPIResponse:
    """Mock API response for testing"""

    def __init__(self, content="Test response", status_code=200):
        self.content = [Mock(text=content)]
        self.status_code = status_code
        self.choices = [Mock(message=Mock(content=content))]


@pytest.fixture
def mock_api_response():
    """Mock API response fixture"""
    return MockAPIResponse()


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    client = Mock()
    client.messages.create.return_value = MockAPIResponse(
        '{"methodology_detailed": "test", "technical_approach": "test", "key_findings": "test", "research_design": "test"}'
    )
    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    client = Mock()
    response_mock = Mock()
    response_mock.choices = [
        Mock(
            message=Mock(
                content='{"methodology_detailed": "test", "technical_approach": "test", "key_findings": "test", "research_design": "test"}'
            )
        )
    ]
    client.chat.completions.create.return_value = response_mock
    return client


@pytest.fixture
def mock_mistral_client():
    """Mock Mistral client for testing"""
    client = Mock()
    response_mock = Mock()
    response_mock.choices = [
        Mock(
            message=Mock(
                content='{"methodology_detailed": "test", "technical_approach": "test", "key_findings": "test", "research_design": "test"}'
            )
        )
    ]
    client.chat.return_value = response_mock
    return client
