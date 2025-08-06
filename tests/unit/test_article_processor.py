"""
Unit tests for article processor
"""

from unittest.mock import Mock, patch

import pytest

from src.exceptions import ConfigurationError, ContentProcessingError
from src.processors.article_processor import (
    ArticleProcessor,
    ProcessingConfig,
    ProcessingResults,
)


class TestProcessingResults:
    """Test ProcessingResults dataclass"""

    def test_initialization(self):
        """Test ProcessingResults initialization"""
        results = ProcessingResults(
            start_time=1234567890.0,
            duration=120.5,
            rss_entries_found=10,
            new_articles=5,
            scraped_articles=4,
            analyzed_articles=3,
            report_generated=True,
            errors=["Error 1", "Error 2"],
        )

        assert results.start_time == 1234567890.0
        assert results.duration == 120.5
        assert results.rss_entries_found == 10
        assert results.new_articles == 5
        assert results.scraped_articles == 4
        assert results.analyzed_articles == 3
        assert results.report_generated is True
        assert results.errors == ["Error 1", "Error 2"]
        assert results.reports is None

    def test_to_dict(self):
        """Test converting ProcessingResults to dictionary"""
        results = ProcessingResults(
            start_time=1234567890.0,
            duration=120.5,
            rss_entries_found=10,
            new_articles=5,
            scraped_articles=4,
            analyzed_articles=3,
            report_generated=True,
            errors=[],
        )

        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["start_time"] == 1234567890.0
        assert result_dict["duration"] == 120.5
        assert result_dict["report_generated"] is True


class TestProcessingConfig:
    """Test ProcessingConfig dataclass"""

    def test_default_values(self):
        """Test ProcessingConfig default values"""
        config = ProcessingConfig()

        assert config.force_refresh is False
        assert config.limit is None
        assert config.follow_links is True
        assert config.max_linked_articles == 3

    def test_custom_values(self):
        """Test ProcessingConfig with custom values"""
        config = ProcessingConfig(
            force_refresh=True, limit=10, follow_links=False, max_linked_articles=5
        )

        assert config.force_refresh is True
        assert config.limit == 10
        assert config.follow_links is False
        assert config.max_linked_articles == 5

    def test_from_dict(self):
        """Test creating ProcessingConfig from dictionary"""
        config_dict = {
            "force_refresh": True,
            "limit": 15,
            "follow_links": False,
            "max_linked_articles": 2,
        }

        config = ProcessingConfig.from_dict(config_dict)

        assert config.force_refresh is True
        assert config.limit == 15
        assert config.follow_links is False
        assert config.max_linked_articles == 2

    def test_from_dict_partial(self):
        """Test creating ProcessingConfig from partial dictionary"""
        config_dict = {
            "force_refresh": True,
            "limit": 20,
            # Missing other fields - should use defaults
        }

        config = ProcessingConfig.from_dict(config_dict)

        assert config.force_refresh is True
        assert config.limit == 20
        assert config.follow_links is True  # Default
        assert config.max_linked_articles == 3  # Default


class TestArticleProcessor:
    """Test ArticleProcessor class"""

    def test_initialization_success(self, mock_config):
        """Test successful ArticleProcessor initialization"""
        with (
            patch("src.processors.article_processor.DatabaseManager") as mock_db,
            patch("src.processors.article_processor.RSSParser") as mock_rss,
            patch("src.processors.article_processor.WebScraper") as mock_scraper,
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.ReportGenerator") as mock_report,
        ):
            mock_factory.create_from_config.return_value = Mock()

            processor = ArticleProcessor(mock_config)

            assert processor.config == mock_config
            assert processor.db is not None
            assert processor.rss_parser is not None
            assert processor.scraper is not None
            assert processor.ai_client is not None
            assert processor.report_generator is not None

    def test_initialization_failure(self, mock_config):
        """Test ArticleProcessor initialization failure"""
        with patch("src.processors.article_processor.DatabaseManager") as mock_db:
            mock_db.side_effect = Exception("Database init failed")

            with pytest.raises(ConfigurationError) as exc_info:
                ArticleProcessor(mock_config)

            assert "Initialization failed" in str(exc_info.value)

    def test_test_api_connection_success(self, mock_config, mock_api_client):
        """Test successful API connection test"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager"),
            patch("src.processors.article_processor.RSSParser"),
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = mock_api_client
            mock_api_client.test_connection.return_value = True

            processor = ArticleProcessor(mock_config)
            processor._test_api_connection()  # Should not raise

    def test_test_api_connection_failure(self, mock_config, mock_api_client):
        """Test failed API connection test"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager"),
            patch("src.processors.article_processor.RSSParser"),
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = mock_api_client
            mock_api_client.test_connection.return_value = False

            processor = ArticleProcessor(mock_config)

            with pytest.raises(ConfigurationError):
                processor._test_api_connection()

    def test_fetch_rss_feed_success(self, mock_config, mock_rss_parser):
        """Test successful RSS feed fetching"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager"),
            patch("src.processors.article_processor.RSSParser") as mock_rss_class,
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = Mock()
            mock_rss_class.return_value = mock_rss_parser

            processor = ArticleProcessor(mock_config)
            results = ProcessingResults(0, 0, 0, 0, 0, 0, False, [])

            entries = processor._fetch_rss_feed(results)

            assert len(entries) == 2
            assert results.rss_entries_found == 2
            mock_rss_parser.fetch_feed.assert_called_once_with(
                mock_config["rss_feed_url"]
            )

    def test_fetch_rss_feed_empty(self, mock_config, mock_rss_parser):
        """Test RSS feed fetching with empty feed"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager"),
            patch("src.processors.article_processor.RSSParser") as mock_rss_class,
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = Mock()
            mock_rss_class.return_value = mock_rss_parser
            mock_rss_parser.fetch_feed.return_value = []

            processor = ArticleProcessor(mock_config)
            results = ProcessingResults(0, 0, 0, 0, 0, 0, False, [])

            entries = processor._fetch_rss_feed(results)

            assert len(entries) == 0
            assert results.rss_entries_found == 0

    def test_fetch_rss_feed_failure(self, mock_config, mock_rss_parser):
        """Test RSS feed fetching failure"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager"),
            patch("src.processors.article_processor.RSSParser") as mock_rss_class,
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = Mock()
            mock_rss_class.return_value = mock_rss_parser
            mock_rss_parser.fetch_feed.side_effect = Exception("RSS fetch failed")

            processor = ArticleProcessor(mock_config)
            results = ProcessingResults(0, 0, 0, 0, 0, 0, False, [])

            with pytest.raises(ContentProcessingError):
                processor._fetch_rss_feed(results)

    def test_filter_articles_force_refresh(
        self, mock_config, mock_db_manager, mock_rss_parser
    ):
        """Test article filtering with force refresh"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager") as mock_db_class,
            patch("src.processors.article_processor.RSSParser") as mock_rss_class,
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = Mock()
            mock_db_class.return_value = mock_db_manager
            mock_rss_class.return_value = mock_rss_parser

            processor = ArticleProcessor(mock_config)
            results = ProcessingResults(0, 0, 0, 0, 0, 0, False, [])
            config = ProcessingConfig(force_refresh=True)

            # Mock RSS entries
            entries = [Mock(), Mock()]

            filtered = processor._filter_articles(entries, config, results)

            assert len(filtered) == 2
            assert results.new_articles == 2
            # Should not call get_analyzed_content_hashes when force refresh
            mock_db_manager.get_analyzed_content_hashes.assert_not_called()

    def test_filter_articles_no_force_refresh(
        self, mock_config, mock_db_manager, mock_rss_parser
    ):
        """Test article filtering without force refresh"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager") as mock_db_class,
            patch("src.processors.article_processor.RSSParser") as mock_rss_class,
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = Mock()
            mock_db_class.return_value = mock_db_manager
            mock_rss_class.return_value = mock_rss_parser

            processor = ArticleProcessor(mock_config)
            results = ProcessingResults(0, 0, 0, 0, 0, 0, False, [])
            config = ProcessingConfig(force_refresh=False)

            # Mock RSS entries and filtering
            entries = [Mock(), Mock()]
            mock_rss_parser.filter_new_entries.return_value = [
                entries[0]
            ]  # Filter out one

            filtered = processor._filter_articles(entries, config, results)

            assert len(filtered) == 1
            assert results.new_articles == 1
            mock_db_manager.get_analyzed_content_hashes.assert_called_once()
            mock_rss_parser.filter_new_entries.assert_called_once()

    def test_filter_articles_with_limit(
        self, mock_config, mock_db_manager, mock_rss_parser
    ):
        """Test article filtering with limit"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager") as mock_db_class,
            patch("src.processors.article_processor.RSSParser") as mock_rss_class,
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = Mock()
            mock_db_class.return_value = mock_db_manager
            mock_rss_class.return_value = mock_rss_parser

            processor = ArticleProcessor(mock_config)
            results = ProcessingResults(0, 0, 0, 0, 0, 0, False, [])
            config = ProcessingConfig(force_refresh=True, limit=1)

            # Mock RSS entries - more than limit
            entries = [Mock(), Mock(), Mock()]

            filtered = processor._filter_articles(entries, config, results)

            assert len(filtered) == 1  # Limited to 1
            assert results.new_articles == 1

    def test_get_client_info(self, mock_config, mock_api_client):
        """Test getting client information"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager"),
            patch("src.processors.article_processor.RSSParser"),
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = mock_api_client

            processor = ArticleProcessor(mock_config)
            info = processor.get_client_info()

            assert info == mock_api_client.get_provider_info.return_value
            mock_api_client.get_provider_info.assert_called_once()

    def test_get_processing_stats(self, mock_config, mock_db_manager):
        """Test getting processing statistics"""
        with (
            patch("src.processors.article_processor.AIClientFactory") as mock_factory,
            patch("src.processors.article_processor.DatabaseManager") as mock_db_class,
            patch("src.processors.article_processor.RSSParser"),
            patch("src.processors.article_processor.WebScraper"),
            patch("src.processors.article_processor.ReportGenerator"),
        ):
            mock_factory.create_from_config.return_value = Mock()
            mock_db_class.return_value = mock_db_manager

            processor = ArticleProcessor(mock_config)
            stats = processor.get_processing_stats()

            assert stats == mock_db_manager.get_processing_statistics.return_value
            mock_db_manager.get_processing_statistics.assert_called_once()
