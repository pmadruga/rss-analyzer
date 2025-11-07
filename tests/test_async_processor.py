"""
Tests for Async Article Processor

Tests concurrent processing capabilities and performance improvements.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.processors.async_article_processor import (
    AsyncArticleProcessor,
    ProcessingConfig,
    ProcessingResults,
)
from src.core.async_scraper import ScrapedContent


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    return {
        "db_path": "test.db",
        "cache_db_path": "test_cache.db",
        "rss_feed_url": "https://example.com/feed",
        "output_dir": "test_output",
        "anthropic_api_key": "test-key-12345678901234567890",
        "api_provider": "anthropic",
        "api_model": "claude-3-5-sonnet-20241022",
        "max_concurrent_articles": 5,
    }


@pytest.fixture
def mock_rss_entry():
    """Create mock RSS entry"""
    entry = Mock()
    entry.title = "Test Article"
    entry.link = "https://example.com/article"
    entry.content_hash = "abc123"
    entry.guid = "guid-123"
    entry.publication_date = datetime.now()
    return entry


@pytest.fixture
def mock_scraped_content():
    """Create mock scraped content"""
    return ScrapedContent(
        url="https://example.com/article",
        title="Test Article",
        content="This is test content for the article.",
        metadata={"source": "test"}
    )


class TestAsyncArticleProcessor:
    """Test suite for AsyncArticleProcessor"""

    @pytest.mark.asyncio
    async def test_processor_initialization(self, mock_config):
        """Test async processor initializes correctly"""
        with patch("src.processors.async_article_processor.DatabaseManager"):
            with patch("src.processors.async_article_processor.ContentCache"):
                with patch("src.processors.async_article_processor.RSSParser"):
                    with patch("src.processors.async_article_processor.AsyncWebScraper"):
                        with patch("src.processors.async_article_processor.AsyncClaudeClient"):
                            processor = AsyncArticleProcessor(mock_config)
                            assert processor is not None
                            assert processor.config == mock_config

    @pytest.mark.asyncio
    async def test_scrape_article_async(self, mock_config, mock_rss_entry, mock_scraped_content):
        """Test async article scraping"""
        with patch("src.processors.async_article_processor.DatabaseManager"):
            with patch("src.processors.async_article_processor.ContentCache") as mock_cache:
                with patch("src.processors.async_article_processor.RSSParser"):
                    with patch("src.processors.async_article_processor.AsyncWebScraper") as mock_scraper_class:
                        with patch("src.processors.async_article_processor.AsyncClaudeClient"):
                            # Setup mocks
                            mock_cache.return_value.get.return_value = None
                            mock_scraper = Mock()
                            mock_scraper.scrape_single = AsyncMock(return_value=mock_scraped_content)
                            mock_scraper_class.return_value = mock_scraper

                            processor = AsyncArticleProcessor(mock_config)
                            processor.db.is_content_already_processed = Mock(return_value=False)
                            processor.db.update_article_content_hash = Mock()

                            processing_config = ProcessingConfig(force_refresh=False)

                            result = await processor._scrape_article_async(
                                mock_rss_entry,
                                processing_config,
                                1
                            )

                            assert result == mock_scraped_content
                            mock_scraper.scrape_single.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_article_async(self, mock_config, mock_rss_entry, mock_scraped_content):
        """Test async article analysis"""
        with patch("src.processors.async_article_processor.DatabaseManager"):
            with patch("src.processors.async_article_processor.ContentCache") as mock_cache:
                with patch("src.processors.async_article_processor.RSSParser"):
                    with patch("src.processors.async_article_processor.AsyncWebScraper"):
                        with patch("src.processors.async_article_processor.AsyncClaudeClient") as mock_client_class:
                            # Setup mocks
                            mock_cache.return_value.get.return_value = None
                            mock_analysis = {
                                "methodology_detailed": "Test analysis",
                                "technical_approach": "Test approach",
                                "key_findings": "Test findings",
                            }
                            mock_client = Mock()
                            mock_client.analyze_article_async = AsyncMock(return_value=mock_analysis)
                            mock_client.model = "claude-3-5-sonnet-20241022"
                            mock_client_class.return_value = mock_client

                            processor = AsyncArticleProcessor(mock_config)

                            result = await processor._analyze_article_async(
                                mock_rss_entry,
                                mock_scraped_content,
                                1
                            )

                            assert result == mock_analysis
                            mock_client.analyze_article_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_config):
        """Test that articles are processed concurrently"""
        with patch("src.processors.async_article_processor.DatabaseManager") as mock_db:
            with patch("src.processors.async_article_processor.ContentCache"):
                with patch("src.processors.async_article_processor.RSSParser"):
                    with patch("src.processors.async_article_processor.AsyncWebScraper") as mock_scraper_class:
                        with patch("src.processors.async_article_processor.AsyncClaudeClient") as mock_client_class:
                            # Setup mocks
                            mock_db.return_value.insert_articles_batch.return_value = [1, 2, 3]

                            # Mock scraper with delay to test concurrency
                            async def mock_scrape_single(*args, **kwargs):
                                await asyncio.sleep(0.1)  # Simulate network delay
                                return ScrapedContent(
                                    url="https://example.com/test",
                                    title="Test",
                                    content="Content",
                                )

                            mock_scraper = Mock()
                            mock_scraper.scrape_single = mock_scrape_single
                            mock_scraper_class.return_value = mock_scraper

                            # Mock AI client
                            async def mock_analyze(*args, **kwargs):
                                await asyncio.sleep(0.1)  # Simulate API delay
                                return {
                                    "methodology_detailed": "Analysis",
                                    "technical_approach": "Approach",
                                    "key_findings": "Findings",
                                }

                            mock_client = Mock()
                            mock_client.analyze_article_async = mock_analyze
                            mock_client.model = "claude-3-5-sonnet-20241022"
                            mock_client_class.return_value = mock_client

                            processor = AsyncArticleProcessor(mock_config)
                            processor.db.is_content_already_processed = Mock(return_value=False)
                            processor.db.update_article_content_hash = Mock()
                            processor.db.update_status_batch = Mock()
                            processor.db.insert_content_batch = Mock()
                            processor.db.log_processing_batch = Mock()

                            # Create mock entries
                            entries = []
                            for i in range(3):
                                entry = Mock()
                                entry.title = f"Article {i}"
                                entry.link = f"https://example.com/article-{i}"
                                entry.content_hash = f"hash-{i}"
                                entry.guid = f"guid-{i}"
                                entry.publication_date = datetime.now()
                                entries.append(entry)

                            processing_config = ProcessingConfig(
                                force_refresh=False,
                                max_concurrent=3
                            )
                            results = ProcessingResults(
                                start_time=time.time(),
                                duration=0.0,
                                rss_entries_found=3,
                                new_articles=3,
                                scraped_articles=0,
                                analyzed_articles=0,
                                report_generated=False,
                                errors=[]
                            )

                            start_time = time.time()
                            processed = await processor._process_articles_async(
                                entries,
                                processing_config,
                                results
                            )
                            elapsed = time.time() - start_time

                            # With concurrent processing (3 articles, 0.1s scrape + 0.1s analysis each):
                            # Sequential: 3 * (0.1 + 0.1) = 0.6s
                            # Concurrent: max(0.1, 0.1) = ~0.2s
                            # Allow some margin for overhead
                            assert elapsed < 0.5, f"Processing took {elapsed}s, expected < 0.5s (concurrent)"
                            assert len(processed) == 3

    @pytest.mark.asyncio
    async def test_cache_integration_async(self, mock_config, mock_rss_entry, mock_scraped_content):
        """Test cache is checked and used in async mode"""
        with patch("src.processors.async_article_processor.DatabaseManager"):
            with patch("src.processors.async_article_processor.ContentCache") as mock_cache_class:
                with patch("src.processors.async_article_processor.RSSParser"):
                    with patch("src.processors.async_article_processor.AsyncWebScraper"):
                        with patch("src.processors.async_article_processor.AsyncClaudeClient"):
                            # Setup cache to return cached content
                            mock_cache = Mock()
                            mock_cache.get.return_value = mock_scraped_content
                            mock_cache_class.return_value = mock_cache

                            processor = AsyncArticleProcessor(mock_config)

                            processing_config = ProcessingConfig(force_refresh=False)

                            result = await processor._scrape_article_async(
                                mock_rss_entry,
                                processing_config,
                                1
                            )

                            # Should return cached content
                            assert result == mock_scraped_content
                            # Cache get should be called
                            mock_cache.get.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_processing(self, mock_config):
        """Test that errors in one article don't stop others"""
        with patch("src.processors.async_article_processor.DatabaseManager") as mock_db:
            with patch("src.processors.async_article_processor.ContentCache"):
                with patch("src.processors.async_article_processor.RSSParser"):
                    with patch("src.processors.async_article_processor.AsyncWebScraper") as mock_scraper_class:
                        with patch("src.processors.async_article_processor.AsyncClaudeClient") as mock_client_class:
                            # Setup mocks
                            mock_db.return_value.insert_articles_batch.return_value = [1, 2, 3]

                            # First scrape succeeds, second fails, third succeeds
                            call_count = 0

                            async def mock_scrape_single(*args, **kwargs):
                                nonlocal call_count
                                call_count += 1
                                if call_count == 2:
                                    raise Exception("Scraping failed")
                                return ScrapedContent(
                                    url=f"https://example.com/test-{call_count}",
                                    title=f"Test {call_count}",
                                    content="Content",
                                )

                            mock_scraper = Mock()
                            mock_scraper.scrape_single = mock_scrape_single
                            mock_scraper_class.return_value = mock_scraper

                            # Mock AI client
                            mock_client = Mock()
                            mock_client.analyze_article_async = AsyncMock(return_value={
                                "methodology_detailed": "Analysis",
                                "technical_approach": "Approach",
                                "key_findings": "Findings",
                            })
                            mock_client.model = "claude-3-5-sonnet-20241022"
                            mock_client_class.return_value = mock_client

                            processor = AsyncArticleProcessor(mock_config)
                            processor.db.is_content_already_processed = Mock(return_value=False)
                            processor.db.update_article_content_hash = Mock()
                            processor.db.update_status_batch = Mock()
                            processor.db.insert_content_batch = Mock()
                            processor.db.log_processing_batch = Mock()

                            # Create mock entries
                            entries = []
                            for i in range(3):
                                entry = Mock()
                                entry.title = f"Article {i}"
                                entry.link = f"https://example.com/article-{i}"
                                entry.content_hash = f"hash-{i}"
                                entry.guid = f"guid-{i}"
                                entry.publication_date = datetime.now()
                                entries.append(entry)

                            processing_config = ProcessingConfig(max_concurrent=3)
                            results = ProcessingResults(
                                start_time=time.time(),
                                duration=0.0,
                                rss_entries_found=3,
                                new_articles=3,
                                scraped_articles=0,
                                analyzed_articles=0,
                                report_generated=False,
                                errors=[]
                            )

                            processed = await processor._process_articles_async(
                                entries,
                                processing_config,
                                results
                            )

                            # Should process 2 successfully (articles 1 and 3)
                            assert len(processed) == 2
                            # Should have 1 error logged
                            assert len(results.errors) >= 1


class TestPerformanceBenchmark:
    """Performance benchmark tests"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_sync_vs_async_performance(self, mock_config):
        """Benchmark async vs sync performance (disabled by default)"""
        # This test is marked with @pytest.mark.benchmark
        # Run with: pytest tests/test_async_processor.py -m benchmark
        pytest.skip("Benchmark test - run manually with -m benchmark")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
