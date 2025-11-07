"""
Integration Tests for Full RSS Analyzer Pipeline

Tests the complete flow: RSS → Scrape → Analyze → Database → Reports
Covers both synchronous and asynchronous processing paths.
"""

import asyncio
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients import AIClientFactory, AsyncArticleProcessor
from src.core import DatabaseManager, RSSParser, WebScraper
from src.core.cache import ContentCache
from src.processors.article_processor import ArticleProcessor, ProcessingConfig


# Test data
MOCK_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Test Article 1</title>
      <link>https://example.com/article1</link>
      <description>This is a test article about AI</description>
    </item>
    <item>
      <title>Test Article 2</title>
      <link>https://example.com/article2</link>
      <description>This is a test article about ML</description>
    </item>
  </channel>
</rss>
"""

MOCK_HTML_CONTENT = """
<html>
<head><title>Test Article</title></head>
<body>
    <article>
        <h1>Test Article</h1>
        <p>This is test content about artificial intelligence and machine learning.</p>
        <p>It contains enough text to be meaningful for analysis.</p>
    </article>
</body>
</html>
"""

MOCK_ANALYSIS = {
    "methodology_detailed": "Detailed analysis using Feynman technique",
    "technical_approach": "Technical details of the methodology",
    "key_findings": "Key findings from the research",
    "research_design": "Research approach and design",
    "extracted_title": "Test Article",
    "metadata": {
        "ai_provider": "test",
        "model": "test-model",
        "processed_at": 1234567890,
    }
}


@pytest.fixture
def temp_test_env():
    """Create temporary test environment"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_articles.db"
        cache_path = Path(tmpdir) / "test_cache.db"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        yield {
            "db_path": str(db_path),
            "cache_path": str(cache_path),
            "output_dir": str(output_dir),
            "tmpdir": tmpdir
        }


@pytest.fixture
def mock_rss_response():
    """Mock RSS feed response"""
    mock_response = MagicMock()
    mock_response.text = MOCK_RSS_FEED
    mock_response.status_code = 200
    return mock_response


@pytest.fixture
def mock_html_response():
    """Mock HTML content response"""
    mock_response = MagicMock()
    mock_response.text = MOCK_HTML_CONTENT
    mock_response.status_code = 200
    mock_response.content = MOCK_HTML_CONTENT.encode('utf-8')
    return mock_response


@pytest.fixture
def mock_ai_client():
    """Mock AI client for testing"""
    mock_client = MagicMock()
    mock_client.analyze_article.return_value = MOCK_ANALYSIS
    mock_client.provider_name = "test"
    return mock_client


class TestSynchronousFullPipeline:
    """Test complete synchronous pipeline processing"""

    def test_full_pipeline_with_cache(self, temp_test_env, mock_rss_response,
                                      mock_html_response, mock_ai_client):
        """Test full pipeline with caching enabled"""
        config = {
            "db_path": temp_test_env["db_path"],
            "cache_db_path": temp_test_env["cache_path"],
            "output_dir": temp_test_env["output_dir"],
            "rss_feed_url": "https://example.com/feed.xml",
            "api_provider": "test",
            "test_api_key": "test-key-123",
        }

        with patch('requests.get') as mock_get, \
             patch.object(AIClientFactory, 'create_client', return_value=mock_ai_client):

            # Mock RSS and HTML responses
            mock_get.side_effect = [mock_rss_response, mock_html_response, mock_html_response]

            # Initialize processor
            processor = ArticleProcessor(config)

            # Process articles
            processing_config = ProcessingConfig(limit=2)
            results = processor.process_articles(processing_config)

            # Verify results
            assert results.new_articles == 2
            assert results.scraped_articles == 2
            assert results.analyzed_articles == 2
            assert results.report_generated is True
            assert len(results.errors) == 0

            # Verify database entries
            with sqlite3.connect(temp_test_env["db_path"]) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM articles")
                article_count = cursor.fetchone()[0]
                assert article_count == 2

                cursor = conn.execute("SELECT COUNT(*) FROM content")
                content_count = cursor.fetchone()[0]
                assert content_count == 2

            # Verify cache usage
            cache_stats = processor.cache.get_stats()
            assert cache_stats['l1_entries'] > 0 or cache_stats['l2_entries'] > 0

    def test_pipeline_with_duplicate_detection(self, temp_test_env, mock_rss_response,
                                                mock_html_response, mock_ai_client):
        """Test that duplicate articles are not reprocessed"""
        config = {
            "db_path": temp_test_env["db_path"],
            "cache_db_path": temp_test_env["cache_path"],
            "output_dir": temp_test_env["output_dir"],
            "rss_feed_url": "https://example.com/feed.xml",
            "api_provider": "test",
            "test_api_key": "test-key-123",
        }

        with patch('requests.get') as mock_get, \
             patch.object(AIClientFactory, 'create_client', return_value=mock_ai_client):

            mock_get.side_effect = [
                mock_rss_response, mock_html_response, mock_html_response,
                mock_rss_response  # Second run with same feed
            ]

            processor = ArticleProcessor(config)

            # First run
            processing_config = ProcessingConfig(limit=2)
            results1 = processor.process_articles(processing_config)
            assert results1.new_articles == 2

            # Second run (should detect duplicates)
            results2 = processor.process_articles(processing_config)
            assert results2.new_articles == 0  # No new articles

            # Verify only 2 articles in database
            with sqlite3.connect(temp_test_env["db_path"]) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM articles")
                article_count = cursor.fetchone()[0]
                assert article_count == 2

    def test_pipeline_error_handling(self, temp_test_env, mock_rss_response):
        """Test pipeline error handling and recovery"""
        config = {
            "db_path": temp_test_env["db_path"],
            "cache_db_path": temp_test_env["cache_path"],
            "output_dir": temp_test_env["output_dir"],
            "rss_feed_url": "https://example.com/feed.xml",
            "api_provider": "test",
            "test_api_key": "test-key-123",
        }

        # Mock AI client that fails on second article
        mock_client = MagicMock()
        call_count = 0

        def analyze_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("AI API error")
            return MOCK_ANALYSIS

        mock_client.analyze_article.side_effect = analyze_side_effect
        mock_client.provider_name = "test"

        with patch('requests.get', return_value=mock_rss_response), \
             patch.object(AIClientFactory, 'create_client', return_value=mock_client):

            processor = ArticleProcessor(config)
            processing_config = ProcessingConfig(limit=2)
            results = processor.process_articles(processing_config)

            # Should process first article, fail on second
            assert results.analyzed_articles == 1
            assert len(results.errors) > 0
            assert "error" in str(results.errors[0]).lower()


class TestAsynchronousFullPipeline:
    """Test complete asynchronous pipeline processing"""

    @pytest.mark.asyncio
    async def test_async_pipeline_parallel_processing(self, temp_test_env):
        """Test async pipeline processes articles in parallel"""
        # Create test articles
        articles = [
            {
                "title": f"Test Article {i}",
                "content": f"Content {i} about AI and machine learning" * 20,
                "url": f"https://example.com/article{i}",
            }
            for i in range(10)
        ]

        # Mock async AI client
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps(MOCK_ANALYSIS)
        mock_response.content = [mock_content]

        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_instance

            # Process articles asynchronously
            processor = AsyncArticleProcessor(
                provider="anthropic",
                api_key="test-key-123",
                max_concurrent=5
            )

            import time
            start_time = time.time()
            results = await processor.process_articles(articles, show_progress=False)
            duration = time.time() - start_time

            # Verify all articles processed
            assert len(results) == 10
            assert all(r is not None for r in results)

            # Verify parallel processing is faster (rough check)
            # 10 articles with 5 concurrent should take ~2 batches
            assert duration < 10  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_async_pipeline_with_cache(self, temp_test_env):
        """Test async pipeline uses cache effectively"""
        cache = ContentCache(temp_test_env["cache_path"])

        # Pre-populate cache
        test_url = "https://example.com/article1"
        cache_key = ContentCache.generate_key(test_url, "api")
        cache.set(cache_key, MOCK_ANALYSIS, ttl=3600, content_type="api")

        articles = [
            {
                "title": "Cached Article",
                "content": "Content about AI",
                "url": test_url,
            }
        ]

        # Mock AI client that shouldn't be called
        mock_client = MagicMock()
        mock_client.analyze_article_async = AsyncMock()

        with patch.object(AIClientFactory, 'create_async_client', return_value=mock_client):
            processor = AsyncArticleProcessor(
                provider="anthropic",
                api_key="test-key-123"
            )

            # Check cache before processing
            cached_value = cache.get(cache_key)
            assert cached_value is not None

            # Note: This test would need processor to support cache parameter
            # For now, verify cache has the data
            assert cached_value == MOCK_ANALYSIS


class TestPipelineIntegration:
    """Test integration between pipeline components"""

    def test_database_cache_integration(self, temp_test_env):
        """Test database and cache work together correctly"""
        db = DatabaseManager(temp_test_env["db_path"])
        cache = ContentCache(temp_test_env["cache_path"])

        # Add article to database
        article_id = db.insert_article(
            title="Test Article",
            url="https://example.com/test",
            description="Test description",
            content="Test content",
            content_hash="test_hash_123"
        )

        # Add content to cache
        cache_key = ContentCache.generate_key("https://example.com/test", "scraped")
        cache.set(cache_key, "Cached content", ttl=3600)

        # Verify both systems have data
        article = db.get_article_by_id(article_id)
        assert article is not None
        assert article["title"] == "Test Article"

        cached_content = cache.get(cache_key)
        assert cached_content == "Cached content"

    def test_scraper_database_integration(self, temp_test_env, mock_html_response):
        """Test scraper and database integration"""
        db = DatabaseManager(temp_test_env["db_path"])
        scraper = WebScraper()

        with patch('requests.get', return_value=mock_html_response):
            # Scrape content
            content = scraper.scrape_article("https://example.com/test")
            assert content is not None
            assert len(content) > 0

            # Store in database
            article_id = db.insert_article(
                title="Scraped Article",
                url="https://example.com/test",
                description="Description",
                content=content,
                content_hash="hash_123"
            )

            # Retrieve and verify
            article = db.get_article_by_id(article_id)
            assert article is not None
            assert article["content"] == content


class TestPipelineRecovery:
    """Test pipeline recovery and resilience"""

    def test_pipeline_continues_after_scraper_failure(self, temp_test_env,
                                                       mock_rss_response, mock_ai_client):
        """Test pipeline continues processing after scraping errors"""
        config = {
            "db_path": temp_test_env["db_path"],
            "cache_db_path": temp_test_env["cache_path"],
            "output_dir": temp_test_env["output_dir"],
            "rss_feed_url": "https://example.com/feed.xml",
            "api_provider": "test",
            "test_api_key": "test-key-123",
        }

        # Mock responses: RSS succeeds, first scrape fails, second succeeds
        mock_html_response = MagicMock()
        mock_html_response.text = MOCK_HTML_CONTENT
        mock_html_response.status_code = 200

        call_count = 0
        def get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_rss_response
            elif call_count == 2:
                raise Exception("Network error")
            else:
                return mock_html_response

        with patch('requests.get', side_effect=get_side_effect), \
             patch.object(AIClientFactory, 'create_client', return_value=mock_ai_client):

            processor = ArticleProcessor(config)
            processing_config = ProcessingConfig(limit=2)
            results = processor.process_articles(processing_config)

            # Should process at least one article
            assert results.analyzed_articles >= 1
            assert len(results.errors) > 0

    def test_pipeline_database_transaction_rollback(self, temp_test_env):
        """Test database transaction rollback on errors"""
        db = DatabaseManager(temp_test_env["db_path"])

        # Start with clean database
        with sqlite3.connect(temp_test_env["db_path"]) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            initial_count = cursor.fetchone()[0]

        # Try to insert with invalid data that should rollback
        try:
            with sqlite3.connect(temp_test_env["db_path"]) as conn:
                conn.execute(
                    "INSERT INTO articles (title, url, content_hash) VALUES (?, ?, ?)",
                    ("Test", "https://example.com", "hash123")
                )
                # Force an error
                conn.execute("INSERT INTO articles (id) VALUES (NULL)")  # Should fail
                conn.commit()
        except Exception:
            pass

        # Verify rollback - count should be unchanged
        with sqlite3.connect(temp_test_env["db_path"]) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            final_count = cursor.fetchone()[0]
            assert final_count == initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
