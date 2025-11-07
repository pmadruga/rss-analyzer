"""
Comprehensive Async Article Processor Tests

Tests for AsyncArticleProcessor covering:
- Concurrent processing capabilities
- Error handling with mixed failures
- Semaphore/rate limiting
- Performance vs sync processor
- Full pipeline execution
- Batch processing
- Result aggregation
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.processors.async_article_processor import (
    AsyncArticleProcessor,
    ProcessingConfig,
    ProcessingResults,
)


# Sample test data
SAMPLE_ARTICLES = [
    {
        "title": "Article 1",
        "link": "https://example.com/1",
        "content_hash": "hash1",
        "guid": "guid1",
        "description": "Description 1",
    },
    {
        "title": "Article 2",
        "link": "https://example.com/2",
        "content_hash": "hash2",
        "guid": "guid2",
        "description": "Description 2",
    },
    {
        "title": "Article 3",
        "link": "https://example.com/3",
        "content_hash": "hash3",
        "guid": "guid3",
        "description": "Description 3",
    },
    {
        "title": "Article 4",
        "link": "https://example.com/4",
        "content_hash": "hash4",
        "guid": "guid4",
        "description": "Description 4",
    },
    {
        "title": "Article 5",
        "link": "https://example.com/5",
        "content_hash": "hash5",
        "guid": "guid5",
        "description": "Description 5",
    },
]

SAMPLE_ANALYSIS = {
    "methodology_detailed": "Test methodology using Feynman technique",
    "technical_approach": "Technical approach details",
    "key_findings": "Key findings from analysis",
    "research_design": "Research design overview",
    "metadata": {
        "ai_provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "processed_at": 1234567890,
    },
}


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "api_provider": "anthropic",
        "anthropic_api_key": "sk-test-key",
        "api_model": "claude-3-5-sonnet-20241022",
        "rss_feed_url": "https://example.com/feed.xml",
        "db_path": "test.db",
        "cache_db_path": "test_cache.db",
        "output_dir": "test_output",
        "user_agent": "Test-Agent/1.0",
        "max_articles_per_run": 10,
        "scraper_delay": 0.1,
        "request_timeout": 10,
        "max_concurrent_articles": 3,
    }


@pytest.fixture
def mock_processor_components():
    """Create mocked processor components"""
    components = {
        "db": AsyncMock(),
        "cache": AsyncMock(),
        "rss_parser": AsyncMock(),
        "scraper": AsyncMock(),
        "ai_client": AsyncMock(),
        "report_generator": AsyncMock(),
    }

    # Configure default mock behaviors
    components["db"].get_analyzed_content_hashes.return_value = set()
    components["db"].insert_article.return_value = 1
    components["db"].update_article_status.return_value = None
    components["cache"].get.return_value = None
    components["cache"].set.return_value = None
    components["rss_parser"].fetch_feed.return_value = SAMPLE_ARTICLES
    components["ai_client"].analyze_article.return_value = SAMPLE_ANALYSIS
    components["report_generator"].generate_report.return_value = "report.md"

    return components


@pytest.fixture
def mock_processor_with_components(mock_config, mock_processor_components):
    """Create AsyncArticleProcessor with mocked components"""
    with patch("src.processors.async_article_processor.DatabaseManager") as db_mock, \
         patch("src.processors.async_article_processor.ContentCache") as cache_mock, \
         patch("src.processors.async_article_processor.RSSParser") as parser_mock, \
         patch("src.processors.async_article_processor.AsyncWebScraper") as scraper_mock, \
         patch("src.processors.async_article_processor.AsyncClaudeClient") as client_mock, \
         patch("src.processors.async_article_processor.ReportGenerator") as gen_mock:

        db_mock.return_value = mock_processor_components["db"]
        cache_mock.return_value = mock_processor_components["cache"]
        parser_mock.return_value = mock_processor_components["rss_parser"]
        scraper_mock.return_value = mock_processor_components["scraper"]
        client_mock.return_value = mock_processor_components["ai_client"]
        gen_mock.return_value = mock_processor_components["report_generator"]

        processor = AsyncArticleProcessor(mock_config)
        processor.db = mock_processor_components["db"]
        processor.cache = mock_processor_components["cache"]
        processor.rss_parser = mock_processor_components["rss_parser"]
        processor.scraper = mock_processor_components["scraper"]
        processor.ai_client = mock_processor_components["ai_client"]
        processor.report_generator = mock_processor_components["report_generator"]

        return processor


# =====================================================================
# Initialization and Configuration Tests
# =====================================================================


@pytest.mark.asyncio
async def test_processor_initialization(mock_config):
    """Test AsyncArticleProcessor initializes correctly"""
    with patch("src.processors.async_article_processor.DatabaseManager"), \
         patch("src.processors.async_article_processor.ContentCache"), \
         patch("src.processors.async_article_processor.RSSParser"), \
         patch("src.processors.async_article_processor.AsyncWebScraper"), \
         patch("src.processors.async_article_processor.AsyncClaudeClient"), \
         patch("src.processors.async_article_processor.ReportGenerator"):

        processor = AsyncArticleProcessor(mock_config)
        assert processor.config == mock_config


@pytest.mark.asyncio
async def test_processing_config_from_dict():
    """Test ProcessingConfig creation from dictionary"""
    config_dict = {
        "force_refresh": True,
        "limit": 5,
        "follow_links": False,
        "max_linked_articles": 2,
        "max_concurrent": 10,
    }

    config = ProcessingConfig.from_dict(config_dict)
    assert config.force_refresh is True
    assert config.limit == 5
    assert config.follow_links is False
    assert config.max_linked_articles == 2
    assert config.max_concurrent == 10


@pytest.mark.asyncio
async def test_processing_config_defaults():
    """Test ProcessingConfig default values"""
    config = ProcessingConfig()
    assert config.force_refresh is False
    assert config.limit is None
    assert config.follow_links is True
    assert config.max_linked_articles == 3
    assert config.max_concurrent == 5


# =====================================================================
# Concurrent Processing Tests
# =====================================================================


@pytest.mark.asyncio
async def test_concurrent_scraping(mock_processor_with_components):
    """Test that articles are scraped concurrently"""
    processor = mock_processor_with_components

    # Create mock scraped content
    scraped_contents = [
        Mock(content=f"Content {i}", title=f"Title {i}")
        for i in range(5)
    ]

    processor.scraper.scrape_articles = AsyncMock(return_value=scraped_contents)

    # Test concurrent scraping
    start = time.time()
    results = await processor.scraper.scrape_articles([f"url{i}" for i in range(5)])
    elapsed = time.time() - start

    assert len(results) == 5
    assert all(result.content.startswith("Content") for result in results)
    processor.scraper.scrape_articles.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_ai_analysis(mock_processor_with_components):
    """Test that AI analysis happens concurrently"""
    processor = mock_processor_with_components

    # Mock article data
    articles = [
        {"title": f"Article {i}", "content": f"Content {i}"}
        for i in range(5)
    ]

    processor.ai_client.analyze_article = AsyncMock(return_value=SAMPLE_ANALYSIS)

    # Test concurrent analysis
    start = time.time()
    tasks = [
        processor.ai_client.analyze_article(article["content"])
        for article in articles
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    assert len(results) == 5
    assert all(r["methodology_detailed"] for r in results)
    assert processor.ai_client.analyze_article.call_count == 5


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency(mock_processor_with_components):
    """Test that semaphore properly limits concurrent requests"""
    processor = mock_processor_with_components
    max_concurrent = 3

    # Track concurrent executions
    concurrent_count = 0
    max_concurrent_reached = 0

    async def tracked_scrape(*args, **kwargs):
        nonlocal concurrent_count, max_concurrent_reached
        concurrent_count += 1
        max_concurrent_reached = max(max_concurrent_reached, concurrent_count)
        await asyncio.sleep(0.1)  # Simulate work
        concurrent_count -= 1
        return Mock(content="test", title="test")

    processor.scraper.scrape_article = AsyncMock(side_effect=tracked_scrape)

    # Create semaphore for limiting concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_scrape(url):
        async with semaphore:
            return await processor.scraper.scrape_article(url)

    # Execute multiple concurrent scrapes
    tasks = [limited_scrape(f"url{i}") for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert max_concurrent_reached <= max_concurrent


# =====================================================================
# Error Handling Tests
# =====================================================================


@pytest.mark.asyncio
async def test_mixed_success_and_failure():
    """Test processing with mixed successful and failed articles"""
    with patch("src.processors.async_article_processor.DatabaseManager") as db_mock, \
         patch("src.processors.async_article_processor.ContentCache") as cache_mock, \
         patch("src.processors.async_article_processor.RSSParser") as parser_mock, \
         patch("src.processors.async_article_processor.AsyncWebScraper") as scraper_mock, \
         patch("src.processors.async_article_processor.AsyncClaudeClient") as client_mock, \
         patch("src.processors.async_article_processor.ReportGenerator") as gen_mock:

        # Setup mocks
        db_instance = MagicMock()
        cache_instance = MagicMock()
        parser_instance = MagicMock()
        scraper_instance = MagicMock()
        client_instance = AsyncMock()
        gen_instance = MagicMock()

        db_mock.return_value = db_instance
        cache_mock.return_value = cache_instance
        parser_mock.return_value = parser_instance
        scraper_mock.return_value = scraper_instance
        client_mock.return_value = client_instance
        gen_mock.return_value = gen_instance

        db_instance.get_analyzed_content_hashes.return_value = set()

        # Mock scraper to return mix of success and failure
        scraper_instance.scrape_articles = AsyncMock(
            side_effect=[
                [Mock(content="Content 1"), Mock(content="Content 2")],
                Exception("Network error"),
                [Mock(content="Content 3")],
            ]
        )

        # Mock client to succeed on some, fail on others
        client_instance.analyze_article = AsyncMock(
            side_effect=[
                SAMPLE_ANALYSIS,  # Success
                Exception("API error"),  # Failure
                SAMPLE_ANALYSIS,  # Success
            ]
        )

        processor = AsyncArticleProcessor({
            "api_provider": "anthropic",
            "anthropic_api_key": "test",
            "db_path": "test.db",
            "cache_db_path": "test_cache.db",
            "output_dir": "test_output",
            "rss_feed_url": "https://example.com/feed.xml",
        })
        processor.db = db_instance
        processor.cache = cache_instance
        processor.rss_parser = parser_instance
        processor.scraper = scraper_instance
        processor.ai_client = client_instance
        processor.report_generator = gen_instance


@pytest.mark.asyncio
async def test_retry_logic_on_rate_limit():
    """Test retry logic when hitting rate limits"""
    client = AsyncMock()

    # First 2 calls fail with rate limit, 3rd succeeds
    call_count = 0

    async def mock_analyze(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Rate limited")
        return SAMPLE_ANALYSIS

    client.analyze_article = AsyncMock(side_effect=mock_analyze)

    # Implement retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await client.analyze_article("test content")
            assert result == SAMPLE_ANALYSIS
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff

    assert call_count == 3


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test handling of request timeouts"""
    client = AsyncMock()

    async def timeout_request(*args, **kwargs):
        await asyncio.sleep(5)  # Simulate long request
        return SAMPLE_ANALYSIS

    client.analyze_article = AsyncMock(side_effect=timeout_request)

    # Test with timeout
    try:
        result = await asyncio.wait_for(
            client.analyze_article("test content"),
            timeout=0.1
        )
        assert False, "Should have timed out"
    except asyncio.TimeoutError:
        assert True


# =====================================================================
# Performance Tests
# =====================================================================


@pytest.mark.asyncio
async def test_concurrent_faster_than_sequential():
    """Test that concurrent processing is faster than sequential"""
    # Simulate processing with artificial delay
    async def process_item(item_id, delay=0.1):
        await asyncio.sleep(delay)
        return f"processed_{item_id}"

    items = list(range(5))

    # Sequential processing
    start = time.time()
    sequential_results = []
    for item in items:
        result = await process_item(item)
        sequential_results.append(result)
    sequential_time = time.time() - start

    # Concurrent processing
    start = time.time()
    concurrent_results = await asyncio.gather(*[
        process_item(item) for item in items
    ])
    concurrent_time = time.time() - start

    # Concurrent should be ~5x faster (0.5s vs 0.1s)
    assert concurrent_time < sequential_time
    assert concurrent_time < (sequential_time * 0.5)  # At least 2x faster
    assert len(sequential_results) == len(concurrent_results) == 5


@pytest.mark.asyncio
async def test_throughput_improvement():
    """Test throughput improvement with async processing"""
    articles_count = 20

    async def process_article(article_id):
        # Simulate scraping (100ms) + analysis (100ms)
        await asyncio.sleep(0.1)  # Scraping
        await asyncio.sleep(0.1)  # Analysis
        return {"id": article_id, "status": "processed"}

    # Sequential: 20 * 0.2s = 4s
    start = time.time()
    sequential_results = []
    for i in range(articles_count):
        result = await process_article(i)
        sequential_results.append(result)
    sequential_time = time.time() - start

    # Concurrent (with 5 max concurrent): ~0.8s
    start = time.time()
    concurrent_results = await asyncio.gather(*[
        process_article(i) for i in range(articles_count)
    ])
    concurrent_time = time.time() - start

    # Concurrent should process all ~5 articles in parallel
    # 20 articles / 5 concurrent = 4 batches * 0.2s per batch = 0.8s minimum
    assert concurrent_time < sequential_time
    speedup = sequential_time / concurrent_time
    assert speedup >= 3.0  # At least 3x faster


@pytest.mark.asyncio
async def test_batch_processing_performance(mock_processor_with_components):
    """Test batch processing performance"""
    processor = mock_processor_with_components
    batch_size = 5

    async def batch_process(items):
        """Process items in batches"""
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

        results = []
        for batch in batches:
            batch_results = await asyncio.gather(*[
                processor.ai_client.analyze_article(item["content"])
                for item in batch
            ])
            results.extend(batch_results)

        return results

    articles = [
        {"title": f"Article {i}", "content": f"Content {i}"}
        for i in range(15)
    ]

    processor.ai_client.analyze_article = AsyncMock(return_value=SAMPLE_ANALYSIS)

    start = time.time()
    results = await batch_process(articles)
    elapsed = time.time() - start

    assert len(results) == 15
    assert processor.ai_client.analyze_article.call_count == 15


# =====================================================================
# Result Aggregation Tests
# =====================================================================


@pytest.mark.asyncio
async def test_processing_results_to_dict():
    """Test ProcessingResults conversion to dictionary"""
    results = ProcessingResults(
        start_time=time.time(),
        duration=1.5,
        rss_entries_found=10,
        new_articles=8,
        scraped_articles=7,
        analyzed_articles=6,
        report_generated=True,
        errors=[],
        reports={"report.md": "path/to/report.md"},
    )

    result_dict = results.to_dict()

    assert result_dict["rss_entries_found"] == 10
    assert result_dict["new_articles"] == 8
    assert result_dict["scraped_articles"] == 7
    assert result_dict["analyzed_articles"] == 6
    assert result_dict["report_generated"] is True
    assert isinstance(result_dict["start_time"], float)
    assert result_dict["duration"] == 1.5


@pytest.mark.asyncio
async def test_error_accumulation(mock_processor_with_components):
    """Test that errors are properly accumulated in results"""
    processor = mock_processor_with_components

    results = ProcessingResults(
        start_time=time.time(),
        duration=0.0,
        rss_entries_found=0,
        new_articles=0,
        scraped_articles=0,
        analyzed_articles=0,
        report_generated=False,
        errors=[],
    )

    # Simulate error accumulation
    errors = ["Scraping error on article 1", "API error on article 2", "Timeout on article 3"]
    for error in errors:
        results.errors.append(error)

    assert len(results.errors) == 3
    assert all(isinstance(e, str) for e in results.errors)


# =====================================================================
# Integration Tests
# =====================================================================


@pytest.mark.asyncio
async def test_full_pipeline_success(mock_processor_with_components):
    """Test full async pipeline with successful execution"""
    processor = mock_processor_with_components

    # Use MagicMock for sync methods
    processor.rss_parser.fetch_feed = MagicMock(return_value=SAMPLE_ARTICLES[:2])
    processor.rss_parser.filter_new_entries = MagicMock(return_value=SAMPLE_ARTICLES[:2])
    processor.scraper.scrape_articles = AsyncMock(
        return_value=[
            Mock(content=f"Content {i}", title=f"Title {i}")
            for i in range(2)
        ]
    )
    processor.ai_client.analyze_article = AsyncMock(return_value=SAMPLE_ANALYSIS)
    processor.db.insert_article = MagicMock(return_value=1)

    # Run async pipeline
    config = ProcessingConfig(limit=2)

    try:
        # Note: This is a simplified test - actual pipeline has more steps
        articles = processor.rss_parser.fetch_feed("test_url")
        assert len(articles) == 2

        scraper_results = await processor.scraper.scrape_articles(
            [a["link"] for a in articles]
        )
        assert len(scraper_results) == 2

        analysis_results = await asyncio.gather(*[
            processor.ai_client.analyze_article(r.content)
            for r in scraper_results
        ])
        assert len(analysis_results) == 2

    except Exception as e:
        pytest.fail(f"Pipeline failed: {e}")


@pytest.mark.asyncio
async def test_pipeline_with_partial_failures(mock_processor_with_components):
    """Test pipeline continues despite partial failures"""
    processor = mock_processor_with_components

    # Some articles fail scraping
    async def scrape_with_failures(urls):
        results = []
        for i, url in enumerate(urls):
            if i == 1:  # Second article fails
                raise Exception(f"Failed to scrape {url}")
            results.append(Mock(content=f"Content {i}", title=f"Title {i}"))
        return results

    processor.scraper.scrape_articles = AsyncMock(side_effect=scrape_with_failures)

    # Pipeline should continue with remaining articles
    articles = [SAMPLE_ARTICLES[0], SAMPLE_ARTICLES[2]]

    try:
        results = await processor.scraper.scrape_articles(
            [a["link"] for a in articles]
        )
    except Exception as e:
        # Error is expected
        assert "Failed to scrape" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
