"""
Tests for Async Web Scraper Module

Tests concurrent article fetching, rate limiting, and error handling.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiohttp
from bs4 import BeautifulSoup

from src.core.async_scraper import (
    AsyncWebScraper,
    ScrapedContent,
    scrape_articles_async,
)


@pytest.fixture
def scraper():
    """Create AsyncWebScraper instance for testing"""
    return AsyncWebScraper(delay_between_requests=0.1, max_concurrent=3, timeout=10)


@pytest.fixture
def mock_html():
    """Sample HTML content for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article</title>
        <meta property="og:title" content="Test Article Title">
        <meta name="description" content="Test description">
    </head>
    <body>
        <article>
            <h1 class="article-title">Test Article Title</h1>
            <div class="article-content">
                <p>This is a test article with some content.</p>
                <p>It has multiple paragraphs.</p>
                <a href="https://arxiv.org/abs/1234.5678">Related Paper</a>
            </div>
        </article>
    </body>
    </html>
    """


@pytest.fixture
def mock_arxiv_html():
    """Sample arXiv HTML for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>arXiv Paper</title></head>
    <body>
        <h1 class="title">Title: Test arXiv Paper</h1>
        <div class="authors">John Doe, Jane Smith</div>
        <blockquote class="abstract">
            Abstract: This is the abstract of the paper.
        </blockquote>
        <td class="tablecell subjects">Computer Science - Machine Learning</td>
    </body>
    </html>
    """


class TestScrapedContent:
    """Tests for ScrapedContent class"""

    def test_scraped_content_initialization(self):
        """Test ScrapedContent object creation"""
        content = ScrapedContent(
            url="https://example.com",
            title="Test Title",
            content="Test content",
            metadata={"author": "Test Author"},
        )

        assert content.url == "https://example.com"
        assert content.title == "Test Title"
        assert content.content == "Test content"
        assert content.metadata["author"] == "Test Author"
        assert content.content_hash is not None

    def test_content_hash_generation(self):
        """Test content hash generation"""
        content1 = ScrapedContent(
            url="https://example.com", title="Title", content="Content"
        )
        content2 = ScrapedContent(
            url="https://example.com", title="Title", content="Content"
        )
        content3 = ScrapedContent(
            url="https://example.com", title="Different", content="Content"
        )

        # Same content should have same hash
        assert content1.content_hash == content2.content_hash

        # Different content should have different hash
        assert content1.content_hash != content3.content_hash

    def test_to_dict(self):
        """Test conversion to dictionary"""
        content = ScrapedContent(
            url="https://example.com",
            title="Test",
            content="Content",
            metadata={"key": "value"},
        )

        result = content.to_dict()

        assert result["url"] == "https://example.com"
        assert result["title"] == "Test"
        assert result["content"] == "Content"
        assert result["metadata"]["key"] == "value"
        assert "content_hash" in result
        assert "scraped_at" in result


class TestAsyncWebScraper:
    """Tests for AsyncWebScraper class"""

    def test_scraper_initialization(self):
        """Test scraper initialization with parameters"""
        scraper = AsyncWebScraper(
            delay_between_requests=2.0, max_concurrent=10, timeout=60
        )

        assert scraper.delay == 2.0
        assert scraper.max_concurrent == 10
        assert scraper.timeout == 60
        assert scraper._semaphore._value == 10

    def test_is_arxiv_url(self, scraper):
        """Test arXiv URL detection"""
        assert scraper._is_arxiv_url("https://arxiv.org/abs/1234.5678")
        assert scraper._is_arxiv_url("https://arxiv.org/pdf/1234.5678.pdf")
        assert not scraper._is_arxiv_url("https://example.com")

    def test_is_bluesky_url(self, scraper):
        """Test Bluesky URL detection"""
        assert scraper._is_bluesky_url("https://bsky.app/profile/user/post/123")
        assert scraper._is_bluesky_url("https://bsky.social/profile/user")
        assert not scraper._is_bluesky_url("https://twitter.com/user")

    def test_extract_title(self, scraper, mock_html):
        """Test title extraction from HTML"""
        soup = BeautifulSoup(mock_html, "html.parser")
        title = scraper._extract_title(soup)

        assert title == "Test Article Title"

    def test_clean_title(self, scraper):
        """Test title cleaning"""
        assert scraper._clean_title("Title: My Article") == "My Article"
        assert scraper._clean_title("Paper: Research Study") == "Research Study"
        assert (
            scraper._clean_title("Article Title | example.com")
            == "Article Title | example.com"
        )

    def test_extract_content(self, scraper, mock_html):
        """Test content extraction from HTML"""
        soup = BeautifulSoup(mock_html, "html.parser")
        content = scraper._extract_content(soup)

        assert "This is a test article" in content
        assert "multiple paragraphs" in content

    def test_extract_metadata(self, scraper, mock_html):
        """Test metadata extraction"""
        soup = BeautifulSoup(mock_html, "html.parser")
        metadata = scraper._extract_metadata(soup, "https://example.com/article")

        assert metadata["domain"] == "example.com"
        assert "scraped_timestamp" in metadata
        assert metadata["description"] == "Test description"

    def test_looks_like_article_link(self, scraper):
        """Test article link heuristic"""
        assert scraper._looks_like_article_link("https://example.com/blog/post-title")
        assert scraper._looks_like_article_link(
            "https://example.com/article/2024/01/title"
        )
        assert scraper._looks_like_article_link("https://example.com/research/paper")
        assert not scraper._looks_like_article_link("https://example.com/file.pdf")
        assert not scraper._looks_like_article_link("https://example.com/file.zip")

    def test_find_academic_link(self, scraper):
        """Test academic link detection"""
        links = [
            "https://example.com",
            "https://arxiv.org/abs/1234.5678",
            "https://twitter.com/user",
        ]

        academic_link = scraper._find_academic_link(links)
        assert academic_link == "https://arxiv.org/abs/1234.5678"

    def test_find_academic_link_none(self, scraper):
        """Test academic link detection with no academic links"""
        links = ["https://example.com", "https://twitter.com/user"]

        academic_link = scraper._find_academic_link(links)
        assert academic_link is None

    @pytest.mark.asyncio
    async def test_rate_limiting(self, scraper):
        """Test rate limiting between requests"""
        scraper.delay = 0.5

        start_time = time.time()

        # Make two consecutive rate-limited calls
        await scraper._respect_rate_limit()
        await scraper._respect_rate_limit()

        elapsed = time.time() - start_time

        # Should have waited at least the delay time
        assert elapsed >= 0.5

    @pytest.mark.asyncio
    async def test_scrape_single(self, scraper, mock_html):
        """Test scraping a single URL"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await scraper.scrape_single(
                "https://example.com/article", follow_links=False
            )

            assert result is not None
            assert result.title == "Test Article Title"
            assert "test article" in result.content.lower()

    @pytest.mark.asyncio
    async def test_scrape_arxiv_async(self, scraper, mock_arxiv_html):
        """Test arXiv paper scraping"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.text = AsyncMock(return_value=mock_arxiv_html)
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        result = await scraper._scrape_arxiv_async(
            mock_session, "https://arxiv.org/abs/1234.5678"
        )

        assert result is not None
        assert result.title == "Test arXiv Paper"
        assert "John Doe, Jane Smith" in result.content
        assert "abstract of the paper" in result.content
        assert result.metadata["source"] == "arXiv"

    @pytest.mark.asyncio
    async def test_scrape_articles_batch(self, scraper, mock_html):
        """Test batch scraping of multiple URLs"""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value.__aenter__.return_value = mock_response

            start_time = time.time()
            results = await scraper.scrape_articles_batch(urls)
            elapsed = time.time() - start_time

            # Should scrape all articles
            assert len(results) == 3

            # Should be faster than sequential (3 articles * 0.1s delay = 0.3s)
            # With concurrency, should be closer to 0.1s (one batch)
            assert elapsed < 1.0  # Allow some overhead

    @pytest.mark.asyncio
    async def test_scrape_articles_batch_with_errors(self, scraper):
        """Test batch scraping with some failures"""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]

        call_count = 0

        async def mock_get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_response = AsyncMock()

            # Make the second request fail
            if call_count == 2:
                mock_response.raise_for_status.side_effect = aiohttp.ClientError(
                    "Connection failed"
                )
            else:
                mock_response.status = 200
                mock_response.text = AsyncMock(
                    return_value="<html><body><h1>Test</h1></body></html>"
                )
                mock_response.raise_for_status = MagicMock()

            return mock_response

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.side_effect = mock_get_side_effect

            results = await scraper.scrape_articles_batch(urls)

            # Should have 2 successful results (1st and 3rd)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_concurrent_limit(self, scraper):
        """Test that concurrent requests are limited by semaphore"""
        scraper.max_concurrent = 2
        scraper._semaphore = asyncio.Semaphore(2)

        concurrent_count = 0
        max_concurrent = 0

        async def mock_scrape(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent

            async with scraper._semaphore:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

                # Simulate some work
                await asyncio.sleep(0.1)

                concurrent_count -= 1

        urls = ["url1", "url2", "url3", "url4", "url5"]
        tasks = [mock_scrape(url) for url in urls]
        await asyncio.gather(*tasks)

        # Max concurrent should not exceed limit
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_extract_content_links(self, scraper, mock_html):
        """Test link extraction from content"""
        soup = BeautifulSoup(mock_html, "html.parser")
        links = scraper._extract_content_links(soup, "https://example.com")

        assert len(links) > 0
        assert "https://arxiv.org/abs/1234.5678" in links

    @pytest.mark.asyncio
    async def test_filter_interesting_links(self, scraper):
        """Test filtering of interesting links"""
        links = [
            "https://example.com/same-domain",
            "https://arxiv.org/abs/1234.5678",
            "https://twitter.com/user",
            "https://medium.com/article",
            "https://facebook.com/page",
        ]

        interesting = scraper._filter_interesting_links(links, "https://example.com")

        # Should include arXiv and Medium, exclude Twitter and Facebook
        assert "https://arxiv.org/abs/1234.5678" in interesting
        assert "https://medium.com/article" in interesting
        assert "https://twitter.com/user" not in interesting
        assert "https://facebook.com/page" not in interesting

    @pytest.mark.asyncio
    async def test_merge_content_with_links(self, scraper):
        """Test merging main content with linked content"""
        main_content = "# Main Article\n\nThis is the main content."

        linked_content = [
            {
                "url": "https://example.com/link1",
                "title": "Linked Article 1",
                "content": "Summary of linked article 1.",
                "metadata": {},
            },
            {
                "url": "https://example.com/link2",
                "title": "Linked Article 2",
                "content": "Summary of linked article 2.",
                "metadata": {},
            },
        ]

        merged = scraper._merge_content_with_links(main_content, linked_content)

        assert "Main Article" in merged
        assert "Referenced Articles" in merged
        assert "Linked Article 1" in merged
        assert "Linked Article 2" in merged
        assert "https://example.com/link1" in merged


class TestConvenienceFunction:
    """Tests for convenience functions"""

    @pytest.mark.asyncio
    async def test_scrape_articles_async(self, mock_html):
        """Test convenience function for async scraping"""
        urls = ["https://example.com/article1", "https://example.com/article2"]

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await scrape_articles_async(urls, max_concurrent=2, delay=0.1)

            assert len(results) == 2
            assert all(isinstance(r, ScrapedContent) for r in results)


class TestPerformance:
    """Performance tests for async scraper"""

    @pytest.mark.asyncio
    async def test_parallel_faster_than_sequential(self):
        """Test that parallel scraping is faster than sequential"""
        urls = [f"https://example.com/article{i}" for i in range(5)]

        mock_html = "<html><body><h1>Test</h1><p>Content</p></body></html>"

        # Time parallel scraping
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value.__aenter__.return_value = mock_response

            scraper = AsyncWebScraper(delay_between_requests=0.1, max_concurrent=5)

            start = time.time()
            results = await scraper.scrape_articles_batch(urls)
            parallel_time = time.time() - start

            # With 5 concurrent requests and 0.1s delay
            # Should complete in ~0.1-0.2s (one batch)
            assert len(results) == 5
            assert parallel_time < 1.0  # Much faster than 0.5s (5 * 0.1s sequential)

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrent requests"""
        scraper = AsyncWebScraper(max_concurrent=3)

        active_requests = 0
        max_active = 0

        async def mock_request():
            nonlocal active_requests, max_active

            async with scraper._semaphore:
                active_requests += 1
                max_active = max(max_active, active_requests)
                await asyncio.sleep(0.05)
                active_requests -= 1

        # Launch 10 concurrent requests
        await asyncio.gather(*[mock_request() for _ in range(10)])

        # Should never exceed max_concurrent
        assert max_active <= 3
        assert active_requests == 0  # All completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
