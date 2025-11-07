"""
Async Web Scraper Module

Provides async/concurrent version of WebScraper for parallel article fetching.
Uses aiohttp for non-blocking HTTP requests with connection pooling and rate limiting.
"""

import asyncio
import hashlib
import logging
import time
from typing import Optional, List, Dict

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup

from .scraper_base import ScraperBase

logger = logging.getLogger(__name__)


class ScrapedContent:
    """Represents scraped article content (async-compatible version)"""

    def __init__(
        self, url: str, title: str = "", content: str = "", metadata: Optional[Dict] = None
    ):
        self.url = url
        self.title = title
        self.content = content
        self.metadata = metadata or {}
        self.scraped_at = time.time()
        self.content_hash = self.generate_content_hash()

    def generate_content_hash(self) -> str:
        """Generate a hash based on the actual article content for accurate duplicate detection"""
        content_for_hash = f"{self.url}{self.title}{self.content[:2000]}"
        return hashlib.md5(content_for_hash.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "scraped_at": self.scraped_at,
            "content_hash": self.content_hash,
        }


class AsyncWebScraper(ScraperBase):
    """Async web scraper with support for concurrent requests and academic publishers"""

    def __init__(
        self,
        delay_between_requests: float = 1.0,
        max_concurrent: int = 5,
        timeout: int = 30,
        rate_limit_rps: float = 10.0,
        rate_limit_burst: int = 20,
    ):
        """
        Initialize async web scraper

        Args:
            delay_between_requests: Minimum delay between requests (rate limiting)
            max_concurrent: Maximum number of concurrent requests
            timeout: Request timeout in seconds
            rate_limit_rps: Maximum requests per second (default: 10)
            rate_limit_burst: Maximum burst size (default: 20)
        """
        self.delay = delay_between_requests
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.last_request_time = 0
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limit_lock = asyncio.Lock()

        # Initialize rate limiter to prevent DoS attacks and IP bans
        self.rate_limiter = AsyncLimiter(
            max_rate=rate_limit_rps,
            time_period=1.0  # 1 second
        )

        logger.info(
            f"Rate limiter initialized: {rate_limit_rps} req/s, "
            f"burst={rate_limit_burst}, max_concurrent={max_concurrent}"
        )

    def _create_session(self) -> ClientSession:
        """Create aiohttp session with connection pooling and headers"""
        timeout_config = ClientTimeout(total=self.timeout, connect=10, sock_read=20)

        # Connection pooling configuration
        connector = TCPConnector(
            limit=self.max_concurrent * 2,  # Total connection pool size
            limit_per_host=5,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            enable_cleanup_closed=True,
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        return ClientSession(
            connector=connector, timeout=timeout_config, headers=headers
        )

    async def _respect_rate_limit(self):
        """Implement async delay between requests"""
        async with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay:
                sleep_time = self.delay - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.time()

    async def scrape_article_async(
        self,
        session: ClientSession,
        url: str,
        follow_links: bool = True,
        max_linked_articles: int = 3,
    ) -> Optional[ScrapedContent]:
        """
        Asynchronously scrape article content from URL

        Args:
            session: aiohttp ClientSession
            url: Article URL
            follow_links: Whether to follow and analyze links found in the content
            max_linked_articles: Maximum number of linked articles to analyze

        Returns:
            ScrapedContent object or None if scraping fails
        """
        async with self._semaphore:
            # Apply rate limiting
            async with self.rate_limiter:
                try:
                    await self._respect_rate_limit()

                    logger.info(f"Scraping article: {url}")

                    # Handle special cases for known publishers
                    if self.is_arxiv_url(url):
                        return await self._scrape_arxiv_async(session, url)
                    elif self.is_bluesky_url(url):
                        return await self._scrape_bluesky_post_async(
                            session, url, follow_links, max_linked_articles
                        )

                    # General scraping approach
                    async with session.get(url) as response:
                        response.raise_for_status()
                        html = await response.text()

                    soup = BeautifulSoup(html, "html.parser")

                    # Extract title
                    title = self.extract_title(soup)
                    logger.debug(f"Extracted title: '{title}' from {url}")

                    # Extract main content
                    content = self.extract_content(soup)

                    # Extract metadata
                    metadata = self.extract_metadata(soup, url)
                    metadata["extracted_title"] = title
                    metadata["title_extraction_method"] = "content_analysis"

                    if not content:
                        logger.warning(f"No content extracted from {url}")
                        return None

                    # Follow and analyze links if requested
                    if follow_links:
                        linked_content = await self._analyze_embedded_links_async(
                            session, soup, url, max_linked_articles
                        )
                        if linked_content:
                            content = self.merge_content_with_links(content, linked_content)
                            metadata["linked_articles"] = [
                                link["url"] for link in linked_content
                            ]

                    logger.info(f"Successfully scraped {len(content)} characters from {url}")

                    return ScrapedContent(
                        url=url, title=title, content=content, metadata=metadata
                    )

                except aiohttp.ClientError as e:
                    logger.error(f"Network error scraping {url}: {e}")
                    return None
                except asyncio.TimeoutError:
                    logger.error(f"Timeout scraping {url}")
                    return None
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    return None

    async def scrape_articles_batch(
        self, urls: List[str], max_concurrent: Optional[int] = None
    ) -> List[ScrapedContent]:
        """
        Scrape multiple articles concurrently

        Args:
            urls: List of URLs to scrape
            max_concurrent: Override default max_concurrent setting

        Returns:
            List of successfully scraped ScrapedContent objects
        """
        if max_concurrent:
            self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"Starting async batch scrape of {len(urls)} articles")
        start_time = time.time()

        async with self._create_session() as session:
            tasks = [
                self.scrape_article_async(session, url, follow_links=True)
                for url in urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures and exceptions
        scraped_articles = []
        errors = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception scraping {urls[i]}: {result}")
                errors += 1
            elif result is not None:
                scraped_articles.append(result)
            else:
                errors += 1

        elapsed = time.time() - start_time
        success_rate = len(scraped_articles) / len(urls) * 100 if urls else 0

        logger.info(
            f"Batch scraping completed in {elapsed:.2f}s: "
            f"{len(scraped_articles)}/{len(urls)} articles scraped "
            f"({success_rate:.1f}% success rate, {errors} errors)"
        )

        return scraped_articles

    async def scrape_single(
        self, url: str, follow_links: bool = True, max_linked_articles: int = 3
    ) -> Optional[ScrapedContent]:
        """
        Convenience method to scrape a single URL (creates its own session)

        Args:
            url: Article URL
            follow_links: Whether to follow and analyze links
            max_linked_articles: Maximum number of linked articles to analyze

        Returns:
            ScrapedContent object or None
        """
        async with self._create_session() as session:
            return await self.scrape_article_async(
                session, url, follow_links, max_linked_articles
            )

    # Publisher-specific async methods

    async def _scrape_arxiv_async(
        self, session: ClientSession, url: str
    ) -> Optional[ScrapedContent]:
        """Async special handling for arXiv papers"""
        try:
            # Convert to abstract page if it's a PDF link
            if "/pdf/" in url:
                url = url.replace("/pdf/", "/abs/")

            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()

            soup = BeautifulSoup(html, "html.parser")

            # Use base class parsing
            parsed = self.parse_arxiv_content(soup, url)

            return ScrapedContent(
                url=url,
                title=parsed["title"],
                content=parsed["content"],
                metadata=parsed["metadata"],
            )

        except Exception as e:
            logger.error(f"Error scraping arXiv paper {url}: {e}")
            return None

    async def _scrape_bluesky_post_async(
        self,
        session: ClientSession,
        url: str,
        follow_links: bool = True,
        max_linked_articles: int = 3,
    ) -> Optional[ScrapedContent]:
        """Async special handling for Bluesky posts"""
        try:
            logger.info(f"Scraping Bluesky post: {url}")

            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()

            soup = BeautifulSoup(html, "html.parser")

            # Extract post content/text
            post_text = self.extract_bluesky_post_text(soup)

            # Look for embedded links
            embedded_links = self.extract_bluesky_links(soup, post_text)

            # Check if any links are academic papers
            academic_link = self.find_academic_link(embedded_links)

            if academic_link:
                logger.info(f"Found academic paper link in Bluesky post: {academic_link}")

                # Scrape the actual academic paper
                if self.is_arxiv_url(academic_link):
                    paper_content = await self._scrape_arxiv_async(session, academic_link)
                else:
                    paper_content = await self._scrape_general_academic_link_async(
                        session, academic_link
                    )

                if paper_content:
                    enhanced_content = self._enhance_with_bluesky_context(
                        paper_content, url, post_text, academic_link
                    )
                    return enhanced_content

            # If no academic paper found, create content from Bluesky post
            logger.info("No academic paper found, using Bluesky post content")

            title = self.extract_bluesky_title(soup, post_text)

            content = "# Bluesky Post Analysis\n\n"
            content += f"**Original Bluesky Post:** {url}\n\n"
            content += f"## Post Content\n\n{post_text}\n\n"

            if embedded_links:
                content += "## Embedded Links\n\n"
                for link in embedded_links:
                    content += f"- {link}\n"
                content += "\n"

            metadata = {
                "source": "Bluesky",
                "original_post_url": url,
                "post_type": "social_media",
                "embedded_links": embedded_links,
            }

            return ScrapedContent(url=url, title=title, content=content, metadata=metadata)

        except Exception as e:
            logger.error(f"Error scraping Bluesky post {url}: {e}")
            return None

    async def _scrape_general_academic_link_async(
        self, session: ClientSession, url: str
    ) -> Optional[ScrapedContent]:
        """Async scraping for non-arXiv academic sources"""
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()

            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title = self.extract_title(soup)

            # Extract content
            content = self.extract_content(soup)

            # Extract metadata
            metadata = self.extract_metadata(soup, url)
            metadata["paper_type"] = "academic"
            metadata["source"] = "academic_link"

            if content:
                return ScrapedContent(
                    url=url, title=title, content=content, metadata=metadata
                )

        except Exception as e:
            logger.error(f"Error scraping academic link {url}: {e}")

        return None

    def _enhance_with_bluesky_context(
        self,
        paper_content: ScrapedContent,
        bluesky_url: str,
        post_text: str,
        paper_url: str,
    ) -> ScrapedContent:
        """Enhance academic paper content with Bluesky post context"""
        enhanced_content = f"# {paper_content.title}\n\n"
        enhanced_content += f"**Shared via Bluesky:** {bluesky_url}\n"
        enhanced_content += f"**Original Paper:** {paper_url}\n\n"

        if post_text and post_text != "Could not extract post text from Bluesky":
            enhanced_content += "## Social Media Context\n\n"
            enhanced_content += f"**Bluesky Post Commentary:**\n{post_text}\n\n"
            enhanced_content += "---\n\n"

        enhanced_content += paper_content.content

        enhanced_metadata = paper_content.metadata.copy()
        enhanced_metadata["shared_via"] = "Bluesky"
        enhanced_metadata["bluesky_post_url"] = bluesky_url
        enhanced_metadata["bluesky_post_text"] = post_text
        enhanced_metadata["discovery_method"] = "social_media_sharing"

        return ScrapedContent(
            url=paper_url,
            title=paper_content.title,
            content=enhanced_content,
            metadata=enhanced_metadata,
        )

    async def _analyze_embedded_links_async(
        self,
        session: ClientSession,
        soup: BeautifulSoup,
        base_url: str,
        max_links: int,
    ) -> List[Dict]:
        """
        Async extract and analyze links found in the main content

        Args:
            session: aiohttp ClientSession
            soup: BeautifulSoup object of the main page
            base_url: Base URL for resolving relative links
            max_links: Maximum number of links to analyze

        Returns:
            List of dictionaries containing analyzed link content
        """
        try:
            # Extract links from the main content area
            content_links = self.extract_content_links(soup, base_url)

            # Filter and prioritize interesting links
            interesting_links = self.filter_interesting_links(content_links, base_url)

            # Limit the number of links to analyze
            links_to_analyze = interesting_links[:max_links]

            # Scrape all links concurrently
            tasks = [
                self._scrape_linked_article_async(session, link_url)
                for link_url in links_to_analyze
                if link_url != base_url
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            analyzed_links = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Exception analyzing link: {result}")
                    continue

                if result:
                    analyzed_links.append(
                        {
                            "url": links_to_analyze[i],
                            "title": result.title,
                            "content": result.content[:2000],  # Limit content for summary
                            "metadata": result.metadata,
                        }
                    )
                    logger.info(f"Successfully analyzed linked content: {result.title}")

            return analyzed_links

        except Exception as e:
            logger.error(f"Error analyzing embedded links: {e}")
            return []

    async def _scrape_linked_article_async(
        self, session: ClientSession, url: str
    ) -> Optional[ScrapedContent]:
        """Async scrape content from a linked article without following further links"""
        try:
            return await self.scrape_article_async(
                session, url, follow_links=False, max_linked_articles=0
            )
        except Exception as e:
            logger.error(f"Error scraping linked article {url}: {e}")
            return None


# Convenience function for backwards compatibility
async def scrape_articles_async(
    urls: List[str], max_concurrent: int = 5, delay: float = 1.0
) -> List[ScrapedContent]:
    """
    Convenience function to scrape multiple URLs asynchronously

    Args:
        urls: List of URLs to scrape
        max_concurrent: Maximum number of concurrent requests
        delay: Delay between requests in seconds

    Returns:
        List of ScrapedContent objects
    """
    scraper = AsyncWebScraper(
        delay_between_requests=delay, max_concurrent=max_concurrent
    )
    return await scraper.scrape_articles_batch(urls)
