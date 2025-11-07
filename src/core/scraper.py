"""
Web Scraper Module

Handles scraping full article content from URLs, with support for
various article formats and academic publishers.
"""

import hashlib
import logging
import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .scraper_base import ScraperBase

logger = logging.getLogger(__name__)


class ArticleContent:
    """Represents scraped article content"""

    def __init__(
        self, url: str, title: str = "", content: str = "", metadata: dict | None = None
    ):
        self.url = url
        self.title = title
        self.content = content
        self.metadata = metadata or {}
        self.scraped_at = time.time()
        self.content_hash = self.generate_content_hash()

    def generate_content_hash(self) -> str:
        """Generate a hash based on the actual article content for accurate duplicate detection"""
        # Create hash from the combination of URL, title, and significant portion of content
        # This ensures we catch duplicates even if RSS metadata slightly differs
        content_for_hash = f"{self.url}{self.title}{self.content[:2000]}"  # Use first 2000 chars
        return hashlib.md5(content_for_hash.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "scraped_at": self.scraped_at,
            "content_hash": self.content_hash,
        }


class WebScraper(ScraperBase):
    """Web scraper with support for academic publishers and various article formats"""

    def __init__(self, delay_between_requests: float = 1.0):
        self.delay = delay_between_requests
        self.session = self._create_session()
        self.last_request_time = 0

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy and headers"""
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Headers to appear as a legitimate browser
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        return session

    def _respect_rate_limit(self):
        """Implement delay between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay:
            sleep_time = self.delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def scrape_article(
        self,
        url: str,
        timeout: int = 30,
        follow_links: bool = True,
        max_linked_articles: int = 3,
    ) -> ArticleContent | None:
        """
        Scrape article content from URL with optional link following

        Args:
            url: Article URL
            timeout: Request timeout in seconds
            follow_links: Whether to follow and analyze links found in the content
            max_linked_articles: Maximum number of linked articles to analyze

        Returns:
            ArticleContent object or None if scraping fails
        """
        try:
            self._respect_rate_limit()

            logger.info(f"Scraping article: {url}")

            # Handle special cases for known publishers
            if self._is_arxiv_url(url):
                return self._scrape_arxiv(url, timeout)
            elif self._is_bluesky_url(url):
                return self._scrape_bluesky_post(
                    url, timeout, follow_links, max_linked_articles
                )

            # General scraping approach
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title using improved extraction logic
            title = self.extract_title(soup)

            # Log title extraction for debugging
            logger.debug(f"Extracted title: '{title}' from {url}")

            # Extract main content
            content = self.extract_content(soup)

            # Extract metadata
            metadata = self.extract_metadata(soup, url)
            
            # Add extracted title to metadata for reference
            metadata["extracted_title"] = title
            metadata["title_extraction_method"] = "content_analysis"

            if not content:
                logger.warning(f"No content extracted from {url}")
                return None

            # Follow and analyze links if requested
            if follow_links:
                linked_content = self._analyze_embedded_links(
                    soup, url, max_linked_articles, timeout
                )
                if linked_content:
                    content = self._merge_content_with_links(content, linked_content)
                    metadata["linked_articles"] = [
                        link["url"] for link in linked_content
                    ]

            logger.info(f"Successfully scraped {len(content)} characters from {url}")

            return ArticleContent(
                url=url, title=title, content=content, metadata=metadata
            )

        except requests.RequestException as e:
            logger.error(f"Network error scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def _is_arxiv_url(self, url: str) -> bool:
        """Check if URL is from arXiv"""
        return self.is_arxiv_url(url)

    def _is_bluesky_url(self, url: str) -> bool:
        """Check if URL is from Bluesky"""
        return self.is_bluesky_url(url)

    def _scrape_arxiv(self, url: str, timeout: int) -> ArticleContent | None:
        """Special handling for arXiv papers"""
        try:
            # Convert to abstract page if it's a PDF link
            if "/pdf/" in url:
                url = url.replace("/pdf/", "/abs/")

            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Use base class parsing
            parsed = self.parse_arxiv_content(soup, url)

            return ArticleContent(
                url=url,
                title=parsed["title"],
                content=parsed["content"],
                metadata=parsed["metadata"],
            )

        except Exception as e:
            logger.error(f"Error scraping arXiv paper {url}: {e}")
            return None

    def _scrape_bluesky_post(
        self,
        url: str,
        timeout: int,
        follow_links: bool = True,
        max_linked_articles: int = 3,
    ) -> ArticleContent | None:
        """Special handling for Bluesky posts that may contain academic paper links"""
        try:
            logger.info(f"Scraping Bluesky post: {url}")

            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract post content/text
            post_text = self.extract_bluesky_post_text(soup)

            # Look for embedded links, especially arXiv papers
            embedded_links = self.extract_bluesky_links(soup, post_text)

            # Check if any links are academic papers (arXiv, etc.)
            academic_link = self.find_academic_link(embedded_links)

            if academic_link:
                logger.info(
                    f"Found academic paper link in Bluesky post: {academic_link}"
                )

                # Scrape the actual academic paper
                if self._is_arxiv_url(academic_link):
                    paper_content = self._scrape_arxiv(academic_link, timeout)
                else:
                    # Try general scraping for other academic sources
                    paper_content = self._scrape_general_academic_link(
                        academic_link, timeout
                    )

                if paper_content:
                    # Enhance the content with Bluesky context
                    enhanced_content = self._enhance_with_bluesky_context(
                        paper_content, url, post_text, academic_link
                    )
                    return enhanced_content

            # If no academic paper found, create content from the Bluesky post itself
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

            return ArticleContent(
                url=url, title=title, content=content, metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error scraping Bluesky post {url}: {e}")
            return None


    def _scrape_general_academic_link(
        self, url: str, timeout: int
    ) -> ArticleContent | None:
        """Scrape academic content from non-arXiv sources"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title = self.extract_title(soup)

            # Extract content using academic-specific selectors
            content = self.extract_content(soup)

            # Extract metadata
            metadata = self.extract_metadata(soup, url)
            metadata["paper_type"] = "academic"
            metadata["source"] = "academic_link"

            if content:
                return ArticleContent(
                    url=url, title=title, content=content, metadata=metadata
                )

        except Exception as e:
            logger.error(f"Error scraping academic link {url}: {e}")

        return None

    def _enhance_with_bluesky_context(
        self,
        paper_content: ArticleContent,
        bluesky_url: str,
        post_text: str,
        paper_url: str,
    ) -> ArticleContent:
        """Enhance academic paper content with Bluesky post context"""

        # Add Bluesky context to the content
        enhanced_content = f"# {paper_content.title}\n\n"
        enhanced_content += f"**Shared via Bluesky:** {bluesky_url}\n"
        enhanced_content += f"**Original Paper:** {paper_url}\n\n"

        if post_text and post_text != "Could not extract post text from Bluesky":
            enhanced_content += "## Social Media Context\n\n"
            enhanced_content += f"**Bluesky Post Commentary:**\n{post_text}\n\n"
            enhanced_content += "---\n\n"

        # Add the original paper content
        enhanced_content += paper_content.content

        # Update metadata
        enhanced_metadata = paper_content.metadata.copy()
        enhanced_metadata["shared_via"] = "Bluesky"
        enhanced_metadata["bluesky_post_url"] = bluesky_url
        enhanced_metadata["bluesky_post_text"] = post_text
        enhanced_metadata["discovery_method"] = "social_media_sharing"

        return ArticleContent(
            url=paper_url,  # Use the paper URL as primary
            title=paper_content.title,
            content=enhanced_content,
            metadata=enhanced_metadata,
        )


    def batch_scrape(
        self, urls: list[str], max_articles: int | None = None
    ) -> list[ArticleContent]:
        """
        Scrape multiple articles with progress tracking

        Args:
            urls: List of URLs to scrape
            max_articles: Maximum number of articles to scrape

        Returns:
            List of successfully scraped ArticleContent objects
        """
        if max_articles:
            urls = urls[:max_articles]

        results = []
        total = len(urls)

        logger.info(f"Starting batch scrape of {total} articles")

        for i, url in enumerate(urls, 1):
            try:
                logger.info(f"Scraping article {i}/{total}: {url}")

                content = self.scrape_article(url)
                if content:
                    results.append(content)
                    logger.info(f"Successfully scraped article {i}/{total}")
                else:
                    logger.warning(f"Failed to scrape article {i}/{total}")

            except Exception as e:
                logger.error(f"Error scraping article {i}/{total} ({url}): {e}")
                continue

        success_rate = len(results) / total * 100 if total > 0 else 0
        logger.info(
            f"Batch scraping completed: {len(results)}/{total} articles scraped ({success_rate:.1f}% success rate)"
        )

        return results

    def _analyze_embedded_links(
        self, soup: BeautifulSoup, base_url: str, max_links: int, timeout: int
    ) -> list[dict]:
        """
        Extract and analyze links found in the main content

        Args:
            soup: BeautifulSoup object of the main page
            base_url: Base URL for resolving relative links
            max_links: Maximum number of links to analyze
            timeout: Request timeout for each link

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

            analyzed_links = []

            for link_url in links_to_analyze:
                try:
                    logger.info(f"Analyzing linked content: {link_url}")

                    # Skip if it's the same as the base URL
                    if link_url == base_url:
                        continue

                    # Scrape the linked content
                    linked_content = self._scrape_linked_article(link_url, timeout)

                    if linked_content:
                        analyzed_links.append(
                            {
                                "url": link_url,
                                "title": linked_content.title,
                                "content": linked_content.content[
                                    :2000
                                ],  # Limit content for summary
                                "metadata": linked_content.metadata,
                            }
                        )
                        logger.info(
                            f"Successfully analyzed linked content: {linked_content.title}"
                        )
                    else:
                        logger.warning(f"Failed to analyze linked content: {link_url}")

                except Exception as e:
                    logger.error(f"Error analyzing link {link_url}: {e}")
                    continue

            return analyzed_links

        except Exception as e:
            logger.error(f"Error analyzing embedded links: {e}")
            return []


    def _scrape_linked_article(self, url: str, timeout: int) -> ArticleContent | None:
        """Scrape content from a linked article without following further links"""
        try:
            # Scrape without following additional links to avoid infinite recursion
            return self.scrape_article(url, timeout, follow_links=False)
        except Exception as e:
            logger.error(f"Error scraping linked article {url}: {e}")
            return None

