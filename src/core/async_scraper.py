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
from urllib.parse import urlparse

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from markdownify import markdownify as md

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


class AsyncWebScraper:
    """Async web scraper with support for concurrent requests and academic publishers"""

    def __init__(
        self,
        delay_between_requests: float = 1.0,
        max_concurrent: int = 5,
        timeout: int = 30,
    ):
        """
        Initialize async web scraper

        Args:
            delay_between_requests: Minimum delay between requests (rate limiting)
            max_concurrent: Maximum number of concurrent requests
            timeout: Request timeout in seconds
        """
        self.delay = delay_between_requests
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.last_request_time = 0
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limit_lock = asyncio.Lock()

        # Content selectors (same as sync version)
        self.content_selectors = [
            # Academic papers
            "div.ltx_page_main",  # arXiv
            "div.abstract-content",
            "div.article-body",
            "div.fulltext-view",  # IEEE
            "div#body",
            "div.body",
            # Blog posts and articles
            "article",
            "main",
            "div.post-content",
            "div.entry-content",
            "div.article-content",
            "div.content",
            "div#content",
            "div.main-content",
            # Fallback selectors
            'div[role="main"]',
            "div.primary",
            "div.text",
            "div.story-body",
        ]

        # Selectors to remove
        self.remove_selectors = [
            "nav",
            "header",
            "footer",
            "aside",
            "div.navigation",
            "div.nav",
            "div.menu",
            "div.sidebar",
            "div.ads",
            "div.advertisement",
            "div.social-share",
            "div.comments",
            "script",
            "style",
            "noscript",
            "div.cookie-notice",
            "div.newsletter",
            "div.related-articles",
            "div.recommended",
        ]

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
            try:
                await self._respect_rate_limit()

                logger.info(f"Scraping article: {url}")

                # Handle special cases for known publishers
                if self._is_arxiv_url(url):
                    return await self._scrape_arxiv_async(session, url)
                elif self._is_bluesky_url(url):
                    return await self._scrape_bluesky_post_async(
                        session, url, follow_links, max_linked_articles
                    )

                # General scraping approach
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()

                soup = BeautifulSoup(html, "html.parser")

                # Extract title
                title = self._extract_title(soup)
                logger.debug(f"Extracted title: '{title}' from {url}")

                # Extract main content
                content = self._extract_content(soup)

                # Extract metadata
                metadata = self._extract_metadata(soup, url)
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
                        content = self._merge_content_with_links(content, linked_content)
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

    def _is_arxiv_url(self, url: str) -> bool:
        """Check if URL is from arXiv"""
        return "arxiv.org" in url.lower()

    def _is_bluesky_url(self, url: str) -> bool:
        """Check if URL is from Bluesky"""
        return "bsky.app" in url.lower() or "bsky.social" in url.lower()

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

            # Extract title
            title_elem = soup.find("h1", class_="title")
            title = (
                title_elem.get_text().replace("Title:", "").strip()
                if title_elem
                else ""
            )

            # Extract abstract
            abstract_elem = soup.find("blockquote", class_="abstract")
            abstract = ""
            if abstract_elem:
                abstract_text = abstract_elem.get_text()
                abstract = abstract_text.replace("Abstract:", "").strip()

            # Extract authors
            authors_elem = soup.find("div", class_="authors")
            authors = authors_elem.get_text().strip() if authors_elem else ""

            # Extract subjects
            subjects_elem = soup.find("td", class_="tablecell subjects")
            subjects = subjects_elem.get_text().strip() if subjects_elem else ""

            content = f"# {title}\n\n"
            if authors:
                content += f"**Authors:** {authors}\n\n"
            if subjects:
                content += f"**Subjects:** {subjects}\n\n"
            if abstract:
                content += f"## Abstract\n\n{abstract}\n\n"

            metadata = {
                "source": "arXiv",
                "authors": authors,
                "subjects": subjects,
                "paper_type": "academic",
            }

            return ScrapedContent(
                url=url, title=title, content=content, metadata=metadata
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
            post_text = self._extract_bluesky_post_text(soup)

            # Look for embedded links
            embedded_links = self._extract_bluesky_links(soup, post_text)

            # Check if any links are academic papers
            academic_link = self._find_academic_link(embedded_links)

            if academic_link:
                logger.info(f"Found academic paper link in Bluesky post: {academic_link}")

                # Scrape the actual academic paper
                if self._is_arxiv_url(academic_link):
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

            title = self._extract_bluesky_title(soup, post_text)

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
            title = self._extract_title(soup)

            # Extract content
            content = self._extract_content(soup)

            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            metadata["paper_type"] = "academic"
            metadata["source"] = "academic_link"

            if content:
                return ScrapedContent(
                    url=url, title=title, content=content, metadata=metadata
                )

        except Exception as e:
            logger.error(f"Error scraping academic link {url}: {e}")

        return None

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
            content_links = self._extract_content_links(soup, base_url)

            # Filter and prioritize interesting links
            interesting_links = self._filter_interesting_links(content_links, base_url)

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

    # Content extraction methods (same as sync version, no async needed)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title, preferring content-specific titles"""
        content_title_selectors = [
            "h1.title",
            "h1.article-title",
            "h1.entry-title",
            "h1.post-title",
            "h1.paper-title",
            "[data-testid='article-title']",
            ".article-header h1",
            ".post-header h1",
            ".content h1",
            "article h1",
            "main h1",
        ]

        for selector in content_title_selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text().strip()
                if title and len(title) > 5:
                    title = self._clean_title(title)
                    return title

        # Try meta tags
        meta_title_selectors = [
            'meta[property="og:title"]',
            'meta[name="twitter:title"]',
            'meta[name="article:title"]',
        ]

        for selector in meta_title_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get("content"):
                title = elem["content"].strip()
                if title and len(title) > 5:
                    title = self._clean_title(title)
                    return title

        # Look for the first substantial H1
        content_area = self._find_main_content_area(soup)
        if content_area:
            h1_elem = content_area.select_one("h1")
            if h1_elem:
                title = h1_elem.get_text().strip()
                if title and len(title) > 5:
                    title = self._clean_title(title)
                    return title

        # Fallback to generic h1 and title tags
        fallback_selectors = ["h1", "title"]
        for selector in fallback_selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text().strip()
                if title and len(title) > 5:
                    title = self._clean_title(title)
                    return title

        return "Untitled"

    def _clean_title(self, title: str) -> str:
        """Clean up title by removing common prefixes and suffixes"""
        import re

        prefixes_to_remove = [
            r"^Title:\s*",
            r"^Article:\s*",
            r"^Paper:\s*",
            r"^Research:\s*",
            r"^Study:\s*",
        ]

        for prefix_pattern in prefixes_to_remove:
            title = re.sub(prefix_pattern, "", title, flags=re.IGNORECASE)

        # Remove site suffixes
        title = re.sub(r"\s+[-|]\s+[A-Za-z\s]+\.(com|org|edu|net|gov).*$", "", title)
        title = re.sub(
            r"\s+[-|]\s+(Home|Homepage|Main|Index)$", "", title, flags=re.IGNORECASE
        )

        # Clean up whitespace
        title = re.sub(r"\s+", " ", title).strip()

        return title

    def _find_main_content_area(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content area of the page"""
        for selector in self.content_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                return content_area
        return None

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content"""
        # Remove unwanted elements
        for selector in self.remove_selectors:
            for elem in soup.select(selector):
                elem.decompose()

        # Try to find main content
        content_elem = None
        for selector in self.content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break

        # Fallback to body
        if not content_elem:
            content_elem = soup.find("body")

        if not content_elem:
            return ""

        # Convert to markdown
        content_html = str(content_elem)
        content_md = md(content_html, heading_style="ATX")

        # Clean up markdown
        content_md = self._clean_markdown(content_md)

        return content_md

    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown content with enhanced formatting"""
        import re

        # Remove excessive whitespace
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        # Fix heading formatting
        content = re.sub(r"\n*(#{1,6})\s*([^\n]+)\n*", r"\n\n\1 \2\n\n", content)

        # Remove empty links
        content = re.sub(r"\[\]\([^)]*\)", "", content)
        content = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", content)

        # Remove standalone brackets
        content = re.sub(r"^\s*\[\s*\]\s*$", "", content, flags=re.MULTILINE)

        # Clean up list items
        content = re.sub(r"^\s*[-*+]\s*$", "", content, flags=re.MULTILINE)
        content = re.sub(
            r"^\s*([-*+])\s*([^\n]+)", r"\1 \2", content, flags=re.MULTILINE
        )
        content = re.sub(
            r"^\s*(\d+\.)\s*([^\n]+)", r"\1 \2", content, flags=re.MULTILINE
        )

        # Remove excessive spaces
        content = re.sub(r" {2,}", " ", content)

        # Clean up code blocks
        content = re.sub(r"\n*```([^\n]*)\n", r"\n\n```\1\n", content)
        content = re.sub(r"\n```\n*", r"\n```\n\n", content)

        # Fix quote formatting
        content = re.sub(r"\n*>\s*([^\n]+)", r"\n\n> \1", content)

        # Remove lines with only punctuation
        content = re.sub(r"^\s*[^\w\s]*\s*$", "", content, flags=re.MULTILINE)

        # Clean up tables
        content = re.sub(r"\n*\|", r"\n|", content)

        # Remove multiple consecutive empty lines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Ensure clean start and end
        content = content.strip()

        if content and not content.endswith("\n"):
            content += "\n"

        return content

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from the page"""
        metadata = {"domain": urlparse(url).netloc, "scraped_timestamp": time.time()}

        # Extract publication date
        date_selectors = [
            "time[datetime]",
            "span.date",
            "div.date",
            'meta[name="article:published_time"]',
            'meta[property="article:published_time"]',
        ]

        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                date_text = (
                    elem.get("datetime") or elem.get("content") or elem.get_text()
                )
                if date_text:
                    metadata["publication_date"] = date_text.strip()
                    break

        # Extract author
        author_selectors = [
            "span.author",
            "div.author",
            'meta[name="author"]',
            'meta[property="article:author"]',
        ]

        for selector in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                author = elem.get("content") or elem.get_text()
                if author:
                    metadata["author"] = author.strip()
                    break

        # Extract description
        desc_selectors = ['meta[name="description"]', 'meta[property="og:description"]']

        for selector in desc_selectors:
            elem = soup.select_one(selector)
            if elem:
                desc = elem.get("content")
                if desc:
                    metadata["description"] = desc.strip()
                    break

        return metadata

    # Bluesky-specific methods

    def _extract_bluesky_post_text(self, soup: BeautifulSoup) -> str:
        """Extract the main text content from a Bluesky post"""
        post_selectors = [
            'div[data-testid="postText"]',
            "div.post-content",
            "div.thread-post-content",
            '[data-testid="postContent"]',
            'div[data-word-wrap="1"]',
        ]

        for selector in post_selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()

        # Fallback: look for divs with substantial text
        divs = soup.find_all("div")
        for div in divs:
            text = div.get_text().strip()
            if 20 < len(text) < 2000 and any(
                keyword in text.lower()
                for keyword in [
                    "paper",
                    "research",
                    "arxiv",
                    "study",
                    "analysis",
                    "method",
                ]
            ):
                return text

        return "Could not extract post text from Bluesky"

    def _extract_bluesky_links(self, soup: BeautifulSoup, post_text: str) -> List[str]:
        """Extract links from Bluesky post"""
        import re

        links = []

        # Extract from HTML links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("http"):
                links.append(href)

        # Extract URLs from text using regex
        url_pattern = r'https?://[^\s<>"\']+(?:[^\s<>"\'.,;!?])'
        urls_in_text = re.findall(url_pattern, post_text)
        links.extend(urls_in_text)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links

    def _find_academic_link(self, links: List[str]) -> Optional[str]:
        """Find the first academic paper link from a list of links"""
        academic_domains = [
            "arxiv.org",
            "ieee.org",
            "acm.org",
            "springer.com",
            "nature.com",
            "science.org",
            "pnas.org",
            "biorxiv.org",
            "medrxiv.org",
            "researchgate.net",
            "semanticscholar.org",
            "doi.org",
        ]

        for link in links:
            for domain in academic_domains:
                if domain in link.lower():
                    return link

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

    def _extract_bluesky_title(self, soup: BeautifulSoup, post_text: str) -> str:
        """Extract or generate a title for Bluesky post"""
        # Try page title
        title_elem = soup.find("title")
        if title_elem:
            title = title_elem.get_text().strip()
            if title and title != "Bluesky":
                return title

        # Generate from post text
        if post_text and len(post_text) > 10:
            first_sentence = post_text.split(".")[0].strip()
            if 10 < len(first_sentence) < 100:
                return f"Bluesky Post: {first_sentence}"
            else:
                return f"Bluesky Post: {post_text[:50]}..."

        return "Bluesky Social Media Post"

    # Link analysis methods

    def _extract_content_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the main content area"""
        links = []

        # Find main content area
        content_area = None
        for selector in self.content_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                break

        if not content_area:
            content_area = soup

        # Extract all links
        for link in content_area.find_all("a", href=True):
            href = link["href"]

            # Convert relative URLs to absolute
            if href.startswith("/"):
                parsed_base = urlparse(base_url)
                absolute_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
            elif href.startswith("http"):
                absolute_url = href
            else:
                continue

            links.append(absolute_url)

        # Remove duplicates
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links

    def _filter_interesting_links(self, links: List[str], base_url: str) -> List[str]:
        """Filter links to find the most interesting ones to analyze"""
        interesting_links = []

        # Priority domains
        priority_domains = [
            # Academic
            "arxiv.org",
            "ieee.org",
            "acm.org",
            "springer.com",
            "nature.com",
            "science.org",
            "pnas.org",
            "biorxiv.org",
            "medrxiv.org",
            "researchgate.net",
            "semanticscholar.org",
            "doi.org",
            # Tech companies
            "openai.com",
            "anthropic.com",
            "deepmind.com",
            "ai.google",
            "research.microsoft.com",
            "research.facebook.com",
            "ai.meta.com",
            # Tech blogs
            "medium.com",
            "substack.com",
            "towards-data-science",
            "hackernews",
            "techcrunch.com",
            "venturebeat.com",
            "wired.com",
            "arstechnica.com",
            # Dev resources
            "github.com",
            "huggingface.co",
            "kaggle.com",
            "colab.research.google.com",
        ]

        # Skip domains
        skip_domains = [
            "twitter.com",
            "x.com",
            "facebook.com",
            "instagram.com",
            "linkedin.com",
            "youtube.com",
            "amazon.com",
            "ebay.com",
            "ads.",
            "analytics.",
            "google.com/search",
            "bing.com",
            "duckduckgo.com",
        ]

        base_domain = urlparse(base_url).netloc

        for link in links:
            parsed_url = urlparse(link)
            domain = parsed_url.netloc.lower()

            # Skip same domain unless priority
            if domain == base_domain and not any(
                priority in domain for priority in priority_domains
            ):
                continue

            # Skip unwanted domains
            if any(skip in domain for skip in skip_domains):
                continue

            # Prioritize interesting domains
            if any(priority in domain for priority in priority_domains):
                interesting_links.insert(0, link)
            else:
                if self._looks_like_article_link(link):
                    interesting_links.append(link)

        return interesting_links

    def _looks_like_article_link(self, url: str) -> bool:
        """Heuristic to determine if a URL might contain interesting content"""
        import re

        url_lower = url.lower()

        article_patterns = [
            "/blog/",
            "/article/",
            "/post/",
            "/news/",
            "/research/",
            "/paper/",
            "/publication/",
            "/tutorial/",
            "/guide/",
            "/analysis/",
            "/review/",
            "/report/",
            "/study/",
        ]

        if any(pattern in url_lower for pattern in article_patterns):
            return True

        # Check for date patterns
        date_patterns = [
            r"/\d{4}/",
            r"/\d{4}/\d{2}/",
            r"/\d{4}-\d{2}/",
            r"-\d{4}-",
            r"_\d{4}_",
        ]

        if any(re.search(pattern, url) for pattern in date_patterns):
            return True

        # Skip non-article extensions
        skip_extensions = [".pdf", ".doc", ".ppt", ".zip", ".tar", ".gz"]
        return not any(url_lower.endswith(ext) for ext in skip_extensions)

    def _merge_content_with_links(
        self, main_content: str, linked_content: List[Dict]
    ) -> str:
        """Merge main content with summaries of linked articles"""
        if not linked_content:
            return main_content

        enhanced_content = main_content + "\n\n"
        enhanced_content += "## Referenced Articles and Links\n\n"
        enhanced_content += (
            "The following articles and resources were referenced in this content:\n\n"
        )

        for i, link_data in enumerate(linked_content, 1):
            enhanced_content += f"### {i}. {link_data['title']}\n"
            enhanced_content += f"**Source:** {link_data['url']}\n\n"

            content_preview = link_data["content"]
            if len(content_preview) > 500:
                content_preview = content_preview[:500] + "..."

            enhanced_content += f"**Summary:** {content_preview}\n\n"
            enhanced_content += "---\n\n"

        return enhanced_content


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
