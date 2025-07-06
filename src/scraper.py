"""
Web Scraper Module

Handles scraping full article content from URLs, with support for
various article formats and academic publishers.
"""

import logging
import re
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "scraped_at": self.scraped_at,
        }


class WebScraper:
    """Web scraper with support for academic publishers and various article formats"""

    def __init__(self, delay_between_requests: float = 1.0):
        self.delay = delay_between_requests
        self.session = self._create_session()
        self.last_request_time = 0

        # Common selectors for different content types
        self.content_selectors = [
            # Academic papers
            "div.ltx_page_main",  # arXiv
            "div.abstract-content",
            "div.article-body",  # Various journals
            "div.fulltext-view",  # IEEE
            "div#body",
            "div.body",  # Generic body
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

        # Selectors to remove (navigation, ads, etc.)
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

            # Extract title
            title = self._extract_title(soup)

            # Extract main content
            content = self._extract_content(soup)

            # Extract metadata
            metadata = self._extract_metadata(soup, url)

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
        return "arxiv.org" in url.lower()

    def _is_bluesky_url(self, url: str) -> bool:
        """Check if URL is from Bluesky"""
        return "bsky.app" in url.lower() or "bsky.social" in url.lower()

    def _scrape_arxiv(self, url: str, timeout: int) -> ArticleContent | None:
        """Special handling for arXiv papers"""
        try:
            # Convert to abstract page if it's a PDF link
            if "/pdf/" in url:
                url = url.replace("/pdf/", "/abs/")

            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

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

            return ArticleContent(
                url=url, title=title, content=content, metadata=metadata
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
            post_text = self._extract_bluesky_post_text(soup)

            # Look for embedded links, especially arXiv papers
            embedded_links = self._extract_bluesky_links(soup, post_text)

            # Check if any links are academic papers (arXiv, etc.)
            academic_link = self._find_academic_link(embedded_links)

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

            return ArticleContent(
                url=url, title=title, content=content, metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error scraping Bluesky post {url}: {e}")
            return None

    def _extract_bluesky_post_text(self, soup: BeautifulSoup) -> str:
        """Extract the main text content from a Bluesky post"""
        # Bluesky post content selectors
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

        # Fallback: look for any div with substantial text content
        divs = soup.find_all("div")
        for div in divs:
            text = div.get_text().strip()
            # Look for divs with substantial text that might be the post
            if 20 < len(text) < 2000 and any(  # Reasonable post length
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

    def _extract_bluesky_links(self, soup: BeautifulSoup, post_text: str) -> list[str]:
        """Extract links from Bluesky post"""
        links = []

        # Extract from HTML links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("http"):
                links.append(href)

        # Extract URLs from text using regex
        import re

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

    def _find_academic_link(self, links: list[str]) -> str | None:
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

    def _scrape_general_academic_link(
        self, url: str, timeout: int
    ) -> ArticleContent | None:
        """Scrape academic content from non-arXiv sources"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title = self._extract_title(soup)

            # Extract content using academic-specific selectors
            content = self._extract_content(soup)

            # Extract metadata
            metadata = self._extract_metadata(soup, url)
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

    def _extract_bluesky_title(self, soup: BeautifulSoup, post_text: str) -> str:
        """Extract or generate a title for Bluesky post"""

        # Try to extract from page title
        title_elem = soup.find("title")
        if title_elem:
            title = title_elem.get_text().strip()
            if title and title != "Bluesky":
                return title

        # Generate title from post text
        if post_text and len(post_text) > 10:
            # Take first sentence or first 50 characters
            first_sentence = post_text.split(".")[0].strip()
            if len(first_sentence) > 10 and len(first_sentence) < 100:
                return f"Bluesky Post: {first_sentence}"
            else:
                return f"Bluesky Post: {post_text[:50]}..."

        return "Bluesky Social Media Post"

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        # Try various title selectors
        title_selectors = [
            "h1.title",
            "h1.article-title",
            "h1.entry-title",
            "h1.post-title",
            "h1",
            "title",
        ]

        for selector in title_selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text().strip()
                if title and len(title) > 5:  # Avoid empty or very short titles
                    return title

        return "Untitled"

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

        # Fallback to body if no specific content found
        if not content_elem:
            content_elem = soup.find("body")

        if not content_elem:
            return ""

        # Convert to markdown and clean up
        content_html = str(content_elem)
        content_md = md(content_html, heading_style="ATX")

        # Clean up the markdown
        content_md = self._clean_markdown(content_md)

        return content_md

    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown content"""
        # Remove excessive whitespace
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        # Remove empty links
        content = re.sub(r"\[\]\([^)]*\)", "", content)

        # Remove standalone brackets
        content = re.sub(r"^\s*\[\s*\]\s*$", "", content, flags=re.MULTILINE)

        # Clean up list items
        content = re.sub(r"^\s*\*\s*$", "", content, flags=re.MULTILINE)

        # Remove excessive spaces
        content = re.sub(r" +", " ", content)

        return content.strip()

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """Extract metadata from the page"""
        metadata = {"domain": urlparse(url).netloc, "scraped_timestamp": time.time()}

        # Try to extract publication date
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

        # Try to extract author
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

        # Extract description/summary
        desc_selectors = ['meta[name="description"]', 'meta[property="og:description"]']

        for selector in desc_selectors:
            elem = soup.select_one(selector)
            if elem:
                desc = elem.get("content")
                if desc:
                    metadata["description"] = desc.strip()
                    break

        return metadata

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
            content_links = self._extract_content_links(soup, base_url)

            # Filter and prioritize interesting links
            interesting_links = self._filter_interesting_links(content_links, base_url)

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

    def _extract_content_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract all links from the main content area"""
        links = []

        # Find the main content area first
        content_area = None
        for selector in self.content_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                break

        # If no content area found, use the whole page
        if not content_area:
            content_area = soup

        # Extract all links from the content area
        for link in content_area.find_all("a", href=True):
            href = link["href"]

            # Convert relative URLs to absolute
            if href.startswith("/"):
                parsed_base = urlparse(base_url)
                absolute_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
            elif href.startswith("http"):
                absolute_url = href
            else:
                # Skip other types of links (mailto, javascript, etc.)
                continue

            links.append(absolute_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links

    def _filter_interesting_links(self, links: list[str], base_url: str) -> list[str]:
        """Filter links to find the most interesting ones to analyze"""
        interesting_links = []

        # Domains that are likely to contain interesting content
        priority_domains = [
            # Academic and research
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
            # Tech and AI companies
            "openai.com",
            "anthropic.com",
            "deepmind.com",
            "ai.google",
            "research.microsoft.com",
            "research.facebook.com",
            "ai.meta.com",
            # Tech blogs and news
            "medium.com",
            "substack.com",
            "towards-data-science",
            "hackernews",
            "techcrunch.com",
            "venturebeat.com",
            "wired.com",
            "arstechnica.com",
            # Documentation and tutorials
            "github.com",
            "huggingface.co",
            "kaggle.com",
            "colab.research.google.com",
        ]

        # Skip domains that are unlikely to be interesting
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

            # Skip links to the same domain (unless it's a priority domain)
            if domain == base_domain and not any(
                priority in domain for priority in priority_domains
            ):
                continue

            # Skip unwanted domains
            if any(skip in domain for skip in skip_domains):
                continue

            # Prioritize interesting domains
            if any(priority in domain for priority in priority_domains):
                interesting_links.insert(0, link)  # Add to front for priority
            else:
                # Check if the link looks like it might contain interesting content
                if self._looks_like_article_link(link):
                    interesting_links.append(link)

        return interesting_links

    def _looks_like_article_link(self, url: str) -> bool:
        """Heuristic to determine if a URL might contain interesting article content"""
        url_lower = url.lower()

        # Common article/blog patterns
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

        # Look for article-like patterns in the URL
        if any(pattern in url_lower for pattern in article_patterns):
            return True

        # Check for date patterns that might indicate blog posts
        import re

        date_patterns = [
            r"/\d{4}/",
            r"/\d{4}/\d{2}/",
            r"/\d{4}-\d{2}/",
            r"-\d{4}-",
            r"_\d{4}_",
        ]

        if any(re.search(pattern, url) for pattern in date_patterns):
            return True

        # Skip common non-article extensions
        skip_extensions = [".pdf", ".doc", ".ppt", ".zip", ".tar", ".gz"]
        return not any(url_lower.endswith(ext) for ext in skip_extensions)

    def _scrape_linked_article(self, url: str, timeout: int) -> ArticleContent | None:
        """Scrape content from a linked article without following further links"""
        try:
            # Scrape without following additional links to avoid infinite recursion
            return self.scrape_article(url, timeout, follow_links=False)
        except Exception as e:
            logger.error(f"Error scraping linked article {url}: {e}")
            return None

    def _merge_content_with_links(
        self, main_content: str, linked_content: list[dict]
    ) -> str:
        """Merge main content with summaries of linked articles"""
        if not linked_content:
            return main_content

        # Add linked content section
        enhanced_content = main_content + "\n\n"
        enhanced_content += "## Referenced Articles and Links\n\n"
        enhanced_content += (
            "The following articles and resources were referenced in this content:\n\n"
        )

        for i, link_data in enumerate(linked_content, 1):
            enhanced_content += f"### {i}. {link_data['title']}\n"
            enhanced_content += f"**Source:** {link_data['url']}\n\n"

            # Add a summary of the linked content
            content_preview = link_data["content"]
            if len(content_preview) > 500:
                content_preview = content_preview[:500] + "..."

            enhanced_content += f"**Summary:** {content_preview}\n\n"
            enhanced_content += "---\n\n"

        return enhanced_content
