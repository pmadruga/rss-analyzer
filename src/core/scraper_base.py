"""
Base Scraper Module

Provides shared scraping logic for both sync and async implementations.
All static/reusable methods are extracted here to eliminate code duplication.
"""

import re
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md


class ScraperBase:
    """Base class containing shared scraping logic for sync and async implementations"""

    # Common content selectors across all implementations
    CONTENT_SELECTORS = [
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

    # Selectors to remove from content
    REMOVE_SELECTORS = [
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

    # Priority domains for link filtering
    PRIORITY_DOMAINS = [
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
    SKIP_DOMAINS = [
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

    # Academic domains for link detection
    ACADEMIC_DOMAINS = [
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

    @staticmethod
    def is_arxiv_url(url: str) -> bool:
        """Check if URL is from arXiv"""
        return "arxiv.org" in url.lower()

    @staticmethod
    def is_bluesky_url(url: str) -> bool:
        """Check if URL is from Bluesky"""
        return "bsky.app" in url.lower() or "bsky.social" in url.lower()

    @staticmethod
    def extract_title(soup: BeautifulSoup) -> str:
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
                    title = ScraperBase.clean_title(title)
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
                    title = ScraperBase.clean_title(title)
                    return title

        # Look for the first substantial H1 in the content area
        content_area = ScraperBase.find_main_content_area(soup)
        if content_area:
            h1_elem = content_area.select_one("h1")
            if h1_elem:
                title = h1_elem.get_text().strip()
                if title and len(title) > 5:
                    title = ScraperBase.clean_title(title)
                    return title

        # Fallback to generic h1 and title tags
        fallback_selectors = ["h1", "title"]
        for selector in fallback_selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text().strip()
                if title and len(title) > 5:
                    title = ScraperBase.clean_title(title)
                    return title

        return "Untitled"

    @staticmethod
    def clean_title(title: str) -> str:
        """Clean up title by removing common prefixes and suffixes"""
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

    @staticmethod
    def find_main_content_area(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content area of the page"""
        for selector in ScraperBase.CONTENT_SELECTORS:
            content_area = soup.select_one(selector)
            if content_area:
                return content_area
        return None

    @staticmethod
    def extract_content(soup: BeautifulSoup) -> str:
        """Extract main article content"""
        # Remove unwanted elements
        for selector in ScraperBase.REMOVE_SELECTORS:
            for elem in soup.select(selector):
                elem.decompose()

        # Try to find main content
        content_elem = None
        for selector in ScraperBase.CONTENT_SELECTORS:
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
        content_md = ScraperBase.clean_markdown(content_md)

        return content_md

    @staticmethod
    def clean_markdown(content: str) -> str:
        """Clean up markdown content with enhanced formatting"""
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

    @staticmethod
    def extract_metadata(soup: BeautifulSoup, url: str) -> Dict:
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

    @staticmethod
    def extract_bluesky_post_text(soup: BeautifulSoup) -> str:
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

    @staticmethod
    def extract_bluesky_links(soup: BeautifulSoup, post_text: str) -> List[str]:
        """Extract links from Bluesky post"""
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

    @staticmethod
    def find_academic_link(links: List[str]) -> Optional[str]:
        """Find the first academic paper link from a list of links"""
        for link in links:
            for domain in ScraperBase.ACADEMIC_DOMAINS:
                if domain in link.lower():
                    return link
        return None

    @staticmethod
    def extract_bluesky_title(soup: BeautifulSoup, post_text: str) -> str:
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

    @staticmethod
    def extract_content_links(soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the main content area"""
        links = []

        # Find main content area
        content_area = None
        for selector in ScraperBase.CONTENT_SELECTORS:
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

    @staticmethod
    def filter_interesting_links(links: List[str], base_url: str) -> List[str]:
        """Filter links to find the most interesting ones to analyze"""
        interesting_links = []
        base_domain = urlparse(base_url).netloc

        for link in links:
            parsed_url = urlparse(link)
            domain = parsed_url.netloc.lower()

            # Skip same domain unless priority
            if domain == base_domain and not any(
                priority in domain for priority in ScraperBase.PRIORITY_DOMAINS
            ):
                continue

            # Skip unwanted domains
            if any(skip in domain for skip in ScraperBase.SKIP_DOMAINS):
                continue

            # Prioritize interesting domains
            if any(priority in domain for priority in ScraperBase.PRIORITY_DOMAINS):
                interesting_links.insert(0, link)
            else:
                if ScraperBase.looks_like_article_link(link):
                    interesting_links.append(link)

        return interesting_links

    @staticmethod
    def looks_like_article_link(url: str) -> bool:
        """Heuristic to determine if a URL might contain interesting content"""
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

    @staticmethod
    def merge_content_with_links(main_content: str, linked_content: List[Dict]) -> str:
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

    @staticmethod
    def parse_arxiv_content(soup: BeautifulSoup, url: str) -> Dict:
        """Parse arXiv-specific content structure"""
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

        return {"title": title, "content": content, "metadata": metadata}
