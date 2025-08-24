"""
Unified Content Fetcher Interface

Coordinates RSS parsing and web scraping to provide a clean interface
for content extraction.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

from .rss_parser import RSSParser, RSSEntry
from .web_scraper import WebScraper, ArticleContent

logger = logging.getLogger(__name__)


@dataclass
class FetchedArticle:
    """Unified representation of a fetched article"""
    rss_entry: RSSEntry
    article_content: Optional[ArticleContent] = None
    fetch_error: Optional[str] = None


class ContentFetcher:
    """Unified interface for content extraction"""
    
    def __init__(self, user_agent: str = "RSS-Article-Analyzer/1.0"):
        self.rss_parser = RSSParser(user_agent)
        self.web_scraper = WebScraper(user_agent)
    
    def fetch_from_rss(self, feed_url: str, existing_hashes: set = None) -> List[RSSEntry]:
        """Fetch new entries from RSS feed"""
        try:
            entries = self.rss_parser.fetch_feed(feed_url)
            
            if existing_hashes:
                entries = self.rss_parser.filter_new_entries(entries, existing_hashes)
            
            return entries
        except Exception as e:
            logger.error(f"Failed to fetch RSS entries: {e}")
            return []
    
    def fetch_article_content(self, rss_entry: RSSEntry) -> FetchedArticle:
        """Fetch full article content for an RSS entry"""
        try:
            article_content = self.web_scraper.scrape_article(rss_entry.link)
            return FetchedArticle(
                rss_entry=rss_entry,
                article_content=article_content
            )
        except Exception as e:
            logger.error(f"Failed to fetch article content for {rss_entry.link}: {e}")
            return FetchedArticle(
                rss_entry=rss_entry,
                fetch_error=str(e)
            )
    
    def fetch_articles_batch(self, rss_entries: List[RSSEntry]) -> List[FetchedArticle]:
        """Fetch content for multiple articles"""
        fetched_articles = []
        
        for entry in rss_entries:
            fetched_article = self.fetch_article_content(entry)
            fetched_articles.append(fetched_article)
        
        return fetched_articles
    
    def get_feed_info(self, feed_url: str) -> dict:
        """Get RSS feed information"""
        return self.rss_parser.get_feed_info(feed_url)