"""
RSS Parser Integration with Deduplication

Example integration showing how to use DeduplicationManager with RSSParser
to skip duplicate articles BEFORE expensive scraping and AI analysis.
"""

import logging
import time
from typing import Dict, List, Optional

from core.database import DatabaseManager
from core.rss_parser import RSSParser
from deduplication_manager import DeduplicationManager

logger = logging.getLogger(__name__)


class RSSParserWithDeduplication:
    """
    Enhanced RSS Parser with integrated duplicate detection.

    Extends standard RSS parsing with:
    - Pre-scraping duplicate detection
    - Batch duplicate checking
    - Performance tracking
    - Cache warming
    """

    def __init__(
        self,
        db_path: str = "data/articles.db",
        cache_capacity: int = 100000
    ):
        """
        Initialize RSS parser with deduplication.

        Args:
            db_path: Path to SQLite database
            cache_capacity: Maximum cached articles
        """
        self.db = DatabaseManager(db_path)
        self.dedup = DeduplicationManager(self.db, cache_capacity)
        self.parser = RSSParser()

        self.stats = {
            "total_entries": 0,
            "new_articles": 0,
            "url_duplicates": 0,
            "content_duplicates": 0,
            "errors": 0,
        }

        logger.info("RSSParserWithDeduplication initialized")

    def parse_feed_with_dedup(
        self,
        feed_url: str,
        max_entries: Optional[int] = None
    ) -> List[Dict]:
        """
        Parse RSS feed and filter out duplicates.

        Args:
            feed_url: RSS feed URL
            max_entries: Maximum entries to process

        Returns:
            List of non-duplicate entries ready for scraping
        """
        logger.info(f"Parsing feed with deduplication: {feed_url}")
        start_time = time.time()

        # Parse RSS feed
        entries = self.parser.parse(feed_url)

        if max_entries:
            entries = entries[:max_entries]

        self.stats["total_entries"] = len(entries)
        logger.info(f"Found {len(entries)} entries in feed")

        # Filter duplicates
        new_entries = self._filter_duplicates(entries)

        duration = time.time() - start_time
        logger.info(
            f"Deduplication completed in {duration:.2f}s. "
            f"New: {len(new_entries)}, Duplicates: {len(entries) - len(new_entries)}"
        )

        return new_entries

    def _filter_duplicates(self, entries: List[Dict]) -> List[Dict]:
        """
        Filter duplicate entries before scraping.

        Args:
            entries: RSS feed entries

        Returns:
            Non-duplicate entries
        """
        new_entries = []

        for entry in entries:
            url = entry.get('link')
            title = entry.get('title', 'Untitled')

            if not url:
                logger.warning(f"Entry missing URL: {title}")
                continue

            # Check URL duplicate (fast, no scraping needed)
            is_duplicate, reason = self.dedup.is_duplicate(url)

            if is_duplicate:
                logger.debug(f"Skipping {reason} duplicate: {title}")

                if reason == "url":
                    self.stats["url_duplicates"] += 1
                else:
                    self.stats["content_duplicates"] += 1

                continue

            # Not a duplicate - add to processing list
            new_entries.append(entry)
            self.stats["new_articles"] += 1

        return new_entries

    def parse_and_process(
        self,
        feed_url: str,
        scraper,
        processor,
        max_entries: Optional[int] = None
    ) -> Dict:
        """
        Complete workflow: parse, deduplicate, scrape, and process.

        Args:
            feed_url: RSS feed URL
            scraper: WebScraper instance
            processor: ArticleProcessor instance
            max_entries: Maximum entries to process

        Returns:
            Processing statistics
        """
        logger.info("Starting parse and process workflow")

        # Parse and filter duplicates
        new_entries = self.parse_feed_with_dedup(feed_url, max_entries)

        if not new_entries:
            logger.info("No new articles to process")
            return self.get_stats()

        logger.info(f"Processing {len(new_entries)} new articles")

        # Process each new article
        for entry in new_entries:
            url = entry.get('link')
            title = entry.get('title', 'Untitled')

            try:
                logger.info(f"Processing: {title}")

                # Scrape content
                content = scraper.scrape(url)

                # Double-check content duplicate (after scraping)
                is_duplicate, reason = self.dedup.is_duplicate(url, content)

                if is_duplicate:
                    logger.warning(f"Content duplicate detected after scraping: {title}")
                    self.stats["content_duplicates"] += 1
                    continue

                # Process article (AI analysis, etc.)
                article_id = processor.process(entry, content)

                # Generate and store hashes
                content_hash = self.dedup.generate_content_hash(content)
                url_hash = self.dedup.generate_url_hash(url)

                # Mark as processed in cache
                self.dedup.mark_processed(article_id, url, content_hash, url_hash)

                logger.info(f"Successfully processed article {article_id}")

            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                self.stats["errors"] += 1

        # Return statistics
        return self.get_stats()

    def batch_parse_feeds(
        self,
        feed_urls: List[str],
        max_entries_per_feed: Optional[int] = None
    ) -> List[Dict]:
        """
        Parse multiple RSS feeds with deduplication.

        Args:
            feed_urls: List of RSS feed URLs
            max_entries_per_feed: Max entries per feed

        Returns:
            List of all non-duplicate entries from all feeds
        """
        logger.info(f"Batch parsing {len(feed_urls)} feeds")

        all_new_entries = []

        for feed_url in feed_urls:
            try:
                new_entries = self.parse_feed_with_dedup(
                    feed_url,
                    max_entries_per_feed
                )
                all_new_entries.extend(new_entries)

            except Exception as e:
                logger.error(f"Error parsing feed {feed_url}: {e}")
                self.stats["errors"] += 1

        logger.info(
            f"Batch parse complete. Total new entries: {len(all_new_entries)}"
        )

        return all_new_entries

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        dedup_stats = self.dedup.get_duplicate_stats()

        return {
            "processing": self.stats.copy(),
            "deduplication": dedup_stats,
            "summary": {
                "total_processed": self.stats["total_entries"],
                "new_articles": self.stats["new_articles"],
                "duplicates_skipped": (
                    self.stats["url_duplicates"] +
                    self.stats["content_duplicates"]
                ),
                "error_count": self.stats["errors"],
                "cache_hit_rate": dedup_stats["cache_stats"]["cache_hit_rate"],
            }
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "total_entries": 0,
            "new_articles": 0,
            "url_duplicates": 0,
            "content_duplicates": 0,
            "errors": 0,
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize parser with deduplication
    parser = RSSParserWithDeduplication()

    # Example feed URLs
    feed_urls = [
        "https://export.arxiv.org/rss/cs.AI",
        "https://export.arxiv.org/rss/cs.LG",
    ]

    # Parse feeds and filter duplicates
    for feed_url in feed_urls:
        print(f"\nParsing: {feed_url}")

        new_entries = parser.parse_feed_with_dedup(feed_url, max_entries=10)

        print(f"  Found {len(new_entries)} new articles")
        for entry in new_entries[:3]:  # Show first 3
            print(f"    - {entry.get('title', 'Untitled')}")

    # Show statistics
    stats = parser.get_stats()
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total entries: {stats['summary']['total_processed']}")
    print(f"New articles: {stats['summary']['new_articles']}")
    print(f"Duplicates skipped: {stats['summary']['duplicates_skipped']}")
    print(f"Cache hit rate: {stats['summary']['cache_hit_rate']}")
    print("=" * 80)
