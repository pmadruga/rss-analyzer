#!/usr/bin/env python3
"""
RSS Feed Sync Checker

This script compares the RSS feed with the database to ensure all latest
articles have been analyzed and prevents duplicate processing.
"""

import hashlib
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import yaml


def setup_logging():
    """Configure logging for the sync checker"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config file"""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def normalize_url(url: str) -> str:
    """Normalize URL for consistent comparison"""
    # Remove fragments and common tracking parameters
    parsed = urlparse(url)
    # Remove common tracking parameters
    query_params = []
    if parsed.query:
        for param in parsed.query.split("&"):
            if not any(
                param.startswith(track) for track in ["utm_", "ref=", "source="]
            ):
                query_params.append(param)

    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if query_params:
        normalized += "?" + "&".join(query_params)

    return normalized.rstrip("/")


def get_url_content_hash(url: str) -> str:
    """Generate a content hash for the URL for deduplication"""
    normalized = normalize_url(url)
    return hashlib.md5(normalized.encode()).hexdigest()


def fetch_rss_entries(rss_url: str, logger: logging.Logger) -> list[dict]:
    """Fetch and parse RSS feed entries"""
    logger.info(f"Fetching RSS feed from: {rss_url}")

    try:
        feed = feedparser.parse(rss_url)
        if feed.bozo:
            logger.warning(f"RSS feed has parsing issues: {feed.bozo_exception}")

        entries = []
        for entry in feed.entries:
            # Extract relevant information
            entry_data = {
                "title": getattr(entry, "title", "No Title"),
                "link": getattr(entry, "link", ""),
                "published": getattr(entry, "published", ""),
                "summary": getattr(entry, "summary", ""),
                "content_hash": get_url_content_hash(getattr(entry, "link", "")),
            }
            entries.append(entry_data)

        logger.info(f"Fetched {len(entries)} entries from RSS feed")
        return entries

    except Exception as e:
        logger.error(f"Failed to fetch RSS feed: {e}")
        raise


def get_database_articles(db_path: str, logger: logging.Logger) -> set[str]:
    """Get set of content hashes from database"""
    logger.info(f"Checking database: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all article URLs and generate content hashes
        cursor.execute("SELECT url FROM articles")
        urls = cursor.fetchall()

        content_hashes = set()
        for (url,) in urls:
            content_hash = get_url_content_hash(url)
            content_hashes.add(content_hash)

        conn.close()

        logger.info(f"Found {len(content_hashes)} articles in database")
        return content_hashes

    except Exception as e:
        logger.error(f"Database error: {e}")
        raise


def check_sync_status(rss_url: str, db_path: str, logger: logging.Logger) -> dict:
    """Check synchronization status between RSS and database"""

    # Fetch RSS entries
    rss_entries = fetch_rss_entries(rss_url, logger)
    rss_hashes = {entry["content_hash"]: entry for entry in rss_entries}

    # Get database articles
    db_hashes = get_database_articles(db_path, logger)

    # Find differences
    rss_only = set(rss_hashes.keys()) - db_hashes  # In RSS but not in DB
    db_only = db_hashes - set(rss_hashes.keys())  # In DB but not in RSS
    common = set(rss_hashes.keys()) & db_hashes  # In both

    # Prepare results
    sync_status = {
        "total_rss_entries": len(rss_entries),
        "total_db_articles": len(db_hashes),
        "common_count": len(common),
        "missing_from_db": len(rss_only),
        "missing_from_rss": len(db_only),
        "sync_percentage": (len(common) / len(rss_hashes) * 100) if rss_hashes else 100,
        "missing_articles": [],
    }

    # Add details about missing articles
    for hash_id in rss_only:
        entry = rss_hashes[hash_id]
        sync_status["missing_articles"].append(
            {
                "title": entry["title"],
                "url": entry["link"],
                "published": entry["published"],
            }
        )

    return sync_status


def main():
    """Main function to check RSS sync status"""
    logger = setup_logging()

    try:
        # Load configuration
        config = load_config()
        rss_url = config.get("rss", {}).get("feed_url")
        db_path = config.get("database", {}).get("path", "data/articles.db")

        if not rss_url:
            logger.error("RSS feed URL not found in configuration")
            sys.exit(1)

        # Check if database exists
        if not Path(db_path).exists():
            logger.error(f"Database not found: {db_path}")
            sys.exit(1)

        logger.info("Starting RSS feed synchronization check")

        # Perform sync check
        sync_status = check_sync_status(rss_url, db_path, logger)

        # Report results
        print("\n" + "=" * 60)
        print("RSS FEED SYNCHRONIZATION REPORT")
        print("=" * 60)
        print(f"Generated at: {datetime.now().isoformat()}")
        print(f"RSS entries: {sync_status['total_rss_entries']}")
        print(f"Database articles: {sync_status['total_db_articles']}")
        print(f"Common articles: {sync_status['common_count']}")
        print(f"Missing from database: {sync_status['missing_from_db']}")
        print(f"Missing from RSS: {sync_status['missing_from_rss']}")
        print(f"Sync percentage: {sync_status['sync_percentage']:.1f}%")

        # Show missing articles if any
        if sync_status["missing_articles"]:
            print("\nMissing articles (in RSS but not in database):")
            for i, article in enumerate(sync_status["missing_articles"][:10], 1):
                print(f"  {i}. {article['title']}")
                print(f"     URL: {article['url']}")
                print(f"     Published: {article['published']}")
                print()

            if len(sync_status["missing_articles"]) > 10:
                remaining = len(sync_status["missing_articles"]) - 10
                print(f"  ... and {remaining} more articles")

        print("=" * 60)

        # Exit with appropriate code
        if sync_status["missing_from_db"] > 0:
            logger.warning(
                f"Found {sync_status['missing_from_db']} articles that need to be processed"
            )
            print(
                f"\n⚠️  ACTION REQUIRED: Run 'docker compose run rss-analyzer run --limit {sync_status['missing_from_db']}' to process missing articles"
            )
            sys.exit(1)  # Exit with error code to indicate action needed
        else:
            logger.info("RSS feed and database are in sync")
            print("\n✅ RSS feed and database are synchronized")
            sys.exit(0)  # Success

    except Exception as e:
        logger.error(f"Sync check failed: {e}")
        print(f"\n❌ Sync check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
