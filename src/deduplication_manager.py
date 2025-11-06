"""
Deduplication Manager Module

High-performance duplicate detection system for RSS analyzer with:
- O(1) hash-based lookups using SHA-256
- In-memory LRU cache for fast duplicate checks
- Batch operations for efficient backfilling
- Integration with existing DatabaseManager
"""

import hashlib
import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation.
    Provides O(1) get/set operations with automatic eviction.
    """

    def __init__(self, capacity: int = 100000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[any]:
        """Get value from cache, moving to end (most recently used)."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: any):
        """Set value in cache, evicting oldest if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self):
        """Clear all entries from cache."""
        self.cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class DeduplicationManager:
    """
    Manages article deduplication with hash-based detection and in-memory caching.

    Features:
    - Content hash (SHA-256) for duplicate detection
    - URL hash for fast URL lookups
    - In-memory LRU cache for < 1ms lookups
    - Batch operations for efficient processing
    - Automatic cache management

    Performance targets:
    - < 1ms per duplicate check
    - < 100MB memory for 100K articles
    - 1000+ articles/sec batch processing
    """

    def __init__(self, database_manager, cache_capacity: int = 100000):
        """
        Initialize deduplication manager.

        Args:
            database_manager: Instance of DatabaseManager
            cache_capacity: Maximum number of cached hashes (default: 100K articles)
        """
        self.db = database_manager
        self.cache_capacity = cache_capacity

        # Separate caches for content and URL hashes
        self.content_hash_cache = LRUCache(cache_capacity)
        self.url_hash_cache = LRUCache(cache_capacity)

        # Statistics tracking
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "duplicates_detected": 0,
            "articles_processed": 0,
            "last_cleanup": datetime.now(),
        }

        # Warm up cache on initialization
        self._warm_cache()

        logger.info(
            f"DeduplicationManager initialized with cache capacity: {cache_capacity}"
        )

    def _warm_cache(self):
        """
        Warm up cache by loading recent article hashes from database.
        Loads most recent articles to optimize for common access patterns.
        """
        try:
            start_time = time.time()

            with self.db.get_connection() as conn:
                # Load content hashes
                cursor = conn.execute("""
                    SELECT content_hash, id
                    FROM articles
                    WHERE content_hash IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (self.cache_capacity,))

                for row in cursor.fetchall():
                    self.content_hash_cache.set(row[0], row[1])

                # Load URL hashes
                cursor = conn.execute("""
                    SELECT url_hash, id
                    FROM articles
                    WHERE url_hash IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (self.cache_capacity,))

                for row in cursor.fetchall():
                    self.url_hash_cache.set(row[0], row[1])

            duration = time.time() - start_time
            logger.info(
                f"Cache warmed up in {duration:.2f}s. "
                f"Content hashes: {self.content_hash_cache.size()}, "
                f"URL hashes: {self.url_hash_cache.size()}"
            )

        except Exception as e:
            logger.error(f"Failed to warm cache: {e}")

    @staticmethod
    def generate_content_hash(content: str) -> str:
        """
        Generate SHA-256 hash for article content.

        Args:
            content: Article content text

        Returns:
            64-character hex string (SHA-256 hash)
        """
        # Normalize content: lowercase, strip whitespace, remove extra spaces
        normalized = " ".join(content.lower().strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_url_hash(url: str) -> str:
        """
        Generate SHA-256 hash for article URL.

        Args:
            url: Article URL

        Returns:
            64-character hex string (SHA-256 hash)
        """
        # Normalize URL: lowercase, strip whitespace
        normalized = url.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(
        self,
        url: str,
        content: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if article is a duplicate (O(1) lookup).

        Args:
            url: Article URL
            content: Article content (optional, for content-based detection)

        Returns:
            Tuple of (is_duplicate: bool, reason: str)
            - is_duplicate: True if duplicate found
            - reason: "url" or "content" indicating duplicate type
        """
        self.stats["articles_processed"] += 1

        # Check URL hash first (fastest)
        url_hash = self.generate_url_hash(url)

        # Check cache first
        cached_id = self.url_hash_cache.get(url_hash)
        if cached_id is not None:
            self.stats["cache_hits"] += 1
            self.stats["duplicates_detected"] += 1
            logger.debug(f"URL duplicate found in cache: {url}")
            return True, "url"

        # Check database
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM articles WHERE url_hash = ? LIMIT 1",
                    (url_hash,)
                )
                result = cursor.fetchone()

                if result:
                    self.stats["cache_misses"] += 1
                    self.stats["duplicates_detected"] += 1
                    # Update cache for future lookups
                    self.url_hash_cache.set(url_hash, result[0])
                    logger.debug(f"URL duplicate found in database: {url}")
                    return True, "url"

        except Exception as e:
            logger.error(f"Error checking URL duplicate: {e}")

        # Check content hash if provided
        if content:
            content_hash = self.generate_content_hash(content)

            # Check cache
            cached_id = self.content_hash_cache.get(content_hash)
            if cached_id is not None:
                self.stats["cache_hits"] += 1
                self.stats["duplicates_detected"] += 1
                logger.debug(f"Content duplicate found in cache")
                return True, "content"

            # Check database
            try:
                with self.db.get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT id FROM articles WHERE content_hash = ? LIMIT 1",
                        (content_hash,)
                    )
                    result = cursor.fetchone()

                    if result:
                        self.stats["cache_misses"] += 1
                        self.stats["duplicates_detected"] += 1
                        # Update cache
                        self.content_hash_cache.set(content_hash, result[0])
                        logger.debug(f"Content duplicate found in database")
                        return True, "content"

            except Exception as e:
                logger.error(f"Error checking content duplicate: {e}")

        # Not a duplicate
        self.stats["cache_misses"] += 1
        return False, None

    def mark_processed(
        self,
        article_id: int,
        url: str,
        content_hash: str,
        url_hash: Optional[str] = None
    ):
        """
        Mark article as processed and update caches.

        Args:
            article_id: Article database ID
            url: Article URL
            content_hash: Content hash (SHA-256)
            url_hash: URL hash (optional, will be generated if not provided)
        """
        if not url_hash:
            url_hash = self.generate_url_hash(url)

        # Update caches
        self.content_hash_cache.set(content_hash, article_id)
        self.url_hash_cache.set(url_hash, article_id)

        logger.debug(f"Marked article {article_id} as processed in cache")

    def batch_check_duplicates(
        self,
        articles: List[Dict[str, str]]
    ) -> List[Dict[str, any]]:
        """
        Batch check multiple articles for duplicates (optimized for throughput).

        Args:
            articles: List of dicts with 'url' and optional 'content' keys

        Returns:
            List of dicts with original article data plus:
            - 'is_duplicate': bool
            - 'duplicate_reason': str or None
        """
        start_time = time.time()
        results = []

        for article in articles:
            url = article.get("url")
            content = article.get("content")

            if not url:
                logger.warning("Article missing URL in batch check")
                continue

            is_dup, reason = self.is_duplicate(url, content)

            results.append({
                **article,
                "is_duplicate": is_dup,
                "duplicate_reason": reason,
            })

        duration = time.time() - start_time
        throughput = len(articles) / duration if duration > 0 else 0

        logger.info(
            f"Batch processed {len(articles)} articles in {duration:.2f}s "
            f"({throughput:.0f} articles/sec)"
        )

        return results

    def batch_mark_processed(self, article_data: List[Dict[str, any]]):
        """
        Batch mark multiple articles as processed (optimized for throughput).

        Args:
            article_data: List of dicts with 'article_id', 'url', 'content_hash'
        """
        for data in article_data:
            self.mark_processed(
                data["article_id"],
                data["url"],
                data["content_hash"],
                data.get("url_hash")
            )

        logger.info(f"Batch marked {len(article_data)} articles as processed")

    def get_duplicate_stats(self) -> Dict[str, any]:
        """
        Get deduplication statistics and performance metrics.

        Returns:
            Dict with statistics including cache performance and duplicate counts
        """
        cache_hit_rate = 0.0
        total_checks = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_checks > 0:
            cache_hit_rate = (self.stats["cache_hits"] / total_checks) * 100

        return {
            "articles_processed": self.stats["articles_processed"],
            "duplicates_detected": self.stats["duplicates_detected"],
            "duplicate_rate": (
                (self.stats["duplicates_detected"] / self.stats["articles_processed"] * 100)
                if self.stats["articles_processed"] > 0
                else 0.0
            ),
            "cache_stats": {
                "content_cache_size": self.content_hash_cache.size(),
                "url_cache_size": self.url_hash_cache.size(),
                "cache_capacity": self.cache_capacity,
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            },
            "last_cleanup": self.stats["last_cleanup"].isoformat(),
        }

    def clean_old_cache(self, max_age_hours: int = 24):
        """
        Clean old entries from cache based on age.
        Note: LRU cache automatically evicts old entries, but this provides
        a manual cleanup option and resets statistics.

        Args:
            max_age_hours: Maximum age in hours before forcing cache refresh
        """
        try:
            now = datetime.now()
            last_cleanup = self.stats.get("last_cleanup", now)

            if (now - last_cleanup) > timedelta(hours=max_age_hours):
                logger.info("Cleaning cache due to age threshold")

                # Clear caches
                self.content_hash_cache.clear()
                self.url_hash_cache.clear()

                # Warm up again
                self._warm_cache()

                # Reset statistics
                self.stats["last_cleanup"] = now

                logger.info("Cache cleanup completed")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def rebuild_cache(self):
        """
        Completely rebuild cache from database.
        Useful after bulk database operations or corruption detection.
        """
        logger.info("Rebuilding cache from database...")

        # Clear existing caches
        self.content_hash_cache.clear()
        self.url_hash_cache.clear()

        # Warm up fresh
        self._warm_cache()

        logger.info("Cache rebuild completed")

    def get_memory_usage_estimate(self) -> Dict[str, any]:
        """
        Estimate memory usage of deduplication manager.

        Returns:
            Dict with memory usage estimates in MB
        """
        # Each hash is 64 bytes (SHA-256 hex string)
        # Plus integer article ID (8 bytes)
        # Plus OrderedDict overhead (~100 bytes per entry)
        bytes_per_entry = 64 + 8 + 100

        content_cache_mb = (
            self.content_hash_cache.size() * bytes_per_entry
        ) / (1024 * 1024)
        url_cache_mb = (
            self.url_hash_cache.size() * bytes_per_entry
        ) / (1024 * 1024)

        total_mb = content_cache_mb + url_cache_mb

        return {
            "content_cache_mb": f"{content_cache_mb:.2f}",
            "url_cache_mb": f"{url_cache_mb:.2f}",
            "total_mb": f"{total_mb:.2f}",
            "estimated_max_mb": f"{(self.cache_capacity * 2 * bytes_per_entry) / (1024 * 1024):.2f}",
        }
