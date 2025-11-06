"""
Two-tier content caching system for RSS analyzer.

This module implements a high-performance caching layer with:
- L1: In-memory LRU cache (256MB limit)
- L2: SQLite-based persistent cache with compression
- Automatic expiration and eviction
- Cache statistics and monitoring
"""

import hashlib
import json
import pickle
import sqlite3
import sys
import zlib
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""

    key: str
    value: Any
    content_type: str
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict:
        """Convert entry to dictionary for storage."""
        return {
            'key': self.key,
            'content_type': self.content_type,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'size_bytes': self.size_bytes
        }


@dataclass
class CacheStats:
    """Cache statistics and metrics."""

    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l1_size_bytes: int = 0
    l1_entries: int = 0
    l2_size_bytes: int = 0
    l2_entries: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def total_hits(self) -> int:
        """Total cache hits across both tiers."""
        return self.l1_hits + self.l2_hits

    @property
    def total_misses(self) -> int:
        """Total cache misses."""
        return self.l1_misses + self.l2_misses

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate (0-1)."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        """L1 cache hit rate (0-1)."""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        """L2 cache hit rate (0-1)."""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return asdict(self)


class L1Cache:
    """In-memory LRU cache with size limit (256MB)."""

    MAX_SIZE_BYTES = 256 * 1024 * 1024  # 256MB

    def __init__(self):
        """Initialize L1 cache."""
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get value from L1 cache.

        Args:
            key: Cache key

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                self._remove(key)
                return None

            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            return entry

    def set(self, key: str, entry: CacheEntry) -> bool:
        """
        Set value in L1 cache.

        Args:
            key: Cache key
            entry: Cache entry to store

        Returns:
            True if stored successfully, False if entry too large
        """
        with self._lock:
            # Calculate entry size
            entry_size = sys.getsizeof(entry.value)
            entry.size_bytes = entry_size

            # Check if single entry exceeds max size
            if entry_size > self.MAX_SIZE_BYTES:
                logger.warning(f"Entry too large for L1 cache: {entry_size} bytes")
                return False

            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)

            # Evict entries if needed to make space
            while self._current_size + entry_size > self.MAX_SIZE_BYTES and self._cache:
                self._evict_oldest()

            # Add new entry
            self._cache[key] = entry
            self._current_size += entry_size

            return True

    def delete(self, key: str) -> bool:
        """
        Delete entry from L1 cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from L1 cache."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    def size_bytes(self) -> int:
        """Get current cache size in bytes."""
        with self._lock:
            return self._current_size

    def count(self) -> int:
        """Get number of entries in cache."""
        with self._lock:
            return len(self._cache)

    def _remove(self, key: str) -> None:
        """Remove entry from cache (internal, no lock)."""
        entry = self._cache.pop(key)
        self._current_size -= entry.size_bytes

    def _evict_oldest(self) -> None:
        """Evict least recently used entry (internal, no lock)."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_size -= entry.size_bytes
            logger.debug(f"Evicted L1 cache entry: {key}")


class L2Cache:
    """SQLite-based persistent cache with compression."""

    def __init__(self, db_path: str = "cache.db"):
        """
        Initialize L2 cache.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    content_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    size_bytes INTEGER DEFAULT 0
                )
            """)

            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON cache(expires_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_type
                ON cache(content_type)
            """)

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get value from L2 cache.

        Args:
            key: Cache key

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT * FROM cache WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()

                    if not row:
                        return None

                    # Parse entry
                    entry = self._row_to_entry(row)

                    # Check expiration
                    if entry.is_expired():
                        self.delete(key)
                        return None

                    # Update access metadata
                    entry.access_count += 1
                    entry.last_accessed = datetime.utcnow()

                    conn.execute(
                        """UPDATE cache
                           SET access_count = ?, last_accessed = ?
                           WHERE key = ?""",
                        (entry.access_count, entry.last_accessed, key)
                    )
                    conn.commit()

                    return entry

            except Exception as e:
                logger.error(f"Error reading from L2 cache: {e}")
                return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        """
        Set value in L2 cache.

        Args:
            key: Cache key
            entry: Cache entry to store

        Returns:
            True if stored successfully
        """
        with self._lock:
            try:
                # Compress and serialize value
                serialized = pickle.dumps(entry.value)
                compressed = zlib.compress(serialized, level=6)
                entry.size_bytes = len(compressed)

                with self._get_connection() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO cache
                           (key, value, content_type, created_at, expires_at,
                            access_count, last_accessed, size_bytes)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            key,
                            compressed,
                            entry.content_type,
                            entry.created_at,
                            entry.expires_at,
                            entry.access_count,
                            entry.last_accessed,
                            entry.size_bytes
                        )
                    )
                    conn.commit()

                return True

            except Exception as e:
                logger.error(f"Error writing to L2 cache: {e}")
                return False

    def delete(self, key: str) -> bool:
        """
        Delete entry from L2 cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error deleting from L2 cache: {e}")
                return False

    def clear(self) -> None:
        """Clear all entries from L2 cache."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("DELETE FROM cache")
                    conn.commit()
            except Exception as e:
                logger.error(f"Error clearing L2 cache: {e}")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from L2 cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            try:
                now = datetime.utcnow()
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache WHERE expires_at < ?",
                        (now,)
                    )
                    conn.commit()
                    return cursor.rowcount
            except Exception as e:
                logger.error(f"Error cleaning up L2 cache: {e}")
                return 0

    def size_bytes(self) -> int:
        """Get total cache size in bytes."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT SUM(size_bytes) FROM cache")
                    result = cursor.fetchone()[0]
                    return result if result else 0
            except Exception as e:
                logger.error(f"Error getting L2 cache size: {e}")
                return 0

    def count(self) -> int:
        """Get number of entries in cache."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM cache")
                    return cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Error getting L2 cache count: {e}")
                return 0

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        """Convert database row to CacheEntry."""
        # Decompress and deserialize value
        compressed = row['value']
        decompressed = zlib.decompress(compressed)
        value = pickle.loads(decompressed)

        return CacheEntry(
            key=row['key'],
            value=value,
            content_type=row['content_type'],
            created_at=datetime.fromisoformat(row['created_at']),
            expires_at=datetime.fromisoformat(row['expires_at']),
            access_count=row['access_count'],
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
            size_bytes=row['size_bytes']
        )


class ContentCache:
    """
    Two-tier content caching system.

    Combines fast in-memory L1 cache with persistent L2 cache.
    Provides automatic expiration, LRU eviction, and statistics.
    """

    # Default TTLs
    TTL_SCRAPED_CONTENT = 7 * 24 * 60 * 60  # 7 days
    TTL_API_RESPONSE = 30 * 24 * 60 * 60    # 30 days

    def __init__(self, db_path: str = "data/cache.db"):
        """
        Initialize content cache.

        Args:
            db_path: Path to SQLite database for L2 cache
        """
        self.l1 = L1Cache()
        self.l2 = L2Cache(db_path)
        self.stats = CacheStats()
        self._lock = threading.RLock()

        logger.info(f"Content cache initialized (L2: {db_path})")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (L1 then L2).

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            # Try L1 first
            entry = self.l1.get(key)
            if entry:
                self.stats.l1_hits += 1
                logger.debug(f"L1 cache hit: {key}")
                return entry.value

            self.stats.l1_misses += 1

            # Try L2
            entry = self.l2.get(key)
            if entry:
                self.stats.l2_hits += 1
                logger.debug(f"L2 cache hit: {key}")

                # Promote to L1
                self.l1.set(key, entry)

                return entry.value

            self.stats.l2_misses += 1
            logger.debug(f"Cache miss: {key}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = TTL_SCRAPED_CONTENT,
        content_type: str = "generic"
    ) -> bool:
        """
        Set value in cache (both L1 and L2).

        Args:
            key: Cache key
            ttl: Time to live in seconds
            value: Value to cache
            content_type: Type of content (for categorization)

        Returns:
            True if cached successfully
        """
        with self._lock:
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl)

            entry = CacheEntry(
                key=key,
                value=value,
                content_type=content_type,
                created_at=now,
                expires_at=expires_at
            )

            # Store in both tiers
            l1_success = self.l1.set(key, entry)
            l2_success = self.l2.set(key, entry)

            logger.debug(
                f"Cached {key} (L1: {l1_success}, L2: {l2_success}, "
                f"TTL: {ttl}s, Type: {content_type})"
            )

            return l2_success  # L2 is source of truth

    def delete(self, key: str) -> bool:
        """
        Delete value from cache (both L1 and L2).

        Args:
            key: Cache key

        Returns:
            True if deleted from at least one tier
        """
        with self._lock:
            l1_deleted = self.l1.delete(key)
            l2_deleted = self.l2.delete(key)

            if l1_deleted or l2_deleted:
                logger.debug(f"Deleted from cache: {key}")
                return True

            return False

    def clear(self) -> None:
        """Clear all entries from cache (both L1 and L2)."""
        with self._lock:
            self.l1.clear()
            self.l2.clear()
            logger.info("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from L2 cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            count = self.l2.cleanup_expired()
            if count > 0:
                self.stats.expirations += count
                logger.info(f"Cleaned up {count} expired cache entries")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            # Update current size/count stats
            self.stats.l1_size_bytes = self.l1.size_bytes()
            self.stats.l1_entries = self.l1.count()
            self.stats.l2_size_bytes = self.l2.size_bytes()
            self.stats.l2_entries = self.l2.count()

            stats_dict = self.stats.to_dict()

            # Add computed metrics
            stats_dict.update({
                'total_hits': self.stats.total_hits,
                'total_misses': self.stats.total_misses,
                'hit_rate': round(self.stats.hit_rate * 100, 2),
                'l1_hit_rate': round(self.stats.l1_hit_rate * 100, 2),
                'l2_hit_rate': round(self.stats.l2_hit_rate * 100, 2),
                'l1_size_mb': round(self.stats.l1_size_bytes / (1024 * 1024), 2),
                'l2_size_mb': round(self.stats.l2_size_bytes / (1024 * 1024), 2),
                'total_size_mb': round(
                    (self.stats.l1_size_bytes + self.stats.l2_size_bytes) / (1024 * 1024),
                    2
                )
            })

            return stats_dict

    @staticmethod
    def generate_key(url: str, content_type: str = "generic") -> str:
        """
        Generate cache key from URL and content type.

        Args:
            url: URL or identifier
            content_type: Type of content

        Returns:
            Cache key (hash)
        """
        key_data = f"{url}:{content_type}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]


# Convenience functions
def create_cache(db_path: str = "data/cache.db") -> ContentCache:
    """
    Create a new content cache instance.

    Args:
        db_path: Path to SQLite database

    Returns:
        ContentCache instance
    """
    return ContentCache(db_path)


__all__ = [
    'ContentCache',
    'CacheEntry',
    'CacheStats',
    'L1Cache',
    'L2Cache',
    'create_cache'
]
