"""
Unit tests for the two-tier content caching system.
"""

import os
import sys
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cache import (
    ContentCache,
    CacheEntry,
    CacheStats,
    L1Cache,
    L2Cache,
    create_cache
)


class TestCacheEntry:
    """Test CacheEntry functionality."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)

        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            content_type="json",
            created_at=now,
            expires_at=expires
        )

        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.content_type == "json"
        assert entry.created_at == now
        assert entry.expires_at == expires
        assert entry.access_count == 0
        assert entry.last_accessed is None

    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        now = datetime.utcnow()

        # Not expired
        entry = CacheEntry(
            key="test",
            value="data",
            content_type="generic",
            created_at=now,
            expires_at=now + timedelta(hours=1)
        )
        assert not entry.is_expired()

        # Expired
        expired_entry = CacheEntry(
            key="test",
            value="data",
            content_type="generic",
            created_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1)
        )
        assert expired_entry.is_expired()

    def test_cache_entry_to_dict(self):
        """Test converting cache entry to dictionary."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)

        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            content_type="json",
            created_at=now,
            expires_at=expires,
            access_count=5,
            last_accessed=now,
            size_bytes=100
        )

        entry_dict = entry.to_dict()

        assert entry_dict['key'] == "test_key"
        assert entry_dict['content_type'] == "json"
        assert entry_dict['access_count'] == 5
        assert entry_dict['size_bytes'] == 100
        assert 'created_at' in entry_dict
        assert 'expires_at' in entry_dict


class TestCacheStats:
    """Test CacheStats functionality."""

    def test_cache_stats_initialization(self):
        """Test initializing cache stats."""
        stats = CacheStats()

        assert stats.l1_hits == 0
        assert stats.l1_misses == 0
        assert stats.l2_hits == 0
        assert stats.l2_misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0

    def test_cache_stats_calculations(self):
        """Test cache statistics calculations."""
        stats = CacheStats(
            l1_hits=70,
            l1_misses=30,
            l2_hits=20,
            l2_misses=10
        )

        assert stats.total_hits == 90
        assert stats.total_misses == 40
        assert stats.hit_rate == pytest.approx(0.692, rel=0.01)
        assert stats.l1_hit_rate == pytest.approx(0.7, rel=0.01)
        assert stats.l2_hit_rate == pytest.approx(0.666, rel=0.01)

    def test_cache_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CacheStats(l1_hits=10, l1_misses=5)
        stats_dict = stats.to_dict()

        assert stats_dict['l1_hits'] == 10
        assert stats_dict['l1_misses'] == 5


class TestL1Cache:
    """Test L1 in-memory cache."""

    def test_l1_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = L1Cache()
        now = datetime.utcnow()

        entry = CacheEntry(
            key="test",
            value="data",
            content_type="generic",
            created_at=now,
            expires_at=now + timedelta(hours=1)
        )

        assert cache.set("test", entry)
        retrieved = cache.get("test")

        assert retrieved is not None
        assert retrieved.value == "data"
        assert retrieved.access_count == 1

    def test_l1_cache_expiration(self):
        """Test that expired entries are removed."""
        cache = L1Cache()
        now = datetime.utcnow()

        entry = CacheEntry(
            key="test",
            value="data",
            content_type="generic",
            created_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1)
        )

        cache.set("test", entry)
        retrieved = cache.get("test")

        assert retrieved is None
        assert cache.count() == 0

    def test_l1_cache_lru_eviction(self):
        """Test LRU eviction when size limit is reached."""
        cache = L1Cache()
        now = datetime.utcnow()

        # Fill cache with large entries
        large_value = "x" * (50 * 1024 * 1024)  # 50MB each

        for i in range(6):  # Should evict first 2 to stay under 256MB
            entry = CacheEntry(
                key=f"test_{i}",
                value=large_value,
                content_type="generic",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )
            cache.set(f"test_{i}", entry)

        # First entries should be evicted
        assert cache.get("test_0") is None
        assert cache.get("test_1") is None

        # Recent entries should still exist
        assert cache.get("test_5") is not None

    def test_l1_cache_delete(self):
        """Test deleting entries."""
        cache = L1Cache()
        now = datetime.utcnow()

        entry = CacheEntry(
            key="test",
            value="data",
            content_type="generic",
            created_at=now,
            expires_at=now + timedelta(hours=1)
        )

        cache.set("test", entry)
        assert cache.count() == 1

        assert cache.delete("test")
        assert cache.count() == 0
        assert cache.get("test") is None

    def test_l1_cache_clear(self):
        """Test clearing all entries."""
        cache = L1Cache()
        now = datetime.utcnow()

        for i in range(5):
            entry = CacheEntry(
                key=f"test_{i}",
                value=f"data_{i}",
                content_type="generic",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )
            cache.set(f"test_{i}", entry)

        assert cache.count() == 5

        cache.clear()
        assert cache.count() == 0
        assert cache.size_bytes() == 0

    def test_l1_cache_move_to_end(self):
        """Test that accessed entries are moved to end (LRU)."""
        cache = L1Cache()
        now = datetime.utcnow()

        # Add multiple entries
        for i in range(3):
            entry = CacheEntry(
                key=f"test_{i}",
                value=f"data_{i}",
                content_type="generic",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )
            cache.set(f"test_{i}", entry)

        # Access first entry (should move to end)
        cache.get("test_0")

        # Fill cache to trigger eviction
        large_value = "x" * (100 * 1024 * 1024)  # 100MB
        for i in range(3, 6):
            entry = CacheEntry(
                key=f"test_{i}",
                value=large_value,
                content_type="generic",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )
            cache.set(f"test_{i}", entry)

        # test_0 should still exist (was recently accessed)
        # test_1 might be evicted (least recently used)
        assert cache.get("test_1") is None


class TestL2Cache:
    """Test L2 persistent cache."""

    def test_l2_cache_set_and_get(self):
        """Test basic set and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = L2Cache(str(db_path))

            now = datetime.utcnow()
            entry = CacheEntry(
                key="test",
                value={"data": "test_value"},
                content_type="json",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )

            assert cache.set("test", entry)
            retrieved = cache.get("test")

            assert retrieved is not None
            assert retrieved.value == {"data": "test_value"}
            assert retrieved.content_type == "json"
            assert retrieved.access_count == 1

    def test_l2_cache_compression(self):
        """Test that data is compressed in storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = L2Cache(str(db_path))

            now = datetime.utcnow()
            # Large repeating data compresses well
            large_data = "x" * 10000

            entry = CacheEntry(
                key="test",
                value=large_data,
                content_type="text",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )

            cache.set("test", entry)

            # Check that stored size is less than original
            stored_size = cache.size_bytes()
            original_size = sys.getsizeof(large_data)

            assert stored_size < original_size

    def test_l2_cache_persistence(self):
        """Test that data persists across cache instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"

            # Create cache and add entry
            cache1 = L2Cache(str(db_path))
            now = datetime.utcnow()

            entry = CacheEntry(
                key="test",
                value="persistent_data",
                content_type="text",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )

            cache1.set("test", entry)

            # Create new cache instance with same database
            cache2 = L2Cache(str(db_path))
            retrieved = cache2.get("test")

            assert retrieved is not None
            assert retrieved.value == "persistent_data"

    def test_l2_cache_expiration(self):
        """Test that expired entries are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = L2Cache(str(db_path))

            now = datetime.utcnow()
            entry = CacheEntry(
                key="test",
                value="data",
                content_type="generic",
                created_at=now - timedelta(hours=2),
                expires_at=now - timedelta(hours=1)
            )

            cache.set("test", entry)
            retrieved = cache.get("test")

            assert retrieved is None

    def test_l2_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = L2Cache(str(db_path))

            now = datetime.utcnow()

            # Add expired entries
            for i in range(3):
                entry = CacheEntry(
                    key=f"expired_{i}",
                    value=f"data_{i}",
                    content_type="generic",
                    created_at=now - timedelta(hours=2),
                    expires_at=now - timedelta(hours=1)
                )
                cache.set(f"expired_{i}", entry)

            # Add valid entry
            valid_entry = CacheEntry(
                key="valid",
                value="valid_data",
                content_type="generic",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )
            cache.set("valid", valid_entry)

            # Cleanup should remove 3 expired entries
            removed = cache.cleanup_expired()
            assert removed == 3
            assert cache.count() == 1

    def test_l2_cache_delete(self):
        """Test deleting entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = L2Cache(str(db_path))

            now = datetime.utcnow()
            entry = CacheEntry(
                key="test",
                value="data",
                content_type="generic",
                created_at=now,
                expires_at=now + timedelta(hours=1)
            )

            cache.set("test", entry)
            assert cache.count() == 1

            assert cache.delete("test")
            assert cache.count() == 0

    def test_l2_cache_clear(self):
        """Test clearing all entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = L2Cache(str(db_path))

            now = datetime.utcnow()
            for i in range(5):
                entry = CacheEntry(
                    key=f"test_{i}",
                    value=f"data_{i}",
                    content_type="generic",
                    created_at=now,
                    expires_at=now + timedelta(hours=1)
                )
                cache.set(f"test_{i}", entry)

            assert cache.count() == 5

            cache.clear()
            assert cache.count() == 0


class TestContentCache:
    """Test integrated two-tier content cache."""

    def test_content_cache_initialization(self):
        """Test cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            assert cache.l1 is not None
            assert cache.l2 is not None
            assert cache.stats is not None

    def test_content_cache_set_and_get(self):
        """Test basic set and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            cache.set("test_key", "test_value", ttl=3600, content_type="text")
            value = cache.get("test_key")

            assert value == "test_value"
            assert cache.stats.l1_hits == 1

    def test_content_cache_l1_miss_l2_hit(self):
        """Test L1 miss followed by L2 hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            # Add to cache
            cache.set("test_key", "test_value", ttl=3600)

            # Clear L1 to force L2 lookup
            cache.l1.clear()

            # Should hit L2 and promote to L1
            value = cache.get("test_key")

            assert value == "test_value"
            assert cache.stats.l1_misses == 1
            assert cache.stats.l2_hits == 1

            # Next access should hit L1
            value2 = cache.get("test_key")
            assert value2 == "test_value"
            assert cache.stats.l1_hits == 1

    def test_content_cache_miss(self):
        """Test cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            value = cache.get("nonexistent")

            assert value is None
            assert cache.stats.l1_misses == 1
            assert cache.stats.l2_misses == 1

    def test_content_cache_ttl(self):
        """Test TTL functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            # Set with short TTL
            cache.set("test_key", "test_value", ttl=1)

            # Should be cached immediately
            assert cache.get("test_key") == "test_value"

            # Wait for expiration
            time.sleep(2)

            # Should be expired
            assert cache.get("test_key") is None

    def test_content_cache_delete(self):
        """Test deleting from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            cache.set("test_key", "test_value")
            assert cache.get("test_key") == "test_value"

            assert cache.delete("test_key")
            assert cache.get("test_key") is None

    def test_content_cache_clear(self):
        """Test clearing entire cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            for i in range(5):
                cache.set(f"key_{i}", f"value_{i}")

            stats = cache.get_stats()
            assert stats['l1_entries'] == 5
            assert stats['l2_entries'] == 5

            cache.clear()

            stats = cache.get_stats()
            assert stats['l1_entries'] == 0
            assert stats['l2_entries'] == 0

    def test_content_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            # Add expired entries
            for i in range(3):
                cache.set(f"expired_{i}", f"value_{i}", ttl=1)

            time.sleep(2)

            # Add valid entry
            cache.set("valid", "valid_value", ttl=3600)

            # Cleanup
            removed = cache.cleanup_expired()
            assert removed == 3

    def test_content_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            # Generate some cache activity
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.get("key1")  # L1 hit
            cache.l1.clear()
            cache.get("key1")  # L1 miss, L2 hit
            cache.get("nonexistent")  # Both miss

            stats = cache.get_stats()

            assert stats['l1_hits'] == 1
            assert stats['l1_misses'] == 2
            assert stats['l2_hits'] == 1
            assert stats['l2_misses'] == 1
            assert stats['total_hits'] == 2
            assert stats['total_misses'] == 3
            assert 'hit_rate' in stats
            assert 'l1_size_mb' in stats
            assert 'l2_size_mb' in stats

    def test_content_cache_generate_key(self):
        """Test cache key generation."""
        key1 = ContentCache.generate_key("https://example.com/article", "scraped")
        key2 = ContentCache.generate_key("https://example.com/article", "scraped")
        key3 = ContentCache.generate_key("https://example.com/article", "api")

        # Same URL and type should generate same key
        assert key1 == key2

        # Different type should generate different key
        assert key1 != key3

        # Should be reasonable length (SHA256 truncated to 32 chars)
        assert len(key1) == 32

    def test_content_cache_content_types(self):
        """Test different content types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = ContentCache(str(db_path))

            # Test different content types
            cache.set("scraped_content", {"html": "<div>test</div>"},
                     ttl=ContentCache.TTL_SCRAPED_CONTENT, content_type="scraped")

            cache.set("api_response", {"result": "success"},
                     ttl=ContentCache.TTL_API_RESPONSE, content_type="api")

            assert cache.get("scraped_content") is not None
            assert cache.get("api_response") is not None

    def test_create_cache_factory(self):
        """Test cache factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = create_cache(str(db_path))

            assert isinstance(cache, ContentCache)
            assert cache.l1 is not None
            assert cache.l2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
