"""
Unit Tests for Deduplication Manager

Tests for hash generation, duplicate detection, caching, and batch operations.
"""

import unittest
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.database import DatabaseManager
from deduplication_manager import DeduplicationManager, LRUCache


class TestLRUCache(unittest.TestCase):
    """Tests for LRU cache implementation."""

    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        cache = LRUCache(capacity=3)

        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Get values
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")

    def test_cache_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(capacity=2)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Add third item (should evict key1)
        cache.set("key3", "value3")

        # key1 should be evicted
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")

    def test_cache_update(self):
        """Test updating existing cache entries."""
        cache = LRUCache(capacity=2)

        cache.set("key1", "value1")
        cache.set("key1", "updated")

        self.assertEqual(cache.get("key1"), "updated")
        self.assertEqual(cache.size(), 1)

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(capacity=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        self.assertEqual(cache.size(), 2)

        cache.clear()
        self.assertEqual(cache.size(), 0)
        self.assertIsNone(cache.get("key1"))


class TestHashGeneration(unittest.TestCase):
    """Tests for hash generation functions."""

    def test_content_hash_deterministic(self):
        """Test content hash is deterministic."""
        content = "This is test article content."

        hash1 = DeduplicationManager.generate_content_hash(content)
        hash2 = DeduplicationManager.generate_content_hash(content)

        self.assertEqual(hash1, hash2)

    def test_content_hash_length(self):
        """Test content hash has correct length (SHA-256 = 64 chars)."""
        content = "Test content"
        hash_value = DeduplicationManager.generate_content_hash(content)

        self.assertEqual(len(hash_value), 64)

    def test_content_hash_normalization(self):
        """Test content normalization in hashing."""
        content1 = "Test Content"
        content2 = "test content"
        content3 = "TEST   CONTENT"

        hash1 = DeduplicationManager.generate_content_hash(content1)
        hash2 = DeduplicationManager.generate_content_hash(content2)
        hash3 = DeduplicationManager.generate_content_hash(content3)

        # All should normalize to same hash
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)

    def test_url_hash_deterministic(self):
        """Test URL hash is deterministic."""
        url = "https://example.com/article"

        hash1 = DeduplicationManager.generate_url_hash(url)
        hash2 = DeduplicationManager.generate_url_hash(url)

        self.assertEqual(hash1, hash2)

    def test_url_hash_normalization(self):
        """Test URL normalization in hashing."""
        url1 = "https://Example.com/Article"
        url2 = "https://example.com/article"

        hash1 = DeduplicationManager.generate_url_hash(url1)
        hash2 = DeduplicationManager.generate_url_hash(url2)

        # Should normalize to same hash
        self.assertEqual(hash1, hash2)

    def test_different_content_different_hash(self):
        """Test different content produces different hashes."""
        content1 = "First article content"
        content2 = "Second article content"

        hash1 = DeduplicationManager.generate_content_hash(content1)
        hash2 = DeduplicationManager.generate_content_hash(content2)

        self.assertNotEqual(hash1, hash2)


class TestDeduplicationManager(unittest.TestCase):
    """Tests for DeduplicationManager class."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self.db = DatabaseManager(self.db_path)
        self.dedup = DeduplicationManager(self.db, cache_capacity=10)

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_duplicate_detection_url(self):
        """Test URL-based duplicate detection."""
        url = "https://example.com/article1"
        title = "Test Article"
        content = "Article content"

        # First check - not duplicate
        is_dup, reason = self.dedup.is_duplicate(url)
        self.assertFalse(is_dup)

        # Insert article
        content_hash = self.dedup.generate_content_hash(content)
        url_hash = self.dedup.generate_url_hash(url)

        article_id = self.db.insert_article(title, url, content_hash)
        self.dedup.mark_processed(article_id, url, content_hash, url_hash)

        # Second check - is duplicate
        is_dup, reason = self.dedup.is_duplicate(url)
        self.assertTrue(is_dup)
        self.assertEqual(reason, "url")

    def test_duplicate_detection_content(self):
        """Test content-based duplicate detection."""
        url1 = "https://example.com/article1"
        url2 = "https://example.com/article2"
        title = "Test Article"
        content = "This is the article content"

        # Insert first article
        content_hash = self.dedup.generate_content_hash(content)
        url_hash1 = self.dedup.generate_url_hash(url1)

        article_id = self.db.insert_article(title, url1, content_hash)
        self.dedup.mark_processed(article_id, url1, content_hash, url_hash1)

        # Check with different URL but same content
        is_dup, reason = self.dedup.is_duplicate(url2, content)
        self.assertTrue(is_dup)
        self.assertEqual(reason, "content")

    def test_not_duplicate(self):
        """Test non-duplicate detection."""
        url = "https://example.com/unique-article"
        content = "Unique article content"

        is_dup, reason = self.dedup.is_duplicate(url, content)

        self.assertFalse(is_dup)
        self.assertIsNone(reason)

    def test_batch_check_duplicates(self):
        """Test batch duplicate checking."""
        # Insert some articles
        articles_to_insert = [
            ("https://example.com/article1", "Content 1"),
            ("https://example.com/article2", "Content 2"),
        ]

        for url, content in articles_to_insert:
            content_hash = self.dedup.generate_content_hash(content)
            article_id = self.db.insert_article("Title", url, content_hash)
            self.dedup.mark_processed(article_id, url, content_hash)

        # Batch check with mix of new and duplicate
        articles_to_check = [
            {"url": "https://example.com/article1", "content": "Content 1"},  # Dup
            {"url": "https://example.com/article3", "content": "Content 3"},  # New
            {"url": "https://example.com/article2", "content": "Content 2"},  # Dup
            {"url": "https://example.com/article4", "content": "Content 4"},  # New
        ]

        results = self.dedup.batch_check_duplicates(articles_to_check)

        # Check results
        self.assertEqual(len(results), 4)
        self.assertTrue(results[0]["is_duplicate"])  # article1
        self.assertFalse(results[1]["is_duplicate"])  # article3
        self.assertTrue(results[2]["is_duplicate"])  # article2
        self.assertFalse(results[3]["is_duplicate"])  # article4

    def test_cache_hit_miss_tracking(self):
        """Test cache hit/miss statistics tracking."""
        url = "https://example.com/test"
        content = "Test content"

        # First check - cache miss
        self.dedup.is_duplicate(url, content)
        stats = self.dedup.get_duplicate_stats()
        self.assertEqual(stats["cache_stats"]["cache_misses"], 2)  # URL + content

        # Insert and mark processed
        content_hash = self.dedup.generate_content_hash(content)
        article_id = self.db.insert_article("Title", url, content_hash)
        self.dedup.mark_processed(article_id, url, content_hash)

        # Second check - cache hit
        self.dedup.is_duplicate(url, content)
        stats = self.dedup.get_duplicate_stats()
        self.assertGreater(stats["cache_stats"]["cache_hits"], 0)

    def test_memory_usage_estimate(self):
        """Test memory usage estimation."""
        memory = self.dedup.get_memory_usage_estimate()

        self.assertIn("content_cache_mb", memory)
        self.assertIn("url_cache_mb", memory)
        self.assertIn("total_mb", memory)
        self.assertIn("estimated_max_mb", memory)

        # Should be reasonable values
        total_mb = float(memory["total_mb"])
        self.assertGreaterEqual(total_mb, 0)
        self.assertLess(total_mb, 1000)  # Should be < 1GB

    def test_statistics(self):
        """Test statistics gathering."""
        stats = self.dedup.get_duplicate_stats()

        self.assertIn("articles_processed", stats)
        self.assertIn("duplicates_detected", stats)
        self.assertIn("duplicate_rate", stats)
        self.assertIn("cache_stats", stats)

        cache_stats = stats["cache_stats"]
        self.assertIn("cache_hit_rate", cache_stats)
        self.assertIn("content_cache_size", cache_stats)
        self.assertIn("url_cache_size", cache_stats)


class TestPerformance(unittest.TestCase):
    """Performance tests for deduplication."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self.db = DatabaseManager(self.db_path)
        self.dedup = DeduplicationManager(self.db, cache_capacity=1000)

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_lookup_speed(self):
        """Test lookup speed is < 1ms."""
        import time

        url = "https://example.com/article"
        content = "Test content"

        # Insert article
        content_hash = self.dedup.generate_content_hash(content)
        article_id = self.db.insert_article("Title", url, content_hash)
        self.dedup.mark_processed(article_id, url, content_hash)

        # Measure lookup time (cache hit)
        start = time.time()
        self.dedup.is_duplicate(url)
        duration = time.time() - start

        # Should be < 1ms
        self.assertLess(duration * 1000, 1.0)

    def test_batch_throughput(self):
        """Test batch processing throughput."""
        import time

        # Create batch of articles
        batch_size = 100
        articles = [
            {
                "url": f"https://example.com/article{i}",
                "content": f"Content for article {i}"
            }
            for i in range(batch_size)
        ]

        # Measure batch processing time
        start = time.time()
        self.dedup.batch_check_duplicates(articles)
        duration = time.time() - start

        throughput = batch_size / duration

        # Should process > 100 articles/sec
        self.assertGreater(throughput, 100)


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
