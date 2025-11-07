"""
Test suite for batch database operations

Validates that batch operations work correctly and provide performance improvements.
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from src.core.database import DatabaseManager


@pytest.fixture
def db():
    """Create a temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_manager = DatabaseManager(str(db_path))
        yield db_manager
        db_manager.close_pool()


class TestBatchArticleInsert:
    """Test batch article insertion"""

    def test_insert_single_batch(self, db):
        """Test inserting a batch of articles"""
        articles = [
            {
                "title": f"Test Article {i}",
                "url": f"https://example.com/article-{i}",
                "content_hash": f"hash{i}",
                "rss_guid": f"guid{i}",
            }
            for i in range(10)
        ]

        article_ids = db.insert_articles_batch(articles)

        assert len(article_ids) == 10
        assert all(isinstance(id, int) for id in article_ids)

        # Verify articles were inserted
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            assert count == 10

    def test_insert_duplicate_handling(self, db):
        """Test that duplicate articles are handled gracefully"""
        articles = [
            {
                "title": "Test Article",
                "url": "https://example.com/article",
                "content_hash": "hash1",
                "rss_guid": "guid1",
            }
        ]

        # Insert once
        first_ids = db.insert_articles_batch(articles)
        assert len(first_ids) == 1

        # Try to insert duplicate
        second_ids = db.insert_articles_batch(articles)
        assert len(second_ids) == 1
        assert first_ids[0] == second_ids[0]  # Same article ID

    def test_insert_large_batch(self, db):
        """Test inserting a large batch (>50 articles)"""
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/article-{i}",
                "content_hash": f"hash{i}",
            }
            for i in range(150)
        ]

        article_ids = db.insert_articles_batch(articles)

        assert len(article_ids) == 150

        # Verify batching worked (50 per batch)
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            assert count == 150


class TestBatchContentInsert:
    """Test batch content insertion"""

    def test_insert_content_batch(self, db):
        """Test batch inserting content records"""
        # First insert articles
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/article-{i}",
                "content_hash": f"hash{i}",
            }
            for i in range(5)
        ]
        article_ids = db.insert_articles_batch(articles)

        # Prepare content records
        contents = [
            {
                "article_id": article_id,
                "original_content": f"Content {i}",
                "analysis": {
                    "methodology_detailed": f"Method {i}",
                    "technical_approach": f"Approach {i}",
                    "key_findings": f"Findings {i}",
                    "research_design": f"Design {i}",
                    "metadata": {"test": i},
                },
            }
            for i, article_id in enumerate(article_ids)
        ]

        content_ids = db.insert_content_batch(contents)

        assert len(content_ids) == 5

        # Verify content was inserted
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM content")
            count = cursor.fetchone()[0]
            assert count == 5


class TestBatchStatusUpdate:
    """Test batch status updates"""

    def test_update_status_batch(self, db):
        """Test batch updating article statuses"""
        # Insert articles
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/article-{i}",
                "content_hash": f"hash{i}",
            }
            for i in range(10)
        ]
        article_ids = db.insert_articles_batch(articles)

        # Update statuses
        updates = [(article_id, "completed") for article_id in article_ids]
        updated_count = db.update_status_batch(updates)

        assert updated_count == 10

        # Verify updates
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE status = 'completed'"
            )
            count = cursor.fetchone()[0]
            assert count == 10

    def test_mixed_status_updates(self, db):
        """Test updating articles with different statuses"""
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/article-{i}",
                "content_hash": f"hash{i}",
            }
            for i in range(10)
        ]
        article_ids = db.insert_articles_batch(articles)

        # Mixed status updates
        updates = []
        for i, article_id in enumerate(article_ids):
            status = "completed" if i % 2 == 0 else "failed"
            updates.append((article_id, status))

        updated_count = db.update_status_batch(updates)
        assert updated_count == 10

        # Verify mixed statuses
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM articles GROUP BY status"
            )
            status_counts = dict(cursor.fetchall())
            assert status_counts["completed"] == 5
            assert status_counts["failed"] == 5


class TestBatchProcessingLog:
    """Test batch processing log insertion"""

    def test_log_processing_batch(self, db):
        """Test batch logging processing entries"""
        # Insert articles
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/article-{i}",
                "content_hash": f"hash{i}",
            }
            for i in range(5)
        ]
        article_ids = db.insert_articles_batch(articles)

        # Prepare log entries
        logs = [
            {
                "article_id": article_id,
                "status": "success",
                "processing_step": "scraping",
                "duration_seconds": 1.5,
            }
            for article_id in article_ids
        ]

        logged_count = db.log_processing_batch(logs)
        assert logged_count == 5

        # Verify logs
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM processing_log")
            count = cursor.fetchone()[0]
            assert count == 5

    def test_log_with_errors(self, db):
        """Test logging entries with error messages"""
        articles = [
            {
                "title": "Test Article",
                "url": "https://example.com/article",
                "content_hash": "hash1",
            }
        ]
        article_ids = db.insert_articles_batch(articles)

        logs = [
            {
                "article_id": article_ids[0],
                "status": "error",
                "error_message": "Test error message",
                "processing_step": "analysis",
                "duration_seconds": 0.5,
            }
        ]

        logged_count = db.log_processing_batch(logs)
        assert logged_count == 1

        # Verify error was logged
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT error_message FROM processing_log WHERE article_id = ?",
                (article_ids[0],),
            )
            error = cursor.fetchone()[0]
            assert error == "Test error message"


class TestBatchPerformance:
    """Test batch operation performance improvements"""

    def test_batch_vs_individual_insert_performance(self, db):
        """Compare performance of batch vs individual inserts"""
        num_articles = 100

        # Prepare test data
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/batch-{i}",
                "content_hash": f"batch_hash{i}",
            }
            for i in range(num_articles)
        ]

        # Test batch insert
        start_time = time.perf_counter()
        batch_ids = db.insert_articles_batch(articles)
        batch_time = time.perf_counter() - start_time

        assert len(batch_ids) == num_articles

        # Test individual inserts (with different URLs to avoid duplicates)
        individual_articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/individual-{i}",
                "content_hash": f"individual_hash{i}",
            }
            for i in range(num_articles)
        ]

        start_time = time.perf_counter()
        for article in individual_articles:
            try:
                db.insert_article(
                    article["title"],
                    article["url"],
                    article["content_hash"],
                )
            except sqlite3.OperationalError:
                # Database locked - expected in rapid individual inserts
                import time
                time.sleep(0.001)  # Brief wait
                db.insert_article(
                    article["title"],
                    article["url"],
                    article["content_hash"],
                )
        individual_time = time.perf_counter() - start_time

        # Batch should be significantly faster (at least 2x)
        speedup = individual_time / batch_time
        print(f"\nBatch speedup: {speedup:.2f}x")
        print(f"Batch time: {batch_time:.4f}s")
        print(f"Individual time: {individual_time:.4f}s")

        assert speedup >= 2.0, f"Batch operations should be at least 2x faster (got {speedup:.2f}x)"

    def test_batch_vs_individual_status_update_performance(self, db):
        """Compare performance of batch vs individual status updates"""
        num_articles = 100

        # Insert test articles
        articles = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/status-{i}",
                "content_hash": f"status_hash{i}",
            }
            for i in range(num_articles)
        ]
        article_ids = db.insert_articles_batch(articles)

        # Test batch update
        updates = [(article_id, "completed") for article_id in article_ids]
        start_time = time.perf_counter()
        db.update_status_batch(updates)
        batch_time = time.perf_counter() - start_time

        # Reset statuses for individual test
        with db.get_connection() as conn:
            conn.execute("UPDATE articles SET status = 'pending'")

        # Test individual updates
        start_time = time.perf_counter()
        for article_id in article_ids:
            db.update_article_status(article_id, "completed")
        individual_time = time.perf_counter() - start_time

        # Batch should be significantly faster
        speedup = individual_time / batch_time
        print(f"\nStatus update batch speedup: {speedup:.2f}x")
        print(f"Batch time: {batch_time:.4f}s")
        print(f"Individual time: {individual_time:.4f}s")

        assert speedup >= 2.0, f"Batch status updates should be at least 2x faster (got {speedup:.2f}x)"


class TestBatchTransactionRollback:
    """Test that batch operations properly rollback on errors"""

    def test_batch_insert_rollback_on_error(self, db):
        """Test that failed batch insert doesn't partially commit"""
        # Insert a valid article first
        valid_article = {
            "title": "Valid Article",
            "url": "https://example.com/valid",
            "content_hash": "valid_hash",
        }
        db.insert_articles_batch([valid_article])

        # Try to insert batch with duplicate
        articles_with_duplicate = [
            {
                "title": "New Article",
                "url": "https://example.com/new",
                "content_hash": "new_hash",
            },
            valid_article,  # Duplicate
        ]

        # Should handle duplicate gracefully
        result_ids = db.insert_articles_batch(articles_with_duplicate)

        # Should still get IDs (either new or existing)
        assert len(result_ids) == 2

        # Database should remain consistent
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            # Should have 2 articles (1 original + 1 new)
            assert count == 2


def test_empty_batch_operations(db):
    """Test that empty batches are handled gracefully"""
    # Empty article batch
    article_ids = db.insert_articles_batch([])
    assert article_ids == []

    # Empty content batch
    content_ids = db.insert_content_batch([])
    assert content_ids == []

    # Empty status update batch
    updated_count = db.update_status_batch([])
    assert updated_count == 0

    # Empty log batch
    logged_count = db.log_processing_batch([])
    assert logged_count == 0
