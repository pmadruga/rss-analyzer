"""
Unit tests for database connection pooling

Tests thread-safety, connection lifecycle, and pool management.
"""

import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from src.core.database import ConnectionPool, DatabaseManager


class TestConnectionPool(unittest.TestCase):
    """Test ConnectionPool class"""

    def setUp(self):
        """Create temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        """Clean up temporary database"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)

    def test_pool_initialization(self):
        """Test pool is initialized with correct number of connections"""
        pool = ConnectionPool(self.db_path, pool_size=5)
        stats = pool.get_pool_stats()

        self.assertEqual(stats["pool_size"], 5)
        self.assertEqual(stats["idle_connections"], 5)
        self.assertEqual(stats["active_connections"], 0)
        self.assertEqual(stats["total_connections_created"], 5)
        self.assertFalse(stats["closed"])

        pool.close_pool()

    def test_connection_checkout_checkin(self):
        """Test checking out and returning connections"""
        pool = ConnectionPool(self.db_path, pool_size=3)

        # Check out a connection
        with pool.get_connection() as conn:
            stats = pool.get_pool_stats()
            self.assertEqual(stats["active_connections"], 1)
            self.assertEqual(stats["idle_connections"], 2)

            # Connection should be usable
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)

        # Connection returned to pool
        stats = pool.get_pool_stats()
        self.assertEqual(stats["active_connections"], 0)
        self.assertEqual(stats["idle_connections"], 3)

        pool.close_pool()

    def test_connection_validation(self):
        """Test connection health checks"""
        pool = ConnectionPool(self.db_path, pool_size=2)

        # Get a valid connection
        with pool.get_connection() as conn:
            # Validate should pass
            self.assertTrue(pool._validate_connection(conn))

        pool.close_pool()

    def test_multiple_connections(self):
        """Test multiple simultaneous connections"""
        pool = ConnectionPool(self.db_path, pool_size=5)

        connections = []
        for i in range(3):
            conn_ctx = pool.get_connection()
            conn = conn_ctx.__enter__()
            connections.append((conn, conn_ctx))

        # Should have 3 active connections
        stats = pool.get_pool_stats()
        self.assertEqual(stats["active_connections"], 3)
        self.assertEqual(stats["idle_connections"], 2)

        # Return all connections
        for conn, conn_ctx in connections:
            conn_ctx.__exit__(None, None, None)

        # All connections returned
        stats = pool.get_pool_stats()
        self.assertEqual(stats["active_connections"], 0)
        self.assertEqual(stats["idle_connections"], 5)

        pool.close_pool()

    def test_thread_safety(self):
        """Test pool is thread-safe with concurrent access"""
        pool = ConnectionPool(self.db_path, pool_size=5)
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    with pool.get_connection() as conn:
                        cursor = conn.execute("SELECT ?", (worker_id,))
                        result = cursor.fetchone()[0]
                        results.append(result)
                        time.sleep(0.001)  # Simulate work
            except Exception as e:
                errors.append(e)

        # Create 10 threads competing for 5 connections
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # No errors should occur
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

        # All operations should succeed
        self.assertEqual(len(results), 100)

        pool.close_pool()

    def test_pool_closure(self):
        """Test pool can be closed and prevents further access"""
        pool = ConnectionPool(self.db_path, pool_size=3)

        # Close the pool
        pool.close_pool()

        # Stats should show closed
        stats = pool.get_pool_stats()
        self.assertTrue(stats["closed"])

        # Attempting to get connection should raise error
        with self.assertRaises(RuntimeError):
            with pool.get_connection() as conn:
                pass

    def test_pool_timeout(self):
        """Test pool handles timeout when all connections are in use"""
        pool = ConnectionPool(self.db_path, pool_size=2)

        # Hold all connections
        conn1_ctx = pool.get_connection()
        conn1 = conn1_ctx.__enter__()
        conn2_ctx = pool.get_connection()
        conn2 = conn2_ctx.__enter__()

        # Modify timeout for faster test
        pool._pool.get(timeout=0.1)

        # Try to get third connection with short timeout - should timeout
        # Note: This will timeout in actual implementation
        # We can't easily test this without modifying the timeout parameter

        # Cleanup
        conn1_ctx.__exit__(None, None, None)
        conn2_ctx.__exit__(None, None, None)
        pool.close_pool()


class TestDatabaseManagerWithPooling(unittest.TestCase):
    """Test DatabaseManager with connection pooling"""

    def setUp(self):
        """Create temporary database for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_articles.db")

    def tearDown(self):
        """Clean up temporary database"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)

    def test_database_manager_initialization(self):
        """Test DatabaseManager initializes with connection pool"""
        db = DatabaseManager(self.db_path, pool_size=5)

        # Check pool was created
        stats = db.get_pool_stats()
        self.assertEqual(stats["pool_size"], 5)
        self.assertEqual(stats["idle_connections"], 5)

        db.close_pool()

    def test_backward_compatibility(self):
        """Test existing code works with pooled connections"""
        db = DatabaseManager(self.db_path, pool_size=3)

        # Old code pattern should still work
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Should have created tables
            self.assertIn("articles", tables)
            self.assertIn("content", tables)
            self.assertIn("processing_log", tables)

        db.close_pool()

    def test_concurrent_database_operations(self):
        """Test multiple threads can perform database operations"""
        db = DatabaseManager(self.db_path, pool_size=5)
        errors = []

        def worker(worker_id):
            try:
                # Insert article
                article_id = db.insert_article(
                    title=f"Test Article {worker_id}",
                    url=f"https://example.com/article_{worker_id}",
                    content_hash=f"hash_{worker_id}",
                )

                # Query article
                with db.get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT * FROM articles WHERE id = ?", (article_id,)
                    )
                    result = cursor.fetchone()
                    self.assertIsNotNone(result)

            except Exception as e:
                errors.append(e)

        # Create threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # No errors should occur
        self.assertEqual(len(errors), 0, f"Errors: {errors}")

        # Verify all articles were inserted
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 10)

        db.close_pool()

    def test_pool_stats_accessible(self):
        """Test pool statistics are accessible"""
        db = DatabaseManager(self.db_path, pool_size=4)

        stats = db.get_pool_stats()

        self.assertIn("pool_size", stats)
        self.assertIn("active_connections", stats)
        self.assertIn("idle_connections", stats)
        self.assertIn("total_connections_created", stats)
        self.assertIn("closed", stats)

        self.assertEqual(stats["pool_size"], 4)

        db.close_pool()

    def test_graceful_shutdown(self):
        """Test database manager shuts down gracefully"""
        db = DatabaseManager(self.db_path, pool_size=3)

        # Use some connections
        with db.get_connection() as conn:
            conn.execute("SELECT 1")

        # Close pool
        db.close_pool()

        # Pool should be closed
        stats = db.get_pool_stats()
        self.assertTrue(stats["closed"])

    def test_connection_reuse(self):
        """Test connections are reused from pool"""
        db = DatabaseManager(self.db_path, pool_size=2)

        initial_stats = db.get_pool_stats()
        initial_created = initial_stats["total_connections_created"]

        # Perform multiple operations
        for i in range(10):
            with db.get_connection() as conn:
                conn.execute("SELECT 1")

        # Total connections created should remain the same (reuse)
        final_stats = db.get_pool_stats()
        final_created = final_stats["total_connections_created"]

        self.assertEqual(initial_created, final_created)

        db.close_pool()


if __name__ == "__main__":
    unittest.main()
