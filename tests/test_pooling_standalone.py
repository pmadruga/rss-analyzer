"""
Standalone test for connection pooling functionality

Tests the connection pool without importing full module dependencies.
"""

import os
import sqlite3
import sys
import tempfile
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_connection_pool():
    """Test basic connection pooling functionality"""
    print("Testing Connection Pool...")

    # Import here to avoid module-level import issues
    from src.core.database import ConnectionPool

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    try:
        # Test 1: Pool initialization
        print("  ✓ Test 1: Pool initialization...")
        pool = ConnectionPool(db_path, pool_size=5)
        stats = pool.get_pool_stats()
        assert stats["pool_size"] == 5, f"Expected pool_size=5, got {stats['pool_size']}"
        assert stats["idle_connections"] == 5, f"Expected 5 idle connections"
        assert stats["active_connections"] == 0, f"Expected 0 active connections"
        print(f"    Pool stats: {stats}")

        # Test 2: Connection checkout
        print("  ✓ Test 2: Connection checkout/checkin...")
        with pool.get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1, "Connection query failed"
            stats = pool.get_pool_stats()
            assert stats["active_connections"] == 1, "Expected 1 active connection"

        stats = pool.get_pool_stats()
        assert stats["active_connections"] == 0, "Connection not returned to pool"
        print("    Connection properly returned to pool")

        # Test 3: Multiple connections
        print("  ✓ Test 3: Multiple simultaneous connections...")
        connections = []
        for i in range(3):
            conn_ctx = pool.get_connection()
            conn = conn_ctx.__enter__()
            connections.append((conn, conn_ctx))

        stats = pool.get_pool_stats()
        assert stats["active_connections"] == 3, f"Expected 3 active, got {stats['active_connections']}"
        assert stats["idle_connections"] == 2, f"Expected 2 idle, got {stats['idle_connections']}"

        for conn, conn_ctx in connections:
            conn_ctx.__exit__(None, None, None)

        stats = pool.get_pool_stats()
        assert stats["active_connections"] == 0, "All connections should be returned"
        print("    All connections properly managed")

        # Test 4: Thread safety
        print("  ✓ Test 4: Thread safety with concurrent access...")
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(5):
                    with pool.get_connection() as conn:
                        cursor = conn.execute("SELECT ?", (worker_id,))
                        result = cursor.fetchone()[0]
                        results.append(result)
                        time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        assert len(results) == 50, f"Expected 50 results, got {len(results)}"
        print(f"    50 concurrent operations completed successfully")

        # Test 5: Connection reuse
        print("  ✓ Test 5: Connection reuse...")
        initial_created = stats["total_connections_created"]
        for i in range(20):
            with pool.get_connection() as conn:
                conn.execute("SELECT 1")

        final_stats = pool.get_pool_stats()
        assert final_stats["total_connections_created"] == initial_created, \
            "Connections should be reused, not created"
        print(f"    Connections reused: {initial_created} total created")

        # Test 6: Pool closure
        print("  ✓ Test 6: Pool closure...")
        pool.close_pool()
        stats = pool.get_pool_stats()
        assert stats["closed"] == True, "Pool should be marked as closed"

        try:
            with pool.get_connection() as conn:
                pass
            assert False, "Should raise error on closed pool"
        except RuntimeError:
            pass  # Expected
        print("    Pool properly closed")

        print("\n✅ All Connection Pool tests passed!\n")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        os.rmdir(temp_dir)


def test_database_manager():
    """Test DatabaseManager with connection pooling"""
    print("Testing DatabaseManager with Pooling...")

    from src.core.database import DatabaseManager

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_articles.db")

    try:
        # Test 1: Initialization
        print("  ✓ Test 1: DatabaseManager initialization...")
        db = DatabaseManager(db_path, pool_size=5)
        stats = db.get_pool_stats()
        assert stats["pool_size"] == 5, "Pool should have 5 connections"
        print(f"    Initialized with pool size: {stats['pool_size']}")

        # Test 2: Backward compatibility
        print("  ✓ Test 2: Backward compatibility...")
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "articles" in tables, "articles table should exist"
            assert "content" in tables, "content table should exist"
        print("    Existing API works correctly")

        # Test 3: Concurrent operations
        print("  ✓ Test 3: Concurrent database operations...")
        errors = []

        def worker(worker_id):
            try:
                article_id = db.insert_article(
                    title=f"Test Article {worker_id}",
                    url=f"https://example.com/article_{worker_id}",
                    content_hash=f"hash_{worker_id}",
                )
                assert article_id is not None
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Concurrent operation errors: {errors}"

        with db.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            assert count == 10, f"Expected 10 articles, got {count}"
        print(f"    10 concurrent inserts completed successfully")

        # Test 4: Pool statistics
        print("  ✓ Test 4: Pool statistics accessible...")
        stats = db.get_pool_stats()
        assert "pool_size" in stats
        assert "active_connections" in stats
        assert "idle_connections" in stats
        print(f"    Stats: {stats}")

        # Test 5: Graceful shutdown
        print("  ✓ Test 5: Graceful shutdown...")
        db.close_pool()
        stats = db.get_pool_stats()
        assert stats["closed"] == True, "Pool should be closed"
        print("    Database manager shut down gracefully")

        print("\n✅ All DatabaseManager tests passed!\n")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        os.rmdir(temp_dir)


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Connection Pooling Test Suite")
    print("=" * 60 + "\n")

    try:
        test_connection_pool()
        test_database_manager()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
