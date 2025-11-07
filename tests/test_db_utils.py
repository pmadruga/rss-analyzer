"""
Test database utilities for concurrent operations
"""

import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.core.db_utils import (
    exponential_backoff_retry,
    safe_executemany,
    batch_write_context,
    BatchWriter,
    AdvisoryLock,
)


def test_exponential_backoff():
    """Test exponential backoff retry mechanism"""
    print("\n🧪 Testing exponential backoff retry...")

    attempts = []

    @exponential_backoff_retry(max_retries=3, base_delay=0.05)
    def flaky_operation():
        attempts.append(time.time())
        if len(attempts) < 3:
            raise sqlite3.OperationalError("database is locked")
        return "success"

    result = flaky_operation()

    assert result == "success"
    assert len(attempts) == 3
    print(f"✅ Retry worked after {len(attempts)} attempts")


def test_advisory_lock():
    """Test advisory locking mechanism"""
    print("\n🧪 Testing advisory locking...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Test acquiring lock
        with AdvisoryLock(conn, "test_lock") as lock:
            assert lock.acquired
            print("✅ Advisory lock acquired")

        # Lock should be released
        assert not lock.acquired
        print("✅ Advisory lock released")

        conn.close()


def test_batch_write_context():
    """Test batch write context manager"""
    print("\n🧪 Testing batch write context...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create test table
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()

        # Use batch write context
        with batch_write_context(conn, "test_batch") as cursor:
            cursor.executemany(
                "INSERT INTO test (value) VALUES (?)",
                [("value1",), ("value2",), ("value3",)]
            )

        # Verify writes
        cursor = conn.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        assert count == 3
        print(f"✅ Batch write inserted {count} records")

        conn.close()


def test_safe_executemany():
    """Test safe_executemany with retry"""
    print("\n🧪 Testing safe_executemany...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create test table
        conn.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT)")
        conn.commit()

        # Use safe_executemany
        data = [(f"Article {i}",) for i in range(10)]
        safe_executemany(
            conn,
            "INSERT INTO articles (title) VALUES (?)",
            data,
            use_advisory_lock=True
        )

        # Verify
        cursor = conn.execute("SELECT COUNT(*) FROM articles")
        count = cursor.fetchone()[0]
        assert count == 10
        print(f"✅ safe_executemany inserted {count} records")

        conn.close()


def test_batch_writer():
    """Test BatchWriter for buffered writes"""
    print("\n🧪 Testing BatchWriter...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create test table
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()

        # Use BatchWriter
        with BatchWriter(conn, "items", batch_size=5) as writer:
            for i in range(12):
                writer.add(
                    "INSERT INTO items (name) VALUES (?)",
                    (f"Item {i}",)
                )

        # Verify all records written
        cursor = conn.execute("SELECT COUNT(*) FROM items")
        count = cursor.fetchone()[0]
        assert count == 12
        print(f"✅ BatchWriter inserted {count} records (2 batches of 5, 1 batch of 2)")

        conn.close()


def test_concurrent_writes():
    """Test concurrent writes with utilities"""
    print("\n🧪 Testing concurrent writes with 20 threads...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Setup database with WAL mode
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 60000")
        conn.execute("CREATE TABLE concurrent_test (id INTEGER PRIMARY KEY, thread_id INTEGER, value TEXT)")
        conn.commit()
        conn.close()

        def write_batch(thread_id):
            """Write a batch of records from a thread"""
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 60000")

            try:
                data = [(thread_id, f"Thread-{thread_id}-Value-{i}") for i in range(10)]
                # Don't use advisory lock for concurrent writes - WAL mode handles it
                safe_executemany(
                    conn,
                    "INSERT INTO concurrent_test (thread_id, value) VALUES (?, ?)",
                    data,
                    use_advisory_lock=False  # Let WAL mode + retry handle concurrency
                )
                return thread_id, True, None
            except Exception as e:
                return thread_id, False, str(e)
            finally:
                conn.close()

        # Run concurrent writes
        successes = 0
        failures = 0
        errors = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(write_batch, i) for i in range(20)]

            for future in as_completed(futures):
                thread_id, success, error = future.result()
                if success:
                    successes += 1
                else:
                    failures += 1
                    errors.append(f"Thread {thread_id}: {error}")

        # Verify results
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM concurrent_test")
        total_records = cursor.fetchone()[0]
        conn.close()

        print(f"📊 Concurrent write results:")
        print(f"   ✅ Successes: {successes}/20")
        print(f"   ❌ Failures: {failures}/20")
        print(f"   📝 Total records: {total_records}/200")

        if errors:
            print(f"\n⚠️ Errors encountered:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   - {error}")

        # Assert expectations
        assert successes >= 18, f"Expected at least 18/20 successes, got {successes}"
        assert total_records >= 180, f"Expected at least 180/200 records, got {total_records}"

        if successes == 20 and total_records == 200:
            print("\n🎉 Perfect! 100% success rate with NO lock errors!")
        else:
            print(f"\n✅ Good! {successes/20*100:.1f}% success rate")


def run_all_tests():
    """Run all database utility tests"""
    print("=" * 60)
    print("🧪 Database Utilities Test Suite")
    print("=" * 60)

    tests = [
        test_exponential_backoff,
        test_advisory_lock,
        test_batch_write_context,
        test_safe_executemany,
        test_batch_writer,
        test_concurrent_writes,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
