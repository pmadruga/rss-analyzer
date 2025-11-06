"""
Minimal direct test of connection pooling

Imports database module directly without going through __init__.py
"""

import os
import sys
import tempfile
import threading
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database module file directly using importlib to avoid __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "database",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "src", "core", "database.py")
)
database = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database)

def main():
    """Run minimal connection pooling tests"""
    print("\n" + "=" * 60)
    print("Minimal Connection Pooling Tests")
    print("=" * 60 + "\n")

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    try:
        # Test 1: ConnectionPool initialization
        print("Test 1: ConnectionPool initialization")
        pool = database.ConnectionPool(db_path, pool_size=5)
        stats = pool.get_pool_stats()
        print(f"  Pool size: {stats['pool_size']}")
        print(f"  Idle connections: {stats['idle_connections']}")
        print(f"  Active connections: {stats['active_connections']}")
        assert stats['pool_size'] == 5
        assert stats['idle_connections'] == 5
        assert stats['active_connections'] == 0
        print("  ✅ PASSED\n")

        # Test 2: Connection checkout/checkin
        print("Test 2: Connection checkout and checkin")
        with pool.get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"  Query result: {result[0]}")
            assert result[0] == 1
            stats = pool.get_pool_stats()
            print(f"  Active while checked out: {stats['active_connections']}")
            assert stats['active_connections'] == 1

        stats = pool.get_pool_stats()
        print(f"  Active after return: {stats['active_connections']}")
        assert stats['active_connections'] == 0
        print("  ✅ PASSED\n")

        # Test 3: Multiple simultaneous connections
        print("Test 3: Multiple simultaneous connections")
        connections = []
        for i in range(3):
            conn_ctx = pool.get_connection()
            conn = conn_ctx.__enter__()
            connections.append((conn, conn_ctx))

        stats = pool.get_pool_stats()
        print(f"  Active connections: {stats['active_connections']}")
        print(f"  Idle connections: {stats['idle_connections']}")
        assert stats['active_connections'] == 3
        assert stats['idle_connections'] == 2

        for conn, conn_ctx in connections:
            conn_ctx.__exit__(None, None, None)

        stats = pool.get_pool_stats()
        assert stats['active_connections'] == 0
        print("  ✅ PASSED\n")

        # Test 4: Thread safety
        print("Test 4: Thread safety with 10 threads")
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
                errors.append(str(e))

        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"  Operations completed: {len(results)}")
        print(f"  Errors: {len(errors)}")
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 50
        print("  ✅ PASSED\n")

        # Test 5: Connection reuse
        print("Test 5: Connection reuse")
        initial_stats = pool.get_pool_stats()
        initial_created = initial_stats['total_connections_created']
        print(f"  Initial connections created: {initial_created}")

        for i in range(20):
            with pool.get_connection() as conn:
                conn.execute("SELECT 1")

        final_stats = pool.get_pool_stats()
        final_created = final_stats['total_connections_created']
        print(f"  Final connections created: {final_created}")
        assert final_created == initial_created, "New connections should not be created"
        print("  ✅ PASSED\n")

        # Test 6: Pool closure
        print("Test 6: Pool closure")
        pool.close_pool()
        stats = pool.get_pool_stats()
        print(f"  Pool closed: {stats['closed']}")
        assert stats['closed'] == True

        try:
            with pool.get_connection() as conn:
                pass
            assert False, "Should raise error"
        except RuntimeError as e:
            print(f"  Expected error: {e}")
        print("  ✅ PASSED\n")

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60 + "\n")
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        os.rmdir(temp_dir)


if __name__ == "__main__":
    sys.exit(main())
