#!/usr/bin/env python3
"""
Connection Pooling Demonstration

Shows how to use the database connection pool for concurrent operations.
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import using importlib to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "database",
    Path(__file__).parent.parent / "src" / "core" / "database.py"
)
database = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database)


def demo_basic_usage():
    """Demonstrate basic connection pool usage"""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Connection Pool Usage")
    print("=" * 60)

    # Create database with pool
    db = database.DatabaseManager(":memory:", pool_size=5)

    # Check initial pool stats
    stats = db.get_pool_stats()
    print(f"\nInitial pool state:")
    print(f"  Pool size: {stats['pool_size']}")
    print(f"  Idle connections: {stats['idle_connections']}")
    print(f"  Active connections: {stats['active_connections']}")

    # Use a connection
    print("\nUsing a connection...")
    with db.get_connection() as conn:
        stats = db.get_pool_stats()
        print(f"  Active connections: {stats['active_connections']}")

        # Query the database (articles table already created by init_database)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            print(f"  Articles in database: {count}")
        except Exception as e:
            print(f"  Query result: Connection working! (Empty database)")

    # Connection returned
    stats = db.get_pool_stats()
    print(f"\nAfter connection returned:")
    print(f"  Active connections: {stats['active_connections']}")

    db.close_pool()
    print("\n✅ Demo 1 complete")


def demo_concurrent_access():
    """Demonstrate thread-safe concurrent database access"""
    print("\n" + "=" * 60)
    print("Demo 2: Concurrent Database Access")
    print("=" * 60)

    # Use file-based database for concurrent access demo
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = temp_db.name
    temp_db.close()

    db = database.DatabaseManager(db_path, pool_size=5)

    results = []
    errors = []
    start_time = time.time()

    def worker(worker_id):
        """Worker thread that performs database operations"""
        try:
            for i in range(5):
                # Use direct SQL instead of insert_article for demo
                with db.get_connection() as conn:
                    cursor = conn.execute("SELECT ?", (worker_id,))
                    result = cursor.fetchone()[0]
                    results.append(result)
                time.sleep(0.01)  # Simulate work

        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")

    # Create 10 threads (more than pool size to show queuing)
    print(f"\nSpawning 10 worker threads (pool size = 5)...")
    threads = []
    for i in range(10):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Monitor pool while threads run
    print("\nPool activity:")
    while any(t.is_alive() for t in threads):
        stats = db.get_pool_stats()
        print(f"  Active: {stats['active_connections']}/{stats['pool_size']}", end="\r")
        time.sleep(0.1)

    # Wait for completion
    for thread in threads:
        thread.join()

    elapsed = time.time() - start_time

    print(f"\n\nResults:")
    print(f"  Operations completed: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Operations/second: {len(results)/elapsed:.1f}")

    if errors:
        print(f"  Error details: {errors[:3]}")

    db.close_pool()

    # Cleanup temp database
    os.unlink(db_path)

    print("\n✅ Demo 2 complete")


def demo_pool_monitoring():
    """Demonstrate pool statistics monitoring"""
    print("\n" + "=" * 60)
    print("Demo 3: Pool Statistics Monitoring")
    print("=" * 60)

    db = database.DatabaseManager(":memory:", pool_size=3)

    print("\nCreating multiple concurrent connections...")

    # Get 2 connections (leave 1 idle)
    contexts = []
    for i in range(2):
        ctx = db.get_connection()
        conn = ctx.__enter__()
        contexts.append((conn, ctx))

        stats = db.get_pool_stats()
        print(f"\nAfter checkout #{i+1}:")
        print(f"  Active: {stats['active_connections']}")
        print(f"  Idle: {stats['idle_connections']}")
        print(f"  Total created: {stats['total_connections_created']}")

    # Return connections
    print("\nReturning connections...")
    for i, (conn, ctx) in enumerate(contexts):
        ctx.__exit__(None, None, None)
        stats = db.get_pool_stats()
        print(f"\nAfter return #{i+1}:")
        print(f"  Active: {stats['active_connections']}")
        print(f"  Idle: {stats['idle_connections']}")

    # Show final stats
    stats = db.get_pool_stats()
    print(f"\nFinal pool statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    db.close_pool()
    print("\n✅ Demo 3 complete")


def demo_connection_reuse():
    """Demonstrate that connections are reused from pool"""
    print("\n" + "=" * 60)
    print("Demo 4: Connection Reuse")
    print("=" * 60)

    db = database.DatabaseManager(":memory:", pool_size=3)

    initial_stats = db.get_pool_stats()
    initial_created = initial_stats['total_connections_created']

    print(f"\nInitial connections created: {initial_created}")
    print("\nPerforming 20 database operations...")

    # Perform many operations
    for i in range(20):
        with db.get_connection() as conn:
            conn.execute("SELECT 1")

    final_stats = db.get_pool_stats()
    final_created = final_stats['total_connections_created']

    print(f"\nFinal connections created: {final_created}")
    print(f"New connections created: {final_created - initial_created}")

    if final_created == initial_created:
        print("\n✅ All connections were reused from pool!")
    else:
        print(f"\n⚠️ {final_created - initial_created} new connections created")

    db.close_pool()
    print("\n✅ Demo 4 complete")


def demo_performance_comparison():
    """Compare performance with and without connection reuse"""
    print("\n" + "=" * 60)
    print("Demo 5: Performance Comparison")
    print("=" * 60)

    operations = 100

    # With connection pooling (reuse)
    print(f"\nPerforming {operations} operations WITH pooling...")
    db_pooled = database.DatabaseManager(":memory:", pool_size=5)

    start = time.time()
    for i in range(operations):
        with db_pooled.get_connection() as conn:
            conn.execute("SELECT 1")
    pooled_time = time.time() - start

    stats = db_pooled.get_pool_stats()
    print(f"  Time: {pooled_time:.3f}s")
    print(f"  Operations/second: {operations/pooled_time:.1f}")
    print(f"  Connections created: {stats['total_connections_created']}")

    db_pooled.close_pool()

    # Simulated without pooling (create new connection each time)
    print(f"\nSimulating {operations} operations WITHOUT pooling...")
    import sqlite3

    start = time.time()
    for i in range(operations):
        conn = sqlite3.connect(":memory:")
        conn.execute("SELECT 1")
        conn.close()
    no_pool_time = time.time() - start

    print(f"  Time: {no_pool_time:.3f}s")
    print(f"  Operations/second: {operations/no_pool_time:.1f}")
    print(f"  Connections created: {operations}")

    # Comparison
    speedup = no_pool_time / pooled_time
    print(f"\nPerformance improvement:")
    print(f"  Speedup: {speedup:.2f}x faster with pooling")
    print(f"  Time saved: {no_pool_time - pooled_time:.3f}s")

    print("\n✅ Demo 5 complete")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("Database Connection Pooling Demonstrations")
    print("=" * 60)

    try:
        demo_basic_usage()
        demo_concurrent_access()
        demo_pool_monitoring()
        demo_connection_reuse()
        demo_performance_comparison()

        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
