"""
Demo script showing the content cache in action.

This script demonstrates:
- Basic cache operations (set, get, delete)
- Cache key generation
- TTL and expiration
- Statistics tracking
- Two-tier cache behavior
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cache import ContentCache, create_cache


def demo_basic_operations():
    """Demonstrate basic cache operations."""
    print("\n=== Basic Cache Operations ===\n")

    # Initialize cache
    cache = create_cache("examples/demo_cache.db")

    # Store some data
    print("1. Storing data in cache...")
    cache.set("article_1", {"title": "Example Article", "content": "Lorem ipsum..."})
    cache.set("article_2", {"title": "Another Article", "content": "Dolor sit amet..."})
    cache.set("api_response", {"analysis": "This is a great article!"})

    # Retrieve data
    print("2. Retrieving data from cache...")
    article_1 = cache.get("article_1")
    print(f"   Retrieved: {article_1['title']}")

    # Cache miss
    print("3. Testing cache miss...")
    missing = cache.get("nonexistent_key")
    print(f"   Missing key returns: {missing}")

    # Delete entry
    print("4. Deleting entry...")
    cache.delete("article_2")
    deleted = cache.get("article_2")
    print(f"   Deleted entry returns: {deleted}")


def demo_key_generation():
    """Demonstrate cache key generation."""
    print("\n=== Cache Key Generation ===\n")

    cache = create_cache("examples/demo_cache.db")

    # Generate keys from URLs
    url = "https://example.com/article/123"

    scraped_key = ContentCache.generate_key(url, "scraped")
    api_key = ContentCache.generate_key(url, "api")

    print(f"URL: {url}")
    print(f"Scraped content key: {scraped_key}")
    print(f"API response key: {api_key}")
    print(f"Keys are different: {scraped_key != api_key}")

    # Store with generated keys
    cache.set(scraped_key, {"html": "<div>content</div>"}, content_type="scraped")
    cache.set(api_key, {"analysis": "Summary..."}, content_type="api")

    # Retrieve
    scraped = cache.get(scraped_key)
    api = cache.get(api_key)

    print(f"\nRetrieved scraped: {scraped is not None}")
    print(f"Retrieved API: {api is not None}")


def demo_ttl_and_expiration():
    """Demonstrate TTL and expiration."""
    print("\n=== TTL and Expiration ===\n")

    cache = create_cache("examples/demo_cache.db")

    # Store with short TTL
    print("1. Storing with 2-second TTL...")
    cache.set("short_lived", "temporary data", ttl=2)

    # Immediately retrieve
    data = cache.get("short_lived")
    print(f"   Immediate retrieval: {data}")

    # Wait for expiration
    print("2. Waiting 3 seconds for expiration...")
    time.sleep(3)

    # Try to retrieve expired entry
    expired = cache.get("short_lived")
    print(f"   After expiration: {expired}")

    # Use predefined TTLs
    print("\n3. Using predefined TTL constants...")
    cache.set(
        "scraped_content",
        "web content",
        ttl=ContentCache.TTL_SCRAPED_CONTENT,
        content_type="scraped"
    )
    print(f"   Scraped content TTL: {ContentCache.TTL_SCRAPED_CONTENT / (24*60*60)} days")

    cache.set(
        "api_response",
        "analysis",
        ttl=ContentCache.TTL_API_RESPONSE,
        content_type="api"
    )
    print(f"   API response TTL: {ContentCache.TTL_API_RESPONSE / (24*60*60)} days")


def demo_statistics():
    """Demonstrate statistics tracking."""
    print("\n=== Cache Statistics ===\n")

    cache = create_cache("examples/demo_cache.db")

    # Generate some activity
    print("1. Generating cache activity...")

    # Populate cache
    for i in range(5):
        cache.set(f"article_{i}", f"content_{i}")

    # Generate hits
    for i in range(3):
        cache.get(f"article_{i}")  # L1 hits

    # Generate L2 hits
    cache.l1.clear()
    cache.get("article_3")  # L1 miss, L2 hit

    # Generate misses
    cache.get("nonexistent_1")
    cache.get("nonexistent_2")

    # Display statistics
    print("\n2. Cache statistics:")
    stats = cache.get_stats()

    print(f"\n   Performance:")
    print(f"   - Hit rate: {stats['hit_rate']}%")
    print(f"   - L1 hits: {stats['l1_hits']}")
    print(f"   - L2 hits: {stats['l2_hits']}")
    print(f"   - Total misses: {stats['total_misses']}")

    print(f"\n   Size:")
    print(f"   - L1 entries: {stats['l1_entries']}")
    print(f"   - L2 entries: {stats['l2_entries']}")
    print(f"   - L1 size: {stats['l1_size_mb']} MB")
    print(f"   - L2 size: {stats['l2_size_mb']} MB")


def demo_two_tier_behavior():
    """Demonstrate L1/L2 cache behavior."""
    print("\n=== Two-Tier Cache Behavior ===\n")

    cache = create_cache("examples/demo_cache.db")

    # Store in both tiers
    print("1. Storing data (goes to both L1 and L2)...")
    cache.set("test_key", {"data": "test_value"})

    # Verify in L1
    print("2. Retrieving from L1 (fast)...")
    start = time.time()
    data = cache.get("test_key")
    l1_time = (time.time() - start) * 1000
    print(f"   Retrieved: {data}")
    print(f"   Time: {l1_time:.3f}ms")
    print(f"   L1 hits: {cache.stats.l1_hits}")

    # Clear L1, force L2 lookup
    print("\n3. Clearing L1, forcing L2 lookup...")
    cache.l1.clear()

    print("4. Retrieving from L2 (slower, but persistent)...")
    start = time.time()
    data = cache.get("test_key")
    l2_time = (time.time() - start) * 1000
    print(f"   Retrieved: {data}")
    print(f"   Time: {l2_time:.3f}ms")
    print(f"   L2 hits: {cache.stats.l2_hits}")

    # Data promoted back to L1
    print("\n5. Data promoted back to L1...")
    print(f"   L1 entries: {cache.l1.count()}")

    # Next access will hit L1 again
    print("\n6. Next retrieval hits L1 again (fast)...")
    start = time.time()
    data = cache.get("test_key")
    l1_time_2 = (time.time() - start) * 1000
    print(f"   Time: {l1_time_2:.3f}ms")
    print(f"   L1 hits: {cache.stats.l1_hits}")


def demo_cleanup():
    """Demonstrate cache cleanup."""
    print("\n=== Cache Cleanup ===\n")

    cache = create_cache("examples/demo_cache.db")

    # Create some entries that will expire
    print("1. Creating entries with short TTL...")
    for i in range(5):
        cache.set(f"temp_{i}", f"data_{i}", ttl=1)

    # Create some long-lived entries
    cache.set("permanent_1", "data_1", ttl=3600)
    cache.set("permanent_2", "data_2", ttl=3600)

    print(f"   Total entries: {cache.l2.count()}")

    # Wait for expiration
    print("\n2. Waiting 2 seconds for expiration...")
    time.sleep(2)

    # Cleanup
    print("\n3. Running cleanup...")
    removed = cache.cleanup_expired()
    print(f"   Removed {removed} expired entries")
    print(f"   Remaining entries: {cache.l2.count()}")


def demo_performance():
    """Demonstrate cache performance."""
    print("\n=== Performance Demo ===\n")

    cache = create_cache("examples/demo_cache.db")

    # Test write performance
    print("1. Testing write performance...")
    num_writes = 1000
    start = time.time()
    for i in range(num_writes):
        cache.set(f"perf_key_{i}", {"data": f"value_{i}"})
    write_time = time.time() - start
    print(f"   {num_writes} writes in {write_time:.2f}s")
    print(f"   {num_writes/write_time:.0f} writes/sec")

    # Test L1 read performance
    print("\n2. Testing L1 read performance...")
    start = time.time()
    for i in range(num_writes):
        cache.get(f"perf_key_{i}")
    l1_read_time = time.time() - start
    print(f"   {num_writes} L1 reads in {l1_read_time:.2f}s")
    print(f"   {num_writes/l1_read_time:.0f} reads/sec")

    # Test L2 read performance
    print("\n3. Testing L2 read performance...")
    cache.l1.clear()
    num_l2_reads = 100
    start = time.time()
    for i in range(num_l2_reads):
        cache.get(f"perf_key_{i}")
    l2_read_time = time.time() - start
    print(f"   {num_l2_reads} L2 reads in {l2_read_time:.2f}s")
    print(f"   {num_l2_reads/l2_read_time:.0f} reads/sec")

    # Compare
    print("\n4. Performance comparison:")
    print(f"   L1 is {l2_read_time/l1_read_time:.1f}x faster than L2")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Content Cache Demo")
    print("="*60)

    try:
        demo_basic_operations()
        demo_key_generation()
        demo_ttl_and_expiration()
        demo_statistics()
        demo_two_tier_behavior()
        demo_cleanup()
        demo_performance()

        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
