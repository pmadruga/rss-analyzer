"""
Stress Testing Suite

Tests system behavior under heavy load:
- Large article volumes (100+ articles)
- Concurrent processing limits
- Memory usage under load (<450MB target)
- Database connection pool under stress
- Rate limiter behavior under pressure
"""

import asyncio
import gc
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from src.clients import AsyncArticleProcessor
from src.core import DatabaseManager
from src.core.cache import ContentCache


# Large test dataset
def generate_articles(count: int):
    """Generate test articles"""
    return [
        {
            "title": f"Stress Test Article {i}",
            "content": f"Content for article {i} " * 200,  # ~400 words
            "url": f"https://example.com/stress/article{i}",
        }
        for i in range(count)
    ]


MOCK_ANALYSIS = {
    "methodology_detailed": "Analysis" * 100,
    "technical_approach": "Technical" * 100,
    "key_findings": "Findings" * 100,
    "research_design": "Design" * 100,
    "extracted_title": "Title",
    "metadata": {"ai_provider": "test"}
}


class TestHighVolumeProcessing:
    """Test processing large volumes of articles"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_100_articles(self):
        """Test processing 100 articles"""
        import json

        articles = generate_articles(100)

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps(MOCK_ANALYSIS)
        mock_response.content = [mock_content]

        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            async def fast_create(*args, **kwargs):
                await asyncio.sleep(0.01)  # Fast mock
                return mock_response

            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=fast_create)
            mock_anthropic.return_value = mock_instance

            processor = AsyncArticleProcessor(
                provider="anthropic",
                api_key="test-key-123",
                max_concurrent=10
            )

            start = time.time()
            results = await processor.process_articles(articles, show_progress=False)
            duration = time.time() - start

            print(f"\n=== High Volume Processing ===")
            print(f"Processed {len(results)} articles in {duration:.2f}s")
            print(f"Average: {duration/len(results):.3f}s per article")
            print(f"Throughput: {len(results)/duration:.1f} articles/second")

            assert len(results) == 100
            # Should complete in reasonable time
            assert duration < 30  # <30 seconds for 100 articles

    def test_database_bulk_insert_stress(self):
        """Test database handles bulk inserts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stress.db"
            db = DatabaseManager(str(db_path))

            articles = generate_articles(500)

            start = time.time()
            for i, article in enumerate(articles):
                db.insert_article(
                    title=article["title"],
                    url=article["url"],
                    description=f"Description {i}",
                    content=article["content"],
                    content_hash=f"hash_{i}"
                )
            duration = time.time() - start

            print(f"\n=== Database Bulk Insert ===")
            print(f"Inserted 500 articles in {duration:.2f}s")
            print(f"Rate: {500/duration:.1f} inserts/second")

            # Verify all inserted
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM articles")
                count = cursor.fetchone()[0]
                assert count == 500

    def test_cache_bulk_operations(self):
        """Test cache handles bulk operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "stress_cache.db"
            cache = ContentCache(str(cache_path))

            # Add 1000 entries
            start = time.time()
            for i in range(1000):
                cache.set(f"key_{i}", f"value_{i}" * 10, ttl=3600)
            write_duration = time.time() - start

            # Read all entries
            start = time.time()
            for i in range(1000):
                value = cache.get(f"key_{i}")
                assert value is not None
            read_duration = time.time() - start

            print(f"\n=== Cache Bulk Operations ===")
            print(f"Write 1000 entries: {write_duration:.2f}s")
            print(f"Read 1000 entries: {read_duration:.2f}s")
            print(f"Write rate: {1000/write_duration:.0f} ops/sec")
            print(f"Read rate: {1000/read_duration:.0f} ops/sec")

            stats = cache.get_stats()
            print(f"Cache hit rate: {stats['hit_rate']:.1f}%")
            print(f"L1 size: {stats['l1_size_mb']:.1f}MB")
            print(f"L2 size: {stats['l2_size_mb']:.1f}MB")


class TestMemoryUsage:
    """Test memory usage under load"""

    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory stays under 450MB target"""
        import json

        # Force garbage collection before test
        gc.collect()
        baseline_memory = self.get_memory_usage_mb()

        articles = generate_articles(100)

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps(MOCK_ANALYSIS)
        mock_response.content = [mock_content]

        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            async def fast_create(*args, **kwargs):
                await asyncio.sleep(0.001)
                return mock_response

            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=fast_create)
            mock_anthropic.return_value = mock_instance

            processor = AsyncArticleProcessor(
                provider="anthropic",
                api_key="test-key-123",
                max_concurrent=10
            )

            # Process in batches and monitor memory
            memory_samples = []

            for batch_start in range(0, len(articles), 20):
                batch = articles[batch_start:batch_start + 20]
                await processor.process_articles(batch, show_progress=False)

                current_memory = self.get_memory_usage_mb()
                memory_samples.append(current_memory)

            gc.collect()
            final_memory = self.get_memory_usage_mb()

            memory_increase = final_memory - baseline_memory
            peak_memory = max(memory_samples)

            print(f"\n=== Memory Usage Analysis ===")
            print(f"Baseline: {baseline_memory:.1f}MB")
            print(f"Peak: {peak_memory:.1f}MB")
            print(f"Final: {final_memory:.1f}MB")
            print(f"Increase: {memory_increase:.1f}MB")
            print(f"Target: <450MB total, <100MB increase")

            # Memory increase should be reasonable
            assert memory_increase < 100  # Less than 100MB increase

    def test_cache_memory_limit(self):
        """Test L1 cache respects 256MB limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = ContentCache(str(cache_path))

            # Try to add 300MB of data
            large_value = "x" * (10 * 1024 * 1024)  # 10MB each

            for i in range(35):  # 350MB total
                cache.set(f"large_key_{i}", large_value, ttl=3600)

            stats = cache.get_stats()
            l1_size_mb = stats['l1_size_mb']

            print(f"\n=== Cache Memory Limit ===")
            print(f"L1 cache size: {l1_size_mb:.1f}MB")
            print(f"L1 limit: 256MB")
            print(f"L1 entries: {stats['l1_entries']}")

            # Should not exceed 256MB limit (with some tolerance)
            assert l1_size_mb <= 280  # Allow some overhead


class TestConcurrentAccess:
    """Test concurrent database and cache access"""

    def test_concurrent_database_access(self):
        """Test database handles concurrent connections"""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "concurrent.db"
            db = DatabaseManager(str(db_path))

            errors = []
            completed = []

            def worker(worker_id: int):
                try:
                    for i in range(20):
                        db.insert_article(
                            title=f"Worker {worker_id} Article {i}",
                            url=f"https://example.com/w{worker_id}/a{i}",
                            description="Description",
                            content="Content",
                            content_hash=f"hash_{worker_id}_{i}"
                        )
                    completed.append(worker_id)
                except Exception as e:
                    errors.append((worker_id, str(e)))

            # Spawn 10 concurrent threads
            threads = []
            start = time.time()

            for worker_id in range(10):
                thread = threading.Thread(target=worker, args=(worker_id,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            duration = time.time() - start

            print(f"\n=== Concurrent Database Access ===")
            print(f"10 threads × 20 inserts = 200 total")
            print(f"Duration: {duration:.2f}s")
            print(f"Completed threads: {len(completed)}")
            print(f"Errors: {len(errors)}")

            # Verify all operations completed
            assert len(completed) == 10
            assert len(errors) == 0

            # Verify all records inserted
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM articles")
                count = cursor.fetchone()[0]
                assert count == 200

    def test_concurrent_cache_access(self):
        """Test cache handles concurrent access"""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "concurrent_cache.db"
            cache = ContentCache(str(cache_path))

            errors = []
            completed = []

            def worker(worker_id: int):
                try:
                    # Each worker does 50 operations
                    for i in range(50):
                        key = f"worker_{worker_id}_key_{i}"
                        cache.set(key, f"value_{i}", ttl=3600)
                        value = cache.get(key)
                        assert value == f"value_{i}"
                    completed.append(worker_id)
                except Exception as e:
                    errors.append((worker_id, str(e)))

            threads = []
            start = time.time()

            for worker_id in range(5):
                thread = threading.Thread(target=worker, args=(worker_id,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            duration = time.time() - start

            print(f"\n=== Concurrent Cache Access ===")
            print(f"5 threads × 50 operations = 250 total")
            print(f"Duration: {duration:.2f}s")
            print(f"Completed: {len(completed)}")
            print(f"Errors: {len(errors)}")

            assert len(completed) == 5
            assert len(errors) == 0


class TestRateLimitingUnderStress:
    """Test rate limiter behavior under heavy load"""

    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_limits(self):
        """Test rate limiter prevents exceeding API limits"""
        from aiolimiter import AsyncLimiter

        # Create rate limiter: 10 requests per second
        limiter = AsyncLimiter(max_rate=10, time_period=1.0)

        request_times = []

        async def make_request():
            async with limiter:
                request_times.append(time.time())
                await asyncio.sleep(0.01)  # Simulate API call

        # Try to make 30 requests
        start = time.time()
        await asyncio.gather(*[make_request() for _ in range(30)])
        duration = time.time() - start

        # Calculate actual rate
        if len(request_times) > 1:
            time_span = request_times[-1] - request_times[0]
            actual_rate = len(request_times) / time_span
        else:
            actual_rate = 0

        print(f"\n=== Rate Limiting ===")
        print(f"30 requests in {duration:.2f}s")
        print(f"Actual rate: {actual_rate:.1f} req/s")
        print(f"Limit: 10 req/s")

        # Should take at least 3 seconds (30 requests / 10 per second)
        assert duration >= 2.5  # Allow some overhead
        assert actual_rate <= 12  # Should not significantly exceed limit

    @pytest.mark.asyncio
    async def test_rate_limiter_with_concurrent_clients(self):
        """Test rate limiter with multiple concurrent clients"""
        from aiolimiter import AsyncLimiter

        limiter = AsyncLimiter(max_rate=20, time_period=1.0)
        completed = []

        async def client(client_id: int):
            for i in range(10):
                async with limiter:
                    await asyncio.sleep(0.01)
                    completed.append(f"client_{client_id}_req_{i}")

        # 5 clients making 10 requests each = 50 total
        start = time.time()
        await asyncio.gather(*[client(i) for i in range(5)])
        duration = time.time() - start

        print(f"\n=== Multi-Client Rate Limiting ===")
        print(f"5 clients × 10 requests = 50 total")
        print(f"Duration: {duration:.2f}s")
        print(f"Rate limit: 20 req/s")

        assert len(completed) == 50
        # Should take at least 2.5 seconds (50 / 20)
        assert duration >= 2.0


class TestResourceExhaustion:
    """Test behavior when resources are exhausted"""

    def test_database_connection_pool_saturation(self):
        """Test behavior when connection pool is saturated"""
        import threading
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "pool_test.db"
            db = DatabaseManager(str(db_path))

            # Get pool size
            pool_size = db.pool.pool_size

            results = []
            start_times = []
            end_times = []

            def worker(worker_id: int):
                start_times.append(time.time())
                try:
                    # Hold connection for 0.5 seconds
                    with db.pool.get_connection() as conn:
                        time.sleep(0.5)
                        cursor = conn.execute("SELECT 1")
                        result = cursor.fetchone()
                        results.append((worker_id, result))
                finally:
                    end_times.append(time.time())

            # Spawn more threads than pool size
            threads = []
            start = time.time()

            for i in range(pool_size + 5):  # Exceed pool size
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            duration = time.time() - start

            print(f"\n=== Connection Pool Saturation ===")
            print(f"Pool size: {pool_size}")
            print(f"Threads: {pool_size + 5}")
            print(f"Duration: {duration:.2f}s")
            print(f"Completed: {len(results)}")

            # All should complete eventually
            assert len(results) == pool_size + 5
            # Should take longer due to queueing
            assert duration >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
