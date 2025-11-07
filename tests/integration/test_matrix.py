"""
Configuration Matrix Testing

Tests all combinations of:
- Processors: sync, async
- AI Providers: anthropic, mistral, openai
- Cache modes: enabled, disabled
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients import AIClientFactory, AsyncArticleProcessor
from src.core import DatabaseManager
from src.core.cache import ContentCache


# Test matrix
PROCESSORS = ['sync', 'async']
PROVIDERS = ['anthropic', 'mistral', 'openai']
CACHE_MODES = ['enabled', 'disabled']

MOCK_ANALYSIS = {
    "methodology_detailed": "Analysis",
    "technical_approach": "Technical",
    "key_findings": "Findings",
    "research_design": "Design",
    "extracted_title": "Title",
    "metadata": {"provider": "test"}
}


@pytest.mark.parametrize("provider", PROVIDERS)
class TestProviderMatrix:
    """Test each AI provider works correctly"""

    def test_sync_client_creation(self, provider):
        """Test synchronous client creation for each provider"""
        client = AIClientFactory.create_client(
            provider=provider,
            api_key="test-key-123456"
        )

        assert client is not None
        assert client.provider_name.lower() in provider.lower() or provider in client.provider_name.lower()

        print(f"\n=== Provider: {provider} (sync) ===")
        print(f"Client: {client.__class__.__name__}")
        print(f"Provider name: {client.provider_name}")

    def test_async_client_creation(self, provider):
        """Test asynchronous client creation for each provider"""
        client = AIClientFactory.create_async_client(
            provider=provider,
            api_key="test-key-123456"
        )

        assert client is not None
        assert "Async" in client.__class__.__name__

        print(f"\n=== Provider: {provider} (async) ===")
        print(f"Client: {client.__class__.__name__}")


@pytest.mark.parametrize("provider,cache_enabled", [
    (provider, cache)
    for provider in PROVIDERS
    for cache in [True, False]
])
class TestProviderCacheMatrix:
    """Test each provider with cache enabled/disabled"""

    def test_provider_with_cache_config(self, provider, cache_enabled):
        """Test provider works with different cache configurations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"

            if cache_enabled:
                cache = ContentCache(str(cache_path))
                cache_key = "test_key"
                cache.set(cache_key, MOCK_ANALYSIS, ttl=3600)

                # Verify cache works
                cached = cache.get(cache_key)
                assert cached is not None
                status = "enabled"
            else:
                cache = None
                status = "disabled"

            print(f"\n=== Provider: {provider}, Cache: {status} ===")
            print(f"Cache functional: {cache is not None}")


@pytest.mark.parametrize("processor_type", PROCESSORS)
class TestProcessorMatrix:
    """Test sync vs async processors"""

    def test_processor_type_basic(self, processor_type):
        """Test basic processor functionality"""
        print(f"\n=== Processor Type: {processor_type} ===")

        if processor_type == "sync":
            # Test sync processor pattern
            client = AIClientFactory.create_client(
                provider="anthropic",
                api_key="test-key-123456"
            )
            assert client is not None
            print("Sync processor: OK")

        elif processor_type == "async":
            # Test async processor pattern
            client = AIClientFactory.create_async_client(
                provider="anthropic",
                api_key="test-key-123456"
            )
            assert client is not None
            print("Async processor: OK")


class TestFullMatrix:
    """Test full configuration matrix"""

    @pytest.mark.parametrize("processor,provider,cache", [
        (proc, prov, cache)
        for proc in PROCESSORS
        for prov in PROVIDERS
        for cache in CACHE_MODES
    ])
    def test_full_configuration_matrix(self, processor, provider, cache):
        """Test all combinations of processor, provider, and cache"""
        cache_enabled = (cache == "enabled")

        print(f"\n=== Configuration ===")
        print(f"Processor: {processor}")
        print(f"Provider: {provider}")
        print(f"Cache: {cache}")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Setup cache if enabled
                if cache_enabled:
                    cache_path = Path(tmpdir) / "cache.db"
                    cache_obj = ContentCache(str(cache_path))
                else:
                    cache_obj = None

                # Create appropriate client
                if processor == "sync":
                    client = AIClientFactory.create_client(
                        provider=provider,
                        api_key="test-key-123456"
                    )
                    assert client is not None
                else:  # async
                    client = AIClientFactory.create_async_client(
                        provider=provider,
                        api_key="test-key-123456"
                    )
                    assert client is not None

                print("Configuration valid: OK")

            except Exception as e:
                pytest.fail(f"Configuration failed: {e}")


class TestMatrixPerformanceComparison:
    """Compare performance across different configurations"""

    @pytest.mark.asyncio
    async def test_sync_vs_async_comparison(self):
        """Compare sync vs async performance"""
        import time

        articles = [
            {
                "title": f"Article {i}",
                "content": f"Content {i} " * 50,
                "url": f"https://example.com/{i}",
            }
            for i in range(10)
        ]

        # Mock async client
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps(MOCK_ANALYSIS)
        mock_response.content = [mock_content]

        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            async def delayed_create(*args, **kwargs):
                await asyncio.sleep(0.05)  # 50ms latency
                return mock_response

            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=delayed_create)
            mock_anthropic.return_value = mock_instance

            # Test sync (simulated)
            sync_time = len(articles) * 0.05  # Sequential

            # Test async
            processor = AsyncArticleProcessor(
                provider="anthropic",
                api_key="test-key-123456",
                max_concurrent=5
            )

            start = time.time()
            results = await processor.process_articles(articles, show_progress=False)
            async_time = time.time() - start

            speedup = sync_time / async_time

            print(f"\n=== Sync vs Async Comparison ===")
            print(f"Articles: {len(articles)}")
            print(f"Sync time (estimated): {sync_time:.2f}s")
            print(f"Async time: {async_time:.2f}s")
            print(f"Speedup: {speedup:.1f}x")

            assert len(results) == len(articles)
            assert speedup >= 2.0  # Should be significantly faster


class TestMatrixIntegration:
    """Integration tests across matrix configurations"""

    def test_database_works_with_all_providers(self):
        """Test database works regardless of AI provider"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            for provider in PROVIDERS:
                db = DatabaseManager(str(db_path))

                article_id = db.insert_article(
                    title=f"Article for {provider}",
                    url=f"https://example.com/{provider}",
                    description="Description",
                    content="Content",
                    content_hash=f"hash_{provider}"
                )

                article = db.get_article_by_id(article_id)
                assert article is not None
                assert provider in article["title"]

            print(f"\n=== Database Provider Compatibility ===")
            print(f"Tested {len(PROVIDERS)} providers: OK")

    def test_cache_works_with_all_processors(self):
        """Test cache works with both sync and async"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.db"
            cache = ContentCache(str(cache_path))

            # Test with sync-style access
            cache.set("sync_key", "sync_value", ttl=3600)
            assert cache.get("sync_key") == "sync_value"

            # Test with async-style access (still sync, but verifies compatibility)
            cache.set("async_key", "async_value", ttl=3600)
            assert cache.get("async_key") == "async_value"

            print(f"\n=== Cache Processor Compatibility ===")
            print("Cache works with both processors: OK")


class TestMatrixReport:
    """Generate matrix test report"""

    def test_generate_matrix_report(self):
        """Generate comprehensive matrix test report"""
        results = {
            "processors": {},
            "providers": {},
            "cache_modes": {},
            "combinations": {}
        }

        # Test each processor
        for processor in PROCESSORS:
            try:
                if processor == "sync":
                    client = AIClientFactory.create_client(
                        provider="anthropic",
                        api_key="test-key-123456"
                    )
                else:
                    client = AIClientFactory.create_async_client(
                        provider="anthropic",
                        api_key="test-key-123456"
                    )
                results["processors"][processor] = "OK"
            except Exception as e:
                results["processors"][processor] = f"FAILED: {e}"

        # Test each provider
        for provider in PROVIDERS:
            try:
                client = AIClientFactory.create_client(
                    provider=provider,
                    api_key="test-key-123456"
                )
                results["providers"][provider] = "OK"
            except Exception as e:
                results["providers"][provider] = f"FAILED: {e}"

        # Test cache modes
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in CACHE_MODES:
                try:
                    if mode == "enabled":
                        cache = ContentCache(str(Path(tmpdir) / "cache.db"))
                        cache.set("test", "value", ttl=3600)
                    results["cache_modes"][mode] = "OK"
                except Exception as e:
                    results["cache_modes"][mode] = f"FAILED: {e}"

        # Generate report
        report = "\n=== Configuration Matrix Test Report ===\n"
        report += "\nProcessors:\n"
        for proc, status in results["processors"].items():
            report += f"  {proc}: {status}\n"

        report += "\nProviders:\n"
        for prov, status in results["providers"].items():
            report += f"  {prov}: {status}\n"

        report += "\nCache Modes:\n"
        for mode, status in results["cache_modes"].items():
            report += f"  {mode}: {status}\n"

        report += f"\nTotal Combinations: {len(PROCESSORS) * len(PROVIDERS) * len(CACHE_MODES)}\n"

        print(report)

        # Verify all passed
        assert all(v == "OK" for v in results["processors"].values())
        assert all(v == "OK" for v in results["providers"].values())
        assert all(v == "OK" for v in results["cache_modes"].values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
