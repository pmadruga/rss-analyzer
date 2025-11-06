"""
Tests for Async AI Clients

Test suite for async clients with concurrent processing capabilities.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients import (
    AsyncArticleProcessor,
    AsyncClaudeClient,
    AsyncMistralClient,
    AsyncOpenAIClient,
    process_articles_async,
    run_async_processing,
)
from src.clients.factory import AIClientFactory
from src.exceptions import (
    APIConnectionError,
    APIRateLimitError,
    ContentProcessingError,
)


# Sample test data
SAMPLE_ARTICLES = [
    {
        "title": "Article 1",
        "content": "Content 1 about AI and machine learning",
        "url": "https://example.com/1",
    },
    {
        "title": "Article 2",
        "content": "Content 2 about neural networks",
        "url": "https://example.com/2",
    },
    {
        "title": "Article 3",
        "content": "Content 3 about deep learning",
        "url": "https://example.com/3",
    },
]

SAMPLE_ANALYSIS = {
    "methodology_detailed": "Detailed analysis using Feynman technique",
    "technical_approach": "Technical details",
    "key_findings": "Key findings",
    "research_design": "Research approach",
    "metadata": {
        "ai_provider": "test",
        "model": "test-model",
        "processed_at": 1234567890,
    },
}


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '{"extracted_title": "Test Title", "analysis": "Test analysis"}'
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"extracted_title": "Test Title", "analysis": "Test analysis"}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_mistral_response():
    """Mock Mistral API response"""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"extracted_title": "Test Title", "analysis": "Test analysis"}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


class TestAsyncClaudeClient:
    """Test AsyncClaudeClient"""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization"""
        client = AsyncClaudeClient(
            api_key="test-api-key-1234567890", model="claude-3-5-sonnet-20241022"
        )

        assert client.provider_name == "Claude"
        assert client.model == "claude-3-5-sonnet-20241022"
        assert client.max_concurrent_requests == 10
        assert client.api_key == "test-api-key-1234567890"

    @pytest.mark.asyncio
    async def test_analyze_article_async(self, mock_anthropic_response):
        """Test async article analysis"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_client.return_value = mock_instance

            client = AsyncClaudeClient(api_key="test-key-1234567890")

            result = await client.analyze_article_async(
                title="Test Article", content="Test content", url="https://test.com"
            )

            assert result is not None
            assert "methodology_detailed" in result
            assert result["extracted_title"] == "Test Title"

    @pytest.mark.asyncio
    async def test_batch_analyze_async(self, mock_anthropic_response):
        """Test batch analysis"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_client.return_value = mock_instance

            client = AsyncClaudeClient(api_key="test-key-1234567890")

            results = await client.batch_analyze_async(
                SAMPLE_ARTICLES, max_concurrent=3
            )

            assert len(results) == len(SAMPLE_ARTICLES)
            assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting enforcement"""
        client = AsyncClaudeClient(api_key="test-key-1234567890")

        # Set short rate limit for testing
        client.rate_limit_delay = 0.1

        import time

        start_time = time.time()
        await client._enforce_rate_limit()
        await client._enforce_rate_limit()
        elapsed = time.time() - start_time

        # Second call should wait at least rate_limit_delay
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager"""
        async with AsyncClaudeClient(api_key="test-key-1234567890") as client:
            assert client is not None
            assert client.provider_name == "Claude"


class TestAsyncOpenAIClient:
    """Test AsyncOpenAIClient"""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization"""
        client = AsyncOpenAIClient(api_key="test-api-key-1234567890", model="gpt-4")

        assert client.provider_name == "OpenAI"
        assert client.model == "gpt-4"
        assert client.max_concurrent_requests == 10

    @pytest.mark.asyncio
    async def test_analyze_article_async(self, mock_openai_response):
        """Test async article analysis"""
        with patch("openai.AsyncOpenAI") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_client.return_value = mock_instance

            client = AsyncOpenAIClient(api_key="test-key-1234567890")

            result = await client.analyze_article_async(
                title="Test Article", content="Test content"
            )

            assert result is not None
            assert "methodology_detailed" in result


class TestAsyncMistralClient:
    """Test AsyncMistralClient"""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization"""
        client = AsyncMistralClient(
            api_key="test-api-key-1234567890", model="mistral-large-latest"
        )

        assert client.provider_name == "Mistral"
        assert client.model == "mistral-large-latest"
        assert client.max_concurrent_requests == 10

    @pytest.mark.asyncio
    async def test_analyze_article_async(self, mock_mistral_response):
        """Test async article analysis"""
        with patch("mistralai.Mistral") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.chat.complete_async = AsyncMock(
                return_value=mock_mistral_response
            )
            mock_client.return_value = mock_instance

            client = AsyncMistralClient(api_key="test-key-1234567890")

            result = await client.analyze_article_async(
                title="Test Article", content="Test content"
            )

            assert result is not None
            assert "methodology_detailed" in result


class TestAIClientFactory:
    """Test AIClientFactory async methods"""

    def test_create_async_client(self):
        """Test creating async client"""
        client = AIClientFactory.create_async_client(
            provider="anthropic", api_key="test-key-1234567890"
        )

        assert isinstance(client, AsyncClaudeClient)
        assert client.provider_name == "Claude"

    def test_create_from_config_async(self):
        """Test creating async client from config"""
        config = {
            "api_provider": "anthropic",
            "anthropic_api_key": "test-key-1234567890",
            "claude_model": "claude-3-5-sonnet-20241022",
        }

        client = AIClientFactory.create_from_config(config, async_mode=True)

        assert isinstance(client, AsyncClaudeClient)
        assert client.model == "claude-3-5-sonnet-20241022"


class TestAsyncArticleProcessor:
    """Test AsyncArticleProcessor"""

    @pytest.mark.asyncio
    async def test_processor_initialization(self):
        """Test processor initialization"""
        processor = AsyncArticleProcessor(
            provider="anthropic",
            api_key="test-key-1234567890",
            max_concurrent=5,
        )

        assert processor.max_concurrent == 5
        assert isinstance(processor.client, AsyncClaudeClient)

    @pytest.mark.asyncio
    async def test_process_articles(self, mock_anthropic_response):
        """Test processing articles"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_client.return_value = mock_instance

            processor = AsyncArticleProcessor(
                provider="anthropic", api_key="test-key-1234567890", max_concurrent=2
            )

            results = await processor.process_articles(
                SAMPLE_ARTICLES, show_progress=False
            )

            assert len(results) == len(SAMPLE_ARTICLES)

    @pytest.mark.asyncio
    async def test_process_in_batches(self, mock_anthropic_response):
        """Test batch processing"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_client.return_value = mock_instance

            processor = AsyncArticleProcessor(
                provider="anthropic", api_key="test-key-1234567890", max_concurrent=2
            )

            results = await processor.process_in_batches(
                SAMPLE_ARTICLES, batch_size=2, delay_between_batches=0.1
            )

            assert len(results) == len(SAMPLE_ARTICLES)


class TestAsyncUtilities:
    """Test async utility functions"""

    @pytest.mark.asyncio
    async def test_process_articles_async(self, mock_anthropic_response):
        """Test process_articles_async function"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_instance.close = AsyncMock()
            mock_client.return_value = mock_instance

            results = await process_articles_async(
                articles=SAMPLE_ARTICLES,
                provider="anthropic",
                api_key="test-key-1234567890",
                max_concurrent=3,
            )

            assert len(results) == len(SAMPLE_ARTICLES)

    def test_run_async_processing(self, mock_anthropic_response):
        """Test synchronous wrapper function"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_instance.close = AsyncMock()
            mock_client.return_value = mock_instance

            results = run_async_processing(
                articles=SAMPLE_ARTICLES,
                provider="anthropic",
                api_key="test-key-1234567890",
                max_concurrent=3,
            )

            assert len(results) == len(SAMPLE_ARTICLES)


class TestErrorHandling:
    """Test error handling in async clients"""

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock to raise connection error
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            mock_client.return_value = mock_instance

            client = AsyncClaudeClient(api_key="test-key-1234567890")

            with pytest.raises(ContentProcessingError):
                await client.analyze_article_async(
                    title="Test", content="Test content"
                )

    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test invalid input handling"""
        client = AsyncClaudeClient(api_key="test-key-1234567890")

        with pytest.raises(ContentProcessingError):
            await client.analyze_article_async(title="", content="")

    @pytest.mark.asyncio
    async def test_batch_error_handling(self, mock_anthropic_response):
        """Test batch processing with some failures"""
        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock to fail on second article
            call_count = 0

            async def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("API error")
                return mock_anthropic_response

            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=mock_create)
            mock_client.return_value = mock_instance

            client = AsyncClaudeClient(api_key="test-key-1234567890")

            results = await client.batch_analyze_async(SAMPLE_ARTICLES)

            # First and third should succeed, second should be None
            assert results[0] is not None
            assert results[1] is None
            assert results[2] is not None


class TestConcurrentPerformance:
    """Test concurrent processing performance"""

    @pytest.mark.asyncio
    async def test_concurrent_execution_speed(self, mock_anthropic_response):
        """Test that concurrent execution is faster than sequential"""
        import time

        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock with artificial delay
            async def delayed_create(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate API latency
                return mock_anthropic_response

            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=delayed_create)
            mock_client.return_value = mock_instance

            client = AsyncClaudeClient(api_key="test-key-1234567890")

            # Test with 10 articles
            articles = SAMPLE_ARTICLES * 4  # 12 articles

            start = time.time()
            results = await client.batch_analyze_async(articles, max_concurrent=5)
            elapsed = time.time() - start

            # With 5 concurrent requests, 12 articles should take ~3 batches
            # Sequential would take 1.2s (12 * 0.1s)
            # Concurrent should take ~0.3s (3 batches * 0.1s)
            assert elapsed < 1.0  # Much faster than sequential
            assert len(results) == len(articles)

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrent requests"""
        client = AsyncClaudeClient(api_key="test-key-1234567890")
        client.max_concurrent_requests = 3

        # Verify semaphore is set correctly
        assert client.semaphore._value == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
