"""
Unit tests for base AI client
"""

import time
from unittest.mock import patch

import pytest

from src.clients.base import BaseAIClient
from src.exceptions import (
    APIClientError,
    APIConnectionError,
    APIRateLimitError,
    ContentProcessingError,
)


class TestableAIClient(BaseAIClient):
    """Testable implementation of BaseAIClient"""

    def __init__(self, api_key="test-key", model="test-model", provider_name="test"):
        self.call_count = 0
        self.should_fail = False
        self.response_text = '{"methodology_detailed": "test", "technical_approach": "test", "key_findings": "test", "research_design": "test"}'
        super().__init__(api_key, model, provider_name)

    def _make_api_call(self, prompt: str) -> str:
        """Mock API call implementation"""
        self.call_count += 1
        if self.should_fail:
            raise APIConnectionError("Test connection error", self.provider_name)
        return self.response_text


class TestBaseAIClient:
    """Test BaseAIClient abstract class"""

    def test_init(self):
        """Test client initialization"""
        client = TestableAIClient("test-key", "test-model", "test-provider")

        assert client.api_key == "test-key"
        assert client.model == "test-model"
        assert client.provider_name == "test-provider"
        assert client.last_request_time == 0.0
        assert client.max_retries == 3  # From config
        assert client.system_prompt is not None

    def test_init_invalid_api_key(self):
        """Test initialization with invalid API key"""
        with pytest.raises(APIClientError):
            TestableAIClient("", "test-model", "test-provider")

        with pytest.raises(APIClientError):
            TestableAIClient("short", "test-model", "test-provider")

    def test_validate_api_key(self):
        """Test API key validation"""
        client = TestableAIClient()

        # Valid key
        assert client._validate_api_key("valid-key-12345") == "valid-key-12345"

        # Invalid keys
        with pytest.raises(APIClientError):
            client._validate_api_key("")

        with pytest.raises(APIClientError):
            client._validate_api_key("short")

    def test_create_system_prompt(self):
        """Test system prompt creation"""
        client = TestableAIClient()
        prompt = client._create_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Feynman technique" in prompt
        assert "JSON" in prompt

    def test_enforce_rate_limit(self):
        """Test rate limiting enforcement"""
        client = TestableAIClient()

        # First call should not delay
        start_time = time.time()
        client._enforce_rate_limit()
        first_call_time = time.time() - start_time
        assert first_call_time < 0.1  # Should be very fast

        # Second immediate call should delay
        start_time = time.time()
        client._enforce_rate_limit()
        second_call_time = time.time() - start_time
        assert second_call_time >= client.rate_limit_delay - 0.1  # Allow some tolerance

    def test_prepare_content(self):
        """Test content preparation"""
        client = TestableAIClient()

        title = "Test Title"
        content = "Test content"
        url = "https://example.com"

        prepared = client._prepare_content(title, content, url)

        assert title in prepared
        assert content in prepared
        assert url in prepared
        assert "Title:" in prepared
        assert "URL:" in prepared
        assert "Content:" in prepared

    def test_prepare_content_truncation(self):
        """Test content truncation for long content"""
        client = TestableAIClient()

        # Mock config to have small limit
        with patch("src.clients.base.CONFIG") as mock_config:
            mock_config.processing.MAX_CONTENT_LENGTH = 100

            long_content = "x" * 200
            prepared = client._prepare_content("Title", long_content)

            assert "[Content truncated due to length]" in prepared
            assert len(prepared) < len(long_content) + 50  # Should be truncated

    def test_parse_analysis_response_valid_json(self):
        """Test parsing valid JSON response"""
        client = TestableAIClient()

        response = '{"methodology_detailed": "test method", "technical_approach": "test tech", "key_findings": "test findings", "research_design": "test design"}'

        result = client._parse_analysis_response(response)

        assert result is not None
        assert result["methodology_detailed"] == "test method"
        assert result["technical_approach"] == "test tech"
        assert result["key_findings"] == "test findings"
        assert result["research_design"] == "test design"
        assert "metadata" in result
        assert result["metadata"]["ai_provider"] == "test"

    def test_parse_analysis_response_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks"""
        client = TestableAIClient()

        response = '```json\n{"methodology_detailed": "test", "technical_approach": "test", "key_findings": "test", "research_design": "test"}\n```'

        result = client._parse_analysis_response(response)

        assert result is not None
        assert result["methodology_detailed"] == "test"

    def test_parse_analysis_response_invalid_json(self):
        """Test parsing invalid JSON response"""
        client = TestableAIClient()

        response = "Invalid JSON response"

        result = client._parse_analysis_response(response)

        assert result is not None
        assert "parsing_fallback" in result["metadata"]
        assert result["methodology_detailed"] == response

    def test_parse_analysis_response_empty(self):
        """Test parsing empty response"""
        client = TestableAIClient()

        result = client._parse_analysis_response("")

        assert result is None

    def test_create_fallback_analysis(self):
        """Test fallback analysis creation"""
        client = TestableAIClient()

        text = "Fallback text"
        result = client._create_fallback_analysis(text)

        assert result["methodology_detailed"] == text
        assert result["technical_approach"] == ""
        assert result["metadata"]["parsing_fallback"] is True

    def test_analyze_article_success(self):
        """Test successful article analysis"""
        client = TestableAIClient()

        result = client.analyze_article(
            "Test Title", "Test content", "https://example.com"
        )

        assert result is not None
        assert client.call_count == 1
        assert "methodology_detailed" in result

    def test_analyze_article_empty_inputs(self):
        """Test article analysis with empty inputs"""
        client = TestableAIClient()

        with pytest.raises(ContentProcessingError):
            client.analyze_article("", "Test content")

        with pytest.raises(ContentProcessingError):
            client.analyze_article("Test Title", "")

    def test_analyze_article_api_failure(self):
        """Test article analysis with API failure"""
        client = TestableAIClient()
        client.should_fail = True

        with pytest.raises(APIConnectionError):
            client.analyze_article("Test Title", "Test content")

    @patch("time.sleep")  # Mock sleep to speed up test
    def test_retry_with_backoff_success_after_retry(self, mock_sleep):
        """Test retry with backoff succeeding after retry"""
        client = TestableAIClient()

        call_count = 0

        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIConnectionError("First failure", "test")
            return "success"

        result = client._retry_with_backoff(mock_func)

        assert result == "success"
        assert call_count == 2
        assert mock_sleep.called

    @patch("time.sleep")
    def test_retry_with_backoff_max_retries_exceeded(self, mock_sleep):
        """Test retry with backoff exceeding max retries"""
        client = TestableAIClient()
        client.max_retries = 2

        def mock_func():
            raise APIConnectionError("Always fails", "test")

        with pytest.raises(APIConnectionError):
            client._retry_with_backoff(mock_func)

        assert mock_sleep.call_count == 2  # Should retry 2 times

    @patch("time.sleep")
    def test_retry_with_backoff_rate_limit(self, mock_sleep):
        """Test retry with backoff for rate limit errors"""
        client = TestableAIClient()

        call_count = 0

        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIRateLimitError("Rate limited", "test", retry_after=5)
            return "success"

        result = client._retry_with_backoff(mock_func)

        assert result == "success"
        mock_sleep.assert_called_with(5)  # Should use retry_after value

    def test_batch_analyze(self):
        """Test batch analysis"""
        client = TestableAIClient()

        articles = [
            {
                "title": "Article 1",
                "content": "Content 1",
                "url": "https://example.com/1",
            },
            {
                "title": "Article 2",
                "content": "Content 2",
                "url": "https://example.com/2",
            },
        ]

        results = client.batch_analyze(articles)

        assert len(results) == 2
        assert client.call_count == 2
        assert all(result is not None for result in results)

    def test_batch_analyze_with_failures(self):
        """Test batch analysis with some failures"""
        client = TestableAIClient()

        articles = [
            {"title": "Article 1", "content": "Content 1"},
            {"title": "", "content": "Content 2"},  # Invalid - empty title
            {"title": "Article 3", "content": "Content 3"},
        ]

        results = client.batch_analyze(articles)

        assert len(results) == 3
        assert results[0] is not None  # Success
        assert results[1] is None  # Failure
        assert results[2] is not None  # Success

    def test_test_connection_success(self):
        """Test successful connection test"""
        client = TestableAIClient()

        result = client.test_connection()

        assert result is True
        assert client.call_count == 1

    def test_test_connection_failure(self):
        """Test failed connection test"""
        client = TestableAIClient()
        client.should_fail = True

        result = client.test_connection()

        assert result is False

    def test_get_provider_info(self):
        """Test getting provider information"""
        client = TestableAIClient("test-key", "test-model", "test-provider")

        info = client.get_provider_info()

        assert info["provider"] == "test-provider"
        assert info["model"] == "test-model"
        assert "max_retries" in info
        assert "rate_limit_delay" in info
        assert "timeout" in info
