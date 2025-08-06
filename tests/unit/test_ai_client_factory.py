"""
Unit tests for AI client factory
"""

from unittest.mock import Mock, patch

import pytest

from src.clients.base import BaseAIClient
from src.clients.factory import AIClientFactory
from src.exceptions import APIClientError, ConfigurationError


class MockClient(BaseAIClient):
    """Mock client for testing factory"""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model, "mock")

    def _make_api_call(self, prompt: str) -> str:
        return "mock response"


class TestAIClientFactory:
    """Test AIClientFactory"""

    def test_get_available_providers(self):
        """Test getting available providers"""
        providers = AIClientFactory.get_available_providers()

        assert isinstance(providers, dict)
        assert "anthropic" in providers
        assert "claude" in providers
        assert "mistral" in providers
        assert "openai" in providers

        # Check default models are set
        assert providers["anthropic"] == "claude-3-5-sonnet-20241022"
        assert providers["mistral"] == "mistral-large-latest"
        assert providers["openai"] == "gpt-4"

    @patch("src.clients.factory.ClaudeClient")
    def test_create_client_anthropic(self, mock_claude_client):
        """Test creating Anthropic client"""
        mock_instance = Mock()
        mock_claude_client.return_value = mock_instance

        client = AIClientFactory.create_client(
            provider="anthropic",
            api_key="sk-test-key",
            model="claude-3-5-sonnet-20241022",
        )

        assert client == mock_instance
        mock_claude_client.assert_called_once_with(
            api_key="sk-test-key", model="claude-3-5-sonnet-20241022"
        )

    @patch("src.clients.factory.ClaudeClient")
    def test_create_client_claude_alias(self, mock_claude_client):
        """Test creating client with 'claude' alias"""
        mock_instance = Mock()
        mock_claude_client.return_value = mock_instance

        client = AIClientFactory.create_client(provider="claude", api_key="sk-test-key")

        assert client == mock_instance
        mock_claude_client.assert_called_once_with(
            api_key="sk-test-key",
            model="claude-3-5-sonnet-20241022",  # Default model
        )

    @patch("src.clients.factory.MistralClient")
    def test_create_client_mistral(self, mock_mistral_client):
        """Test creating Mistral client"""
        mock_instance = Mock()
        mock_mistral_client.return_value = mock_instance

        client = AIClientFactory.create_client(
            provider="mistral", api_key="test-mistral-key", model="mistral-large-latest"
        )

        assert client == mock_instance
        mock_mistral_client.assert_called_once_with(
            api_key="test-mistral-key", model="mistral-large-latest"
        )

    @patch("src.clients.factory.OpenAIClient")
    def test_create_client_openai(self, mock_openai_client):
        """Test creating OpenAI client"""
        mock_instance = Mock()
        mock_openai_client.return_value = mock_instance

        client = AIClientFactory.create_client(
            provider="openai", api_key="sk-openai-key", model="gpt-4"
        )

        assert client == mock_instance
        mock_openai_client.assert_called_once_with(
            api_key="sk-openai-key", model="gpt-4"
        )

    def test_create_client_unsupported_provider(self):
        """Test creating client with unsupported provider"""
        with pytest.raises(ConfigurationError) as exc_info:
            AIClientFactory.create_client(provider="unsupported", api_key="test-key")

        assert "Unsupported AI provider: unsupported" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_create_client_empty_api_key(self):
        """Test creating client with empty API key"""
        with pytest.raises(ConfigurationError) as exc_info:
            AIClientFactory.create_client(provider="anthropic", api_key="")

        assert "API key is required" in str(exc_info.value)

    def test_create_client_none_api_key(self):
        """Test creating client with None API key"""
        with pytest.raises(ConfigurationError) as exc_info:
            AIClientFactory.create_client(provider="anthropic", api_key=None)

        assert "API key is required" in str(exc_info.value)

    def test_create_client_case_insensitive(self):
        """Test creating client with case insensitive provider"""
        with patch("src.clients.factory.ClaudeClient") as mock_claude_client:
            mock_instance = Mock()
            mock_claude_client.return_value = mock_instance

            client = AIClientFactory.create_client(
                provider="ANTHROPIC", api_key="sk-test-key"
            )

            assert client == mock_instance

    def test_create_client_with_whitespace(self):
        """Test creating client with provider containing whitespace"""
        with patch("src.clients.factory.ClaudeClient") as mock_claude_client:
            mock_instance = Mock()
            mock_claude_client.return_value = mock_instance

            client = AIClientFactory.create_client(
                provider="  anthropic  ", api_key="sk-test-key"
            )

            assert client == mock_instance

    @patch("src.clients.factory.ClaudeClient")
    def test_create_client_initialization_failure(self, mock_claude_client):
        """Test handling client initialization failure"""
        mock_claude_client.side_effect = Exception("Initialization failed")

        with pytest.raises(APIClientError) as exc_info:
            AIClientFactory.create_client(provider="anthropic", api_key="sk-test-key")

        assert "Client creation failed" in str(exc_info.value)

    def test_register_client(self):
        """Test registering new client"""
        AIClientFactory.register_client(
            provider="mock", client_class=MockClient, default_model="mock-model"
        )

        # Check it was registered
        providers = AIClientFactory.get_available_providers()
        assert "mock" in providers
        assert providers["mock"] == "mock-model"

        # Test creating the registered client
        client = AIClientFactory.create_client(provider="mock", api_key="test-key")

        assert isinstance(client, MockClient)
        assert client.provider_name == "mock"

    def test_register_client_invalid_class(self):
        """Test registering client with invalid class"""

        class InvalidClient:
            pass

        with pytest.raises(ConfigurationError) as exc_info:
            AIClientFactory.register_client(
                provider="invalid",
                client_class=InvalidClient,
                default_model="test-model",
            )

        assert "must inherit from BaseAIClient" in str(exc_info.value)

    def test_create_from_config_anthropic(self):
        """Test creating client from config dictionary - Anthropic"""
        config = {
            "api_provider": "anthropic",
            "anthropic_api_key": "sk-test-key",
            "claude_model": "claude-3-5-sonnet-20241022",
        }

        with patch("src.clients.factory.ClaudeClient") as mock_claude_client:
            mock_instance = Mock()
            mock_claude_client.return_value = mock_instance

            client = AIClientFactory.create_from_config(config)

            assert client == mock_instance
            mock_claude_client.assert_called_once_with(
                api_key="sk-test-key", model="claude-3-5-sonnet-20241022"
            )

    def test_create_from_config_mistral(self):
        """Test creating client from config dictionary - Mistral"""
        config = {
            "api_provider": "mistral",
            "mistral_api_key": "test-mistral-key",
            "mistral_model": "mistral-large-latest",
        }

        with patch("src.clients.factory.MistralClient") as mock_mistral_client:
            mock_instance = Mock()
            mock_mistral_client.return_value = mock_instance

            client = AIClientFactory.create_from_config(config)

            assert client == mock_instance

    def test_create_from_config_openai(self):
        """Test creating client from config dictionary - OpenAI"""
        config = {
            "api_provider": "openai",
            "openai_api_key": "sk-openai-key",
            "openai_model": "gpt-4",
        }

        with patch("src.clients.factory.OpenAIClient") as mock_openai_client:
            mock_instance = Mock()
            mock_openai_client.return_value = mock_instance

            client = AIClientFactory.create_from_config(config)

            assert client == mock_instance

    def test_create_from_config_unknown_provider(self):
        """Test creating client from config with unknown provider"""
        config = {"api_provider": "unknown", "unknown_api_key": "test-key"}

        with pytest.raises(ConfigurationError) as exc_info:
            AIClientFactory.create_from_config(config)

        assert "Unknown provider: unknown" in str(exc_info.value)

    def test_create_from_config_missing_api_key(self):
        """Test creating client from config with missing API key"""
        config = {
            "api_provider": "anthropic"
            # Missing anthropic_api_key
        }

        with pytest.raises(ConfigurationError) as exc_info:
            AIClientFactory.create_from_config(config)

        assert "Missing API key: anthropic_api_key" in str(exc_info.value)

    def test_create_from_config_default_provider(self):
        """Test creating client from config with default provider"""
        config = {
            "anthropic_api_key": "sk-test-key"
            # No api_provider specified, should default to anthropic
        }

        with patch("src.clients.factory.ClaudeClient") as mock_claude_client:
            mock_instance = Mock()
            mock_claude_client.return_value = mock_instance

            client = AIClientFactory.create_from_config(config)

            assert client == mock_instance
