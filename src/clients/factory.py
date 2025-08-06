"""
AI Client Factory Module

Factory pattern implementation for creating AI clients.
Provides a clean interface for client instantiation and configuration.
"""

import logging
from typing import Any

from ..exceptions import APIClientError, ConfigurationError
from .base import BaseAIClient
from .claude import ClaudeClient
from .mistral import MistralClient
from .openai import OpenAIClient

logger = logging.getLogger(__name__)


class AIClientFactory:
    """Factory for creating AI clients"""

    # Registry of available clients
    _clients = {
        "anthropic": ClaudeClient,
        "claude": ClaudeClient,  # Alias
        "mistral": MistralClient,
        "openai": OpenAIClient,
    }

    # Default models for each provider
    _default_models = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "claude": "claude-3-5-sonnet-20241022",
        "mistral": "mistral-large-latest",
        "openai": "gpt-4",
    }

    @classmethod
    def create_client(
        self, provider: str, api_key: str, model: str | None = None, **kwargs
    ) -> BaseAIClient:
        """
        Create an AI client instance

        Args:
            provider: Provider name (anthropic, mistral, openai)
            api_key: API key for the provider
            model: Model name (optional, uses default if not provided)
            **kwargs: Additional configuration parameters

        Returns:
            Configured AI client instance

        Raises:
            ConfigurationError: If provider is not supported or configuration is invalid
            APIClientError: If client initialization fails
        """
        provider = provider.lower().strip()

        if provider not in self._clients:
            available = ", ".join(self._clients.keys())
            raise ConfigurationError(
                f"Unsupported AI provider: {provider}. Available: {available}"
            )

        if not api_key:
            raise ConfigurationError(f"API key is required for {provider}")

        # Use default model if not specified
        if not model:
            model = self._default_models[provider]

        client_class = self._clients[provider]

        try:
            logger.info(f"Creating {provider} client with model {model}")
            return client_class(api_key=api_key, model=model, **kwargs)

        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise APIClientError(f"Client creation failed: {e}", provider)

    @classmethod
    def get_available_providers(cls) -> dict[str, str]:
        """Get list of available providers and their default models"""
        return {
            provider: cls._default_models[provider] for provider in cls._clients.keys()
        }

    @classmethod
    def register_client(cls, provider: str, client_class: type, default_model: str):
        """
        Register a new client provider

        Args:
            provider: Provider name
            client_class: Client class (must inherit from BaseAIClient)
            default_model: Default model for this provider
        """
        if not issubclass(client_class, BaseAIClient):
            raise ConfigurationError("Client class must inherit from BaseAIClient")

        cls._clients[provider.lower()] = client_class
        cls._default_models[provider.lower()] = default_model

        logger.info(f"Registered new AI client: {provider}")

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> BaseAIClient:
        """
        Create client from configuration dictionary

        Args:
            config: Configuration dictionary containing provider, api_key, model etc.

        Returns:
            Configured AI client instance
        """
        provider = config.get("api_provider", "anthropic")

        # Get API key based on provider
        api_key_mapping = {
            "anthropic": "anthropic_api_key",
            "claude": "anthropic_api_key",
            "mistral": "mistral_api_key",
            "openai": "openai_api_key",
        }

        api_key_field = api_key_mapping.get(provider)
        if not api_key_field:
            raise ConfigurationError(f"Unknown provider: {provider}")

        api_key = config.get(api_key_field)
        if not api_key:
            raise ConfigurationError(f"Missing API key: {api_key_field}")

        # Get model
        model_mapping = {
            "anthropic": "claude_model",
            "claude": "claude_model",
            "mistral": "mistral_model",
            "openai": "openai_model",
        }

        model_field = model_mapping.get(provider)
        model = config.get(model_field) if model_field else None

        return cls.create_client(provider, api_key, model)
