"""
Refactored Mistral Client Module

Mistral AI API client using the base client architecture.
Eliminates code duplication while maintaining Mistral-specific functionality.
"""

import logging
from typing import Any

from mistralai.client import MistralClient as MistralAPIClient
from mistralai.models.chat_completion import ChatMessage

from ..config import CONFIG
from ..exceptions import (
    APIConnectionError,
    APIQuotaExceededError,
    APIRateLimitError,
    APIResponseError,
)
from .base import BaseAIClient

logger = logging.getLogger(__name__)


class MistralClient(BaseAIClient):
    """Mistral AI API client"""

    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        """
        Initialize Mistral client

        Args:
            api_key: Mistral API key
            model: Mistral model to use
        """
        super().__init__(api_key, model, "Mistral")

        # Initialize Mistral client
        try:
            self.client = MistralAPIClient(api_key=api_key)
        except Exception as e:
            raise APIConnectionError(
                f"Failed to initialize Mistral client: {e}", "Mistral"
            )

        # Mistral-specific configuration
        self.max_tokens = CONFIG.api.MAX_TOKENS
        self.temperature = CONFIG.api.TEMPERATURE

    def _make_api_call(self, prompt: str) -> str:
        """
        Make API call to Mistral

        Args:
            prompt: The prompt to send to Mistral

        Returns:
            Response text from Mistral

        Raises:
            APIConnectionError: If connection fails
            APIRateLimitError: If rate limited
            APIResponseError: If response is invalid
            APIQuotaExceededError: If quota exceeded
        """
        try:
            # Create chat message
            messages = [ChatMessage(role="user", content=prompt)]

            # Make API call
            response = self.client.chat(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Extract response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise APIResponseError("Empty response from Mistral", "Mistral")

        except Exception as e:
            error_msg = str(e).lower()

            # Handle rate limiting
            if "rate limit" in error_msg or "too many requests" in error_msg:
                logger.warning(f"Mistral rate limit exceeded: {e}")
                raise APIRateLimitError(str(e), "Mistral")

            # Handle quota exceeded
            elif "quota" in error_msg or "billing" in error_msg:
                raise APIQuotaExceededError(f"Quota exceeded: {e}", "Mistral")

            # Handle connection errors
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"Mistral connection error: {e}")
                raise APIConnectionError(f"Connection failed: {e}", "Mistral")

            # Generic API error
            else:
                logger.error(f"Mistral API error: {e}")
                raise APIResponseError(f"API error: {e}", "Mistral")

    def get_model_info(self) -> dict[str, Any]:
        """Get Mistral-specific model information"""
        info = self.get_provider_info()
        info.update(
            {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "context_window": self._get_context_window(),
            }
        )
        return info

    def _get_context_window(self) -> int:
        """Get context window size for the model"""
        context_windows = {
            "mistral-large-latest": 32000,
            "mistral-medium-latest": 32000,
            "mistral-small-latest": 32000,
        }
        return context_windows.get(self.model, 32000)  # Default to 32k
