"""
Async Claude Client Module

Anthropic Claude async API client for concurrent request processing.
"""

import logging
from typing import Any

import anthropic

from ..config import CONFIG
from ..exceptions import (
    APIConnectionError,
    APIQuotaExceededError,
    APIRateLimitError,
    APIResponseError,
)
from .async_base import AsyncAIClient

logger = logging.getLogger(__name__)


class AsyncClaudeClient(AsyncAIClient):
    """Async Anthropic Claude API client"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize async Claude client

        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        super().__init__(api_key, model, "Claude")

        # Initialize Anthropic async client
        try:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except Exception as e:
            raise APIConnectionError(
                f"Failed to initialize async Claude client: {e}", "Claude"
            )

        # Claude-specific configuration
        self.max_tokens = CONFIG.api.MAX_TOKENS
        self.temperature = CONFIG.api.TEMPERATURE

    async def _make_api_call_async(self, prompt: str) -> str:
        """
        Make async API call to Claude

        Args:
            prompt: The prompt to send to Claude

        Returns:
            Response text from Claude

        Raises:
            APIConnectionError: If connection fails
            APIRateLimitError: If rate limited
            APIResponseError: If response is invalid
            APIQuotaExceededError: If quota exceeded
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise APIResponseError("Empty response from Claude", "Claude")

        except anthropic.RateLimitError as e:
            logger.warning(f"Claude rate limit exceeded: {e}")
            # Try to extract retry-after from headers
            retry_after = getattr(e, "retry_after", None)
            raise APIRateLimitError(str(e), "Claude", retry_after)

        except anthropic.APIStatusError as e:
            if e.status_code == 429:
                raise APIRateLimitError(f"Rate limited: {e.message}", "Claude")
            elif e.status_code == 402:
                raise APIQuotaExceededError(f"Quota exceeded: {e.message}", "Claude")
            else:
                raise APIResponseError(
                    f"API error: {e.message}", "Claude", e.status_code
                )

        except anthropic.APIConnectionError as e:
            logger.error(f"Claude connection error: {e}")
            raise APIConnectionError(f"Connection failed: {e}", "Claude")

        except Exception as e:
            logger.error(f"Unexpected Claude API error: {e}")
            raise APIResponseError(f"Unexpected error: {e}", "Claude")

    def get_model_info(self) -> dict[str, Any]:
        """Get Claude-specific model information"""
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
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-opus-20240229": 200000,
        }
        return context_windows.get(self.model, 200000)  # Default to 200k

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.close()
