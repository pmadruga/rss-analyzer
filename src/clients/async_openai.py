"""
Async OpenAI Client Module

OpenAI async API client for concurrent request processing.
"""

import logging
from typing import Any

import openai

from ..config import CONFIG
from ..exceptions import (
    APIConnectionError,
    APIQuotaExceededError,
    APIRateLimitError,
    APIResponseError,
)
from .async_base import AsyncAIClient

logger = logging.getLogger(__name__)


class AsyncOpenAIClient(AsyncAIClient):
    """Async OpenAI API client"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize async OpenAI client

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        super().__init__(api_key, model, "OpenAI")

        # Initialize OpenAI async client
        try:
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except Exception as e:
            raise APIConnectionError(
                f"Failed to initialize async OpenAI client: {e}", "OpenAI"
            )

        # OpenAI-specific configuration
        self.max_tokens = CONFIG.api.MAX_TOKENS
        self.temperature = CONFIG.api.TEMPERATURE

    async def _make_api_call_async(self, prompt: str) -> str:
        """
        Make async API call to OpenAI

        Args:
            prompt: The prompt to send to OpenAI

        Returns:
            Response text from OpenAI

        Raises:
            APIConnectionError: If connection fails
            APIRateLimitError: If rate limited
            APIResponseError: If response is invalid
            APIQuotaExceededError: If quota exceeded
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Extract response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise APIResponseError("Empty response from OpenAI", "OpenAI")

        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}")
            # Try to extract retry-after from headers
            retry_after = getattr(e, "retry_after", None)
            raise APIRateLimitError(str(e), "OpenAI", retry_after)

        except openai.APIStatusError as e:
            if e.status_code == 429:
                raise APIRateLimitError(f"Rate limited: {e.message}", "OpenAI")
            elif e.status_code == 402:
                raise APIQuotaExceededError(f"Quota exceeded: {e.message}", "OpenAI")
            else:
                raise APIResponseError(
                    f"API error: {e.message}", "OpenAI", e.status_code
                )

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise APIConnectionError(f"Connection failed: {e}", "OpenAI")

        except Exception as e:
            logger.error(f"Unexpected OpenAI API error: {e}")
            raise APIResponseError(f"Unexpected error: {e}", "OpenAI")

    def get_model_info(self) -> dict[str, Any]:
        """Get OpenAI-specific model information"""
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
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }
        return context_windows.get(self.model, 8192)  # Default to 8k

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.close()
