"""
Base AI Client Module

Abstract base class for AI clients to eliminate code duplication
and provide consistent interface across all providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ..config import CONFIG
from ..exceptions import (
    APIClientError,
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
    ContentProcessingError,
)

logger = logging.getLogger(__name__)


class BaseAIClient(ABC):
    """Abstract base class for AI clients"""

    def __init__(self, api_key: str, model: str, provider_name: str):
        """
        Initialize base AI client

        Args:
            api_key: API key for the service
            model: Model name to use
            provider_name: Name of the provider (for logging/errors)
        """
        self.api_key = self._validate_api_key(api_key)
        self.model = model
        self.provider_name = provider_name
        self.last_request_time = 0.0

        # Configuration
        self.max_retries = CONFIG.api.MAX_RETRIES
        self.base_delay = CONFIG.api.BASE_DELAY
        self.rate_limit_delay = CONFIG.api.RATE_LIMIT_DELAY
        self.timeout = CONFIG.api.TIMEOUT

        # Common system prompt for analysis
        self.system_prompt = self._create_system_prompt()

        logger.info(f"Initialized {provider_name} client with model {model}")

    def _validate_api_key(self, api_key: str) -> str:
        """Validate API key format"""
        if not api_key or len(api_key) < 10:
            raise APIClientError(
                "Invalid API key provided",
                self.provider_name if hasattr(self, "provider_name") else "unknown",
            )
        return api_key

    def _create_system_prompt(self) -> str:
        """Create standardized system prompt for analysis"""
        return (
            "Analyze this content in-depth and explain it using the Feynman technique"
        )

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _prepare_content(self, title: str, content: str, url: str = "") -> str:
        """Prepare content for analysis with length limits"""
        # Truncate content if too long
        max_length = CONFIG.processing.MAX_CONTENT_LENGTH
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[Content truncated due to length]"
            logger.warning(f"Content truncated to {max_length} chars for analysis")

        return f"""Title: {title}
URL: {url}

Content:
{content}"""

    def _parse_analysis_response(self, response_text: str) -> dict[str, Any] | None:
        """Parse and validate AI response"""
        if not response_text:
            return None

        try:
            response_text = response_text.strip()

            # Store the comprehensive Feynman technique explanation in methodology_detailed
            # Leave other fields empty since we're getting a single unified analysis
            analysis = {
                "methodology_detailed": response_text,
                "technical_approach": "",
                "key_findings": "",
                "research_design": "",
                "metadata": {
                    "ai_provider": self.provider_name.lower(),
                    "model": self.model,
                    "processed_at": time.time(),
                },
            }

            return analysis

        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}...")

            # Fallback: create structured response from text
            return self._create_fallback_analysis(response_text)

    def _create_fallback_analysis(self, text: str) -> dict[str, Any]:
        """Create fallback analysis when JSON parsing fails"""
        return {
            "methodology_detailed": text[:2000] if text else "Analysis parsing failed",
            "technical_approach": "",
            "key_findings": "",
            "research_design": "",
            "metadata": {
                "ai_provider": self.provider_name.lower(),
                "model": self.model,
                "processed_at": time.time(),
                "parsing_fallback": True,
            },
        }

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except APIRateLimitError as e:
                sleep_time = e.retry_after or self.base_delay * (2**attempt)

                logger.warning(
                    f"Rate limited on attempt {attempt + 1}/{self.max_retries}. "
                    f"Sleeping for {sleep_time:.2f}s"
                )
                time.sleep(sleep_time)
                last_exception = e

            except (APIConnectionError, APIResponseError) as e:
                sleep_time = self.base_delay * (2**attempt)
                logger.warning(
                    f"API error on attempt {attempt + 1}/{self.max_retries}: {e}. "
                    f"Retrying in {sleep_time:.2f}s"
                )
                time.sleep(sleep_time)
                last_exception = e

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                raise APIClientError(
                    f"Unexpected error: {e}", self.provider_name
                ) from e

        # All retries failed
        if last_exception:
            raise last_exception
        else:
            raise APIClientError(
                f"Max retries ({self.max_retries}) exceeded", self.provider_name
            )

    @abstractmethod
    def _make_api_call(self, prompt: str) -> str:
        """
        Make API call to the specific provider

        Args:
            prompt: The prompt to send to the API

        Returns:
            Raw response text from the API

        Raises:
            APIClientError: If the API call fails
        """

    def analyze_article(
        self, title: str, content: str, url: str = ""
    ) -> dict[str, Any] | None:
        """
        Analyze article content using AI

        Args:
            title: Article title
            content: Article content
            url: Article URL (optional)

        Returns:
            Analysis dictionary or None if failed

        Raises:
            ContentProcessingError: If content processing fails
            APIClientError: If API call fails
        """
        try:
            # Validate input
            if not title or not content:
                raise ContentProcessingError("Title and content are required")

            # Prepare content
            formatted_content = self._prepare_content(title, content, url)

            # Create analysis prompt
            prompt = f"{self.system_prompt}\n\nArticle to analyze:\n{formatted_content}"

            # Enforce rate limiting
            self._enforce_rate_limit()

            # Make API call with retry logic
            response_text = self._retry_with_backoff(self._make_api_call, prompt)

            # Parse response
            analysis = self._parse_analysis_response(response_text)

            if analysis:
                logger.info(f"Successfully analyzed article: {title[:50]}...")
                return analysis
            else:
                logger.error(f"Failed to parse analysis for: {title[:50]}...")
                return None

        except (ContentProcessingError, APIClientError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error analyzing article '{title}': {e}")
            raise ContentProcessingError(f"Analysis failed: {e}") from e

    def batch_analyze(
        self, articles: list[dict[str, str]]
    ) -> list[dict[str, Any] | None]:
        """
        Analyze multiple articles

        Args:
            articles: List of dicts with 'title', 'content', and optional 'url'

        Returns:
            List of analysis results (same order as input)
        """
        results = []

        for i, article in enumerate(articles):
            logger.info(f"Analyzing article {i + 1}/{len(articles)}")

            try:
                result = self.analyze_article(
                    title=article["title"],
                    content=article["content"],
                    url=article.get("url", ""),
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to analyze article {i + 1}: {e}")
                results.append(None)

        return results

    def test_connection(self) -> bool:
        """
        Test API connection

        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_prompt = "Test connection. Please respond with 'OK'."
            response = self._make_api_call(test_prompt)
            return bool(response and len(response.strip()) > 0)

        except Exception as e:
            logger.error(f"Connection test failed for {self.provider_name}: {e}")
            return False

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information"""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "max_retries": self.max_retries,
            "rate_limit_delay": self.rate_limit_delay,
            "timeout": self.timeout,
        }
