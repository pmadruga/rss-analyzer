"""
Async Base AI Client Module

Abstract base class for async AI clients to eliminate code duplication
and provide consistent interface across all providers with concurrent request support.
"""

import asyncio
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


class AsyncAIClient(ABC):
    """Abstract base class for async AI clients with concurrent request support"""

    def __init__(self, api_key: str, model: str, provider_name: str):
        """
        Initialize async base AI client

        Args:
            api_key: API key for the service
            model: Model name to use
            provider_name: Name of the provider (for logging/errors)
        """
        self.api_key = self._validate_api_key(api_key)
        self.model = model
        self.provider_name = provider_name

        # Configuration
        self.max_retries = CONFIG.api.MAX_RETRIES
        self.base_delay = CONFIG.api.BASE_DELAY
        self.rate_limit_delay = CONFIG.api.RATE_LIMIT_DELAY
        self.timeout = CONFIG.api.TIMEOUT

        # Concurrent request configuration
        self.max_concurrent_requests = 10
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Rate limiting
        self._last_request_times = []
        self._rate_limit_lock = asyncio.Lock()

        # Common system prompt for analysis
        self.system_prompt = self._create_system_prompt()

        logger.info(
            f"Initialized async {provider_name} client with model {model} "
            f"(max concurrent: {self.max_concurrent_requests})"
        )

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
        return """You are an expert analyst. Your task is to:
1. FIRST, identify the actual title of this article/paper from the content (not from the provided title which may be generic)
2. Then analyze the content using the Feynman technique as if you were its author

Please respond in this JSON format:
{
    "extracted_title": "The actual, specific title found in the content",
    "analysis": "Your detailed Feynman technique analysis explaining this content in depth..."
}

Focus on finding the real title from headings, paper titles, or the main subject matter - not generic page titles like 'Home' or site names."""

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests with async support"""
        async with self._rate_limit_lock:
            current_time = time.time()

            # Remove old timestamps outside the rate limit window
            cutoff_time = current_time - self.rate_limit_delay
            self._last_request_times = [
                t for t in self._last_request_times if t > cutoff_time
            ]

            # If we have recent requests, wait before making a new one
            if self._last_request_times:
                time_since_last = current_time - self._last_request_times[-1]
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)

            # Record this request time
            self._last_request_times.append(time.time())

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

            # Try to parse JSON response first
            extracted_title = None
            analysis_content = response_text

            if response_text.startswith("{") and response_text.endswith("}"):
                try:
                    import json

                    parsed_response = json.loads(response_text)
                    extracted_title = parsed_response.get("extracted_title")
                    analysis_content = parsed_response.get("analysis", response_text)
                    logger.info(f"Extracted title from analysis: '{extracted_title}'")
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat entire response as analysis
                    logger.debug("Response not valid JSON, using as plain text")

            # Try to extract title from markdown-style response
            if not extracted_title and response_text.startswith("**Title:"):
                import re

                title_match = re.search(r"\*\*Title:\*\*\s*([^\n]+)", response_text)
                if title_match:
                    extracted_title = title_match.group(1).strip()
                    # Remove the title line from the analysis content
                    analysis_content = re.sub(
                        r"\*\*Title:\*\*[^\n]+\n?", "", response_text
                    ).strip()

            # Store the comprehensive Feynman technique explanation in methodology_detailed
            analysis = {
                "methodology_detailed": analysis_content,
                "technical_approach": "",
                "key_findings": "",
                "research_design": "",
                "extracted_title": extracted_title,  # Add extracted title to metadata
                "metadata": {
                    "ai_provider": self.provider_name.lower(),
                    "model": self.model,
                    "processed_at": time.time(),
                    "title_extraction_attempted": True,
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

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute async function with exponential backoff retry logic"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except APIRateLimitError as e:
                sleep_time = e.retry_after or self.base_delay * (2**attempt)

                logger.warning(
                    f"Rate limited on attempt {attempt + 1}/{self.max_retries}. "
                    f"Sleeping for {sleep_time:.2f}s"
                )
                await asyncio.sleep(sleep_time)
                last_exception = e

            except (APIConnectionError, APIResponseError) as e:
                sleep_time = self.base_delay * (2**attempt)
                logger.warning(
                    f"API error on attempt {attempt + 1}/{self.max_retries}: {e}. "
                    f"Retrying in {sleep_time:.2f}s"
                )
                await asyncio.sleep(sleep_time)
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
    async def _make_api_call_async(self, prompt: str) -> str:
        """
        Make async API call to the specific provider

        Args:
            prompt: The prompt to send to the API

        Returns:
            Raw response text from the API

        Raises:
            APIClientError: If the API call fails
        """

    async def analyze_article_async(
        self, title: str, content: str, url: str = ""
    ) -> dict[str, Any] | None:
        """
        Analyze article content using AI asynchronously

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
        async with self.semaphore:
            try:
                # Validate input
                if not title or not content:
                    raise ContentProcessingError("Title and content are required")

                # Prepare content
                formatted_content = self._prepare_content(title, content, url)

                # Create analysis prompt
                prompt = (
                    f"{self.system_prompt}\n\nArticle to analyze:\n{formatted_content}"
                )

                # Enforce rate limiting
                await self._enforce_rate_limit()

                # Make API call with retry logic
                response_text = await self._retry_with_backoff(
                    self._make_api_call_async, prompt
                )

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

    async def batch_analyze_async(
        self, articles: list[dict[str, str]], max_concurrent: int | None = None
    ) -> list[dict[str, Any] | None]:
        """
        Analyze multiple articles concurrently

        Args:
            articles: List of dicts with 'title', 'content', and optional 'url'
            max_concurrent: Override max concurrent requests (default: use instance setting)

        Returns:
            List of analysis results (same order as input)
        """
        if max_concurrent:
            # Temporarily adjust semaphore
            self.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"Starting batch analysis of {len(articles)} articles "
            f"(max concurrent: {self.semaphore._value})"
        )

        # Create tasks for all articles
        tasks = [
            self.analyze_article_async(
                title=article["title"],
                content=article["content"],
                url=article.get("url", ""),
            )
            for article in articles
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze article {i + 1}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)

        success_count = sum(1 for r in processed_results if r is not None)
        logger.info(
            f"Batch analysis complete: {success_count}/{len(articles)} successful"
        )

        return processed_results

    async def test_connection_async(self) -> bool:
        """
        Test API connection asynchronously

        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_prompt = "Test connection. Please respond with 'OK'."
            response = await self._make_api_call_async(test_prompt)
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
            "max_concurrent_requests": self.max_concurrent_requests,
            "async_enabled": True,
        }
