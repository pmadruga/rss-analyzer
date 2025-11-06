"""
Async Utilities Module

Utility functions for concurrent article processing with async AI clients.
"""

import asyncio
import logging
from typing import Any

from ..config import CONFIG
from .async_base import AsyncAIClient
from .factory import AIClientFactory

logger = logging.getLogger(__name__)


class AsyncArticleProcessor:
    """Async article processor with concurrent request management"""

    def __init__(
        self,
        client: AsyncAIClient | None = None,
        max_concurrent: int = 10,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize async article processor

        Args:
            client: Existing async client (optional)
            max_concurrent: Maximum concurrent requests (default: 10)
            provider: AI provider name (required if client not provided)
            api_key: API key (required if client not provided)
            model: Model name (optional)
        """
        if client:
            self.client = client
        elif provider and api_key:
            self.client = AIClientFactory.create_async_client(
                provider=provider, api_key=api_key, model=model
            )
        else:
            raise ValueError("Either client or (provider, api_key) must be provided")

        self.max_concurrent = max_concurrent
        self.client.max_concurrent_requests = max_concurrent

    async def process_articles(
        self, articles: list[dict[str, str]], show_progress: bool = True
    ) -> list[dict[str, Any] | None]:
        """
        Process multiple articles concurrently

        Args:
            articles: List of article dicts with 'title', 'content', 'url'
            show_progress: Show progress updates

        Returns:
            List of analysis results (same order as input)
        """
        if not articles:
            return []

        logger.info(
            f"Processing {len(articles)} articles with {self.max_concurrent} concurrent requests"
        )

        # Process articles with progress tracking
        if show_progress:
            results = await self._process_with_progress(articles)
        else:
            results = await self.client.batch_analyze_async(
                articles, max_concurrent=self.max_concurrent
            )

        return results

    async def _process_with_progress(
        self, articles: list[dict[str, str]]
    ) -> list[dict[str, Any] | None]:
        """Process articles with progress updates"""
        results = [None] * len(articles)
        completed = 0
        total = len(articles)

        # Create tasks with indices
        tasks = [
            self._process_with_index(i, article)
            for i, article in enumerate(articles)
        ]

        # Process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result
            completed += 1

            if completed % 5 == 0 or completed == total:
                logger.info(f"Progress: {completed}/{total} articles processed")

        return results

    async def _process_with_index(
        self, idx: int, article: dict[str, str]
    ) -> tuple[int, dict[str, Any] | None]:
        """Process single article and return with index"""
        try:
            result = await self.client.analyze_article_async(
                title=article["title"],
                content=article["content"],
                url=article.get("url", ""),
            )
            return idx, result
        except Exception as e:
            logger.error(f"Failed to process article {idx}: {e}")
            return idx, None

    async def process_in_batches(
        self,
        articles: list[dict[str, str]],
        batch_size: int = 50,
        delay_between_batches: float = 1.0,
    ) -> list[dict[str, Any] | None]:
        """
        Process articles in batches with delays

        Args:
            articles: List of article dicts
            batch_size: Number of articles per batch
            delay_between_batches: Delay in seconds between batches

        Returns:
            List of analysis results
        """
        if not articles:
            return []

        all_results = []
        num_batches = (len(articles) + batch_size - 1) // batch_size

        logger.info(
            f"Processing {len(articles)} articles in {num_batches} batches "
            f"(batch size: {batch_size}, delay: {delay_between_batches}s)"
        )

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(articles))
            batch = articles[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_num + 1}/{num_batches} "
                f"({len(batch)} articles)"
            )

            batch_results = await self.client.batch_analyze_async(
                batch, max_concurrent=self.max_concurrent
            )
            all_results.extend(batch_results)

            # Delay between batches (except after last batch)
            if batch_num < num_batches - 1 and delay_between_batches > 0:
                logger.debug(f"Waiting {delay_between_batches}s before next batch")
                await asyncio.sleep(delay_between_batches)

        return all_results

    async def close(self):
        """Close the async client"""
        if hasattr(self.client, "__aexit__"):
            await self.client.__aexit__(None, None, None)


async def process_articles_async(
    articles: list[dict[str, str]],
    provider: str,
    api_key: str,
    model: str | None = None,
    max_concurrent: int = 10,
    batch_size: int | None = None,
) -> list[dict[str, Any] | None]:
    """
    Convenience function to process articles asynchronously

    Args:
        articles: List of article dicts with 'title', 'content', 'url'
        provider: AI provider name
        api_key: API key
        model: Model name (optional)
        max_concurrent: Maximum concurrent requests
        batch_size: Process in batches if specified

    Returns:
        List of analysis results
    """
    processor = AsyncArticleProcessor(
        provider=provider, api_key=api_key, model=model, max_concurrent=max_concurrent
    )

    try:
        if batch_size:
            results = await processor.process_in_batches(articles, batch_size=batch_size)
        else:
            results = await processor.process_articles(articles)
        return results
    finally:
        await processor.close()


def run_async_processing(
    articles: list[dict[str, str]],
    provider: str,
    api_key: str,
    model: str | None = None,
    max_concurrent: int = 10,
    batch_size: int | None = None,
) -> list[dict[str, Any] | None]:
    """
    Synchronous wrapper for async article processing

    Args:
        articles: List of article dicts
        provider: AI provider name
        api_key: API key
        model: Model name (optional)
        max_concurrent: Maximum concurrent requests
        batch_size: Process in batches if specified

    Returns:
        List of analysis results
    """
    return asyncio.run(
        process_articles_async(
            articles=articles,
            provider=provider,
            api_key=api_key,
            model=model,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
        )
    )
