"""
AI Client Module

Provides unified interface for AI clients (Claude, Mistral, OpenAI).
Supports both synchronous and asynchronous clients for concurrent processing.
"""

from .async_base import AsyncAIClient
from .async_claude import AsyncClaudeClient
from .async_mistral import AsyncMistralClient
from .async_openai import AsyncOpenAIClient
from .async_utils import (
    AsyncArticleProcessor,
    process_articles_async,
    run_async_processing,
)
from .base import BaseAIClient
from .claude import ClaudeClient
from .factory import AIClientFactory
from .mistral import MistralClient
from .openai import OpenAIClient

__all__ = [
    # Synchronous clients
    "BaseAIClient",
    "ClaudeClient",
    "MistralClient",
    "OpenAIClient",
    # Asynchronous clients
    "AsyncAIClient",
    "AsyncClaudeClient",
    "AsyncMistralClient",
    "AsyncOpenAIClient",
    # Factory
    "AIClientFactory",
    # Async utilities
    "AsyncArticleProcessor",
    "process_articles_async",
    "run_async_processing",
]
