# Async/Await Guide for RSS Analyzer

## Overview

This guide covers asynchronous programming patterns used in the RSS Analyzer project, including async clients, concurrent operations, and best practices for async/await in Python.

## Table of Contents

1. [Introduction to Async](#introduction-to-async)
2. [Async API Clients](#async-api-clients)
3. [Async Pipeline Processing](#async-pipeline-processing)
4. [Concurrent Operations](#concurrent-operations)
5. [Async Best Practices](#async-best-practices)
6. [Migration from Sync to Async](#migration-from-sync-to-async)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Testing Async Code](#testing-async-code)

---

## Introduction to Async

### Why Async?

Async programming allows multiple operations to run concurrently without blocking, leading to:

- **Better throughput**: Handle multiple API calls simultaneously
- **Lower latency**: Don't wait for slow I/O operations
- **Resource efficiency**: Use fewer threads/processes
- **Scalability**: Handle more concurrent users

### When to Use Async

✅ **Good for**:
- API calls (HTTP requests)
- Database queries
- File I/O operations
- Network operations
- Multiple independent tasks

❌ **Not ideal for**:
- CPU-intensive computations
- Simple sequential operations
- Single-threaded bottlenecks

---

## Async API Clients

### Basic Async Client Pattern

```python
import aiohttp
import asyncio
from typing import Dict, Any

class AsyncClaudeClient:
    """Async Anthropic Claude API client"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.session = None

    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()

    async def analyze_content(
        self,
        content: str,
        prompt: str = None
    ) -> Dict[str, Any]:
        """Analyze content asynchronously"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with.")

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": prompt or f"Analyze this content: {content}"
                }
            ]
        }

        async with self.session.post(
            f"{self.base_url}/messages",
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()
```

### Usage Example

```python
async def main():
    """Example async client usage"""
    async with AsyncClaudeClient(api_key="sk-xxx") as client:
        result = await client.analyze_content("Article content here")
        print(result)

# Run async function
asyncio.run(main())
```

---

## Async Pipeline Processing

### Sequential vs Concurrent Processing

#### Sequential (Slow)

```python
async def process_articles_sequential(articles: list) -> list:
    """Process articles one by one"""
    results = []
    for article in articles:
        # Each operation waits for the previous one
        content = await scrape_article(article.url)
        analysis = await analyze_content(content)
        results.append(analysis)
    return results

# Total time = sum of all operation times
# 10 articles × 2 seconds = 20 seconds
```

#### Concurrent (Fast)

```python
async def process_articles_concurrent(articles: list) -> list:
    """Process articles concurrently"""
    # Create tasks for all articles
    tasks = [
        process_single_article(article)
        for article in articles
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    return results

async def process_single_article(article):
    """Process one article"""
    content = await scrape_article(article.url)
    analysis = await analyze_content(content)
    return analysis

# Total time = longest single operation
# 10 articles concurrent = ~2 seconds
```

### Complete Pipeline Example

```python
from typing import List
import aiohttp
import asyncio

class AsyncArticleProcessor:
    """Async article processing pipeline"""

    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_articles(
        self,
        articles: List[dict]
    ) -> List[dict]:
        """Process multiple articles with concurrency limit"""
        tasks = [
            self._process_with_semaphore(article)
            for article in articles
        ]
        return await asyncio.gather(*tasks)

    async def _process_with_semaphore(
        self,
        article: dict
    ) -> dict:
        """Process article with concurrency control"""
        async with self.semaphore:
            return await self._process_single(article)

    async def _process_single(self, article: dict) -> dict:
        """Process single article"""
        try:
            # Step 1: Scrape content
            content = await self._scrape(article['url'])

            # Step 2: Analyze with AI
            analysis = await self._analyze(content)

            # Step 3: Store results
            await self._store(article['id'], analysis)

            return {
                'id': article['id'],
                'status': 'success',
                'analysis': analysis
            }

        except Exception as e:
            return {
                'id': article['id'],
                'status': 'error',
                'error': str(e)
            }

    async def _scrape(self, url: str) -> str:
        """Scrape article content"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    async def _analyze(self, content: str) -> dict:
        """Analyze content with AI"""
        async with AsyncClaudeClient(self.api_key) as client:
            return await client.analyze_content(content)

    async def _store(self, article_id: int, analysis: dict):
        """Store analysis in database"""
        # Use asyncio.to_thread for sync database operations
        await asyncio.to_thread(
            self._store_sync,
            article_id,
            analysis
        )

    def _store_sync(self, article_id: int, analysis: dict):
        """Sync database operation"""
        # SQLite operations are synchronous
        # Run in thread pool to avoid blocking
        db = DatabaseManager()
        db.store_analysis(article_id, analysis)
```

---

## Concurrent Operations

### Using asyncio.gather

```python
async def fetch_multiple_urls(urls: List[str]) -> List[str]:
    """Fetch multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_one(session, url)
            for url in urls
        ]
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results

async def fetch_one(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch single URL"""
    async with session.get(url) as response:
        return await response.text()
```

### Using asyncio.gather with Error Handling

```python
async def fetch_with_error_handling(urls: List[str]) -> List[dict]:
    """Fetch URLs with individual error handling"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_safe(session, url)
            for url in urls
        ]
        # return_exceptions=True prevents one failure from stopping all
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        return [
            {'url': url, 'content': result}
            if not isinstance(result, Exception)
            else {'url': url, 'error': str(result)}
            for url, result in zip(urls, results)
        ]

async def fetch_safe(
    session: aiohttp.ClientSession,
    url: str
) -> str:
    """Fetch with automatic error handling"""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")
```

### Concurrent Limits with Semaphore

```python
async def fetch_with_limit(
    urls: List[str],
    max_concurrent: int = 5
) -> List[str]:
    """Limit concurrent requests"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_limited(url: str) -> str:
        async with semaphore:
            # Only max_concurrent requests run at once
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()

    tasks = [fetch_limited(url) for url in urls]
    return await asyncio.gather(*tasks)
```

---

## Async Best Practices

### 1. Always Use Context Managers

✅ **Good**:
```python
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()
```

❌ **Bad**:
```python
session = aiohttp.ClientSession()
response = await session.get(url)
data = await response.json()
# Forgot to close session!
```

### 2. Don't Mix Sync and Async

✅ **Good**:
```python
# Use asyncio.to_thread for sync operations
result = await asyncio.to_thread(sync_function, arg1, arg2)
```

❌ **Bad**:
```python
# Blocking the event loop!
result = sync_function(arg1, arg2)
```

### 3. Handle Timeouts

✅ **Good**:
```python
try:
    async with asyncio.timeout(30):
        result = await slow_operation()
except asyncio.TimeoutError:
    logger.error("Operation timed out")
```

❌ **Bad**:
```python
# No timeout - could hang forever
result = await slow_operation()
```

### 4. Use Semaphores for Rate Limiting

```python
class RateLimitedClient:
    """Client with rate limiting"""

    def __init__(self, max_per_second: int = 10):
        self.semaphore = asyncio.Semaphore(max_per_second)
        self.rate_limit_delay = 1.0 / max_per_second

    async def make_request(self, url: str):
        async with self.semaphore:
            result = await self._request(url)
            # Enforce minimum delay between requests
            await asyncio.sleep(self.rate_limit_delay)
            return result
```

### 5. Proper Error Propagation

```python
async def process_with_errors(items: List[str]) -> dict:
    """Process items and collect errors"""
    results = {'success': [], 'errors': []}

    tasks = [process_one(item) for item in items]
    results_or_errors = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )

    for item, result in zip(items, results_or_errors):
        if isinstance(result, Exception):
            results['errors'].append({
                'item': item,
                'error': str(result)
            })
        else:
            results['success'].append(result)

    return results
```

---

## Migration from Sync to Async

### Step-by-Step Migration

#### Step 1: Identify I/O Operations

```python
# Sync version
def process_article(url: str) -> dict:
    content = requests.get(url).text  # I/O
    analysis = call_api(content)       # I/O
    save_to_db(analysis)              # I/O
    return analysis
```

#### Step 2: Convert Functions to Async

```python
# Async version
async def process_article(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()  # async I/O

    analysis = await call_api_async(content)  # async I/O
    await save_to_db_async(analysis)         # async I/O
    return analysis
```

#### Step 3: Update Function Calls

```python
# Sync
results = [process_article(url) for url in urls]

# Async
tasks = [process_article(url) for url in urls]
results = await asyncio.gather(*tasks)
```

#### Step 4: Run Async Code

```python
# In script
if __name__ == "__main__":
    asyncio.run(main())

# In Jupyter/IPython
await main()

# From sync code
def sync_wrapper(urls):
    return asyncio.run(main(urls))
```

### Handling Mixed Sync/Async Code

```python
class HybridProcessor:
    """Processor with both sync and async methods"""

    def __init__(self):
        self.db = DatabaseManager()  # Sync

    async def process_async(self, articles: List[dict]) -> List[dict]:
        """Async processing with sync database"""
        results = []

        for article in articles:
            # Async operations
            content = await self.scrape_async(article['url'])
            analysis = await self.analyze_async(content)

            # Sync database operation in thread pool
            await asyncio.to_thread(
                self.db.store_analysis,
                article['id'],
                analysis
            )

            results.append(analysis)

        return results

    async def scrape_async(self, url: str) -> str:
        """Async scraping"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    async def analyze_async(self, content: str) -> dict:
        """Async AI analysis"""
        async with AsyncClaudeClient(self.api_key) as client:
            return await client.analyze_content(content)
```

---

## Error Handling

### Try-Except in Async

```python
async def robust_operation():
    """Async operation with error handling"""
    try:
        async with asyncio.timeout(30):
            result = await fetch_data()
            return result

    except asyncio.TimeoutError:
        logger.error("Operation timed out after 30s")
        raise

    except aiohttp.ClientError as e:
        logger.error(f"Network error: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

    finally:
        # Cleanup always runs
        await cleanup()
```

### Retry Logic

```python
async def retry_async(
    func,
    max_retries: int = 3,
    delay: float = 1.0
):
    """Retry async function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = delay * (2 ** attempt)
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {wait_time}s..."
            )
            await asyncio.sleep(wait_time)
```

---

## Performance Optimization

### Connection Pooling

```python
class AsyncAPIClient:
    """API client with connection pooling"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        # TCPConnector provides connection pooling
        self.connector = aiohttp.TCPConnector(
            limit=100,           # Max total connections
            limit_per_host=30,   # Max per host
            ttl_dns_cache=300    # DNS cache TTL
        )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        await self.connector.close()
```

### Batch Processing

```python
async def process_in_batches(
    items: List[str],
    batch_size: int = 10
) -> List[dict]:
    """Process items in batches"""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_item(item) for item in batch]
        )
        results.extend(batch_results)

        # Optional: delay between batches
        if i + batch_size < len(items):
            await asyncio.sleep(0.1)

    return results
```

---

## Testing Async Code

### Using pytest-asyncio

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await my_async_function()
    assert result == expected_value

@pytest.mark.asyncio
async def test_api_client():
    """Test async API client"""
    async with AsyncClaudeClient(api_key="test") as client:
        result = await client.analyze_content("test content")
        assert "analysis" in result
```

### Mocking Async Functions

```python
from unittest.mock import AsyncMock
import pytest

@pytest.mark.asyncio
async def test_with_mock():
    """Test with mocked async dependency"""
    # Create async mock
    mock_api = AsyncMock()
    mock_api.analyze.return_value = {"result": "mocked"}

    # Use mock
    processor = ArticleProcessor(api_client=mock_api)
    result = await processor.process("content")

    # Verify
    mock_api.analyze.assert_called_once()
    assert result["result"] == "mocked"
```

---

## Complete Example: Async RSS Analyzer

```python
import asyncio
import aiohttp
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Article:
    url: str
    title: str
    content: str = None
    analysis: dict = None

class AsyncRSSAnalyzer:
    """Complete async RSS analyzer"""

    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 5
    ):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_feed(
        self,
        feed_url: str
    ) -> List[Dict]:
        """Analyze all articles in RSS feed"""
        # Fetch and parse feed
        articles = await self._fetch_feed(feed_url)

        # Process articles concurrently
        results = await asyncio.gather(
            *[self._process_article(article) for article in articles],
            return_exceptions=True
        )

        # Separate successes from errors
        return self._process_results(articles, results)

    async def _fetch_feed(self, feed_url: str) -> List[Article]:
        """Fetch and parse RSS feed"""
        async with aiohttp.ClientSession() as session:
            async with session.get(feed_url) as response:
                feed_data = await response.text()
                return self._parse_feed(feed_data)

    async def _process_article(
        self,
        article: Article
    ) -> Article:
        """Process single article"""
        async with self.semaphore:
            # Scrape content
            article.content = await self._scrape(article.url)

            # Analyze with AI
            article.analysis = await self._analyze(article.content)

            return article

    async def _scrape(self, url: str) -> str:
        """Scrape article content"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    async def _analyze(self, content: str) -> dict:
        """Analyze content with AI"""
        async with AsyncClaudeClient(self.api_key) as client:
            return await client.analyze_content(content)

    def _parse_feed(self, feed_data: str) -> List[Article]:
        """Parse RSS feed data"""
        # Implementation here
        pass

    def _process_results(
        self,
        articles: List[Article],
        results: List
    ) -> List[Dict]:
        """Process results and errors"""
        processed = []
        for article, result in zip(articles, results):
            if isinstance(result, Exception):
                processed.append({
                    'url': article.url,
                    'status': 'error',
                    'error': str(result)
                })
            else:
                processed.append({
                    'url': article.url,
                    'status': 'success',
                    'analysis': result.analysis
                })
        return processed

# Usage
async def main():
    analyzer = AsyncRSSAnalyzer(api_key="sk-xxx")
    results = await analyzer.analyze_feed(
        "https://example.com/feed.xml"
    )
    print(f"Processed {len(results)} articles")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Resources

### Documentation
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- [aiohttp documentation](https://docs.aiohttp.org/)
- [Real Python async tutorial](https://realpython.com/async-io-python/)

### Tools
- `pytest-asyncio`: Testing async code
- `aiohttp`: Async HTTP client/server
- `asyncio`: Built-in async framework

### Best Practices
1. Use `async with` for resource management
2. Handle exceptions properly
3. Set timeouts on all I/O operations
4. Use semaphores to limit concurrency
5. Test async code thoroughly
6. Profile and monitor performance
7. Document async behavior clearly

---

## Conclusion

Async programming enables the RSS Analyzer to:
- Process multiple articles simultaneously
- Reduce total processing time by 60-80%
- Handle API calls efficiently
- Scale to larger workloads

Follow the patterns and practices in this guide to build robust async applications.
