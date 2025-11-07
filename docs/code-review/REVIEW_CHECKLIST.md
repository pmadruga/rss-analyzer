# Code Review Checklist - RSS Analyzer

## Comprehensive review checklist for Python async best practices

This checklist is used by both human reviewers and automated review bots to ensure code quality, security, and adherence to async Python best practices.

---

## 1. Async/Await Patterns

### Critical Checks

- [ ] **No Blocking I/O in Async Context**
  - [ ] No `requests.get()` - Use `aiohttp.ClientSession.get()` instead
  - [ ] No `time.sleep()` - Use `asyncio.sleep()` instead
  - [ ] No synchronous database calls - Use async database client
  - [ ] No synchronous file I/O in async functions - Use `aiofiles` if needed

- [ ] **Proper Async Function Definitions**
  - [ ] All I/O operations are defined as `async def`
  - [ ] All async functions are properly `await`ed
  - [ ] No `asyncio.run()` inside async functions (use `await` instead)

- [ ] **Concurrent Execution**
  - [ ] Use `asyncio.gather()` for parallel operations
  - [ ] Use `asyncio.create_task()` for background tasks
  - [ ] Proper error handling in concurrent operations
  - [ ] Task cancellation is handled correctly

### Example Patterns

```python
# ❌ BAD: Blocking I/O in async context
async def fetch_data():
    response = requests.get(url)  # Blocks entire event loop!
    return response.text

# ✅ GOOD: Non-blocking async I/O
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# ❌ BAD: Sequential operations
async def process_articles(articles):
    results = []
    for article in articles:
        result = await analyze_article(article)
        results.append(result)
    return results

# ✅ GOOD: Concurrent operations
async def process_articles(articles):
    tasks = [analyze_article(article) for article in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

---

## 2. Database Connection Pooling

### Critical Checks

- [ ] **Connection Pool Usage**
  - [ ] Database operations use connection pool (not direct connections)
  - [ ] Pool size configured appropriately (5-10 for typical workloads)
  - [ ] Connection validation before use
  - [ ] Proper connection release after use

- [ ] **Pool Configuration**
  - [ ] `max_connections` set based on workload
  - [ ] `min_connections` >= 2 for connection reuse
  - [ ] `timeout` configured to prevent hanging
  - [ ] Pool health checks enabled

- [ ] **Async Pool Operations**
  - [ ] Use async-aware connection pool
  - [ ] Context managers for automatic cleanup
  - [ ] No blocking pool operations

### Example Patterns

```python
# ❌ BAD: Direct connection creation
async def save_article(article_data):
    conn = sqlite3.connect("articles.db")  # Creates new connection every time
    cursor = conn.cursor()
    cursor.execute("INSERT INTO articles VALUES (?)", (article_data,))
    conn.commit()
    conn.close()

# ✅ GOOD: Connection pool with context manager
async def save_article(article_data):
    async with db_manager.get_connection() as conn:
        await conn.execute("INSERT INTO articles VALUES (?)", (article_data,))
        await conn.commit()
    # Connection automatically returned to pool
```

---

## 3. Cache Utilization

### Critical Checks

- [ ] **Two-Tier Cache Strategy**
  - [ ] L1 (Memory) cache for hot data
  - [ ] L2 (Disk) cache for persistent storage
  - [ ] Proper TTL configuration for each cache tier
  - [ ] Cache key generation is deterministic

- [ ] **Cache Usage Patterns**
  - [ ] Check cache before expensive operations
  - [ ] Store results in cache after computation
  - [ ] Cache invalidation strategy implemented
  - [ ] No cache stampede issues (use locking)

- [ ] **Performance Optimization**
  - [ ] API responses cached (TTL: 30 days)
  - [ ] Web scraping results cached (TTL: 7 days)
  - [ ] RSS feed parsed data cached (TTL: 1 hour)
  - [ ] Cache hit rate monitoring enabled

### Example Patterns

```python
# ❌ BAD: No caching for expensive operations
async def analyze_article(url: str) -> dict:
    content = await scrape_url(url)
    analysis = await ai_client.analyze(content)
    return analysis

# ✅ GOOD: Two-tier caching
async def analyze_article(url: str) -> dict:
    cache_key = f"analysis:{hashlib.md5(url.encode()).hexdigest()}"

    # Check L1 (memory) cache
    if result := cache.get_l1(cache_key):
        return result

    # Check L2 (disk) cache
    if result := await cache.get_l2(cache_key):
        cache.set_l1(cache_key, result)  # Promote to L1
        return result

    # Cache miss - perform operation
    content = await scrape_url(url)
    analysis = await ai_client.analyze(content)

    # Store in both cache tiers
    cache.set_l1(cache_key, analysis)
    await cache.set_l2(cache_key, analysis, ttl=30*24*3600)

    return analysis
```

---

## 4. Error Handling and Logging

### Critical Checks

- [ ] **Exception Handling**
  - [ ] All async operations wrapped in try/except
  - [ ] Specific exception types caught (not bare `except:`)
  - [ ] Proper error propagation to caller
  - [ ] Resources cleaned up in finally blocks

- [ ] **Logging Standards**
  - [ ] Structured logging with context
  - [ ] Log levels used appropriately (DEBUG, INFO, WARNING, ERROR)
  - [ ] Sensitive data not logged (API keys, passwords)
  - [ ] Performance-critical paths use appropriate log level

- [ ] **Retry Logic**
  - [ ] Exponential backoff for transient failures
  - [ ] Maximum retry attempts configured
  - [ ] Idempotent operations for safe retries
  - [ ] Circuit breaker pattern for cascading failures

### Example Patterns

```python
# ❌ BAD: Bare except hides errors
async def fetch_article(url: str):
    try:
        return await scraper.scrape(url)
    except:
        return None  # Silent failure!

# ✅ GOOD: Specific exceptions with logging
async def fetch_article(url: str) -> Optional[dict]:
    for attempt in range(3):
        try:
            result = await scraper.scrape(url)
            logger.info(f"Successfully scraped {url}", extra={"attempt": attempt + 1})
            return result

        except aiohttp.ClientError as e:
            logger.warning(
                f"Network error scraping {url}: {e}",
                extra={"attempt": attempt + 1, "error_type": type(e).__name__}
            )
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            logger.error(
                f"Unexpected error scraping {url}: {e}",
                exc_info=True,
                extra={"attempt": attempt + 1}
            )
            break

    return None
```

---

## 5. Type Hints and Documentation

### Critical Checks

- [ ] **Type Annotations**
  - [ ] All function parameters have type hints
  - [ ] All function return types annotated
  - [ ] Generic types properly specified (`List[str]`, `Dict[str, Any]`)
  - [ ] Optional types used for nullable values

- [ ] **Documentation**
  - [ ] Public functions have docstrings
  - [ ] Docstrings include Args, Returns, Raises sections
  - [ ] Complex algorithms have explanatory comments
  - [ ] Module-level docstrings for file purpose

- [ ] **Type Checking**
  - [ ] Code passes mypy static type checking
  - [ ] No `# type: ignore` comments without explanation
  - [ ] Proper use of Protocol and TypedDict for complex types

### Example Patterns

```python
# ❌ BAD: No type hints or documentation
def process_articles(articles):
    results = []
    for article in articles:
        result = analyze(article)
        results.append(result)
    return results

# ✅ GOOD: Complete type hints and documentation
from typing import List, Dict, Optional, Any

async def process_articles(
    articles: List[Dict[str, Any]],
    max_concurrent: int = 5
) -> List[Optional[Dict[str, Any]]]:
    """
    Process multiple articles concurrently using AI analysis.

    Args:
        articles: List of article dictionaries with 'url' and 'title' keys
        max_concurrent: Maximum number of concurrent processing tasks

    Returns:
        List of analysis results, with None for failed analyses

    Raises:
        ValueError: If articles list is empty
        RuntimeError: If all articles fail to process
    """
    if not articles:
        raise ValueError("Articles list cannot be empty")

    # Create tasks with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with semaphore:
            return await analyze_article(article)

    tasks = [process_with_limit(article) for article in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to None
    processed_results = [
        result if not isinstance(result, Exception) else None
        for result in results
    ]

    if all(result is None for result in processed_results):
        raise RuntimeError("All articles failed to process")

    return processed_results
```

---

## 6. Test Coverage

### Critical Checks

- [ ] **Unit Tests**
  - [ ] All new functions have unit tests
  - [ ] Async functions tested with pytest-asyncio
  - [ ] Edge cases covered (empty input, errors, timeouts)
  - [ ] Mock external dependencies (API calls, database)

- [ ] **Integration Tests**
  - [ ] End-to-end workflows tested
  - [ ] Database integration tested
  - [ ] Cache integration tested
  - [ ] Error scenarios tested

- [ ] **Coverage Metrics**
  - [ ] Minimum 80% code coverage for new code
  - [ ] Critical paths have 100% coverage
  - [ ] No untested error handling branches

### Example Test Patterns

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_process_articles_success():
    """Test successful concurrent article processing."""
    articles = [
        {"url": "https://example.com/1", "title": "Article 1"},
        {"url": "https://example.com/2", "title": "Article 2"}
    ]

    mock_analyze = AsyncMock(return_value={"summary": "test"})

    with patch('src.ai_client.analyze_article', mock_analyze):
        results = await process_articles(articles)

    assert len(results) == 2
    assert all(r is not None for r in results)
    assert mock_analyze.call_count == 2

@pytest.mark.asyncio
async def test_process_articles_handles_failures():
    """Test graceful handling of individual article failures."""
    articles = [
        {"url": "https://example.com/1", "title": "Article 1"},
        {"url": "https://example.com/2", "title": "Article 2"}
    ]

    # First article succeeds, second fails
    mock_analyze = AsyncMock(side_effect=[
        {"summary": "test"},
        aiohttp.ClientError("Network error")
    ])

    with patch('src.ai_client.analyze_article', mock_analyze):
        results = await process_articles(articles)

    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is None  # Failed article returns None

@pytest.mark.asyncio
async def test_process_articles_empty_input():
    """Test error handling for empty article list."""
    with pytest.raises(ValueError, match="Articles list cannot be empty"):
        await process_articles([])
```

---

## 7. Security Considerations

### Critical Checks

- [ ] **API Key Management**
  - [ ] No hardcoded API keys in source code
  - [ ] All secrets loaded from environment variables
  - [ ] API keys not logged or displayed
  - [ ] Proper key rotation strategy

- [ ] **SQL Injection Prevention**
  - [ ] All database queries use parameterized statements
  - [ ] No string formatting or concatenation in SQL
  - [ ] Input validation before database operations
  - [ ] ORM or prepared statements used exclusively

- [ ] **Input Validation**
  - [ ] All user input sanitized
  - [ ] URL validation for web scraping
  - [ ] File path validation to prevent directory traversal
  - [ ] Rate limiting on external API calls

- [ ] **Dependency Security**
  - [ ] All dependencies up to date
  - [ ] No known vulnerabilities in dependencies
  - [ ] Security scanning in CI/CD pipeline

### Example Patterns

```python
# ❌ BAD: Hardcoded API key and SQL injection risk
API_KEY = "sk-1234567890abcdef"  # Never do this!

def save_article(title: str):
    query = f"INSERT INTO articles (title) VALUES ('{title}')"  # SQL injection!
    conn.execute(query)

# ✅ GOOD: Environment variables and parameterized queries
import os
from typing import Optional

API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

async def save_article(title: str) -> int:
    """
    Save article to database with proper sanitization.

    Args:
        title: Article title (will be sanitized)

    Returns:
        ID of inserted article

    Raises:
        ValueError: If title is empty or invalid
    """
    if not title or not title.strip():
        raise ValueError("Article title cannot be empty")

    # Parameterized query prevents SQL injection
    async with db_manager.get_connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO articles (title) VALUES (?)",
            (title.strip(),)
        )
        await conn.commit()
        return cursor.lastrowid
```

---

## 8. Performance Considerations

### Critical Checks

- [ ] **Concurrency Limits**
  - [ ] Semaphores used to limit concurrent operations
  - [ ] Rate limiting for external API calls
  - [ ] Connection pool size appropriate for workload
  - [ ] No unbounded task creation

- [ ] **Memory Management**
  - [ ] Large data structures processed in chunks
  - [ ] Streaming used for large file operations
  - [ ] Cache size limits enforced
  - [ ] Proper cleanup of resources

- [ ] **Monitoring**
  - [ ] Performance metrics collected
  - [ ] Slow operations logged
  - [ ] Resource usage monitored
  - [ ] Alerts for performance degradation

### Example Patterns

```python
# ❌ BAD: Unbounded concurrency can exhaust resources
async def process_many_articles(articles: List[dict]) -> List[dict]:
    tasks = [analyze_article(article) for article in articles]
    return await asyncio.gather(*tasks)  # Could create 1000+ concurrent tasks!

# ✅ GOOD: Bounded concurrency with semaphore
async def process_many_articles(
    articles: List[dict],
    max_concurrent: int = 8
) -> List[dict]:
    """
    Process articles with controlled concurrency.

    Args:
        articles: List of articles to process
        max_concurrent: Maximum concurrent processing tasks (default: 8)

    Returns:
        List of processed article results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(article: dict) -> dict:
        async with semaphore:
            start_time = time.time()
            try:
                result = await analyze_article(article)
                duration = time.time() - start_time

                # Log slow operations
                if duration > 5.0:
                    logger.warning(
                        f"Slow article processing: {duration:.2f}s",
                        extra={"url": article.get("url"), "duration": duration}
                    )

                return result

            except Exception as e:
                logger.error(f"Failed to process article: {e}", exc_info=True)
                raise

    tasks = [process_with_limit(article) for article in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    return [r for r in results if not isinstance(r, Exception)]
```

---

## Quick Reference: Common Async Mistakes

| Mistake | Impact | Solution |
|---------|--------|----------|
| `requests.get()` in async function | Blocks event loop | Use `aiohttp.ClientSession` |
| `time.sleep()` in async function | Blocks event loop | Use `asyncio.sleep()` |
| Synchronous DB calls | Blocks event loop | Use async database client |
| No connection pooling | Slow performance | Use DatabaseManager pool |
| Missing cache checks | High API costs | Check L1/L2 cache first |
| Bare `except:` clauses | Hides errors | Catch specific exceptions |
| No type hints | Type errors at runtime | Add type annotations |
| Missing tests | Bugs in production | Write unit/integration tests |
| Hardcoded secrets | Security vulnerability | Use environment variables |
| Unbounded concurrency | Resource exhaustion | Use semaphores/limits |

---

## Automated Review Bot Integration

This checklist is used by automated review bots in the following ways:

1. **Static Analysis**: Ruff, pylint, mypy check for code quality issues
2. **Security Scanning**: Bandit scans for security vulnerabilities
3. **Pattern Matching**: GitHub Actions checks for common anti-patterns
4. **Coverage Enforcement**: pytest-cov ensures minimum 80% coverage
5. **Comment Generation**: Bot posts review comments with specific line references

See [AUTOMATED_CHECKS.md](./AUTOMATED_CHECKS.md) for detailed configuration.

---

## Review Process Workflow

```
1. PR Created
   ↓
2. Automated Triage (assigns agents based on changed files)
   ↓
3. Parallel Agent Reviews:
   - Security Agent (checks for vulnerabilities)
   - Code Quality Agent (runs static analysis)
   - Python Practices Agent (checks best practices)
   - Async Patterns Agent (validates async/await usage)
   ↓
4. Review Summary Generated
   ↓
5. Quality Gates Checked
   ↓
6. Approve or Request Changes
```

See [QUALITY_GATES.md](./QUALITY_GATES.md) for merge criteria.

---

## Resources

- [Python Async Best Practices](https://docs.python.org/3/library/asyncio-task.html)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [OWASP Python Security](https://owasp.org/www-project-python-security/)
- [RSS Analyzer Optimization Guide](../OPTIMIZATION_RESULTS.md)
