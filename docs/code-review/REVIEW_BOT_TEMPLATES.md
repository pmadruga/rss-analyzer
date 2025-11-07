# Review Bot Comment Templates

## Automated review comment templates for common code issues

This document contains templates used by the automated review bot to post helpful, constructive comments on pull requests.

---

## Template Categories

1. Async/Await Pattern Issues
2. Database Connection Pool Issues
3. Cache Usage Issues
4. Security Issues
5. Type Hint Issues
6. Documentation Issues
7. Performance Issues
8. Test Coverage Issues

---

## 1. Async/Await Pattern Issues

### Blocking I/O in Async Function

```markdown
## ⚠️ Blocking I/O Detected in Async Function

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Synchronous blocking call `{blocking_call}` found in async function `{function_name}`

```python
# Current code (blocking):
async def {function_name}():
    response = {blocking_call}(url)  # ❌ Blocks entire event loop!
    return response
```

**Impact**:
- Blocks the entire async event loop
- Prevents other async operations from running
- Degrades concurrency performance by 10-100x
- Makes async operations run slower than sync code

**Suggested Fix**:
```python
# Use async equivalent:
async def {function_name}():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()  # ✅ Non-blocking
```

**Why This Matters**:
The RSS Analyzer uses async/await for concurrent processing (6-8 articles simultaneously). Blocking I/O defeats this purpose and can make the entire system slower than the synchronous version.

**Resources**:
- [Python Asyncio Best Practices](https://docs.python.org/3/library/asyncio-task.html)
- [Project Async Guide](../ASYNC_GUIDE.md)
- [aiohttp Documentation](https://docs.aiohttp.org/)

---
*Posted by Automated Review Bot - [Async Pattern Check]*
```

### Missing Await on Async Function

```markdown
## ⚠️ Missing `await` on Async Function Call

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Async function `{function_name}` called without `await`

```python
# Current code:
result = {async_function}(args)  # ❌ Returns coroutine, not result!
```

**Impact**:
- Function doesn't actually execute
- Returns a coroutine object instead of the result
- Silent failure (may cause RuntimeWarning)
- Downstream code receives wrong type

**Suggested Fix**:
```python
# Add await:
result = await {async_function}(args)  # ✅ Executes and returns result
```

**How to Verify**:
```bash
# Run type checker:
mypy src/ --strict

# Check for warnings:
python -Werror::RuntimeWarning -m pytest
```

---
*Posted by Automated Review Bot - [Async Pattern Check]*
```

### Using time.sleep() Instead of asyncio.sleep()

```markdown
## ⚠️ Blocking Sleep in Async Function

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: `time.sleep({duration})` used in async function

```python
# Current code (blocking):
async def retry_operation():
    for attempt in range(3):
        try:
            return await operation()
        except Exception:
            time.sleep(2 ** attempt)  # ❌ Blocks event loop!
```

**Impact**:
- Blocks all async operations for {duration} seconds
- Prevents concurrent article processing
- Degrades throughput to sequential performance

**Suggested Fix**:
```python
# Use async sleep:
async def retry_operation():
    for attempt in range(3):
        try:
            return await operation()
        except Exception:
            await asyncio.sleep(2 ** attempt)  # ✅ Non-blocking
```

**Performance Difference**:
- `time.sleep(1)`: Blocks all 8 concurrent operations for 1 second = 8 seconds lost
- `asyncio.sleep(1)`: Only suspends this operation, others continue = 1 second lost

---
*Posted by Automated Review Bot - [Async Pattern Check]*
```

---

## 2. Database Connection Pool Issues

### Direct Connection Creation

```markdown
## ⚠️ Direct Database Connection Detected

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Direct `{db_library}.connect()` call bypasses connection pool

```python
# Current code (inefficient):
def save_article(article):
    conn = sqlite3.connect("articles.db")  # ❌ New connection every time
    cursor = conn.cursor()
    cursor.execute("INSERT INTO articles VALUES (?)", (article,))
    conn.commit()
    conn.close()
```

**Impact**:
- Creates new connection for every operation (expensive)
- Connection overhead: 5-20ms per operation
- No connection reuse
- Potential connection leak if exception occurs

**Suggested Fix**:
```python
# Use connection pool:
async def save_article(article):
    async with db_manager.get_connection() as conn:  # ✅ Reuses pooled connection
        await conn.execute("INSERT INTO articles VALUES (?)", (article,))
        await conn.commit()
    # Connection automatically returned to pool
```

**Performance Benefit**:
With connection pooling, the RSS Analyzer achieved **2.78x faster** database operations.

**Resources**:
- [Connection Pooling Guide](../CONNECTION_POOLING.md)
- [Pool Statistics](../OPTIMIZATION_RESULTS.md#phase-1-connection-pooling)

---
*Posted by Automated Review Bot - [Connection Pool Check]*
```

### Missing Context Manager

```markdown
## ⚠️ Connection Not Using Context Manager

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Database connection not released with context manager

```python
# Current code (potential leak):
async def query_articles():
    conn = db_manager.get_connection()  # ❌ Manual management
    cursor = await conn.execute("SELECT * FROM articles")
    results = await cursor.fetchall()
    # What if exception occurs here? Connection never released!
    await conn.close()
    return results
```

**Impact**:
- Connection may not be returned to pool if exception occurs
- Pool exhaustion after errors
- Resource leaks
- Database lock issues

**Suggested Fix**:
```python
# Use context manager:
async def query_articles():
    async with db_manager.get_connection() as conn:  # ✅ Automatic cleanup
        cursor = await conn.execute("SELECT * FROM articles")
        results = await cursor.fetchall()
        return results
    # Connection automatically returned even if exception occurs
```

**Pool Statistics**:
Check current pool health:
```bash
docker compose run rss-analyzer python -c "
from src.core.database import DatabaseManager
print(DatabaseManager().get_pool_stats())
"
```

---
*Posted by Automated Review Bot - [Connection Pool Check]*
```

---

## 3. Cache Usage Issues

### Missing Cache Check

```markdown
## 💡 Consider Adding Cache Check

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Expensive operation `{operation_name}` without cache check

```python
# Current code (expensive):
async def analyze_article(url: str):
    content = await scrape_url(url)  # ❌ Always scrapes
    analysis = await ai_client.analyze(content)  # ❌ Always calls API
    return analysis
```

**Impact**:
- Unnecessary API calls cost $0.10 per analysis
- Repeated scraping of same content
- Slower response times (2-5s per uncached call)
- Higher resource usage

**Suggested Fix**:
```python
# Add two-tier cache:
async def analyze_article(url: str):
    cache_key = f"analysis:{hashlib.md5(url.encode()).hexdigest()}"

    # Check L1 (memory) cache - microseconds
    if result := cache.get_l1(cache_key):
        return result

    # Check L2 (disk) cache - milliseconds
    if result := await cache.get_l2(cache_key):
        cache.set_l1(cache_key, result)  # Promote to L1
        return result

    # Cache miss - perform operation
    content = await scrape_url(url)
    analysis = await ai_client.analyze(content)

    # Store in both tiers
    cache.set_l1(cache_key, analysis)
    await cache.set_l2(cache_key, analysis, ttl=30*24*3600)  # 30 days

    return analysis
```

**Cost Savings**:
The RSS Analyzer achieved **72% cache hit rate**, saving:
- $107.80/month on API costs
- 72% reduction in processing time
- 90% reduction in network calls

**Resources**:
- [Cache Usage Guide](../CACHE_USAGE.md)
- [Two-Tier Cache Architecture](../OPTIMIZATION_RESULTS.md#phase-2-two-tier-caching)

---
*Posted by Automated Review Bot - [Cache Optimization]*
```

### Incorrect TTL Configuration

```markdown
## ⚠️ Cache TTL May Be Inappropriate

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Cache TTL of `{current_ttl}` seconds seems too {short|long} for `{data_type}`

**Current Configuration**:
```python
cache.set(key, value, ttl={current_ttl})
```

**Recommended TTLs**:
- **API responses**: 30 days (2,592,000s) - rarely change
- **Scraped content**: 7 days (604,800s) - updated periodically
- **RSS feed data**: 1 hour (3,600s) - frequently updated
- **Metadata**: 24 hours (86,400s) - stable but may change

**Suggested Fix**:
```python
# Use appropriate TTL for data type:
cache.set(key, value, ttl={recommended_ttl})  # {recommended_ttl_days}
```

**Impact of Current TTL**:
- Too short: Unnecessary cache misses, higher costs
- Too long: Stale data, missed updates

---
*Posted by Automated Review Bot - [Cache Optimization]*
```

---

## 4. Security Issues

### Hardcoded Secret

```markdown
## 🔴 CRITICAL: Hardcoded Secret Detected

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: API key or password hardcoded in source code

```python
# Current code (INSECURE):
API_KEY = "sk-1234567890abcdef"  # ❌ NEVER commit secrets!
```

**Security Risk**:
- **Severity**: CRITICAL
- Secret exposed in version control history
- Accessible to anyone with repository access
- Cannot be rotated without code change
- May violate API provider terms of service

**Immediate Action Required**:
1. Revoke the exposed API key immediately
2. Generate new API key
3. Remove from code and commit history

**Suggested Fix**:
```python
# Use environment variables:
import os

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
```

**Configuration**:
```bash
# .env file (DO NOT commit):
ANTHROPIC_API_KEY=sk-1234567890abcdef

# Docker Compose:
environment:
  - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

**Cleanup Git History**:
```bash
# Remove from git history:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch {file_path}" \
  --prune-empty --tag-name-filter cat -- --all
```

**Resources**:
- [OWASP Secrets Management](https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)

---
*Posted by Automated Review Bot - [Security Scan] - 🚨 REQUIRES IMMEDIATE ACTION*
```

### SQL Injection Risk

```markdown
## 🔴 HIGH: SQL Injection Vulnerability

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: SQL query constructed with string formatting

```python
# Current code (VULNERABLE):
query = f"SELECT * FROM articles WHERE title = '{title}'"  # ❌ SQL injection!
cursor.execute(query)
```

**Security Risk**:
- **Severity**: HIGH
- Attacker can execute arbitrary SQL commands
- Data theft, modification, or deletion
- Database compromise

**Attack Example**:
```python
# Malicious input:
title = "'; DROP TABLE articles; --"

# Resulting query:
"SELECT * FROM articles WHERE title = ''; DROP TABLE articles; --'"
# This deletes the entire articles table!
```

**Suggested Fix**:
```python
# Use parameterized queries:
query = "SELECT * FROM articles WHERE title = ?"  # ✅ Safe
cursor.execute(query, (title,))
```

**Why Parameterized Queries Are Safe**:
- Database treats parameter as literal data, not SQL code
- Automatic escaping of special characters
- No way to inject SQL commands

**Resources**:
- [OWASP SQL Injection](https://owasp.org/www-community/attacks/SQL_Injection)
- [Python DB-API](https://peps.python.org/pep-0249/)

---
*Posted by Automated Review Bot - [Security Scan] - 🚨 BLOCKS MERGE*
```

### Missing Input Validation

```markdown
## 🟡 MEDIUM: Missing Input Validation

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: User input `{param_name}` not validated before use

```python
# Current code (risky):
async def scrape_url(url: str):
    async with session.get(url) as response:  # ❌ No validation
        return await response.text()
```

**Security Risk**:
- **Severity**: MEDIUM
- May access internal resources (SSRF)
- May cause application errors
- May be exploited for DoS

**Attack Examples**:
```python
# Local file access:
url = "file:///etc/passwd"

# Internal network:
url = "http://localhost:6379/redis-command"

# Invalid URL:
url = "not-a-url"
```

**Suggested Fix**:
```python
# Add input validation:
from urllib.parse import urlparse

async def scrape_url(url: str):
    # Validate URL
    parsed = urlparse(url)

    # Check scheme
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

    # Check not localhost/internal
    if parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        raise ValueError(f"Internal URLs not allowed: {parsed.hostname}")

    # Check for file:// etc
    if parsed.scheme == "file":
        raise ValueError("File URLs not allowed")

    # Proceed with validated URL
    async with session.get(url) as response:
        return await response.text()
```

**Resources**:
- [OWASP Input Validation](https://owasp.org/www-project-proactive-controls/v3/en/c5-validate-inputs)

---
*Posted by Automated Review Bot - [Security Scan]*
```

---

## 5. Type Hint Issues

### Missing Type Hints

```markdown
## 💡 Missing Type Hints

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Function `{function_name}` missing type annotations

```python
# Current code (no types):
async def process_articles(articles, max_concurrent=5):  # ❌ No type hints
    results = []
    for article in articles:
        result = await analyze_article(article)
        results.append(result)
    return results
```

**Impact**:
- Type errors only discovered at runtime
- No IDE autocomplete
- Harder to understand code
- Mypy cannot verify correctness

**Suggested Fix**:
```python
# Add complete type hints:
from typing import List, Dict, Optional, Any

async def process_articles(
    articles: List[Dict[str, Any]],
    max_concurrent: int = 5
) -> List[Optional[Dict[str, Any]]]:  # ✅ Clear types
    """
    Process articles concurrently.

    Args:
        articles: List of article dictionaries
        max_concurrent: Max concurrent processing (default: 5)

    Returns:
        List of analysis results (None for failures)
    """
    results: List[Optional[Dict[str, Any]]] = []
    for article in articles:
        result = await analyze_article(article)
        results.append(result)
    return results
```

**Benefits**:
- Catch type errors before runtime
- Better IDE support
- Self-documenting code
- Easier refactoring

**Type Checking**:
```bash
# Verify types:
mypy src/ --strict
```

---
*Posted by Automated Review Bot - [Type Check]*
```

### Incorrect Return Type

```markdown
## ⚠️ Incorrect Return Type Annotation

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Function returns `{actual_type}` but annotated as `{declared_type}`

```python
# Current code (type mismatch):
async def get_article(article_id: int) -> Dict[str, Any]:  # ❌ Wrong return type
    article = await db.query("SELECT * FROM articles WHERE id = ?", (article_id,))
    if not article:
        return None  # Returns None, not Dict!
    return article
```

**Impact**:
- Caller expects Dict, receives None
- Causes AttributeError at runtime
- Mypy reports error

**Suggested Fix**:
```python
# Use Optional for nullable returns:
async def get_article(article_id: int) -> Optional[Dict[str, Any]]:  # ✅ Correct
    article = await db.query("SELECT * FROM articles WHERE id = ?", (article_id,))
    if not article:
        return None  # Now matches type annotation
    return article
```

**Usage**:
```python
# Caller must handle None:
article = await get_article(123)
if article is not None:  # Type narrowing
    print(article["title"])  # Safe access
else:
    print("Article not found")
```

---
*Posted by Automated Review Bot - [Type Check]*
```

---

## 6. Documentation Issues

### Missing Docstring

```markdown
## 💡 Missing Docstring

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Public function `{function_name}` has no docstring

```python
# Current code (undocumented):
async def process_articles(articles: List[Dict], max_concurrent: int = 5):
    # Function body...
```

**Impact**:
- Unclear purpose and usage
- No parameter documentation
- Missing return value description
- Harder for other developers to use

**Suggested Fix**:
```python
# Add comprehensive docstring:
async def process_articles(
    articles: List[Dict[str, Any]],
    max_concurrent: int = 5
) -> List[Optional[Dict[str, Any]]]:
    """
    Process multiple articles concurrently using async patterns.

    This function implements bounded concurrency with semaphores to prevent
    resource exhaustion while maximizing throughput. Articles are analyzed
    using AI and results are cached for future use.

    Args:
        articles: List of article dictionaries with 'url' and 'title' keys.
                 Each article must have at minimum a valid URL.
        max_concurrent: Maximum number of concurrent processing tasks. Higher
                       values increase throughput but use more resources.
                       Default is 5. Recommended range: 3-10.

    Returns:
        List of analysis result dictionaries, one per article. Returns None
        for articles that failed to process. Order matches input order.

    Raises:
        ValueError: If articles list is empty or contains invalid data
        RuntimeError: If all articles fail to process
        aiohttp.ClientError: If network errors occur during scraping

    Example:
        >>> articles = [
        ...     {"url": "https://example.com/article1", "title": "Article 1"},
        ...     {"url": "https://example.com/article2", "title": "Article 2"}
        ... ]
        >>> results = await process_articles(articles, max_concurrent=3)
        >>> successful = [r for r in results if r is not None]
        >>> print(f"Processed {len(successful)}/{len(articles)} articles")
        Processed 2/2 articles
    """
    # Function body...
```

**Docstring Sections**:
- **Brief description**: One-line summary
- **Detailed description**: Extended explanation
- **Args**: Parameter descriptions with types and constraints
- **Returns**: Return value description
- **Raises**: Possible exceptions
- **Example**: Usage example with expected output

---
*Posted by Automated Review Bot - [Documentation Check]*
```

---

## 7. Performance Issues

### Unbounded Concurrency

```markdown
## ⚠️ Unbounded Concurrency Detected

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: Creating tasks without concurrency limit

```python
# Current code (resource exhaustion risk):
async def process_many_articles(articles: List[Dict]):
    tasks = [analyze_article(article) for article in articles]
    return await asyncio.gather(*tasks)  # ❌ Could create 1000+ tasks!
```

**Impact**:
- May exhaust system resources (memory, connections)
- May overwhelm external APIs (rate limiting)
- May cause connection pool exhaustion
- Unpredictable performance

**Problem Scenario**:
```python
# With 1000 articles:
articles = load_articles(limit=1000)
results = await process_many_articles(articles)
# Creates 1000 concurrent:
# - HTTP connections
# - Database connections
# - Memory buffers
# - API requests
# Result: System crash or API ban
```

**Suggested Fix**:
```python
# Use semaphore for bounded concurrency:
async def process_many_articles(
    articles: List[Dict],
    max_concurrent: int = 8  # Configurable limit
) -> List[Dict]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(article: Dict) -> Dict:
        async with semaphore:  # Only max_concurrent execute simultaneously
            return await analyze_article(article)

    tasks = [process_with_limit(article) for article in articles]
    return await asyncio.gather(*tasks)
```

**Performance Tuning**:
```python
# Optimal concurrency levels:
MAX_CONCURRENT_ARTICLES = 6-8  # For CPU-bound work
MAX_CONCURRENT_API_CALLS = 10  # For API rate limits
MAX_CONCURRENT_DB_OPS = pool_size  # Match pool size
```

**Configuration**:
```bash
# Environment variable:
export MAX_CONCURRENT_ARTICLES=8
```

---
*Posted by Automated Review Bot - [Performance Check]*
```

### Missing Rate Limiting

```markdown
## ⚠️ Missing Rate Limiting for External API

**File**: `{file_path}`
**Line**: {line_number}

**Issue**: External API calls without rate limiting

```python
# Current code (may exceed rate limits):
async def call_api_multiple_times(items: List[str]):
    tasks = [api_client.call(item) for item in items]
    return await asyncio.gather(*tasks)  # ❌ No rate limiting
```

**Impact**:
- May exceed API provider rate limits
- Risk of account suspension or API ban
- Failed requests and wasted retries
- Unpredictable costs

**API Rate Limits**:
- Anthropic Claude: 50 req/min, 40,000 tokens/min
- OpenAI GPT-4: 500 req/min, 150,000 tokens/min
- Most web APIs: 100-1000 req/min

**Suggested Fix**:
```python
# Add rate limiter:
from aiolimiter import AsyncLimiter

# Create rate limiter (50 requests per minute)
rate_limiter = AsyncLimiter(max_rate=50, time_period=60)

async def call_api_multiple_times(items: List[str]):
    results = []
    for item in items:
        async with rate_limiter:  # ✅ Respects rate limit
            result = await api_client.call(item)
            results.append(result)
    return results
```

**Advanced Pattern (Concurrent + Rate Limited)**:
```python
# Combine concurrency with rate limiting:
async def call_api_concurrent_limited(items: List[str]):
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent
    rate_limiter = AsyncLimiter(50, 60)  # Max 50/minute

    async def call_with_limits(item: str):
        async with semaphore:  # Limit concurrency
            async with rate_limiter:  # Limit rate
                return await api_client.call(item)

    tasks = [call_with_limits(item) for item in items]
    return await asyncio.gather(*tasks)
```

**Resources**:
- [aiolimiter Documentation](https://aiolimiter.readthedocs.io/)

---
*Posted by Automated Review Bot - [Performance Check]*
```

---

## 8. Test Coverage Issues

### Missing Tests

```markdown
## ⚠️ Missing Test Coverage

**File**: `{file_path}`
**Function**: `{function_name}`

**Issue**: New function has no unit tests

**Coverage**: {coverage}% (target: 80%+)

**Missing Test Scenarios**:
- [ ] Success case with valid input
- [ ] Error handling for invalid input
- [ ] Edge case: empty input
- [ ] Edge case: None input
- [ ] Async behavior verification
- [ ] Exception handling

**Suggested Tests**:

```python
# tests/test_{module_name}.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_{function_name}_success():
    """Test {function_name} with valid input."""
    # Arrange
    input_data = {...}
    expected_output = {...}

    # Act
    result = await {function_name}(input_data)

    # Assert
    assert result == expected_output

@pytest.mark.asyncio
async def test_{function_name}_handles_error():
    """Test {function_name} error handling."""
    # Arrange
    invalid_input = None

    # Act & Assert
    with pytest.raises(ValueError, match="Input cannot be None"):
        await {function_name}(invalid_input)

@pytest.mark.asyncio
async def test_{function_name}_with_mock():
    """Test {function_name} with mocked dependencies."""
    # Arrange
    mock_dependency = AsyncMock(return_value={"status": "success"})

    # Act
    with patch('module.dependency', mock_dependency):
        result = await {function_name}(input_data)

    # Assert
    assert result["status"] == "success"
    mock_dependency.assert_called_once()
```

**Run Tests**:
```bash
# Run with coverage:
pytest tests/test_{module_name}.py --cov=src.{module_name} --cov-report=term

# Check coverage:
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**Coverage Report**:
- Current: {current_coverage}%
- Target: 80%
- Gap: {gap}%

---
*Posted by Automated Review Bot - [Coverage Check] - ⚠️ BLOCKS MERGE*
```

### Insufficient Error Testing

```markdown
## 💡 Add Error Scenario Tests

**File**: `{file_path}`
**Function**: `{function_name}`

**Issue**: Tests only cover success cases, missing error scenarios

**Current Tests**:
- ✅ Success case with valid input
- ❌ Missing: Invalid input handling
- ❌ Missing: Network error handling
- ❌ Missing: Timeout scenarios
- ❌ Missing: Database errors

**Why Error Tests Matter**:
Error handling is where most production bugs occur. Testing only the happy path leaves critical code paths untested.

**Suggested Error Tests**:

```python
@pytest.mark.asyncio
async def test_{function_name}_handles_network_error():
    """Test resilience to network failures."""
    mock_client = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))

    with patch('module.client', mock_client):
        result = await {function_name}(url)

    # Should handle gracefully, not crash
    assert result is None or result["error"] is not None

@pytest.mark.asyncio
async def test_{function_name}_handles_timeout():
    """Test timeout handling."""
    mock_client = AsyncMock(side_effect=asyncio.TimeoutError())

    with patch('module.client', mock_client):
        with pytest.raises(asyncio.TimeoutError):
            await {function_name}(url)

@pytest.mark.asyncio
async def test_{function_name}_validates_input():
    """Test input validation."""
    invalid_inputs = [None, "", {}, [], "invalid-url"]

    for invalid_input in invalid_inputs:
        with pytest.raises((ValueError, TypeError)):
            await {function_name}(invalid_input)
```

**Test Error Scenarios**:
1. Network failures (connection errors, timeouts)
2. Invalid input (None, empty, wrong type)
3. Resource exhaustion (out of memory, too many connections)
4. External service failures (API down, database unavailable)
5. Rate limiting (429 responses)

---
*Posted by Automated Review Bot - [Coverage Check]*
```

---

## Bot Comment Formatting Guidelines

### Structure

All bot comments should follow this structure:

```markdown
## {Icon} {Severity}: {Title}

**File**: `{file_path}`
**Line**: {line_number}
[**Function**: `{function_name}`]  # Optional

**Issue**: {clear_description}

{code_example_showing_problem}

**Impact**:
- {bullet_point_impacts}

[**Attack Example**:]  # For security issues
{example_of_exploit}

**Suggested Fix**:
{code_example_showing_solution}

[**Why This Matters**:]  # Optional context
{explanation}

[**Performance Benefit**:]  # For optimizations
{metrics}

**Resources**:
- [Link to documentation]

---
*Posted by Automated Review Bot - [{Check_Type}]* [- {Urgency_Flag}]
```

### Icons and Severity

```yaml
Severity Levels:
  - 🔴 CRITICAL: Security vulnerabilities, data loss risks
  - 🟡 HIGH: Major bugs, significant security issues
  - 🟠 MEDIUM: Performance issues, moderate bugs
  - 💡 LOW: Suggestions, optimizations, style issues
  - ✅ INFO: Informational, positive feedback

Urgency Flags:
  - 🚨 REQUIRES IMMEDIATE ACTION: Critical security issues
  - ⚠️ BLOCKS MERGE: Must be fixed before merge
  - 💡 SUGGESTION: Optional improvement
```

### Tone Guidelines

1. **Be Constructive**: Focus on solutions, not just problems
2. **Be Specific**: Provide exact line numbers and code examples
3. **Be Educational**: Explain why the issue matters
4. **Be Helpful**: Include resources and documentation links
5. **Be Respectful**: Assume good intent, avoid blame

### Example Good vs Bad Comments

❌ **Bad Comment**:
```markdown
Your code is wrong. Fix it.
```

✅ **Good Comment**:
```markdown
## 💡 Consider Using Async Context Manager

**File**: `src/scraper.py`
**Line**: 45

**Issue**: HTTP session not closed properly, potential resource leak

**Suggested Fix**:
```python
async with aiohttp.ClientSession() as session:
    # Session automatically closed
```

**Why This Matters**:
Unclosed sessions can exhaust file descriptors and cause connection pool issues.
```

---

## Configuration

Bot comment templates are configured in:

```yaml
# .github/review-bot-config.yml
review_bot:
  enabled: true
  comment_style: detailed  # or 'concise'
  post_on:
    - blocking_issues: always
    - suggestions: pr_description_only
    - info: never

  severity_thresholds:
    critical: always_comment
    high: always_comment
    medium: summarize
    low: pr_summary_only

  max_comments_per_pr: 20
  group_similar_issues: true
```

---

## Resources

- [Code Review Checklist](./REVIEW_CHECKLIST.md)
- [Automated Checks](./AUTOMATED_CHECKS.md)
- [Quality Gates](./QUALITY_GATES.md)
- [GitHub Actions Workflows](../../.github/workflows/)
