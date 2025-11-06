# Implementation Roadmap - RSS Analyzer Enhancements

**Based on:** [ARCHITECTURAL_REVIEW.md](./ARCHITECTURAL_REVIEW.md)
**Created:** 2025-11-06
**Timeline:** 8-12 weeks

## Quick Reference

| Phase | Duration | Effort | Impact | Status |
|-------|----------|--------|--------|--------|
| **Phase 1: Quick Wins** | 1-2 weeks | Low | High | ⏸️ Not Started |
| **Phase 2: Architecture** | 2-4 weeks | Medium | High | ⏸️ Not Started |
| **Phase 3: Advanced** | 1-2 months | High | Medium | ⏸️ Not Started |
| **Phase 4: Scalability** | 3-6 months | High | Future | ⏸️ Not Started |

---

## Phase 1: Quick Wins (1-2 weeks)

### 🎯 Goal: Maximize ROI with minimal effort

**Expected Impact:**
- 70% cost reduction on re-runs
- Better resilience to API failures
- Cleaner codebase (-1000 lines)

### Task 1.1: Integrate Cache into Processing Pipeline

**Priority:** 🔴 CRITICAL
**Effort:** 2 days
**Dependencies:** None

**Subtasks:**
1. Add `ContentCache` initialization to `ArticleProcessor.__init__()`
2. Modify `_scrape_article()` to check cache before scraping
3. Modify `_analyze_article()` to check cache before API call
4. Add cache key generation using `ContentCache.generate_key()`
5. Update tests to verify cache integration

**Acceptance Criteria:**
- [ ] Cache hit logged for scraped content
- [ ] Cache hit logged for AI analysis
- [ ] Cache miss triggers scraping/analysis as before
- [ ] Cache stats command shows hit rate >60% after re-runs
- [ ] Force refresh flag bypasses cache

**Files to Modify:**
- `src/processors/article_processor.py` (main changes)
- `src/main.py` (add cache stats command)

**Example Implementation:**
```python
# In ArticleProcessor._scrape_article()
cache_key = ContentCache.generate_key(entry.link, "scraped_content")
cached = self.cache.get(cache_key)

if cached and not processing_config.force_refresh:
    logger.info(f"Cache hit for scraped content: {entry.link}")
    return cached

# ... existing scraping logic ...

if scraped_content:
    self.cache.set(
        cache_key,
        scraped_content,
        ttl=ContentCache.TTL_SCRAPED_CONTENT,
        content_type="scraped_content"
    )
```

---

### Task 1.2: Add Circuit Breaker Pattern to AI Clients

**Priority:** 🔴 HIGH
**Effort:** 2 days
**Dependencies:** None

**Subtasks:**
1. Create `src/clients/circuit_breaker.py` with `CircuitBreaker` class
2. Add circuit breaker to `BaseAIClient.__init__()`
3. Wrap `analyze_article()` calls with circuit breaker
4. Add unit tests for circuit breaker states (CLOSED, OPEN, HALF_OPEN)
5. Update documentation

**Acceptance Criteria:**
- [ ] Circuit opens after 3 consecutive failures
- [ ] Circuit rejects requests when OPEN
- [ ] Circuit attempts reset after 60 seconds
- [ ] Circuit closes after successful HALF_OPEN test
- [ ] Logs circuit state transitions

**Files to Create:**
- `src/clients/circuit_breaker.py` (new)

**Files to Modify:**
- `src/clients/base.py`
- `src/clients/async_base.py`

**Testing:**
```python
# Mock API to simulate failures
def test_circuit_breaker_opens():
    client = ClaudeClient(api_key="test", model="test")

    # Trigger 3 failures
    for i in range(3):
        with pytest.raises(APIClientError):
            client.analyze_article("test", "test", "test")

    # Circuit should be OPEN
    assert client.circuit_breaker.state == CircuitState.OPEN

    # Next call should fail immediately
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        client.analyze_article("test", "test", "test")
```

---

### Task 1.3: Clean Up Legacy ETL Code

**Priority:** 🟡 MEDIUM
**Effort:** 3 days
**Dependencies:** None

**Subtasks:**
1. Audit usage of `src/etl_orchestrator.py`
2. Audit usage of `src/main_etl.py`
3. Audit usage of `src/rss_parser_dedup_integration.py`
4. Check GitHub Actions workflows for references
5. Migrate unique functionality (if any) to main architecture
6. Create `src/legacy/` directory for archival
7. Move legacy files to archive or delete
8. Update documentation references

**Acceptance Criteria:**
- [ ] All GitHub Actions use `src/main.py` (not `main_etl.py`)
- [ ] No references to `etl_orchestrator` in active code
- [ ] Unique functionality migrated or documented
- [ ] README updated to remove legacy references
- [ ] Codebase reduced by ~500-1000 lines

**Files to Review:**
- `src/etl_orchestrator.py`
- `src/main_etl.py`
- `src/rss_parser_dedup_integration.py`
- `.github/workflows/*.yml`

**Decision Tree:**
```
For each legacy file:
  1. Is it used in GitHub Actions? → Migrate to main.py
  2. Does it have unique features? → Extract and merge
  3. Is it referenced anywhere? → Update references
  4. Otherwise → Archive or delete
```

---

## Phase 2: Architectural Improvements (2-4 weeks)

### 🎯 Goal: Reduce code duplication and enable async processing

**Expected Impact:**
- -1500 lines of duplicate code
- 3-5x faster processing for batches
- Easier maintenance (single source of truth)

### Task 2.1: Consolidate Sync/Async Client Implementations

**Priority:** 🔴 HIGH
**Effort:** 1 week
**Dependencies:** Task 1.2 (Circuit Breaker)

**Subtasks:**
1. Create `src/clients/unified_base.py` with dual-mode base class
2. Migrate `ClaudeClient` to unified implementation
3. Migrate `MistralClient` to unified implementation
4. Migrate `OpenAIClient` to unified implementation
5. Update `AIClientFactory` to use unified clients
6. Remove duplicate `async_*.py` files
7. Update all imports across codebase
8. Update tests to cover both sync and async modes

**Acceptance Criteria:**
- [ ] Single implementation supports both `analyze_article()` and `analyze_article_async()`
- [ ] Sync mode uses `asyncio.run()` wrapper
- [ ] Async mode uses native async implementation
- [ ] All existing tests pass
- [ ] Codebase reduced by ~1500 lines
- [ ] No behavioral changes to existing functionality

**Files to Create:**
- `src/clients/unified_base.py` (new)

**Files to Modify:**
- `src/clients/claude.py`
- `src/clients/mistral.py`
- `src/clients/openai.py`
- `src/clients/factory.py`

**Files to Delete:**
- `src/clients/async_base.py`
- `src/clients/async_claude.py`
- `src/clients/async_mistral.py`
- `src/clients/async_openai.py`

**Implementation Pattern:**
```python
# src/clients/unified_base.py
import asyncio
from abc import ABC, abstractmethod

class UnifiedAIClient(ABC):
    """Base class supporting both sync and async operations"""

    @abstractmethod
    async def _analyze_impl(self, title: str, content: str, url: str) -> dict:
        """Core async implementation (subclasses override this)"""
        pass

    def analyze_article(self, title: str, content: str, url: str) -> dict:
        """Synchronous interface"""
        return asyncio.run(self._analyze_impl(title, content, url))

    async def analyze_article_async(self, title: str, content: str, url: str) -> dict:
        """Asynchronous interface"""
        return await self._analyze_impl(title, content, url)
```

---

### Task 2.2: Implement Async Article Processing

**Priority:** 🔴 HIGH
**Effort:** 1 week
**Dependencies:** Task 2.1 (Unified Clients)

**Subtasks:**
1. Create `src/processors/async_article_processor.py`
2. Add `AsyncArticleProcessor` class inheriting from `ArticleProcessor`
3. Implement `run_async()` method with concurrent processing
4. Add semaphore for concurrency control (default: 5)
5. Implement `_process_articles_async()` with `asyncio.gather()`
6. Add async wrappers for scraping and analysis
7. Update `src/main.py` CLI with `--async` flag
8. Add performance benchmarks comparing sync vs async
9. Write integration tests for async mode

**Acceptance Criteria:**
- [ ] `--async` flag enables concurrent processing
- [ ] `--concurrency N` controls max concurrent articles
- [ ] Processing 10 articles is 3-5x faster in async mode
- [ ] Memory usage stays within reasonable bounds
- [ ] Error handling works correctly (one failure doesn't stop others)
- [ ] Results are identical to sync mode

**Files to Create:**
- `src/processors/async_article_processor.py` (new)
- `tests/test_async_processor.py` (new)

**Files to Modify:**
- `src/main.py` (add async CLI option)
- `src/core/async_scraper.py` (verify async support)

**CLI Usage:**
```bash
# Sync mode (current behavior)
uv run python -m src.main run --limit 10

# Async mode with 5 concurrent articles
uv run python -m src.main run --limit 10 --async --concurrency 5

# Expected output:
# Processing 10 articles in async mode (concurrency: 5)...
# ✓ Completed in 12.3 seconds (vs 58.7 seconds sync)
# Speedup: 4.8x faster
```

**Performance Testing:**
```python
# benchmarks/async_vs_sync.py
import time
from src.processors import ArticleProcessor, AsyncArticleProcessor

def benchmark_sync(entries):
    processor = ArticleProcessor(config)
    start = time.time()
    processor._process_articles(entries, ProcessingConfig(), results)
    return time.time() - start

def benchmark_async(entries):
    processor = AsyncArticleProcessor(config)
    start = time.time()
    asyncio.run(processor._process_articles_async(entries, ProcessingConfig(), results))
    return time.time() - start

# Expected results for 10 articles:
# Sync: ~50 seconds
# Async (concurrency=5): ~10 seconds
# Speedup: 5x
```

---

## Phase 3: Advanced Features (1-2 months)

### 🎯 Goal: Production-grade resilience and observability

**Expected Impact:**
- Better resilience to rate limits
- Improved debugging capabilities
- Production-ready reliability

### Task 3.1: Add Rate Limiting per API Provider

**Priority:** 🟡 MEDIUM
**Effort:** 3 days
**Dependencies:** None

**Subtasks:**
1. Install `ratelimit` library: `pip install ratelimit`
2. Create `src/clients/rate_limiter.py` with token bucket implementation
3. Add rate limits to `BaseAIClient` configuration
4. Configure per-provider limits (Claude: 50 req/min, OpenAI: 60 req/min)
5. Add backoff when rate limit hit
6. Add unit tests for rate limiting
7. Update documentation

**Acceptance Criteria:**
- [ ] Requests are throttled per provider limits
- [ ] No rate limit errors from APIs
- [ ] Graceful degradation (waits instead of fails)
- [ ] Configurable limits via environment variables
- [ ] Logs rate limit events

**Implementation:**
```python
# src/clients/rate_limiter.py
from ratelimit import limits, sleep_and_retry

class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period

    def __call__(self, func):
        @sleep_and_retry
        @limits(calls=self.calls, period=self.period)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

# Usage in AI clients
class ClaudeClient(UnifiedAIClient):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.rate_limiter = RateLimiter(calls=50, period=60)  # 50 req/min

    async def _analyze_impl(self, title: str, content: str, url: str) -> dict:
        @self.rate_limiter
        async def rate_limited_call():
            return await self._make_api_request(...)

        return await rate_limited_call()
```

---

### Task 3.2: Implement Retry with Exponential Backoff

**Priority:** 🟡 MEDIUM
**Effort:** 2 days
**Dependencies:** None

**Subtasks:**
1. Install `tenacity` library: `pip install tenacity`
2. Add retry decorators to API calls
3. Configure exponential backoff (1s, 2s, 4s, 8s)
4. Add retry only for transient errors (500, 503, timeout)
5. Don't retry for auth errors (401, 403)
6. Add logging for retry attempts
7. Update tests

**Acceptance Criteria:**
- [ ] Transient errors trigger retry with backoff
- [ ] Max 4 retry attempts
- [ ] Auth errors fail immediately (no retry)
- [ ] Logs each retry attempt with reason
- [ ] Eventually succeeds if API recovers

**Implementation:**
```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class ClaudeClient(UnifiedAIClient):
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(TransientError),
        reraise=True
    )
    async def _make_api_request(self, ...):
        try:
            return await self.client.messages.create(...)
        except anthropic.APIError as e:
            if e.status_code in (500, 503, 504):
                raise TransientError(e)  # Will retry
            else:
                raise  # Won't retry (auth error, etc.)
```

---

### Task 3.3: Add Distributed Tracing (OpenTelemetry)

**Priority:** 🟢 LOW
**Effort:** 1 week
**Dependencies:** None

**Subtasks:**
1. Install OpenTelemetry: `pip install opentelemetry-api opentelemetry-sdk`
2. Configure tracing exporter (Jaeger or OTLP)
3. Add traces to ArticleProcessor pipeline
4. Add traces to scraper requests
5. Add traces to AI API calls
6. Add traces to database operations
7. Create Jaeger docker-compose service
8. Write tracing documentation

**Acceptance Criteria:**
- [ ] End-to-end traces visible in Jaeger UI
- [ ] Can see full article processing flow
- [ ] Database query times visible
- [ ] API call durations tracked
- [ ] Errors captured in traces

**Implementation:**
```python
# src/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

provider = TracerProvider()
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Usage in ArticleProcessor
class ArticleProcessor:
    def run(self, processing_config):
        with tracer.start_as_current_span("article_processing_pipeline") as span:
            span.set_attribute("config.limit", processing_config.limit)

            with tracer.start_as_current_span("fetch_rss"):
                rss_entries = self._fetch_rss_feed(results)

            with tracer.start_as_current_span("process_articles"):
                processed = self._process_articles(...)
```

---

## Phase 4: Future Scalability (3-6 months)

### 🎯 Goal: Prepare for horizontal scaling

**Expected Impact:**
- Horizontal scalability (multi-node)
- Better cache efficiency across nodes
- Support for 1M+ articles

### Task 4.1: Message Queue Integration

**Priority:** 🟢 LOW (Future)
**Effort:** 2 weeks
**Dependencies:** None

**Architecture:**
```
RSS Parser → RabbitMQ Queue → Worker Pool (N nodes) → Database
```

**Implementation Plan:**
1. Choose message queue (RabbitMQ, Redis Streams, or AWS SQS)
2. Create producer to push RSS entries to queue
3. Create worker consumer to process from queue
4. Add queue management CLI commands
5. Add monitoring for queue depth
6. Write horizontal scaling documentation

---

### Task 4.2: Distributed Cache Support (Redis)

**Priority:** 🟢 LOW (Future)
**Effort:** 1 week
**Dependencies:** None

**Implementation Plan:**
1. Add Redis backend to ContentCache
2. Keep L1 as local memory
3. Replace L2 SQLite with Redis
4. Add cache invalidation across nodes
5. Benchmark Redis vs SQLite performance

---

## Implementation Guidelines

### Before Starting Any Task

1. **Create feature branch:** `git checkout -b feature/task-name`
2. **Review dependencies:** Ensure prerequisite tasks are complete
3. **Read acceptance criteria:** Understand success metrics
4. **Estimate time:** Confirm effort estimate is realistic

### During Implementation

1. **Write tests first** (TDD where applicable)
2. **Commit frequently** with descriptive messages
3. **Update documentation** as you code
4. **Run tests before PR:** `pytest`, `ruff check`

### After Completing Task

1. **Run full test suite:** Ensure no regressions
2. **Update CHANGELOG:** Document changes
3. **Create PR** with link to this roadmap task
4. **Mark task complete** in this document
5. **Update architecture score** in ARCHITECTURAL_REVIEW.md

### Performance Validation

After each phase, run benchmarks:
```bash
# Benchmark processing speed
uv run python -m src.main benchmark --iterations 10 --output results.json

# Check cache performance
uv run python -m src.main metrics --format json

# Verify database pool
uv run python -c "from src.core import DatabaseManager; print(DatabaseManager().get_pool_stats())"
```

---

## Success Metrics

### Phase 1 Metrics
- [ ] Cache hit rate: >60%
- [ ] Cost reduction: 70% on re-runs
- [ ] Codebase reduction: -1000 lines

### Phase 2 Metrics
- [ ] Async speedup: 3-5x for batches
- [ ] Code duplication: <10%
- [ ] Codebase reduction: -1500 lines

### Phase 3 Metrics
- [ ] Zero rate limit errors
- [ ] 95% success rate with retries
- [ ] Trace coverage: 100% of pipeline

### Phase 4 Metrics
- [ ] Horizontal scaling: Linear
- [ ] Cache efficiency: 80%+ hit rate
- [ ] Queue latency: <100ms

---

## Notes

- **Flexibility:** Tasks can be reordered based on business needs
- **Parallel Work:** Phase 1 tasks are independent and can run in parallel
- **Optional Tasks:** Phase 4 tasks are for future scalability (>100k articles/month)
- **Testing:** Every task requires tests (unit, integration, or benchmarks)
- **Documentation:** Update docs as you implement (don't defer)

---

**Roadmap Maintained By:** Development Team
**Next Review:** Every 2 weeks
**Questions?** See [ARCHITECTURAL_REVIEW.md](./ARCHITECTURAL_REVIEW.md) for context
