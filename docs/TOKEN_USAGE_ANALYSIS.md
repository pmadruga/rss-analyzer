# Token Usage Analysis Report
*Generated: 2025-10-29*

## Executive Summary

This report identifies where tokens are being spent during every RSS pipeline run and provides recommendations for optimization.

## Current Token Configuration

**API Settings** (`src/config/settings.py:16`):
- `MAX_TOKENS`: 4000 tokens per API response
- `TEMPERATURE`: 0.3
- `MAX_CONTENT_LENGTH`: 50,000 characters

**Pipeline Settings** (`.github/workflows/rss-complete-pipeline.yml:27`):
- `MAX_ARTICLES_PER_RUN`: 10 articles (default)
- `API_PROVIDER`: mistral (in GitHub Actions)

## Token Expenditure Breakdown

### 1. **Per-Article Analysis** (PRIMARY TOKEN SINK)
**Location**: `src/processors/article_processor.py:398-433`

**Token Usage Per Article**:
- **Input tokens**: ~2,000-12,000 tokens per article
  - System prompt: ~200 tokens (`src/clients/async_base.py:75-85`)
  - Article title: ~10-50 tokens
  - Article content: 1,800-12,000 tokens (truncated at 50K chars ≈ 12.5K tokens)
- **Output tokens**: Up to 4,000 tokens (MAX_TOKENS setting)
- **Total per article**: ~6,000-16,000 tokens

**Annual calculation**:
```
10 articles/day × 365 days = 3,650 articles/year
3,650 articles × 10,000 tokens avg = 36.5M tokens/year
```

### 2. **API Connection Tests**
**Location**: `src/processors/article_processor.py:183-187`

**Token Usage**:
- Test prompt: ~10 tokens input + ~5 tokens output = 15 tokens/run
- **Frequency**: Once per pipeline run (daily)
- **Annual**: 15 tokens × 365 = 5,475 tokens/year (negligible)

### 3. **Duplicate Articles** (OPTIMIZED)
**Location**: `src/processors/article_processor.py:206-237`

**Token Savings**:
- Hash-based deduplication (`src/rss_parser.py`) filters duplicates **BEFORE** API calls
- O(1) duplicate detection (`src/core/database.py`) prevents re-processing
- **No tokens spent** on duplicate articles ✅

### 4. **Failed Scraping** (COST AVOIDANCE)
**Location**: `src/processors/article_processor.py:344-396`

**Token Savings**:
- Articles that fail scraping are **NOT** sent to AI for analysis
- Prevents wasting tokens on articles with no content
- **Estimated savings**: 1-2 failed scrapes/day × 5,000 tokens = 1.8M-3.6M tokens/year saved ✅

### 5. **Caching** (OPTIMIZATION)
**Location**: `src/core/cache.py`

**Current State**:
- Two-tier cache (L1 memory + L2 disk) implemented
- **HOWEVER**: AI analysis results are **NOT currently cached** ❌
- Cache is used for scraped content (RSS + web scraping)
- **Opportunity**: Cache API responses to avoid re-analyzing same content

**Potential Savings**:
- If 30% of content is re-encountered: ~11M tokens/year saved

### 6. **GitHub Actions Workflow**
**Location**: `.github/workflows/rss-complete-pipeline.yml`

**Token Usage Pattern**:
- **Daily automated run**: 10 articles × 10K tokens = 100K tokens/day
- **Manual runs**: Variable (test mode: 1 article = 10K tokens)
- **Annual**: 100K tokens/day × 365 = 36.5M tokens/year

**Optimization Opportunities**:
- Workflow already implements deduplication checks (lines 131-156) ✅
- Post-processing duplicate detection (lines 210-236) ✅

## Token Cost Estimation

### By Provider

**Mistral (current GitHub Actions provider)**:
- Input: ~$0.002 per 1K tokens
- Output: ~$0.006 per 1K tokens
- Average cost: ~$0.004 per 1K tokens (mixed)

**Annual cost calculation**:
```
36.5M tokens/year × $0.004/1K = $146/year (Mistral)
```

**Claude (if using Anthropic)**:
- Input: $3.00 per 1M tokens (claude-3-5-sonnet)
- Output: $15.00 per 1M tokens
- Average cost: ~$9.00 per 1M tokens (mixed)

```
36.5M tokens/year × $9.00/1M = $328.50/year (Claude)
```

**OpenAI (if using GPT-4)**:
- Input: $10.00 per 1M tokens
- Output: $30.00 per 1M tokens
- Average cost: ~$20.00 per 1M tokens (mixed)

```
36.5M tokens/year × $20.00/1M = $730/year (OpenAI GPT-4)
```

## Optimization Recommendations

### 🔴 HIGH IMPACT

#### 1. **Reduce MAX_TOKENS for Output**
**Current**: 4000 tokens per response
**Recommendation**: Reduce to 2000-2500 tokens

**Reasoning**:
- Most analyses don't need 4000 tokens
- Output tokens cost 2-3x more than input tokens
- Can save 40-50% on output token costs

**Implementation**:
```python
# src/config/settings.py
MAX_TOKENS: int = 2000  # Changed from 4000
```

**Estimated savings**: 7.3M tokens/year = $29-146/year

#### 2. **Implement API Response Caching**
**Current**: No caching of AI analysis results
**Recommendation**: Cache API responses with 30-day TTL

**Reasoning**:
- Same content may be re-encountered
- Scraped content hash can be cache key
- Already have cache infrastructure

**Implementation location**: `src/processors/article_processor.py:398-433`

**Estimated savings**: 11M tokens/year (30% duplicate content) = $44-220/year

#### 3. **Content Length Optimization**
**Current**: 50,000 characters max (≈12,500 tokens)
**Recommendation**: Reduce to 30,000 characters (≈7,500 tokens)

**Reasoning**:
- Extremely long articles are rare
- First 30K chars usually contain full context
- Reduces input token costs

**Implementation**:
```python
# src/config/settings.py
MAX_CONTENT_LENGTH: int = 30000  # Changed from 50000
```

**Estimated savings**: 5.5M tokens/year = $22-110/year

### 🟡 MEDIUM IMPACT

#### 4. **Smarter System Prompt**
**Current**: 200 tokens per request
**Recommendation**: Reduce to 100 tokens with more concise instructions

**Location**: `src/clients/async_base.py:73-85`

**Estimated savings**: 1.8M tokens/year = $7-36/year

#### 5. **Batch Processing with Shared Context**
**Current**: Each article analyzed independently
**Recommendation**: Batch 5 articles together with shared system prompt

**Estimated savings**: 20% on system prompt overhead = 1.4M tokens/year = $6-28/year

### 🟢 LOW IMPACT

#### 6. **Rate Limiting Optimization**
**Current**: `RATE_LIMIT_DELAY: 3.0` seconds
**Impact**: No direct token savings, but prevents rate limit errors that cause retries

**Status**: Already optimized ✅

#### 7. **Monitoring Token Usage**
**Recommendation**: Add token counting to `src/core/monitoring.py`

**Implementation**:
```python
# Track actual tokens used per request
self.api_tokens_used += response.usage.total_tokens
```

## Token Usage by Pipeline Phase

### Phase 1: RSS Parsing (0 tokens)
- RSS feed fetch: No API calls ✅
- Duplicate filtering: Hash-based, no API calls ✅

### Phase 2: Content Scraping (0 tokens)
- Web scraping: No API calls ✅
- Content extraction: No API calls ✅
- Failed scrapes: Skip API call ✅

### Phase 3: AI Analysis (100% of tokens)
- **10 articles × 10,000 tokens = 100,000 tokens/day** ⚠️
- This is the **ONLY phase** where tokens are spent

### Phase 4: Report Generation (0 tokens)
- Markdown/JSON/CSV generation: No API calls ✅
- Website data generation: No API calls ✅

## Implementation Priority

### Immediate Actions (Week 1)
1. ✅ Reduce `MAX_TOKENS` from 4000 to 2000
2. ✅ Implement API response caching
3. ✅ Reduce `MAX_CONTENT_LENGTH` from 50K to 30K

**Expected savings**: 23.8M tokens/year = $95-476/year (65% reduction)

### Short-term Actions (Month 1)
4. ✅ Optimize system prompt
5. ✅ Add token usage monitoring
6. ✅ Implement batch processing

**Additional savings**: 3.2M tokens/year = $13-64/year

### Total Potential Savings
**Before optimization**: 36.5M tokens/year = $146-730/year
**After optimization**: 9.5M tokens/year = $38-190/year
**Savings**: 27M tokens/year = $108-540/year (74% reduction)

## Monitoring and Alerts

### Recommended Metrics
1. **Tokens per article** (input + output)
2. **Daily token usage**
3. **Monthly token costs**
4. **Cache hit rate** for API responses
5. **Failed analysis rate** (to prevent wasted retries)

### Alert Thresholds
- Daily usage > 150K tokens (50% over budget)
- Monthly cost > $15 (Mistral) or $35 (Claude)
- Cache hit rate < 20%

## Conclusion

**Primary token sink**: AI article analysis (100% of token usage)

**Key findings**:
1. ✅ Deduplication system prevents duplicate API calls (working well)
2. ✅ Failed scrapes don't consume tokens (good optimization)
3. ❌ No API response caching (major opportunity)
4. ❌ MAX_TOKENS too high (easy fix)
5. ❌ Content length too high (easy fix)

**Recommended actions**:
- Reduce MAX_TOKENS: 4000 → 2000 (saves 50% on output tokens)
- Implement API caching (saves 30% on duplicate content)
- Reduce content length: 50K → 30K chars (saves 40% on input tokens)

**Total savings**: Up to 74% reduction in token usage and costs.
