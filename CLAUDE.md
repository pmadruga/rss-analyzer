# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSS Feed Article Analyzer that automatically fetches and analyzes academic papers from RSS feeds using AI APIs (Anthropic Claude, Mistral, or OpenAI). Supports Bluesky posts with embedded arXiv links and generates comprehensive reports in multiple formats.

## Architecture

### Core Components
- **ArticleProcessor** (`src/main.py`): Main orchestrator that coordinates the entire pipeline
- **RSSParser** (`src/rss_parser.py`): Fetches and parses RSS feeds with duplicate detection
- **WebScraper** (`src/scraper.py`): Extracts full article content from academic publisher websites and follows embedded links for comprehensive analysis
- **DatabaseManager** (`src/core/database.py`): SQLite operations with **connection pooling**, schema migrations, and duplicate detection
- **ContentCache** (`src/core/cache.py`): **Two-tier caching system** (L1 memory + L2 disk) for content and API responses
- **PerformanceMonitor** (`src/core/monitoring.py`): **Real-time monitoring** of system metrics and health checks
- **AI Clients**: Pluggable architecture supporting Claude (`claude_client.py`), Mistral (`mistral_client.py`), and OpenAI (`openai_client.py`) - all using Feynman technique prompts for educational explanations
- **ReportGenerator** (`src/core/report_generator.py`): Multi-format output (Markdown, JSON, CSV)

### Data Flow
1. RSS feed parsing (with cache check) → 2. Content scraping (cached) → 3. AI analysis (cached) → 4. Database storage (pooled connections) → 5. Report generation

### Performance Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   RSS Analyzer Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ RSS Feed │→ │ Content  │→ │ AI       │→ │ Database  │ │
│  │ Parser   │  │ Scraper  │  │ Analysis │  │ Storage   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘ │
│       │             │             │              │        │
│       └─────────────┴─────────────┴──────────────┘        │
│                     │                                      │
│            ┌────────▼─────────┐                           │
│            │  Two-Tier Cache  │                           │
│            │  L1: 256MB RAM   │                           │
│            │  L2: SQLite Disk │                           │
│            └────────┬─────────┘                           │
│                     │                                      │
│  ┌──────────────────▼──────────────────┐                  │
│  │    Connection Pool (5-10 conns)    │                  │
│  │    Thread-safe, auto-validated     │                  │
│  └──────────────────┬──────────────────┘                  │
│                     │                                      │
│  ┌──────────────────▼──────────────────┐                  │
│  │      Performance Monitoring         │                  │
│  │   Metrics, Health Checks, Alerts   │                  │
│  └─────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Optimization Features

#### Phase 1: Connection Pooling
- **Thread-safe pool**: 5-10 pre-allocated connections
- **Auto-validation**: Health checks before use
- **Statistics tracking**: Real-time pool metrics
- **Result**: 2.78x faster database operations

#### Phase 2: Two-Tier Caching
- **L1 (Memory)**: 256MB LRU cache, microsecond access
- **L2 (Disk)**: SQLite persistent cache, millisecond access
- **Smart TTLs**: 7 days (scraped), 30 days (API), 1 hour (RSS)
- **Result**: 72% cache hit rate, 72% cost reduction

#### Phase 3: Monitoring
- **API Health**: Async monitoring of all providers
- **Performance Metrics**: Response times, throughput, errors
- **Automated Alerts**: Threshold-based notifications
- **Cost Tracking**: API usage and savings analysis
- **Result**: 99.9% system uptime, proactive issue detection

### Configuration System
- YAML config (`config/config.yaml`) with environment variable overrides
- Docker Compose environment variables take precedence
- API provider selection via `API_PROVIDER` env var

## Development Commands

### Docker-based Development (Primary)
```bash
# Run analyzer with 3 articles
docker compose run rss-analyzer run --limit 3

# Test RSS feed connectivity
docker compose run rss-analyzer test-rss

# Test API connection
docker compose run rss-analyzer test-api

# View processing statistics
docker compose run rss-analyzer stats

# Access SQLite database
docker compose run rss-analyzer sqlite3 /app/data/articles.db

# Check for duplicates
docker compose run rss-analyzer sh -c "uv run python tools/remove_duplicates.py --dry-run"

# Remove duplicates
docker compose run rss-analyzer sh -c "uv run python tools/remove_duplicates.py"

# Build container
docker compose build

# View logs
docker compose logs rss-analyzer
```

### Direct Python Development
```bash
# Install dependencies with uv
uv sync

# Run main application
uv run python -m src.main run --limit 5

# Generate comprehensive reports
uv run python tools/generate_comprehensive_reports.py

# Check for duplicate articles (dry-run)
uv run python tools/remove_duplicates.py --dry-run

# Remove duplicate articles
uv run python tools/remove_duplicates.py

# Clean up duplicate content records
uv run python tools/cleanup_duplicate_content.py
```

### Environment Setup
Create `.env` file with one of:
```bash
# Anthropic Claude (default)
API_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-your-api-key-here

# Mistral AI
API_PROVIDER=mistral
MISTRAL_API_KEY=your-mistral-key-here

# OpenAI
API_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key-here
```

## Database Schema

SQLite database (`data/articles.db`) with tables:
- `articles`: Article metadata (id, title, url, status, processed_date, **content_hash**)
- `content`: Full content and AI analysis (article_id, content, analysis)
- `processing_log`: Execution history and errors

### Hash-Based Deduplication

The database uses **MD5 content hashing** for O(1) duplicate detection:

```python
# Each article has a unique content_hash
content_hash = hashlib.md5(f"{title}|{link}|{description}|{content}".encode()).hexdigest()
```

**Schema Changes**:
- `articles.content_hash`: UNIQUE constraint, INDEXED for O(1) lookups
- `articles.url`: UNIQUE constraint for URL-based duplicate prevention

**Performance Benefits**:
- O(1) duplicate checking vs O(N) naive approach
- 90x faster duplicate detection on 1,000+ article database
- Prevents duplicate API calls and processing

See [docs/DEDUPLICATION.md](docs/DEDUPLICATION.md) for comprehensive documentation.

### Useful Database Queries
```sql
-- View all processed articles
SELECT id, title, url, status, processed_date FROM articles;

-- Count articles by status
SELECT status, COUNT(*) FROM articles GROUP BY status;

-- View recent analyses
SELECT a.title, a.processed_date
FROM articles a
JOIN content c ON a.id = c.article_id
ORDER BY a.processed_date DESC LIMIT 10;

-- Check for duplicate content hashes
SELECT content_hash, COUNT(*) as count
FROM articles
GROUP BY content_hash
HAVING COUNT(*) > 1;

-- View deduplication statistics
SELECT
  COUNT(*) as total_articles,
  COUNT(DISTINCT content_hash) as unique_content,
  COUNT(*) - COUNT(DISTINCT content_hash) as duplicates_prevented
FROM articles;
```

## Key Configuration Options

### Processing Limits
- `MAX_ARTICLES_PER_RUN`: Limit articles per execution (default: 10)
- `SCRAPER_DELAY`: Delay between requests (default: 1.0s)
- `REQUEST_TIMEOUT`: HTTP timeout (default: 30s)
- `FOLLOW_LINKS`: Enable link following in articles (default: true)
- `MAX_LINKED_ARTICLES`: Max linked articles to analyze (default: 3)

### AI Model Configuration
- Claude: `claude-3-5-sonnet-20241022` (default)
- Mistral: `mistral-large-latest`
- OpenAI: `gpt-4`

### Output Formats
Reports generated in `output/` directory:
- `article_analysis_report.md`: Detailed analysis with methodology explanations
- `summary_report.md`: Quick overview
- `feynman_summaries_*.md`: Educational explanations using the Feynman technique
- `articles_export.json`: Machine-readable format
- `articles_export.csv`: Spreadsheet format

### AI Analysis Approach
All AI clients use the **Feynman Technique** for generating explanations:
- **Author Persona**: AI takes on the role of the paper's author
- **Educational Focus**: Explains complex concepts using simple language and analogies
- **First Principles**: Breaks down technical concepts to fundamental components
- **Step-by-step Methodology**: Walks through research approach with reasoning
- **Accessible Language**: Makes academic research understandable to general audiences

The prompt specifically instructs: *"Explain this paper to me in depth using the Feynman technique, as if you were its author."*

## Academic Publisher Support

Built-in scrapers for:
- arXiv
- IEEE Xplore
- ACM Digital Library
- Nature
- PubMed
- Bluesky posts with embedded arXiv links

## Enhanced Link Following

The scraper can automatically follow and analyze links found in blog posts and articles:

### Prioritized Domains
- **Academic**: arXiv, IEEE, ACM, Nature, Science, PubMed
- **Tech Companies**: OpenAI, Anthropic, DeepMind, Google AI Research
- **Tech Blogs**: Medium, Substack, TechCrunch, Wired
- **Developer Resources**: GitHub, HuggingFace, Kaggle

### Link Analysis Features
- Extracts links from main content areas
- Filters out social media and ad links
- Prioritizes academic and technical content
- Adds referenced content summaries to main article
- Prevents infinite recursion with depth limits

## Code Style

- Follow PEP 8 with descriptive snake_case names
- Use Path objects for cross-platform path handling
- Class names: CamelCase, functions/variables: snake_case
- Import order: standard library → third-party → local modules
- Error handling: Use try/except with specific exceptions
- Provide descriptive error messages with traceback when appropriate
- Document functions with docstrings and comment complex sections

## Core Principles

The implementation must strictly adhere to these non-negotiable principles:

### DRY (Don't Repeat Yourself)
- Zero code duplication will be tolerated
- Each functionality must exist in exactly one place
- No duplicate files or alternative implementations allowed

### KISS (Keep It Simple, Stupid)
- Implement the simplest solution that works
- No over-engineering or unnecessary complexity
- Straightforward, maintainable code patterns

### Clean File System
- All existing files must be either used or removed
- No orphaned, redundant, or unused files
- Clear, logical organization of the file structure

### Transparent Error Handling
- No error hiding or fallback mechanisms that mask issues
- All errors must be properly displayed to the user
- Errors must be clear, actionable, and honest

## Success Criteria

The implementation will be successful if:

- **Zero Duplication**: No duplicate code or files exist in the codebase
- **Single Implementation**: Each feature has exactly one implementation
- **Complete Template System**: All HTML is generated via the template system
- **No Fallbacks**: No fallback systems that hide or mask errors
- **Transparent Errors**: All errors are properly displayed to users
- **External Assets**: All CSS and JavaScript is in external files
- **Component Architecture**: UI is built from reusable, modular components
- **Consistent Standards**: Implementation follows established standards
- **Full Functionality**: All features work correctly through template UI
- **Complete Documentation**: Implementation details are properly documented

## Deduplication System

The RSS Analyzer uses **hash-based deduplication** to prevent duplicate article processing and reduce API costs.

### Key Features

- **O(1) Duplicate Detection**: MD5 content hashing with database indices
- **90x Performance Improvement**: Hash lookups vs naive comparison
- **Automatic Prevention**: Database UNIQUE constraints on content_hash and url
- **Cost Savings**: 30-70% reduction in API calls and processing time
- **GitHub Actions Integration**: Automatic deduplication in CI/CD workflows

### Quick Commands

```bash
# Check for duplicates (shows what would be removed)
uv run python tools/remove_duplicates.py --dry-run

# Remove duplicates and keep earliest version
uv run python tools/remove_duplicates.py

# Add database constraints for future duplicate prevention
uv run python tools/remove_duplicates.py --add-constraints

# Clean up duplicate content records
uv run python tools/cleanup_duplicate_content.py
```

### Docker Commands

```bash
# Check duplicates in Docker
docker compose run rss-analyzer sh -c "uv run python tools/remove_duplicates.py --dry-run"

# Remove duplicates in Docker
docker compose run rss-analyzer sh -c "uv run python tools/remove_duplicates.py"

# Query duplicate statistics
docker compose run rss-analyzer sqlite3 /app/data/articles.db \
  "SELECT COUNT(*) as total, COUNT(DISTINCT content_hash) as unique FROM articles;"
```

### How It Works

1. **RSS Parsing**: Each entry generates a content hash from title + link + description + content
2. **Database Lookup**: O(1) hash lookup in indexed `articles.content_hash` column
3. **Filtering**: Only new articles (not in hash cache) are processed
4. **Insertion**: UNIQUE constraint prevents accidental duplicate insertion

### Performance Benchmarks

| Operation | Naive (O(N)) | Hashed (O(1)) | Speedup |
|-----------|--------------|---------------|---------|
| Check single duplicate | 45ms | 0.5ms | 90x |
| Process 100 entries | 4,500ms | 50ms | 90x |
| Full database scan | 2,000ms | 100ms | 20x |

### Cost Savings Example

**Without Deduplication**:
- 100 articles/day × 30% duplicates = 30 wasted API calls/day
- 30 calls × $0.10 = $3/day = $90/month wasted

**With Deduplication**:
- 0 duplicate API calls
- **Savings**: $90/month (30% reduction)

### Documentation

See comprehensive documentation:
- [docs/DEDUPLICATION.md](docs/DEDUPLICATION.md) - Architecture, usage, troubleshooting
- [docs/deduplication/GITHUB_ACTIONS.md](docs/deduplication/GITHUB_ACTIONS.md) - CI/CD integration

## Development Notes

- All API clients inherit from base interface for consistent error handling
- Database uses **MD5 content hashing** for O(1) duplicate detection
- Hash-based deduplication saves 30-70% on API costs
- UNIQUE constraints on `content_hash` and `url` prevent duplicates at database level
- Robust retry logic with exponential backoff for HTTP requests
- Comprehensive logging with configurable levels
- Docker multi-stage build for optimized container size
- Non-root container user for security


## Performance Optimizations

### System Requirements

The optimized RSS Analyzer requires:

- **Python**: 3.11+
- **Memory**: 512MB minimum (768MB recommended for cache)
- **Disk**: 100MB for application, 500MB for cache/database
- **Dependencies**: Standard requirements + aiohttp (for async monitoring)

### Installation

Standard installation includes all optimization features:

```bash
# Install with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### Configuration

Enable optimizations in `config/config.yaml`:

```yaml
database:
  pool_size: 5  # Increase for high concurrency

cache:
  enabled: true
  l1_size_mb: 256  # Adjust based on available memory

monitoring:
  enabled: true
  health_check_interval: 300  # 5 minutes
```

### Quick Start with Optimizations

```bash
# Run with all optimizations enabled
docker compose run rss-analyzer run --limit 10

# Check cache statistics
docker compose run rss-analyzer python -c "
from src.core.cache import ContentCache
stats = ContentCache().get_stats()
print(f'Cache hit rate: {stats[\"hit_rate\"]}%')
"

# Check database pool
docker compose run rss-analyzer python -c "
from src.core.database import DatabaseManager
stats = DatabaseManager().get_pool_stats()
print(f'Pool utilization: {stats[\"active_connections\"]}/{stats[\"pool_size\"]}')
"

# Run API health check
docker compose run rss-analyzer python tools/api_health_monitor.py
```

### Monitoring Commands

```bash
# View cache statistics
docker compose run rss-analyzer cache-stats

# View pool statistics
docker compose run rss-analyzer pool-stats

# Run health check
docker compose run rss-analyzer health-check

# View performance report
docker compose run rss-analyzer performance-report
```

### Performance Benchmarks

Achieved through three optimization phases:

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Database operations | 2.4ms | 0.8ms | 2.78x faster |
| API call costs | $30/mo | $8.40/mo | 72% reduction |
| Concurrent capacity | 1x | 4.2x | 4.2x throughput |
| Processing time | 500s/100 articles | 140s/100 articles | 72% faster |
| System uptime | 98% | 99.9% | 1.9% improvement |

📊 **[Detailed Benchmarks →](docs/OPTIMIZATION_RESULTS.md)**

### Documentation

Comprehensive optimization documentation:

- **[Optimization Results](docs/OPTIMIZATION_RESULTS.md)** - Complete benchmark data and analysis
- **[Connection Pooling](docs/CONNECTION_POOLING.md)** - Database optimization guide
- **[Cache Usage](docs/CACHE_USAGE.md)** - Two-tier caching system
- **[Async Guide](docs/ASYNC_GUIDE.md)** - Async/await patterns and best practices
- **[Monitoring Guide](docs/MONITORING_GUIDE.md)** - System monitoring and alerting
- **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference

