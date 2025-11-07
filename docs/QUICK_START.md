# Quick Start Guide - RSS Analyzer

Get up and running with the RSS Analyzer in minutes!

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- API key from one of: Anthropic Claude, Mistral AI, or OpenAI
- (Optional) Lefthook for git hooks

## Installation

### 1. Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### 2. Clone and Install Dependencies

```bash
# Clone repository
git clone https://github.com/your-repo/rss-analyzer.git
cd rss-analyzer

# Install dependencies (creates .venv automatically)
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Configure Environment

Create `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit with your API keys
nano .env  # or vim, code, etc.
```

Example `.env` configuration:

```bash
# Choose ONE provider
API_PROVIDER=anthropic  # or 'mistral' or 'openai'

# Anthropic Claude (default, recommended)
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Mistral AI (alternative)
# MISTRAL_API_KEY=your-mistral-key-here
# MISTRAL_MODEL=mistral-large-latest

# OpenAI (alternative)
# OPENAI_API_KEY=sk-your-openai-key-here
# OPENAI_MODEL=gpt-4

# RSS Feed Configuration
RSS_FEED_URL=https://rss.arxiv.org/rss/cs.AI

# Processing Options
MAX_ARTICLES_PER_RUN=10
SCRAPER_DELAY=1.0
FOLLOW_LINKS=true
MAX_LINKED_ARTICLES=3

# Performance Options
CACHE_ENABLED=true
DB_POOL_SIZE=5
```

### 4. Initialize Database

```bash
# Database is created automatically on first run
uv run python -m src.main test-api
```

### 5. Run Your First Analysis

```bash
# Analyze 3 articles
uv run python -m src.main run --limit 3

# Expected output:
# 🚀 Starting RSS article analysis pipeline...
# 📡 Using anthropic with model claude-3-5-sonnet-20241022
# 📊 Processing limited to 3 articles
# ...
# ✅ Success rate: 100%
# 📄 Reports generated in: output/
```

## Installation with Lefthook (Git Hooks)

Lefthook automatically runs linting and formatting on git commits.

### 1. Install Lefthook

```bash
# macOS
brew install lefthook

# Linux/WSL
# Download from https://github.com/evilmartians/lefthook/releases
curl -1sLf 'https://dl.cloudsmith.io/public/evilmartians/lefthook/setup.deb.sh' | sudo -E bash
sudo apt install lefthook

# Windows
scoop install lefthook

# Or via NPM (all platforms)
npm install -g @arkweid/lefthook

# Or via Homebrew on Linux
brew install lefthook
```

### 2. Initialize Lefthook

```bash
# Install git hooks
lefthook install

# Verify installation
lefthook run pre-commit --dry-run
```

### 3. Test Git Hooks

```bash
# Make a small change
echo "# Test" >> README.md

# Stage and commit (hooks will run automatically)
git add README.md
git commit -m "test: verify lefthook integration"

# Expected output:
# RUNNING HOOK: pre-commit
# ✓ ruff-check: checking 32 staged files...
# ✓ ruff-format: formatting 32 staged files...
```

## Common Commands

### Analysis Commands

```bash
# Run with default settings (10 articles)
uv run python -m src.main run

# Limit number of articles
uv run python -m src.main run --limit 5

# Force refresh (ignore cache and reprocess)
uv run python -m src.main run --force-refresh

# Disable link following
uv run python -m src.main run --no-follow-links
```

### Testing & Diagnostics

```bash
# Test API connection
uv run python -m src.main test-api

# Test RSS feed parsing
uv run python -m src.main test-rss

# View processing statistics
uv run python -m src.main stats

# Health check
uv run python -m src.main health

# Performance metrics
uv run python -m src.main metrics

# Run benchmarks
uv run python -m src.main benchmark
```

### Cache Management

```bash
# View cache statistics
uv run python -m src.main cache-stats

# Remove expired cache entries
uv run python -m src.main cleanup-cache

# Clear all cache (confirmation required)
uv run python -m src.main clear-cache

# Clear cache without confirmation
uv run python -m src.main clear-cache --confirm
```

### Provider Management

```bash
# List available AI providers
uv run python -m src.main providers

# Test specific provider
uv run python -m src.main providers --provider mistral
```

### Reports

```bash
# List generated reports
uv run python -m src.main list-reports

# Reports are automatically generated in output/ directory:
# - article_analysis_report_YYYYMMDD_HHMMSS.md (detailed)
# - summary_report_YYYYMMDD_HHMMSS.md (overview)
# - articles_export_YYYYMMDD_HHMMSS.json (machine-readable)
# - articles_export_YYYYMMDD_HHMMSS.csv (spreadsheet)
```

## Docker Usage

### Quick Start with Docker

```bash
# Build container
docker compose build

# Run analyzer
docker compose run rss-analyzer run --limit 3

# View stats
docker compose run rss-analyzer stats

# Access database
docker compose run rss-analyzer sqlite3 /app/data/articles.db
```

### Docker Commands

```bash
# Test RSS feed
docker compose run rss-analyzer test-rss

# Test API
docker compose run rss-analyzer test-api

# Check cache stats
docker compose run rss-analyzer cache-stats

# View logs
docker compose logs rss-analyzer
```

## Verification

### 1. Check Installation

```bash
# Python version
python --version  # Should be 3.11 or 3.12

# UV version
uv --version

# Lefthook version (if installed)
lefthook version

# Check virtual environment
which python  # Should show .venv path
```

### 2. Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_cache.py -v
```

### 3. Check Code Quality

```bash
# Lint code
uv run ruff check src/

# Auto-fix linting issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/

# Type checking (if mypy installed)
uv run mypy src/
```

## Troubleshooting

### Issue: "API key not found"

**Solution:**
```bash
# Verify .env file exists
ls -la .env

# Check API key is set
cat .env | grep API_KEY

# Ensure .env is loaded (uv loads automatically)
# Manual load for testing:
export $(cat .env | xargs)
```

### Issue: "Database locked"

**Solution:**
```bash
# Check for zombie processes
ps aux | grep python | grep rss-analyzer

# Kill if found
pkill -f "python -m src.main"

# Or restart with increased pool size
echo "DB_POOL_SIZE=10" >> .env
```

### Issue: "Cache full / Out of memory"

**Solution:**
```bash
# Clear cache
uv run python -m src.main clear-cache --confirm

# Reduce L1 cache size in src/core/cache.py:
# L1Cache.MAX_SIZE_BYTES = 128 * 1024 * 1024  # 128MB instead of 256MB

# Or disable cache temporarily
echo "CACHE_ENABLED=false" >> .env
```

### Issue: "Lefthook not running"

**Solution:**
```bash
# Reinstall hooks
lefthook install --force

# Check configuration
cat lefthook.yml

# Run manually
lefthook run pre-commit

# Skip hooks for one commit
git commit --no-verify -m "message"
```

## Performance Tips

### 1. Optimize Cache

```bash
# Monitor cache hit rate (target >60%)
uv run python -m src.main cache-stats

# Increase cache if you have memory
# Edit .env:
# L1_CACHE_SIZE_MB=512  # Double the cache size
```

### 2. Adjust Concurrency

```bash
# Increase database pool for faster queries
echo "DB_POOL_SIZE=10" >> .env

# Reduce scraper delay for faster scraping (be respectful!)
echo "SCRAPER_DELAY=0.5" >> .env
```

### 3. Use Async Mode (Future)

```bash
# When async implementation is ready:
# uv run python -m src.main run --async --concurrency 5
```

## Next Steps

1. **Read Architecture Review:** [docs/ARCHITECTURAL_REVIEW.md](./ARCHITECTURAL_REVIEW.md)
2. **See Implementation Roadmap:** [docs/IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md)
3. **Review Cache Documentation:** [docs/CACHE_USAGE.md](./CACHE_USAGE.md)
4. **Check Deduplication Guide:** [docs/DEDUPLICATION.md](./DEDUPLICATION.md)
5. **Explore API Docs:** [docs/API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

## Support

- **Documentation:** `docs/` directory
- **Issues:** GitHub Issues
- **Tests:** `tests/` directory
- **Examples:** `examples/` directory

Happy analyzing! 🚀
