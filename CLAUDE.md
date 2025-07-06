# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSS Feed Article Analyzer that automatically fetches and analyzes academic papers from RSS feeds using AI APIs (Anthropic Claude, Mistral, or OpenAI). Supports Bluesky posts with embedded arXiv links and generates comprehensive reports in multiple formats.

## Architecture

### Core Components
- **ArticleProcessor** (`src/main.py`): Main orchestrator that coordinates the entire pipeline
- **RSSParser** (`src/rss_parser.py`): Fetches and parses RSS feeds with duplicate detection
- **WebScraper** (`src/scraper.py`): Extracts full article content from academic publisher websites and follows embedded links for comprehensive analysis
- **DatabaseManager** (`src/database.py`): SQLite operations with schema migrations and duplicate detection
- **AI Clients**: Pluggable architecture supporting Claude (`claude_client.py`), Mistral (`mistral_client.py`), and OpenAI (`openai_client.py`)
- **ReportGenerator** (`src/report_generator.py`): Multi-format output (Markdown, JSON, CSV)

### Data Flow
1. RSS feed parsing → 2. Content scraping → 3. AI analysis → 4. Database storage → 5. Report generation

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
uv run python generate_comprehensive_reports.py
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
- `articles`: Article metadata (id, title, url, status, processed_date)
- `content`: Full content and AI analysis (article_id, content, analysis)
- `processing_log`: Execution history and errors

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
- `articles_export.json`: Machine-readable format
- `articles_export.csv`: Spreadsheet format

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

## Development Notes

- All API clients inherit from base interface for consistent error handling
- Database uses content hashing for duplicate detection
- Robust retry logic with exponential backoff for HTTP requests
- Comprehensive logging with configurable levels
- Docker multi-stage build for optimized container size
- Non-root container user for security
