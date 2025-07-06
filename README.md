# RSS Feed Article Analyzer

[![RSS Analyzer](https://github.com/your-username/rss-analyzer/actions/workflows/rss-analyzer.yml/badge.svg)](https://github.com/your-username/rss-analyzer/actions/workflows/rss-analyzer.yml)

Automatically fetches and analyzes academic papers from RSS feeds using AI APIs (Anthropic Claude, Mistral, or OpenAI). Supports Bluesky posts with embedded arXiv links and follows referenced links for comprehensive analysis.

## 🚀 Quick Start Options

### Option 1: GitHub Actions (Recommended)
**Fully automated cloud solution - no local setup required!**

```bash
# 1. Fork this repository
# 2. Run the setup script
./setup_github_action.sh
# 3. Configure your API key
# 4. Push to GitHub - starts running automatically once daily at 2 AM UTC!
```

📚 **[Full GitHub Actions Setup Guide →](GITHUB_ACTION_SETUP.md)**

### Option 2: Local Docker Setup

#### Prerequisites
- Docker and Docker Compose

#### Setup & Run

1. **Clone the project**:
   ```bash
   git clone <repository-url>
   cd rss-analyzer
   ```

2. **Configure your API provider**:

   Choose one of the supported AI providers and add your API key to the .env file:

   **Anthropic Claude (default)**
   ```bash
   echo "API_PROVIDER=anthropic" > .env
   echo "ANTHROPIC_API_KEY=sk-your-api-key-here" >> .env
   ```

   **Mistral AI**
   ```bash
   echo "API_PROVIDER=mistral" > .env
   echo "MISTRAL_API_KEY=your-mistral-key-here" >> .env
   ```

   **OpenAI**
   ```bash
   echo "API_PROVIDER=openai" > .env
   echo "OPENAI_API_KEY=sk-your-openai-key-here" >> .env
   ```

3. **Run the analyzer**:
   ```bash
   docker compose run rss-analyzer run --limit 3
   ```

## Usage

```bash
# Analyze 5 newest articles
docker compose run rss-analyzer run --limit 5

# Test RSS feed
docker compose run rss-analyzer test-rss

# Test API connection
docker compose run rss-analyzer test-api

# View processing statistics
docker compose run rss-analyzer stats

# View help
docker compose run rss-analyzer --help
```

## Database Queries

The SQLite database is stored in `data/articles.db`. You can query it directly:

```bash
# Access the database
docker compose run rss-analyzer sqlite3 /app/data/articles.db

# Or use a local SQLite client
sqlite3 data/articles.db
```

### Useful Queries

```sql
-- View all processed articles
SELECT id, title, url, status, processed_date FROM articles;

-- Count articles by status
SELECT status, COUNT(*) FROM articles GROUP BY status;

-- View recent analyses with confidence scores
SELECT a.title, c.confidence_score, a.processed_date
FROM articles a
JOIN content c ON a.id = c.article_id
ORDER BY a.processed_date DESC
LIMIT 10;

-- Find articles from specific sources
SELECT title, url FROM articles WHERE url LIKE '%arxiv%';

-- View processing errors
SELECT timestamp, status, error_message
FROM processing_log
WHERE status LIKE '%failed%'
ORDER BY timestamp DESC;
```

## Output

Reports are generated in the `output/` directory:
- `article_analysis_report.md` - Detailed analysis with methodology explanations
- `summary_report.md` - Quick overview
- `articles_export.json` - Machine-readable format
- `articles_export.csv` - Spreadsheet format

## ✨ Features

### Core Functionality
- **RSS Feed Processing** - Fetches from any RSS feed
- **Bluesky Support** - Extracts embedded arXiv links from Bluesky posts
- **Academic Publishers** - Supports arXiv, IEEE, ACM, Nature, etc.
- **Multiple AI Providers** - Supports Anthropic Claude, Mistral, and OpenAI
- **AI Analysis** - Explains methodology in simple terms
- **Multiple Formats** - Markdown, JSON, CSV outputs

### 🔗 Enhanced Link Following
- **Smart Link Detection** - Follows links in blog posts and articles
- **Prioritized Domains** - Academic papers, tech companies, research blogs
- **Content Enhancement** - Adds summaries of referenced articles
- **Intelligent Filtering** - Skips ads, social media, irrelevant links

### 🤖 Automation Options
- **GitHub Actions** - Fully automated cloud execution
- **Local Scheduling** - Hourly background service (macOS/Linux)
- **Docker Ready** - No local Python setup needed
- **Continuous Updates** - Maintains growing analysis database
