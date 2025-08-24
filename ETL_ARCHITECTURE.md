# ETL Architecture Reorganization

The codebase has been reorganized into a clean **Extract, Transform, Load (ETL)** architecture for better maintainability and clarity.

## 📁 New Directory Structure

```
src/
├── etl/                          # ETL pipeline components
│   ├── extract/                  # Data collection layer
│   │   ├── rss_parser.py        # RSS feed parsing
│   │   ├── web_scraper.py       # Website content extraction  
│   │   └── content_fetcher.py   # Unified content collection interface
│   ├── transform/               # Data processing layer
│   │   ├── ai_clients/          # AI service integrations (Claude, OpenAI, Mistral)
│   │   ├── prompts.py           # Centralized prompt templates
│   │   ├── content_processor.py # Text cleaning and preprocessing
│   │   └── analysis_engine.py   # AI analysis coordination
│   └── load/                    # Data output layer
│       ├── database.py          # Database operations
│       ├── report_generator.py  # Multi-format reports
│       ├── website_generator.py # Static website generation
│       └── data_exporter.py     # Various export formats
├── etl_orchestrator.py          # Main ETL coordination
├── main_etl.py                  # New clean CLI entry point
├── config/                      # Configuration management
├── exceptions/                  # Error handling
└── main.py                      # Legacy entry point (deprecated)
```

## 🔄 ETL Pipeline Stages

### 📥 **Extract** (`src/etl/extract/`)

**Purpose**: Collect and fetch content from external sources

**Components**:
- **RSSParser**: Fetches and parses RSS feeds with duplicate detection
- **WebScraper**: Extracts full article content from websites  
- **ContentFetcher**: Unified interface coordinating RSS and web scraping

**Key Features**:
- Robust error handling and retries
- Duplicate detection using content hashes
- Support for multiple content sources (ArXiv, blogs, news)

### 🔄 **Transform** (`src/etl/transform/`)

**Purpose**: Process and analyze content using AI

**Components**:
- **AI Clients**: Pluggable AI providers (Claude, OpenAI, Mistral)
- **Prompts**: Centralized prompt template management
- **ContentProcessor**: Text cleaning and preprocessing
- **AnalysisEngine**: Coordinates AI analysis workflow

**Key Features**:
- **Centralized Prompts**: All AI prompts in one place (`prompts.py`)
- **Multiple Analysis Types**: Feynman technique, technical analysis, context engineering
- **Content Quality Validation**: Checks for scraping issues and content quality
- **Flexible AI Provider Selection**: Easy switching between AI services

### 💾 **Load** (`src/etl/load/`)

**Purpose**: Store results and generate outputs

**Components**:
- **Database**: SQLite operations with schema management
- **ReportGenerator**: Markdown and multi-format reports
- **WebsiteGenerator**: Static website creation
- **DataExporter**: JSON, CSV, and statistics export

**Key Features**:
- Multiple output formats
- Website with enhanced JSON rendering
- Comprehensive statistics and reporting

## 🚀 **ETL Orchestrator**

The `ETLOrchestrator` class coordinates the entire pipeline:

```python
orchestrator = ETLOrchestrator(config)
results = orchestrator.run_full_pipeline(feed_urls, max_articles=10)
```

**Benefits**:
- Clean separation of concerns
- Easy to test individual components
- Comprehensive error handling and reporting
- Flexible execution (can run individual stages)

## 🎯 **Centralized Prompt Management**

All AI prompts are now in `src/etl/transform/prompts.py`:

```python
# Available prompt types
FEYNMAN_ANALYSIS_PROMPT      # Educational explanations
TECHNICAL_ANALYSIS_PROMPT     # Technical deep-dives  
CONTEXT_ENGINEERING_PROMPT    # System design focus
RESEARCH_SUMMARY_PROMPT       # Concise research summaries
BLOG_POST_ANALYSIS_PROMPT     # Practical insights

# Smart prompt selection
prompt = select_prompt_for_url(url)  # Auto-selects based on URL
```

## 🌐 **Enhanced Website Features**

- **Extracted Titles**: Now uses `extracted_title` from JSON analyses
- **Improved JSON Rendering**: Better handling of malformed JSON
- **Sorting Options**: Sort by scraped date vs processed date
- **Multiple Fallback Layers**: Graceful degradation for parsing issues

## 🔧 **Usage Examples**

### Run Complete Pipeline
```bash
python -m src.main_etl run --limit 10 --feeds "https://example.com/rss"
```

### Test Individual Components
```bash
python -m src.main_etl test-rss --feed-url "https://example.com/rss"
python -m src.main_etl test-ai
python -m src.main_etl stats
```

### Run Individual Stages
```bash
python -m src.main_etl extract    # Extract only
python -m src.main_etl transform  # Transform only  
python -m src.main_etl load      # Load only
```

## 📈 **Benefits of ETL Architecture**

1. **Modularity**: Each component has single responsibility
2. **Testability**: Easy to unit test individual stages
3. **Maintainability**: Clear organization, easy to find and modify code
4. **Scalability**: Can optimize or replace individual components
5. **Debugging**: Clear error boundaries and comprehensive logging
6. **Flexibility**: Can run partial pipelines or custom workflows

## 🔄 **Migration Path**

- **Current**: Legacy `main.py` still works for backward compatibility
- **New**: Use `main_etl.py` for all new development
- **Prompts**: Centralized in `prompts.py` instead of scattered across files
- **Configuration**: Unchanged, uses existing `config/settings.py`

## 🧹 **Code Quality Improvements**

- **No Code Duplication**: ETL pattern eliminates redundant functionality
- **Single Responsibility**: Each module has one clear purpose  
- **Transparent Error Handling**: No hidden fallbacks, clear error reporting
- **Clean File Organization**: Logical grouping by functionality
- **Centralized Configuration**: All prompts and settings in dedicated files

This ETL architecture provides a solid foundation for future enhancements while making the codebase much more maintainable and understandable.