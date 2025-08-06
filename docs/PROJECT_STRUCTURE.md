# RSS Analyzer - Project Structure

This document provides a comprehensive overview of the project structure after refactoring and organization.

## 📁 Root Directory Structure

```
rss-analyzer/
├── 📋 CLAUDE.md                    # Project instructions for Claude Code
├── 📄 README.md                    # Main project documentation
├── 🔒 SECURITY.md                  # Security policies and guidelines
├── ⚙️ pyproject.toml               # Python project configuration
├── 📦 requirements.txt             # Python dependencies (legacy)
├── 🔒 uv.lock                      # UV lock file for dependencies
├── 🐳 Dockerfile                   # Main Docker configuration
├── 🐳 docker-compose.yml           # Docker Compose for development
├── 📁 archive/                     # Archived files and old reports
├── 📁 config/                      # Configuration files
├── 📁 data/                        # Database and persistent data
├── 📁 deployment/                  # Deployment-specific files
├── 📁 docs/                        # Comprehensive documentation
├── 📁 logs/                        # Application logs
├── 📁 output/                      # Generated reports and exports
├── 📁 scripts/                     # Utility scripts
├── 📁 src/                         # Source code (refactored)
├── 📁 tests/                       # Test suite
└── 📁 tools/                       # Development and utility tools
```

## 📖 Documentation Structure (`docs/`)

```
docs/
├── 📄 README.md                    # Website documentation
├── 🌐 index.html                   # Main website interface
├── 📊 data.json                    # Website data file
├── 🚫 404.html                     # Error page
├── 📁 architecture/                # System architecture docs
│   ├── 📊 code_architecture_chart.md
│   ├── 🐍 architecture_diagram.py
│   └── 📋 REFACTORING_SUMMARY.md
├── 📁 setup/                       # Setup and installation guides
│   ├── ⚙️ CLAUDE_CODE_SETUP.md
│   ├── 🔧 GITHUB_ACTION_SETUP.md
│   └── 🌐 GITHUB_PAGES_SETUP.md
├── 📁 deployment/                  # Deployment documentation
└── 📁 development/                 # Development guidelines
```

## 💻 Source Code Structure (`src/`)

```
src/
├── 📄 __init__.py                  # Package initialization
├── 🎯 main.py                      # CLI entry point
├── 📁 clients/                     # AI client implementations
│   ├── 📄 __init__.py
│   ├── 🏗️ base.py                  # Abstract base client
│   ├── 🧠 claude.py                # Anthropic Claude client
│   ├── 🔮 mistral.py               # Mistral AI client
│   ├── 💡 openai.py                # OpenAI client
│   └── 🏭 factory.py               # Client factory pattern
├── 📁 config/                      # Configuration management
│   ├── 📄 __init__.py
│   └── ⚙️ settings.py              # Centralized settings
├── 📁 core/                        # Core business logic
│   ├── 📄 __init__.py
│   ├── 📊 database.py              # Database operations
│   ├── ⚠️ error_logger.py          # Error logging
│   ├── 📋 report_generator.py      # Report generation
│   ├── 📰 rss_parser.py            # RSS feed parsing
│   ├── 🕷️ scraper.py               # Web scraping
│   └── 🔧 utils.py                 # Utility functions
├── 📁 exceptions/                  # Custom exception system
│   ├── 📄 __init__.py
│   └── ⚠️ exceptions.py            # Exception hierarchy
└── 📁 processors/                  # Main processing logic
    ├── 📄 __init__.py
    └── 🏭 article_processor.py     # Main orchestrator
```

## 🧪 Test Structure (`tests/`)

```
tests/
├── 📄 __init__.py                  # Test package init
├── ⚙️ conftest.py                  # Pytest configuration & fixtures
├── 📁 fixtures/                    # Test data fixtures
├── 📁 integration/                 # Integration tests
├── 📁 mocks/                       # Mock objects and data
└── 📁 unit/                        # Unit tests
    ├── 🧪 test_config.py
    ├── 🧪 test_exceptions.py
    ├── 🧪 test_base_ai_client.py
    ├── 🧪 test_ai_client_factory.py
    └── 🧪 test_article_processor.py
```

## 🛠️ Tools and Scripts

### Scripts Directory (`scripts/`)
```
scripts/
├── 📄 README.md                    # Scripts documentation
├── 🔍 check_api_status.sh          # API health check
├── 🔍 check_service_status.sh      # Service monitoring
├── 🧹 lint.sh                      # Code linting
├── ⏰ run_daily.sh                 # Daily automation
├── ⚙️ setup_daily_service.sh       # Service setup
└── 🔧 setup_github_action.sh      # GitHub Actions setup
```

### Tools Directory (`tools/`)
```
tools/
├── 📄 README.md                    # Tools documentation
├── 📄 __init__.py
├── 🏥 api_health_monitor.py        # API monitoring
├── 🔗 claude_code_integration.py   # Claude Code integration
├── 📅 generate_articles_by_date.py # Date-based reports
├── 📊 generate_comprehensive_reports.py # Complete reports
├── 🌐 generate_hugo_content.py     # Hugo website content
├── 📊 generate_website_data.py     # Website data generation
├── ⚡ quick_api_check.py           # Quick API tests
├── 🔗 test_link_following.py       # Link following tests
├── 🔄 update_website_data.py       # Website updates
└── ✅ validate_imports.py          # Import validation
```

## ⚙️ Configuration Structure (`config/`)

```
config/
├── 📋 config.yaml                  # Main configuration file
└── 🧹 ruff.toml                    # Code formatting rules
```

## 📊 Data and Output

### Data Directory (`data/`)
```
data/
└── 📊 articles.db                  # SQLite database
```

### Output Directory (`output/`)
```
output/
├── 📋 articles_export.csv          # Current CSV export
├── 📊 articles_export.json         # Current JSON export
├── 📄 comprehensive_analysis_report.md
├── 📊 database_statistics.json
├── ⚠️ errors.log
├── 📄 feynman_summaries_20250117.md
├── ⚠️ processing_errors.json
├── 📊 processing_status.json
└── 📊 run_summary.json
```

### Archive Directory (`archive/`)
```
archive/
├── 📄 article_analysis_report_*.md  # Historical reports
├── 📊 articles_export_*.csv         # Historical CSV exports
├── 📊 articles_export_*.json        # Historical JSON exports
└── 📄 summary_report_*.md           # Historical summaries
```

## 🚀 Deployment Structure (`deployment/`)

```
deployment/
├── 🐳 Dockerfile.claude            # Claude Code Docker config
├── 🐳 docker-compose.claude.yml    # Claude Code compose
└── ⚙️ setup_claude_code.sh         # Claude Code setup script
```

## 📝 Logs Directory (`logs/`)

```
logs/
├── 📊 analyzer.log                 # Main application logs
├── 🔍 api_status.log              # API monitoring logs
├── ⏰ hourly_runs.log             # Scheduled run logs
└── 🌐 website_generator.log       # Website generation logs
```

## 🏗️ Architecture Principles

### 1. **Separation of Concerns**
- **`src/clients/`**: AI service integration
- **`src/core/`**: Business logic
- **`src/processors/`**: Workflow orchestration
- **`src/config/`**: Configuration management
- **`src/exceptions/`**: Error handling

### 2. **Clean Directory Structure**
- **Configuration**: Centralized in `config/`
- **Documentation**: Organized in `docs/` with subfolders
- **Deployment**: Isolated in `deployment/`
- **Archives**: Historical data in `archive/`
- **Tools**: Development utilities in `tools/`

### 3. **Test Organization**
- **Unit Tests**: Component isolation
- **Integration Tests**: End-to-end workflows
- **Fixtures**: Reusable test data
- **Mocks**: Service simulation

### 4. **Output Management**
- **Current**: Latest reports in `output/`
- **Historical**: Timestamped files in `archive/`
- **Logs**: Separate logging directory
- **Data**: Persistent storage in `data/`

## 📋 File Naming Conventions

### Python Files
- **Snake case**: `article_processor.py`
- **Descriptive names**: `ai_client_factory.py`
- **Module prefixes**: `test_*.py` for tests

### Documentation
- **UPPERCASE**: Important docs (`README.md`, `SECURITY.md`)
- **Descriptive**: `code_architecture_chart.md`
- **Categorized**: Organized in appropriate subdirectories

### Configuration
- **Lowercase**: `config.yaml`
- **Tool-specific**: `ruff.toml`
- **Environment-specific**: `docker-compose.yml`

## 🔄 Maintenance

### Regular Cleanup
1. **Archive old reports** from `output/` to `archive/`
2. **Clean logs** older than 30 days
3. **Review configuration** files for updates
4. **Update documentation** as needed

### Development Workflow
1. **Code changes** in `src/`
2. **Tests** in `tests/`
3. **Documentation** updates in `docs/`
4. **Configuration** adjustments in `config/`

This organized structure promotes maintainability, scalability, and ease of development while following Python best practices and clean architecture principles.
