# Project Structure

This document outlines the clean, professional structure of the RSS Analyzer project.

## 📁 Directory Organization

```
rss-analyzer/
├── 📋 Project Documentation
│   ├── README.md                    # Main project documentation
│   ├── CLAUDE.md                    # Claude Code instructions (ignored)
│   ├── GITHUB_ACTION_SETUP.md       # GitHub Actions setup guide
│   ├── SECURITY.md                  # Security documentation
│   └── PROJECT_STRUCTURE.md         # This file
│
├── 🏗️ Core Application
│   └── src/                         # Main application source code
│       ├── __init__.py              # Package initialization
│       ├── main.py                  # Application entry point
│       ├── database.py              # Database operations
│       ├── rss_parser.py            # RSS feed parsing
│       ├── scraper.py               # Web scraping functionality
│       ├── report_generator.py      # Report generation
│       ├── utils.py                 # Utility functions
│       ├── claude_client.py         # Anthropic Claude API client
│       ├── mistral_client.py        # Mistral AI API client
│       └── openai_client.py         # OpenAI API client
│
├── 🛠️ Tools & Utilities
│   └── tools/                       # Python utility scripts
│       ├── __init__.py              # Tools package initialization
│       ├── README.md                # Tools documentation
│       ├── generate_website_data.py # Website data generator
│       ├── generate_articles_by_date.py # Date-based article generator
│       ├── generate_comprehensive_reports.py # Report generator
│       └── test_link_following.py   # Link following tests
│
├── 📜 Scripts
│   └── scripts/                     # Shell scripts for operations
│       ├── README.md                # Scripts documentation
│       ├── lint.sh                  # Code quality checks
│       ├── check_service_status.sh  # Service monitoring
│       ├── setup_daily_service.sh   # Daily service setup
│       ├── run_daily.sh             # Daily execution
│       └── setup_github_action.sh   # GitHub Actions setup
│
├── ⚙️ Configuration
│   └── config/                      # Configuration files
│       ├── config.yaml              # Main application config
│       └── ruff.toml                # Python linting config
│
├── 🌐 Website
│   └── docs/                        # GitHub Pages website
│       ├── README.md                # Website documentation
│       ├── index.html               # Main website page
│       ├── styles.css               # Website styling
│       ├── script.js                # Website functionality
│       └── data.json                # Article data (auto-generated)
│
├── 🔄 Automation
│   └── .github/
│       └── workflows/               # GitHub Actions workflows
│           ├── rss-analyzer.yml     # Main RSS processing
│           ├── update-website.yml   # Website data updates
│           ├── deploy-pages.yml     # GitHub Pages deployment
│           └── test-rss-analyzer.yml # Testing workflow
│
├── 📊 Data & Output
│   ├── data/                        # Database and data files
│   │   └── articles.db              # SQLite database (ignored)
│   ├── output/                      # Generated reports
│   │   ├── articles_by_date.md      # Date-organized articles
│   │   └── run_summary.json         # Execution summary
│   └── logs/                        # Log files (ignored)
│
├── 🐳 Docker Configuration
│   ├── Dockerfile                   # Container definition
│   ├── docker-compose.yml           # Multi-container setup
│   └── requirements.txt             # Python dependencies
│
└── 📝 Git Configuration
    └── .gitignore                   # Git ignore rules
```

## 🎯 Design Principles

### 1. **Separation of Concerns**
- **src/**: Core application logic
- **tools/**: Utility scripts and generators
- **scripts/**: Shell scripts for operations
- **config/**: Configuration management
- **docs/**: User-facing website

### 2. **Clear Naming Conventions**
- Descriptive directory names
- Consistent file naming patterns
- README files in each major directory

### 3. **Scalability**
- Modular structure for easy extension
- Clean interfaces between components
- Standardized configuration management

### 4. **Professional Standards**
- Comprehensive documentation
- Automated testing and deployment
- Security best practices

## 🚀 Quick Navigation

### Core Development
- **Main Application**: `src/main.py`
- **Configuration**: `config/config.yaml`
- **Dependencies**: `requirements.txt`

### Tools & Utilities
- **Website Generator**: `tools/generate_website_data.py`
- **Report Generator**: `tools/generate_comprehensive_reports.py`
- **Testing Tools**: `tools/test_link_following.py`

### Operations
- **Setup Scripts**: `scripts/setup_*.sh`
- **Monitoring**: `scripts/check_service_status.sh`
- **Linting**: `scripts/lint.sh`

### Website
- **Frontend**: `docs/index.html`, `docs/styles.css`, `docs/script.js`
- **Data**: `docs/data.json` (auto-generated)
- **Documentation**: `docs/README.md`

### Automation
- **CI/CD**: `.github/workflows/`
- **Docker**: `Dockerfile`, `docker-compose.yml`

## 🔧 Development Workflow

1. **Core Changes**: Edit files in `src/`
2. **Tool Development**: Add/modify files in `tools/`
3. **Operations**: Update scripts in `scripts/`
4. **Website Updates**: Modify files in `docs/`
5. **Configuration**: Edit files in `config/`

## 📈 Benefits of This Structure

### ✅ **Maintainability**
- Clear separation of responsibilities
- Easy to locate specific functionality
- Consistent organization patterns

### ✅ **Scalability**
- Easy to add new tools and scripts
- Modular architecture supports growth
- Clean interfaces between components

### ✅ **Professional Quality**
- Industry-standard project organization
- Comprehensive documentation
- Automated quality assurance

### ✅ **Ease of Use**
- Intuitive directory structure
- Clear documentation in each section
- Standardized command patterns

---

*This structure follows modern software development best practices and provides a solid foundation for continued development and maintenance.*
