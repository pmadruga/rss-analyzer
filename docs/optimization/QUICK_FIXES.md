# Quick Fixes - Code Quality Improvements

**Priority List for RSS Analyzer - Immediate Actions**

---

## 🔴 CRITICAL - Do Today

### 1. Remove Duplicate Files (30 minutes)

**Problem:** 2,301 lines duplicated between `src/core/` and `src/etl/`

```bash
# Remove duplicate files
rm src/etl/extract/web_scraper.py
rm src/etl/extract/rss_parser.py
rm src/etl/load/database.py
rm src/etl/load/report_generator.py

# Remove duplicate AI clients
rm -rf src/etl/transform/ai_clients/
```

**Update imports in affected files:**
```python
# Before:
from src.core.scraper import WebScraper
from src.core.rss_parser import RSSParser

# After:
from src.core.scraper import WebScraper
from src.core.rss_parser import RSSParser
```

**Files to update:**
- `src/main_etl.py`
- `src/etl_orchestrator.py`
- Any files in `src/etl/` directories

**Test:**
```bash
# Run tests to ensure nothing broke
docker compose run rss-analyzer sh -c "uv run pytest tests/"
```

---

### 2. Fix Bare Except Clause (5 minutes)

**Location:** `src/etl/load/data_exporter.py`

**Find and replace:**
```python
# Before:
try:
    # operation
    except:
        pass

# After:
try:
    # operation
except (ValueError, KeyError, TypeError) as e:
    logger.error(f"Data export failed: {e}")
    raise
```

---

## 🟡 HIGH PRIORITY - Do This Week

### 3. Create Constants File (30 minutes)

**Create:** `src/core/constants.py`

```python
"""
Constants for RSS Analyzer
Centralized location for all magic numbers and configuration values.
"""

# Content processing
MAX_CONTENT_HASH_LENGTH = 2000
MAX_CONTENT_LENGTH = 50000
MAX_LINKED_ARTICLES = 3

# Title extraction
MIN_TITLE_LENGTH = 10
MAX_TITLE_LENGTH = 150
MAX_TITLE_EXTRACT_LINES = 10

# Caching
DEFAULT_CACHE_CAPACITY = 100_000
CACHE_CLEANUP_HOURS = 24

# Scraping
DEFAULT_SCRAPER_DELAY = 1.0
DEFAULT_REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0
RATE_LIMIT_DELAY = 1.0

# Database
MAX_LOG_AGE_DAYS = 30
BATCH_SIZE = 100

# Content extraction
CONTENT_PREVIEW_LENGTH = 500
CONTENT_TRUNCATE_LENGTH = 2000
```

**Update files to use constants:**
```python
# Before:
if len(title) > 10 and len(title) < 150:

# After:
from src.core.constants import MIN_TITLE_LENGTH, MAX_TITLE_LENGTH
if MIN_TITLE_LENGTH < len(title) < MAX_TITLE_LENGTH:
```

---

### 4. Add Type Hints to Statistics Methods (15 minutes)

**Create:** `src/core/types.py`

```python
"""Type definitions for RSS Analyzer"""
from typing import TypedDict, Any

class ProcessingStats(TypedDict):
    """Processing statistics structure"""
    total_articles: int
    by_status: dict[str, int]
    recent_activity: list[dict[str, Any]]

class ArticleMetadata(TypedDict, total=False):
    """Article metadata structure"""
    domain: str
    author: str
    publication_date: str
    description: str
    linked_articles: list[str]
    ai_provider: str
```

**Update database.py:**
```python
from .types import ProcessingStats

def get_processing_statistics(self) -> ProcessingStats:
    """Get processing statistics"""
    ...
```

---

### 5. Document Title Override System (10 minutes)

**Update:** `src/etl/load/website_generator.py`

**Add comprehensive comment before title_overrides dict:**

```python
# TITLE OVERRIDE SYSTEM
#
# This dictionary provides manual title corrections when automatic extraction fails.
# Common scenarios requiring overrides:
#
# 1. RSS feeds with generic titles (e.g., usernames like "@user.bsky.social")
# 2. Web scraping extracts page titles instead of article titles
# 3. AI analysis produces generic headers like "Analysis Using Feynman Technique"
#
# HOW TO ADD AN OVERRIDE:
# 1. Find the article ID in the database:
#    docker compose run rss-analyzer sqlite3 /app/data/articles.db \
#      "SELECT id, title FROM articles WHERE title LIKE '%generic%';"
# 2. Add entry below: article_id: "Proper Title Here"
# 3. Regenerate website data: docker compose run rss-analyzer sh -c \
#      "uv run python src/etl/load/website_generator.py"
#
# TODO: Consider moving to database table for runtime updates
title_overrides = {
    44: "CRUX: Enhanced Evaluation Metrics for Long-Form RAG Systems",
    # ... etc
}
```

---

## 🟢 MEDIUM PRIORITY - Do Next Week

### 6. Add Input Validation Helper (1 hour)

**Create:** `src/core/validation.py`

```python
"""Input validation utilities"""
from typing import Any

class ValidationError(Exception):
    """Validation error"""
    pass

def validate_non_empty_string(value: Any, field_name: str) -> str:
    """
    Validate that value is a non-empty string.

    Args:
        value: Value to validate
        field_name: Field name for error messages

    Returns:
        Stripped string value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string, got {type(value).__name__}")

    stripped = value.strip()
    if not stripped:
        raise ValidationError(f"{field_name} cannot be empty")

    return stripped

def validate_url(url: Any, field_name: str = "URL") -> str:
    """
    Validate that value is a valid URL.

    Args:
        url: URL to validate
        field_name: Field name for error messages

    Returns:
        Validated URL

    Raises:
        ValidationError: If validation fails
    """
    url = validate_non_empty_string(url, field_name)

    from urllib.parse import urlparse
    parsed = urlparse(url)

    if not all([parsed.scheme, parsed.netloc]):
        raise ValidationError(f"{field_name} must be a valid URL with scheme and domain")

    if parsed.scheme not in ('http', 'https'):
        raise ValidationError(f"{field_name} must use http or https protocol")

    return url

def validate_positive_int(value: Any, field_name: str, min_value: int = 1) -> int:
    """
    Validate that value is a positive integer.

    Args:
        value: Value to validate
        field_name: Field name for error messages
        min_value: Minimum allowed value (default: 1)

    Returns:
        Validated integer

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer, got {type(value).__name__}")

    if value < min_value:
        raise ValidationError(f"{field_name} must be at least {min_value}, got {value}")

    return value
```

**Usage example:**
```python
from src.core.validation import validate_non_empty_string, validate_url

def _extract_better_title(self, analysis: str, url: str, original_title: str) -> str:
    """Extract better title with validation"""
    # Validate inputs
    analysis = validate_non_empty_string(analysis, "analysis") if analysis else ""
    url = validate_url(url, "url")
    original_title = validate_non_empty_string(original_title, "original_title")

    # ... rest of logic
```

---

### 7. Add Circular Dependency Check (15 minutes)

**Create:** `.github/workflows/code-quality.yml`

```yaml
name: Code Quality Checks

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  code-quality:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pydeps

      - name: Check circular dependencies
        run: |
          pydeps src --max-bacon 2 --show-dot > deps.dot
          if grep -q "cycle" deps.dot; then
            echo "❌ Circular dependencies detected!"
            cat deps.dot
            exit 1
          else
            echo "✅ No circular dependencies found"
          fi

      - name: Check for duplicate code
        run: |
          # Check for exact file duplication
          find src -type f -name "*.py" -exec md5sum {} + | sort | uniq -d -w32
          if [ $? -eq 0 ]; then
            echo "⚠️  Potential duplicate files detected"
          fi
```

---

## 📊 Verification Checklist

After completing quick fixes, verify:

- [ ] All tests pass: `docker compose run rss-analyzer sh -c "uv run pytest tests/"`
- [ ] No duplicate files: `find src -type f -name "*.py" -exec md5sum {} + | sort | uniq -d -w32`
- [ ] No bare except: `grep -r "except:" src/ --include="*.py"`
- [ ] Type hints complete: Manual review of modified files
- [ ] Documentation updated: Check CLAUDE.md and docstrings
- [ ] Git diff reasonable: `git diff --stat`

---

## 🎯 Expected Impact

After completing all quick fixes:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Code Duplication | 21% | <1% | 95% reduction |
| Type Safety | 85% | 95% | +10% |
| Maintainability | 7.5/10 | 8.5/10 | +1.0 |
| Documentation | 80% | 90% | +10% |

---

## 📝 Commit Messages

Use these standardized commit messages:

```bash
# After removing duplicates
git commit -m "refactor: eliminate file duplication in core and ETL modules

- Remove duplicate files in src/etl/ (2,301 lines)
- Update all imports to use src/core/
- Maintain backward compatibility with import aliases
- Tests pass, no functionality changed"

# After fixing bare except
git commit -m "fix: replace bare except clause with specific exceptions

- Update data_exporter.py with proper exception handling
- Add logging for error context
- Prevents silent failures"

# After adding constants
git commit -m "refactor: extract magic numbers to constants.py

- Create src/core/constants.py with all constants
- Update all files to use named constants
- Improves maintainability and reduces errors"
```

---

## ⚡ Speed Tips

**Parallel execution:**
```bash
# While tests are running, work on next fix
docker compose run rss-analyzer sh -c "uv run pytest tests/" &
# Work on constants.py in another terminal
```

**Use search and replace:**
```bash
# Find all magic numbers
grep -rn "[:=] [0-9]\{2,\}" src/ --include="*.py"

# Find all magic strings
grep -rn '["\']\w\{10,\}["\']' src/ --include="*.py"
```

**Auto-format after changes:**
```bash
docker compose run rss-analyzer sh -c "uv run black src/"
docker compose run rss-analyzer sh -c "uv run isort src/"
```

---

**Last Updated:** 2025-10-12
**Next Review:** After completing Priority 1 fixes
