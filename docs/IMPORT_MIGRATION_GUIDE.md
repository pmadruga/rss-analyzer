# Import Migration Guide

**Date:** 2025-10-12
**Purpose:** Guide for updating imports after duplicate code removal

---

## Overview

This guide helps you migrate code that previously imported from the `src/etl/` directory to use the consolidated modules in `src/core/` and `src/clients/`.

## Quick Reference Table

| Old Import (Removed) | New Import (Use This) |
|---------------------|----------------------|
| `from src.etl.extract.web_scraper import WebScraper` | `from src.core.scraper import WebScraper` |
| `from src.etl.extract.rss_parser import RSSParser` | `from src.core.rss_parser import RSSParser` |
| `from src.etl.load.database import DatabaseManager` | `from src.core.database import DatabaseManager` |
| `from src.etl.load.report_generator import ReportGenerator` | `from src.core.report_generator import ReportGenerator` |
| `from src.etl.transform.ai_clients.factory import AIClientFactory` | `from src.clients.factory import AIClientFactory` |
| `from src.etl.transform.ai_clients.claude import ClaudeClient` | `from src.clients.claude import ClaudeClient` |
| `from src.etl.transform.ai_clients.mistral import MistralClient` | `from src.clients.mistral import MistralClient` |
| `from src.etl.transform.ai_clients.openai import OpenAIClient` | `from src.clients.openai import OpenAIClient` |
| `from src.etl.transform.ai_clients.base import BaseAIClient` | `from src.clients.base import BaseAIClient` |
| `from src.config.settings import load_config, setup_logging` | `from src.core import load_config, setup_logging` |

## Detailed Migration Examples

### Example 1: RSS Parsing

**Before:**
```python
from src.etl.extract.rss_parser import RSSParser, RSSEntry

parser = RSSParser()
entries = parser.fetch_feed("https://example.com/rss")
```

**After:**
```python
from src.core.rss_parser import RSSParser, RSSEntry

parser = RSSParser()
entries = parser.fetch_feed("https://example.com/rss")
```

### Example 2: Web Scraping

**Before:**
```python
from src.etl.extract.web_scraper import WebScraper, ArticleContent

scraper = WebScraper(delay_between_requests=1.0)
content = scraper.scrape_article("https://example.com/article")
```

**After:**
```python
from src.core.scraper import WebScraper, ArticleContent

scraper = WebScraper(delay_between_requests=1.0)
content = scraper.scrape_article("https://example.com/article")
```

### Example 3: Database Operations

**Before:**
```python
from src.etl.load.database import DatabaseManager

db = DatabaseManager("data/articles.db")
articles = db.get_all_articles()
```

**After:**
```python
from src.core.database import DatabaseManager

db = DatabaseManager("data/articles.db")
articles = db.get_all_articles()
```

### Example 4: AI Client Usage

**Before:**
```python
from src.etl.transform.ai_clients.factory import AIClientFactory
from src.etl.transform.ai_clients.claude import ClaudeClient

factory = AIClientFactory()
client = factory.create_client("anthropic", config)
```

**After:**
```python
from src.clients.factory import AIClientFactory
from src.clients.claude import ClaudeClient

# Simplified factory usage
client = AIClientFactory.create_from_config(config)

# Or create directly
client = ClaudeClient(api_key="your-key")
```

### Example 5: Configuration Loading

**Before:**
```python
from src.config.settings import load_config, setup_logging, validate_config

config = load_config("config.yaml")
setup_logging(config["log_level"])
```

**After:**
```python
from src.core import load_config, setup_logging, validate_config

config = load_config("config.yaml")
setup_logging(config["log_level"])
```

### Example 6: Report Generation

**Before:**
```python
from src.etl.load.report_generator import ReportGenerator

generator = ReportGenerator("output/")
generator.generate_markdown_report(articles)
```

**After:**
```python
from src.core.report_generator import ReportGenerator

generator = ReportGenerator("output/")
generator.generate_markdown_report(articles)
```

## Batch Migration Script

Use this script to automatically update imports in your Python files:

```bash
#!/bin/bash
# migrate_imports.sh - Update imports to use consolidated modules

# Find all Python files (excluding venv, .git, etc.)
find . -name "*.py" -not -path "*/venv/*" -not -path "*/.git/*" | while read file; do
    echo "Processing: $file"

    # Backup original file
    cp "$file" "$file.bak"

    # Update imports
    sed -i 's|from src.etl.extract.web_scraper|from src.core.scraper|g' "$file"
    sed -i 's|from src.etl.extract.rss_parser|from src.core.rss_parser|g' "$file"
    sed -i 's|from src.etl.load.database|from src.core.database|g' "$file"
    sed -i 's|from src.etl.load.report_generator|from src.core.report_generator|g' "$file"
    sed -i 's|from src.etl.transform.ai_clients|from src.clients|g' "$file"
    sed -i 's|from src.config.settings import load_config|from src.core import load_config|g' "$file"

    echo "Updated: $file"
done

echo "Migration complete! Review changes and remove .bak files if satisfied."
```

Usage:
```bash
chmod +x migrate_imports.sh
./migrate_imports.sh
```

## Python Migration Script

For more precise control, use this Python script:

```python
#!/usr/bin/env python3
"""
migrate_imports.py - Migrate imports from src.etl to src.core/src.clients
"""

import re
from pathlib import Path
from typing import Dict, List

# Define import mapping
IMPORT_MAPPINGS: Dict[str, str] = {
    r'from src\.etl\.extract\.web_scraper': 'from src.core.scraper',
    r'from src\.etl\.extract\.rss_parser': 'from src.core.rss_parser',
    r'from src\.etl\.load\.database': 'from src.core.database',
    r'from src\.etl\.load\.report_generator': 'from src.core.report_generator',
    r'from src\.etl\.transform\.ai_clients\.factory': 'from src.clients.factory',
    r'from src\.etl\.transform\.ai_clients\.claude': 'from src.clients.claude',
    r'from src\.etl\.transform\.ai_clients\.mistral': 'from src.clients.mistral',
    r'from src\.etl\.transform\.ai_clients\.openai': 'from src.clients.openai',
    r'from src\.etl\.transform\.ai_clients\.base': 'from src.clients.base',
}


def migrate_file(file_path: Path) -> bool:
    """Migrate imports in a single file."""
    try:
        content = file_path.read_text()
        original_content = content

        # Apply all import mappings
        for old_pattern, new_import in IMPORT_MAPPINGS.items():
            content = re.sub(old_pattern, new_import, content)

        # Only write if changes were made
        if content != original_content:
            # Create backup
            backup_path = file_path.with_suffix('.py.bak')
            backup_path.write_text(original_content)

            # Write updated content
            file_path.write_text(content)
            print(f"✅ Updated: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def main():
    """Main migration function."""
    print("🔄 Starting import migration...\n")

    # Find all Python files
    project_root = Path.cwd()
    python_files = list(project_root.rglob("*.py"))

    # Exclude certain directories
    exclude_dirs = {'venv', '.git', '__pycache__', 'node_modules'}
    python_files = [
        f for f in python_files
        if not any(part in exclude_dirs for part in f.parts)
    ]

    print(f"Found {len(python_files)} Python files to check\n")

    # Migrate each file
    updated_count = 0
    for file_path in python_files:
        if migrate_file(file_path):
            updated_count += 1

    print(f"\n✨ Migration complete!")
    print(f"📊 Updated {updated_count} file(s)")
    print(f"\n💡 Tip: Review changes and run tests before removing .bak files")


if __name__ == "__main__":
    main()
```

Usage:
```bash
python migrate_imports.py
```

## Verification

After migration, verify your changes:

### 1. Check for Remaining Old Imports
```bash
# Search for any remaining src.etl imports
grep -r "from src.etl" --include="*.py" .
grep -r "import src.etl" --include="*.py" .
```

### 2. Verify Syntax
```bash
# Check all Python files compile
find . -name "*.py" -exec python3 -m py_compile {} \;
```

### 3. Run Tests
```bash
# Run your test suite
pytest tests/

# Or specific tests
python -m pytest tests/unit/
```

### 4. Test Import Paths
```bash
# Test critical imports
python3 -c "from src.core import DatabaseManager; print('✅ DatabaseManager')"
python3 -c "from src.core import RSSParser; print('✅ RSSParser')"
python3 -c "from src.core import WebScraper; print('✅ WebScraper')"
python3 -c "from src.clients import AIClientFactory; print('✅ AIClientFactory')"
```

## Common Issues and Solutions

### Issue 1: Import Not Found
**Error:** `ModuleNotFoundError: No module named 'src.etl'`

**Solution:** The import path is using the old structure. Update to new import:
```python
# Change from:
from src.etl.extract.web_scraper import WebScraper
# To:
from src.core.scraper import WebScraper
```

### Issue 2: Circular Import
**Error:** `ImportError: cannot import name 'X' from partially initialized module`

**Solution:** This shouldn't occur with the new structure, but if it does:
1. Check for circular dependencies in your code
2. Use lazy imports if necessary
3. Restructure code to avoid circular dependencies

### Issue 3: Missing Dependencies
**Error:** `ModuleNotFoundError: No module named 'anthropic'`

**Solution:** Install dependencies:
```bash
uv sync
# or
pip install -r requirements.txt
```

## Support

If you encounter issues not covered in this guide:

1. Check the main documentation: `docs/DUPLICATE_CODE_REMOVAL.md`
2. Review the codebase structure in `src/core/` and `src/clients/`
3. Look at working examples in `src/main.py` and `src/etl_orchestrator.py`
4. Check Git history: `git log --follow -- src/etl_orchestrator.py`

## Rollback Instructions

If you need to rollback changes:

```bash
# Restore from backup
find . -name "*.py.bak" | while read backup; do
    original="${backup%.bak}"
    mv "$backup" "$original"
    echo "Restored: $original"
done

# Or restore from Git
git checkout HEAD -- src/
```

---

**Last Updated:** 2025-10-12
**Status:** Complete - All duplicate code removed
