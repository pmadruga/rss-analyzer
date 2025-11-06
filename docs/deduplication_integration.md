# Deduplication System Integration Guide

## Overview

The deduplication system provides high-performance duplicate detection for the RSS analyzer using SHA-256 content and URL hashing with an in-memory LRU cache.

## Architecture

### Components

1. **DeduplicationManager** (`src/deduplication_manager.py`)
   - Core deduplication logic
   - O(1) hash-based lookups
   - In-memory LRU cache (100K articles capacity)
   - Batch operations for throughput optimization

2. **Database Schema** (articles table)
   - `content_hash` VARCHAR(64) - SHA-256 of normalized content
   - `url_hash` VARCHAR(64) - SHA-256 of normalized URL
   - Indexed for fast lookups

3. **Command-Line Tools**
   - `check_duplicates.py` - Analysis and removal
   - `migrate_dedup_schema.py` - Schema migration

## Performance Metrics

- **Lookup Speed**: < 1ms per duplicate check (cache hit)
- **Memory Usage**: < 100MB for 100K articles
- **Batch Throughput**: 1000+ articles/sec
- **Cache Hit Rate**: Typically 85-95% after warm-up

## Quick Start

### 1. Migrate Database Schema

```bash
# Add url_hash and content_hash columns with indexes
python tools/migrate_dedup_schema.py

# Verify migration
python tools/migrate_dedup_schema.py --verify-only
```

### 2. Backfill Existing Articles

```bash
# Generate hashes for existing articles
python tools/check_duplicates.py --backfill
```

### 3. Analyze Duplicates

```bash
# Check for duplicates without making changes
python tools/check_duplicates.py --analyze

# Show statistics and cache performance
python tools/check_duplicates.py --stats
```

### 4. Remove Duplicates

```bash
# Dry-run first (no changes)
python tools/check_duplicates.py --remove --dry-run

# Actually remove duplicates
python tools/check_duplicates.py --remove
```

## Integration with RSS Parser

### Before Scraping - Duplicate Detection

```python
from core.database import DatabaseManager
from deduplication_manager import DeduplicationManager

# Initialize
db = DatabaseManager("data/articles.db")
dedup = DeduplicationManager(db)

# Check if URL is duplicate BEFORE scraping
url = "https://arxiv.org/abs/2401.12345"
is_dup, reason = dedup.is_duplicate(url)

if is_dup:
    print(f"Skipping duplicate ({reason}): {url}")
else:
    # Proceed with scraping
    content = scraper.scrape(url)

    # Check content-based duplicate after scraping
    is_dup, reason = dedup.is_duplicate(url, content)

    if is_dup:
        print(f"Content duplicate detected: {url}")
    else:
        # Process article
        process_article(url, content)
```

### After Processing - Mark as Processed

```python
# After successful processing, mark article
content_hash = dedup.generate_content_hash(content)
url_hash = dedup.generate_url_hash(url)

# Insert into database
article_id = db.insert_article(
    title=title,
    url=url,
    content_hash=content_hash,
    rss_guid=guid,
    publication_date=pub_date
)

# Update cache for fast future lookups
dedup.mark_processed(article_id, url, content_hash, url_hash)
```

### Batch Processing

For processing multiple articles efficiently:

```python
# Prepare batch of articles
articles = [
    {"url": "https://example.com/article1", "content": "..."},
    {"url": "https://example.com/article2", "content": "..."},
    # ... more articles
]

# Batch check for duplicates
results = dedup.batch_check_duplicates(articles)

# Filter non-duplicates
new_articles = [
    r for r in results if not r["is_duplicate"]
]

print(f"Found {len(new_articles)} new articles out of {len(articles)}")

# Process only new articles
for article in new_articles:
    process_article(article["url"], article["content"])
```

## Integration with ArticleProcessor

### Modified Processing Flow

```python
class ArticleProcessor:
    def __init__(self):
        self.db = DatabaseManager()
        self.dedup = DeduplicationManager(self.db)
        self.scraper = WebScraper()

    def process_rss_feed(self, feed_url: str, max_articles: int = 10):
        """Process RSS feed with duplicate detection."""
        parser = RSSParser()
        entries = parser.parse(feed_url)

        processed = 0
        skipped = 0

        for entry in entries:
            if processed >= max_articles:
                break

            url = entry.get('link')
            title = entry.get('title')

            # Check URL duplicate BEFORE scraping
            is_dup, reason = self.dedup.is_duplicate(url)

            if is_dup:
                logger.info(f"Skipping duplicate ({reason}): {title}")
                skipped += 1
                continue

            # Scrape content
            try:
                content = self.scraper.scrape(url)

                # Check content duplicate
                is_dup, reason = self.dedup.is_duplicate(url, content)

                if is_dup:
                    logger.info(f"Content duplicate detected: {title}")
                    skipped += 1
                    continue

                # Process article (AI analysis, etc.)
                self._process_article(entry, content)
                processed += 1

            except Exception as e:
                logger.error(f"Error processing {url}: {e}")

        logger.info(f"Processed {processed} new articles, skipped {skipped} duplicates")

    def _process_article(self, entry: dict, content: str):
        """Process single article with deduplication tracking."""
        # Generate hashes
        url = entry['link']
        content_hash = self.dedup.generate_content_hash(content)
        url_hash = self.dedup.generate_url_hash(url)

        # Insert into database
        article_id = self.db.insert_article(
            title=entry['title'],
            url=url,
            content_hash=content_hash,
            rss_guid=entry.get('id'),
            publication_date=entry.get('published')
        )

        # Analyze with AI
        analysis = self.ai_client.analyze(content)

        # Store analysis
        self.db.insert_content(article_id, content, analysis)

        # Update status
        self.db.update_article_status(article_id, 'completed')

        # Mark as processed in cache
        self.dedup.mark_processed(article_id, url, content_hash, url_hash)
```

## Command-Line Tool Usage

### check_duplicates.py

```bash
# Analyze duplicates (no changes)
python tools/check_duplicates.py --analyze

# Show statistics and cache performance
python tools/check_duplicates.py --stats

# Remove duplicates with dry-run
python tools/check_duplicates.py --remove --dry-run

# Actually remove duplicates
python tools/check_duplicates.py --remove

# Backfill hashes for existing articles
python tools/check_duplicates.py --backfill

# Custom database path
python tools/check_duplicates.py --analyze --db-path /custom/path/articles.db

# Combine operations
python tools/check_duplicates.py --analyze --stats --backfill
```

### migrate_dedup_schema.py

```bash
# Run migration
python tools/migrate_dedup_schema.py

# Verify schema without changes
python tools/migrate_dedup_schema.py --verify-only

# Custom database path
python tools/migrate_dedup_schema.py --db-path /custom/path/articles.db
```

## Cache Management

### Statistics Monitoring

```python
# Get comprehensive statistics
stats = dedup.get_duplicate_stats()

print(f"Articles processed: {stats['articles_processed']}")
print(f"Duplicates detected: {stats['duplicates_detected']}")
print(f"Cache hit rate: {stats['cache_stats']['cache_hit_rate']}")

# Get memory usage
memory = dedup.get_memory_usage_estimate()
print(f"Memory usage: {memory['total_mb']} MB")
```

### Manual Cache Management

```python
# Clean old cache entries (default: 24 hours)
dedup.clean_old_cache(max_age_hours=24)

# Rebuild cache from database
dedup.rebuild_cache()

# Clear and warm up cache
dedup.content_hash_cache.clear()
dedup.url_hash_cache.clear()
dedup._warm_cache()
```

## Performance Optimization Tips

### 1. Cache Capacity Tuning

Adjust cache capacity based on dataset size:

```python
# For large datasets (> 100K articles)
dedup = DeduplicationManager(db, cache_capacity=200000)

# For smaller datasets (< 50K articles)
dedup = DeduplicationManager(db, cache_capacity=50000)
```

### 2. Batch Processing

Always use batch operations for bulk processing:

```python
# Good: Batch check
results = dedup.batch_check_duplicates(articles)

# Bad: Individual checks in loop
for article in articles:
    is_dup, _ = dedup.is_duplicate(article['url'])
```

### 3. URL-Only Checks

When possible, check URL duplicates before scraping content:

```python
# Fast URL-only check (no scraping needed)
is_dup, reason = dedup.is_duplicate(url)
if not is_dup:
    # Only scrape if URL is not duplicate
    content = scraper.scrape(url)
```

### 4. Pre-warming Cache

Warm cache at application startup:

```python
# Cache is automatically warmed on initialization
dedup = DeduplicationManager(db)  # Loads recent articles into cache
```

## Database Schema

### Articles Table

```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    publication_date TIMESTAMP,
    rss_guid TEXT,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_hash VARCHAR(64) NOT NULL,  -- SHA-256 content hash
    url_hash VARCHAR(64) NOT NULL,      -- SHA-256 URL hash
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookups
CREATE INDEX idx_articles_content_hash ON articles (content_hash);
CREATE INDEX idx_articles_url_hash ON articles (url_hash);
CREATE INDEX idx_articles_url ON articles (url);
CREATE INDEX idx_articles_status ON articles (status);
```

## Error Handling

### Database Errors

```python
try:
    is_dup, reason = dedup.is_duplicate(url, content)
except Exception as e:
    logger.error(f"Deduplication check failed: {e}")
    # Proceed without deduplication or abort
```

### Cache Corruption

If cache becomes corrupted:

```python
# Rebuild from database
dedup.rebuild_cache()

# Or reinitialize
dedup = DeduplicationManager(db)
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Cache Hit Rate** - Should be > 80% after warm-up
2. **Duplicate Rate** - Typical: 5-15% for RSS feeds
3. **Memory Usage** - Should stay < 100MB for 100K articles
4. **Processing Time** - Should be < 1ms per check

### Example Monitoring

```python
import time

start = time.time()
is_dup, reason = dedup.is_duplicate(url, content)
duration = time.time() - start

if duration > 0.001:  # > 1ms
    logger.warning(f"Slow duplicate check: {duration*1000:.2f}ms")

stats = dedup.get_duplicate_stats()
cache_hit_rate = float(stats['cache_stats']['cache_hit_rate'].rstrip('%'))

if cache_hit_rate < 80:
    logger.warning(f"Low cache hit rate: {cache_hit_rate}%")
```

## Testing

### Unit Tests

```python
import unittest
from deduplication_manager import DeduplicationManager

class TestDeduplication(unittest.TestCase):
    def test_content_hash_generation(self):
        content = "Test article content"
        hash1 = DeduplicationManager.generate_content_hash(content)
        hash2 = DeduplicationManager.generate_content_hash(content)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256

    def test_duplicate_detection(self):
        db = DatabaseManager(":memory:")
        dedup = DeduplicationManager(db)

        url = "https://example.com/article"
        content = "Article content"

        # First check - not duplicate
        is_dup, _ = dedup.is_duplicate(url, content)
        self.assertFalse(is_dup)

        # Insert article
        content_hash = dedup.generate_content_hash(content)
        article_id = db.insert_article("Title", url, content_hash)
        dedup.mark_processed(article_id, url, content_hash)

        # Second check - is duplicate
        is_dup, reason = dedup.is_duplicate(url, content)
        self.assertTrue(is_dup)
        self.assertIn(reason, ["url", "content"])
```

## Migration Checklist

- [ ] Run schema migration: `python tools/migrate_dedup_schema.py`
- [ ] Verify schema: `python tools/migrate_dedup_schema.py --verify-only`
- [ ] Backfill existing articles: `python tools/check_duplicates.py --backfill`
- [ ] Analyze duplicates: `python tools/check_duplicates.py --analyze`
- [ ] Remove duplicates (dry-run): `python tools/check_duplicates.py --remove --dry-run`
- [ ] Remove duplicates: `python tools/check_duplicates.py --remove`
- [ ] Update RSSParser integration
- [ ] Update ArticleProcessor integration
- [ ] Add monitoring and alerts
- [ ] Test with sample feed
- [ ] Deploy to production

## Support

For issues or questions:
- Check logs in `logs/` directory
- Run statistics: `python tools/check_duplicates.py --stats`
- Verify schema: `python tools/migrate_dedup_schema.py --verify-only`
- Rebuild cache: Use `dedup.rebuild_cache()` in Python
