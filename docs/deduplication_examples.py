#!/usr/bin/env python3
"""
Deduplication Integration Examples

Practical examples of integrating the deduplication system with various
components of the RSS analyzer.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.database import DatabaseManager
from deduplication_manager import DeduplicationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Basic Duplicate Detection
def example_basic_duplicate_detection():
    """Basic duplicate detection workflow."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Duplicate Detection")
    print("=" * 80)

    # Initialize
    db = DatabaseManager("data/articles.db")
    dedup = DeduplicationManager(db)

    # Sample article
    url = "https://arxiv.org/abs/2401.12345"
    title = "Example Research Paper"
    content = "This is the article content..."

    # Check for duplicate
    is_duplicate, reason = dedup.is_duplicate(url, content)

    if is_duplicate:
        print(f"✗ Duplicate detected ({reason}): {title}")
        print(f"  URL: {url}")
    else:
        print(f"✓ New article: {title}")
        print(f"  URL: {url}")

        # Generate hashes
        content_hash = dedup.generate_content_hash(content)
        url_hash = dedup.generate_url_hash(url)

        print(f"  Content hash: {content_hash[:16]}...")
        print(f"  URL hash: {url_hash[:16]}...")

        # Insert into database
        article_id = db.insert_article(title, url, content_hash)
        print(f"  Inserted with ID: {article_id}")

        # Mark as processed
        dedup.mark_processed(article_id, url, content_hash, url_hash)
        print(f"  Marked as processed in cache")


# Example 2: RSS Feed Processing with Deduplication
def example_rss_feed_processing():
    """Process RSS feed entries with duplicate detection."""
    print("\n" + "=" * 80)
    print("Example 2: RSS Feed Processing")
    print("=" * 80)

    db = DatabaseManager("data/articles.db")
    dedup = DeduplicationManager(db)

    # Simulated RSS feed entries
    feed_entries = [
        {
            "title": "Paper 1: Machine Learning",
            "link": "https://arxiv.org/abs/2401.00001",
            "content": "Content about machine learning..."
        },
        {
            "title": "Paper 2: Neural Networks",
            "link": "https://arxiv.org/abs/2401.00002",
            "content": "Content about neural networks..."
        },
        {
            "title": "Paper 1: Machine Learning",  # Duplicate
            "link": "https://arxiv.org/abs/2401.00001",
            "content": "Content about machine learning..."
        },
    ]

    processed = 0
    skipped = 0

    for entry in feed_entries:
        url = entry['link']
        title = entry['title']

        # Check URL duplicate BEFORE scraping
        is_duplicate, reason = dedup.is_duplicate(url)

        if is_duplicate:
            print(f"✗ Skipping duplicate ({reason}): {title}")
            skipped += 1
            continue

        # In real implementation, scrape content here
        content = entry['content']

        # Check content duplicate
        is_duplicate, reason = dedup.is_duplicate(url, content)

        if is_duplicate:
            print(f"✗ Skipping content duplicate: {title}")
            skipped += 1
            continue

        # Process article
        print(f"✓ Processing: {title}")
        content_hash = dedup.generate_content_hash(content)
        article_id = db.insert_article(title, url, content_hash)
        dedup.mark_processed(article_id, url, content_hash)
        processed += 1

    print(f"\nResults: Processed {processed} articles, skipped {skipped} duplicates")


# Example 3: Batch Processing
def example_batch_processing():
    """Efficient batch processing of multiple articles."""
    print("\n" + "=" * 80)
    print("Example 3: Batch Processing")
    print("=" * 80)

    db = DatabaseManager("data/articles.db")
    dedup = DeduplicationManager(db)

    # Batch of articles
    articles = [
        {
            "url": f"https://example.com/article{i}",
            "content": f"Article content {i}..."
        }
        for i in range(1, 11)
    ]

    print(f"Checking {len(articles)} articles in batch...")

    # Batch check for duplicates
    results = dedup.batch_check_duplicates(articles)

    # Filter new articles
    new_articles = [r for r in results if not r["is_duplicate"]]
    duplicates = [r for r in results if r["is_duplicate"]]

    print(f"✓ Found {len(new_articles)} new articles")
    print(f"✗ Found {len(duplicates)} duplicates")

    # Process new articles
    article_data = []
    for article in new_articles:
        url = article['url']
        content = article['content']
        title = f"Article for {url}"

        content_hash = dedup.generate_content_hash(content)
        url_hash = dedup.generate_url_hash(url)

        article_id = db.insert_article(title, url, content_hash)

        article_data.append({
            "article_id": article_id,
            "url": url,
            "content_hash": content_hash,
            "url_hash": url_hash
        })

    # Batch mark as processed
    if article_data:
        dedup.batch_mark_processed(article_data)
        print(f"✓ Batch marked {len(article_data)} articles as processed")


# Example 4: Performance Monitoring
def example_performance_monitoring():
    """Monitor deduplication performance and cache statistics."""
    print("\n" + "=" * 80)
    print("Example 4: Performance Monitoring")
    print("=" * 80)

    db = DatabaseManager("data/articles.db")
    dedup = DeduplicationManager(db)

    # Get comprehensive statistics
    stats = dedup.get_duplicate_stats()

    print("\nProcessing Statistics:")
    print(f"  Articles processed: {stats['articles_processed']}")
    print(f"  Duplicates detected: {stats['duplicates_detected']}")
    print(f"  Duplicate rate: {stats['duplicate_rate']:.2f}%")

    print("\nCache Performance:")
    cache_stats = stats['cache_stats']
    print(f"  Content cache size: {cache_stats['content_cache_size']:,}")
    print(f"  URL cache size: {cache_stats['url_cache_size']:,}")
    print(f"  Cache capacity: {cache_stats['cache_capacity']:,}")
    print(f"  Cache hits: {cache_stats['cache_hits']:,}")
    print(f"  Cache misses: {cache_stats['cache_misses']:,}")
    print(f"  Cache hit rate: {cache_stats['cache_hit_rate']}")

    # Get memory usage
    memory = dedup.get_memory_usage_estimate()
    print("\nMemory Usage:")
    print(f"  Content cache: {memory['content_cache_mb']} MB")
    print(f"  URL cache: {memory['url_cache_mb']} MB")
    print(f"  Total: {memory['total_mb']} MB")
    print(f"  Estimated max: {memory['estimated_max_mb']} MB")


# Example 5: Cache Management
def example_cache_management():
    """Demonstrate cache management operations."""
    print("\n" + "=" * 80)
    print("Example 5: Cache Management")
    print("=" * 80)

    db = DatabaseManager("data/articles.db")
    dedup = DeduplicationManager(db)

    # Check initial cache state
    print("\nInitial cache state:")
    print(f"  Content cache: {dedup.content_hash_cache.size()} entries")
    print(f"  URL cache: {dedup.url_hash_cache.size()} entries")

    # Clean old cache (24+ hours)
    print("\nCleaning old cache entries...")
    dedup.clean_old_cache(max_age_hours=24)

    print("After cleanup:")
    print(f"  Content cache: {dedup.content_hash_cache.size()} entries")
    print(f"  URL cache: {dedup.url_hash_cache.size()} entries")

    # Rebuild cache
    print("\nRebuilding cache from database...")
    dedup.rebuild_cache()

    print("After rebuild:")
    print(f"  Content cache: {dedup.content_hash_cache.size()} entries")
    print(f"  URL cache: {dedup.url_hash_cache.size()} entries")


# Example 6: Error Handling
def example_error_handling():
    """Demonstrate proper error handling."""
    print("\n" + "=" * 80)
    print("Example 6: Error Handling")
    print("=" * 80)

    db = DatabaseManager("data/articles.db")
    dedup = DeduplicationManager(db)

    # Test with various inputs
    test_cases = [
        ("Valid URL", "https://example.com/article", "Valid content"),
        ("Empty content", "https://example.com/article2", ""),
        ("Unicode content", "https://example.com/article3", "Content with émojis 🎉"),
    ]

    for name, url, content in test_cases:
        try:
            print(f"\nTesting: {name}")
            print(f"  URL: {url}")

            # Check duplicate
            is_duplicate, reason = dedup.is_duplicate(url, content if content else None)

            if is_duplicate:
                print(f"  ✗ Duplicate ({reason})")
            else:
                print(f"  ✓ Not duplicate")

                # Generate hashes
                content_hash = dedup.generate_content_hash(content or url)
                url_hash = dedup.generate_url_hash(url)

                print(f"  Content hash: {content_hash[:16]}...")
                print(f"  URL hash: {url_hash[:16]}...")

        except Exception as e:
            print(f"  ✗ Error: {e}")


# Example 7: Integration with ArticleProcessor
def example_article_processor_integration():
    """Example integration with ArticleProcessor class."""
    print("\n" + "=" * 80)
    print("Example 7: ArticleProcessor Integration")
    print("=" * 80)

    class ArticleProcessor:
        """Example article processor with deduplication."""

        def __init__(self):
            self.db = DatabaseManager("data/articles.db")
            self.dedup = DeduplicationManager(self.db)
            self.stats = {"processed": 0, "duplicates": 0}

        def process_article(self, title: str, url: str, content: str) -> bool:
            """Process single article with duplicate detection."""
            # Check URL duplicate first (fast)
            is_duplicate, reason = self.dedup.is_duplicate(url)

            if is_duplicate:
                print(f"  ✗ URL duplicate: {title}")
                self.stats["duplicates"] += 1
                return False

            # Check content duplicate
            is_duplicate, reason = self.dedup.is_duplicate(url, content)

            if is_duplicate:
                print(f"  ✗ Content duplicate: {title}")
                self.stats["duplicates"] += 1
                return False

            # Process article
            print(f"  ✓ Processing: {title}")

            # Generate hashes
            content_hash = self.dedup.generate_content_hash(content)
            url_hash = self.dedup.generate_url_hash(url)

            # Insert into database
            article_id = self.db.insert_article(title, url, content_hash)

            # Mark as processed
            self.dedup.mark_processed(article_id, url, content_hash, url_hash)

            self.stats["processed"] += 1
            return True

        def get_stats(self) -> dict:
            """Get processing statistics."""
            return {
                **self.stats,
                "total": self.stats["processed"] + self.stats["duplicates"],
                "dedup_stats": self.dedup.get_duplicate_stats()
            }

    # Use processor
    processor = ArticleProcessor()

    # Sample articles
    articles = [
        ("Article 1", "https://example.com/1", "Content 1"),
        ("Article 2", "https://example.com/2", "Content 2"),
        ("Article 1 Duplicate", "https://example.com/1", "Content 1"),  # Duplicate
    ]

    print("\nProcessing articles:")
    for title, url, content in articles:
        processor.process_article(title, url, content)

    # Show statistics
    stats = processor.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Duplicates: {stats['duplicates']}")


def main():
    """Run all examples."""
    examples = [
        ("Basic Duplicate Detection", example_basic_duplicate_detection),
        ("RSS Feed Processing", example_rss_feed_processing),
        ("Batch Processing", example_batch_processing),
        ("Performance Monitoring", example_performance_monitoring),
        ("Cache Management", example_cache_management),
        ("Error Handling", example_error_handling),
        ("ArticleProcessor Integration", example_article_processor_integration),
    ]

    print("\n" + "=" * 80)
    print("DEDUPLICATION SYSTEM EXAMPLES")
    print("=" * 80)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRun examples with: python docs/deduplication_examples.py [example_number]")
    print("Or run all: python docs/deduplication_examples.py all\n")

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg.lower() == "all":
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    logger.error(f"Error in {name}: {e}")
        else:
            try:
                index = int(arg) - 1
                if 0 <= index < len(examples):
                    name, func = examples[index]
                    func()
                else:
                    print(f"Invalid example number. Choose 1-{len(examples)}")
            except ValueError:
                print("Invalid argument. Use example number or 'all'")


if __name__ == "__main__":
    main()
