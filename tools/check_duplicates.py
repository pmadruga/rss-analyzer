#!/usr/bin/env python3
"""
Duplicate Analysis and Removal Tool

Command-line tool for analyzing and managing duplicate articles in the RSS analyzer database.
Provides statistics, dry-run mode, and bulk operations.

Usage:
    # Analyze duplicates (no changes)
    python tools/check_duplicates.py --analyze

    # Remove duplicates (dry-run first)
    python tools/check_duplicates.py --remove --dry-run
    python tools/check_duplicates.py --remove

    # Show statistics
    python tools/check_duplicates.py --stats

    # Backfill hashes for existing articles
    python tools/check_duplicates.py --backfill
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.database import DatabaseManager
from deduplication_manager import DeduplicationManager


class DuplicateAnalyzer:
    """Analyzes and manages duplicate articles."""

    def __init__(self, db_path: str = "data/articles.db"):
        self.db = DatabaseManager(db_path)
        self.dedup = DeduplicationManager(self.db)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
            ]
        )
        return logging.getLogger(__name__)

    def analyze_duplicates(self) -> Dict[str, any]:
        """
        Analyze database for duplicates without making changes.

        Returns:
            Dict with analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("DUPLICATE ANALYSIS REPORT")
        self.logger.info("=" * 80)

        results = {
            "total_articles": 0,
            "url_duplicates": [],
            "content_duplicates": [],
            "missing_hashes": 0,
        }

        try:
            with self.db.get_connection() as conn:
                # Total articles
                cursor = conn.execute("SELECT COUNT(*) as count FROM articles")
                results["total_articles"] = cursor.fetchone()[0]

                self.logger.info(f"\nTotal articles in database: {results['total_articles']}")

                # Check for missing hashes
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM articles
                    WHERE content_hash IS NULL OR url_hash IS NULL
                """)
                results["missing_hashes"] = cursor.fetchone()[0]

                if results["missing_hashes"] > 0:
                    self.logger.warning(
                        f"Found {results['missing_hashes']} articles missing hash columns"
                    )
                    self.logger.warning("Run with --backfill to add missing hashes")

                # Find URL duplicates
                cursor = conn.execute("""
                    SELECT url_hash, COUNT(*) as count, GROUP_CONCAT(id) as ids
                    FROM articles
                    WHERE url_hash IS NOT NULL
                    GROUP BY url_hash
                    HAVING count > 1
                    ORDER BY count DESC
                """)

                url_dups = cursor.fetchall()
                results["url_duplicates"] = [
                    {
                        "url_hash": row[0],
                        "count": row[1],
                        "ids": [int(x) for x in row[2].split(",")],
                    }
                    for row in url_dups
                ]

                self.logger.info(f"\nURL-based duplicates found: {len(results['url_duplicates'])}")
                if results["url_duplicates"]:
                    self.logger.info("Top URL duplicates:")
                    for i, dup in enumerate(results["url_duplicates"][:5], 1):
                        self.logger.info(
                            f"  {i}. Hash {dup['url_hash'][:16]}... : "
                            f"{dup['count']} copies (IDs: {dup['ids'][:5]})"
                        )

                # Find content duplicates
                cursor = conn.execute("""
                    SELECT content_hash, COUNT(*) as count, GROUP_CONCAT(id) as ids
                    FROM articles
                    WHERE content_hash IS NOT NULL
                    GROUP BY content_hash
                    HAVING count > 1
                    ORDER BY count DESC
                """)

                content_dups = cursor.fetchall()
                results["content_duplicates"] = [
                    {
                        "content_hash": row[0],
                        "count": row[1],
                        "ids": [int(x) for x in row[2].split(",")],
                    }
                    for row in content_dups
                ]

                self.logger.info(
                    f"\nContent-based duplicates found: {len(results['content_duplicates'])}"
                )
                if results["content_duplicates"]:
                    self.logger.info("Top content duplicates:")
                    for i, dup in enumerate(results["content_duplicates"][:5], 1):
                        self.logger.info(
                            f"  {i}. Hash {dup['content_hash'][:16]}... : "
                            f"{dup['count']} copies (IDs: {dup['ids'][:5]})"
                        )

                # Total duplicate articles
                total_dup_articles = sum(
                    dup["count"] - 1 for dup in results["url_duplicates"]
                )
                total_dup_articles += sum(
                    dup["count"] - 1 for dup in results["content_duplicates"]
                )

                self.logger.info("\n" + "-" * 80)
                self.logger.info(f"Total duplicate articles: {total_dup_articles}")
                self.logger.info(
                    f"Unique articles: {results['total_articles'] - total_dup_articles}"
                )
                self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            raise

        return results

    def remove_duplicates(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Remove duplicate articles from database.

        Args:
            dry_run: If True, only simulate removal without making changes

        Returns:
            Dict with removal statistics
        """
        if dry_run:
            self.logger.info("=" * 80)
            self.logger.info("DRY RUN MODE - No changes will be made")
            self.logger.info("=" * 80)
        else:
            self.logger.warning("=" * 80)
            self.logger.warning("REMOVING DUPLICATES - Changes will be permanent!")
            self.logger.warning("=" * 80)

        stats = {
            "url_duplicates_removed": 0,
            "content_duplicates_removed": 0,
            "total_removed": 0,
            "errors": 0,
        }

        try:
            with self.db.get_connection() as conn:
                # Remove URL-based duplicates (keep oldest article)
                cursor = conn.execute("""
                    SELECT url_hash, MIN(id) as keep_id, GROUP_CONCAT(id) as all_ids
                    FROM articles
                    WHERE url_hash IS NOT NULL
                    GROUP BY url_hash
                    HAVING COUNT(*) > 1
                """)

                url_dups = cursor.fetchall()

                for row in url_dups:
                    keep_id = row[1]
                    all_ids = [int(x) for x in row[2].split(",")]
                    remove_ids = [x for x in all_ids if x != keep_id]

                    self.logger.info(
                        f"URL duplicate: Keep ID {keep_id}, "
                        f"remove IDs {remove_ids}"
                    )

                    if not dry_run:
                        for remove_id in remove_ids:
                            try:
                                # Delete content first (foreign key)
                                conn.execute(
                                    "DELETE FROM content WHERE article_id = ?",
                                    (remove_id,)
                                )
                                # Delete article
                                conn.execute(
                                    "DELETE FROM articles WHERE id = ?",
                                    (remove_id,)
                                )
                                stats["url_duplicates_removed"] += 1
                            except Exception as e:
                                self.logger.error(
                                    f"Error removing article {remove_id}: {e}"
                                )
                                stats["errors"] += 1

                # Remove content-based duplicates (keep oldest article)
                cursor = conn.execute("""
                    SELECT content_hash, MIN(id) as keep_id, GROUP_CONCAT(id) as all_ids
                    FROM articles
                    WHERE content_hash IS NOT NULL
                    AND url_hash NOT IN (
                        SELECT url_hash FROM articles
                        WHERE url_hash IS NOT NULL
                        GROUP BY url_hash
                        HAVING COUNT(*) > 1
                    )
                    GROUP BY content_hash
                    HAVING COUNT(*) > 1
                """)

                content_dups = cursor.fetchall()

                for row in content_dups:
                    keep_id = row[1]
                    all_ids = [int(x) for x in row[2].split(",")]
                    remove_ids = [x for x in all_ids if x != keep_id]

                    self.logger.info(
                        f"Content duplicate: Keep ID {keep_id}, "
                        f"remove IDs {remove_ids}"
                    )

                    if not dry_run:
                        for remove_id in remove_ids:
                            try:
                                # Delete content first (foreign key)
                                conn.execute(
                                    "DELETE FROM content WHERE article_id = ?",
                                    (remove_id,)
                                )
                                # Delete article
                                conn.execute(
                                    "DELETE FROM articles WHERE id = ?",
                                    (remove_id,)
                                )
                                stats["content_duplicates_removed"] += 1
                            except Exception as e:
                                self.logger.error(
                                    f"Error removing article {remove_id}: {e}"
                                )
                                stats["errors"] += 1

                if not dry_run:
                    conn.commit()

        except Exception as e:
            self.logger.error(f"Error during duplicate removal: {e}")
            raise

        stats["total_removed"] = (
            stats["url_duplicates_removed"] + stats["content_duplicates_removed"]
        )

        self.logger.info("\n" + "-" * 80)
        self.logger.info("REMOVAL SUMMARY")
        self.logger.info("-" * 80)
        self.logger.info(f"URL duplicates removed: {stats['url_duplicates_removed']}")
        self.logger.info(f"Content duplicates removed: {stats['content_duplicates_removed']}")
        self.logger.info(f"Total articles removed: {stats['total_removed']}")
        self.logger.info(f"Errors: {stats['errors']}")
        self.logger.info("=" * 80)

        return stats

    def backfill_hashes(self, batch_size: int = 1000) -> Dict[str, int]:
        """
        Backfill missing hash columns for existing articles.

        Args:
            batch_size: Number of articles to process per batch

        Returns:
            Dict with backfill statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("BACKFILLING MISSING HASHES")
        self.logger.info("=" * 80)

        stats = {
            "articles_processed": 0,
            "hashes_added": 0,
            "errors": 0,
        }

        start_time = time.time()

        try:
            with self.db.get_connection() as conn:
                # Get articles missing hashes
                cursor = conn.execute("""
                    SELECT id, url, content_hash
                    FROM articles
                    WHERE url_hash IS NULL OR content_hash IS NULL
                """)

                articles = cursor.fetchall()
                total = len(articles)

                self.logger.info(f"Found {total} articles needing hash backfill")

                if total == 0:
                    self.logger.info("No articles need backfilling")
                    return stats

                # Process in batches
                for i in range(0, total, batch_size):
                    batch = articles[i:i + batch_size]

                    for article in batch:
                        article_id = article[0]
                        url = article[1]
                        existing_content_hash = article[2]

                        try:
                            # Generate URL hash
                            url_hash = self.dedup.generate_url_hash(url)

                            # Use existing content hash or generate placeholder
                            if not existing_content_hash:
                                # Generate temporary content hash from URL
                                # (will be updated when article is re-processed)
                                content_hash = self.dedup.generate_content_hash(url)
                            else:
                                content_hash = existing_content_hash

                            # Update database
                            conn.execute("""
                                UPDATE articles
                                SET url_hash = ?, content_hash = ?
                                WHERE id = ?
                            """, (url_hash, content_hash, article_id))

                            stats["hashes_added"] += 1
                            stats["articles_processed"] += 1

                        except Exception as e:
                            self.logger.error(
                                f"Error backfilling article {article_id}: {e}"
                            )
                            stats["errors"] += 1

                    # Progress update
                    progress = min(i + batch_size, total)
                    self.logger.info(
                        f"Progress: {progress}/{total} "
                        f"({progress/total*100:.1f}%)"
                    )

                conn.commit()

        except Exception as e:
            self.logger.error(f"Error during backfill: {e}")
            raise

        duration = time.time() - start_time
        throughput = stats["articles_processed"] / duration if duration > 0 else 0

        self.logger.info("\n" + "-" * 80)
        self.logger.info("BACKFILL SUMMARY")
        self.logger.info("-" * 80)
        self.logger.info(f"Articles processed: {stats['articles_processed']}")
        self.logger.info(f"Hashes added: {stats['hashes_added']}")
        self.logger.info(f"Errors: {stats['errors']}")
        self.logger.info(f"Duration: {duration:.2f}s")
        self.logger.info(f"Throughput: {throughput:.0f} articles/sec")
        self.logger.info("=" * 80)

        return stats

    def show_stats(self):
        """Display deduplication statistics and cache performance."""
        self.logger.info("=" * 80)
        self.logger.info("DEDUPLICATION STATISTICS")
        self.logger.info("=" * 80)

        # Get deduplication stats
        dedup_stats = self.dedup.get_duplicate_stats()

        self.logger.info("\nProcessing Statistics:")
        self.logger.info(f"  Articles processed: {dedup_stats['articles_processed']}")
        self.logger.info(f"  Duplicates detected: {dedup_stats['duplicates_detected']}")
        self.logger.info(f"  Duplicate rate: {dedup_stats['duplicate_rate']:.2f}%")

        self.logger.info("\nCache Performance:")
        cache_stats = dedup_stats['cache_stats']
        self.logger.info(f"  Content cache size: {cache_stats['content_cache_size']}")
        self.logger.info(f"  URL cache size: {cache_stats['url_cache_size']}")
        self.logger.info(f"  Cache capacity: {cache_stats['cache_capacity']}")
        self.logger.info(f"  Cache hits: {cache_stats['cache_hits']}")
        self.logger.info(f"  Cache misses: {cache_stats['cache_misses']}")
        self.logger.info(f"  Cache hit rate: {cache_stats['cache_hit_rate']}")

        # Get memory usage
        memory_stats = self.dedup.get_memory_usage_estimate()
        self.logger.info("\nMemory Usage:")
        self.logger.info(f"  Content cache: {memory_stats['content_cache_mb']} MB")
        self.logger.info(f"  URL cache: {memory_stats['url_cache_mb']} MB")
        self.logger.info(f"  Total: {memory_stats['total_mb']} MB")
        self.logger.info(f"  Estimated max: {memory_stats['estimated_max_mb']} MB")

        self.logger.info("\n" + "=" * 80)


def main():
    """Main entry point for duplicate checking tool."""
    parser = argparse.ArgumentParser(
        description="Analyze and manage duplicate articles in RSS analyzer database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze duplicates without making changes
  python tools/check_duplicates.py --analyze

  # Show statistics and cache performance
  python tools/check_duplicates.py --stats

  # Remove duplicates (dry-run first!)
  python tools/check_duplicates.py --remove --dry-run
  python tools/check_duplicates.py --remove

  # Backfill hashes for existing articles
  python tools/check_duplicates.py --backfill

  # Custom database path
  python tools/check_duplicates.py --analyze --db-path /path/to/articles.db
        """
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze database for duplicates (no changes)"
    )

    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove duplicate articles from database"
    )

    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill missing hash columns for existing articles"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show deduplication statistics and cache performance"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate removal without making changes (use with --remove)"
    )

    parser.add_argument(
        "--db-path",
        default="data/articles.db",
        help="Path to SQLite database (default: data/articles.db)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for backfill operations (default: 1000)"
    )

    args = parser.parse_args()

    # Require at least one action
    if not (args.analyze or args.remove or args.backfill or args.stats):
        parser.print_help()
        print("\nError: Please specify at least one action (--analyze, --remove, --backfill, or --stats)")
        return 1

    try:
        analyzer = DuplicateAnalyzer(args.db_path)

        # Execute requested actions
        if args.stats:
            analyzer.show_stats()

        if args.analyze:
            analyzer.analyze_duplicates()

        if args.backfill:
            analyzer.backfill_hashes(args.batch_size)

        if args.remove:
            analyzer.remove_duplicates(dry_run=args.dry_run)

        return 0

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
