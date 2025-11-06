#!/usr/bin/env python3
"""
Database Schema Migration for Deduplication

Adds url_hash and content_hash columns to the articles table with indexes.
Backfills existing articles with hash values.

Usage:
    python tools/migrate_dedup_schema.py
    python tools/migrate_dedup_schema.py --db-path /path/to/articles.db
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.database import DatabaseManager
from deduplication_manager import DeduplicationManager


class SchemaManager:
    """Manages database schema migrations for deduplication."""

    def __init__(self, db_path: str = "data/articles.db"):
        self.db_path = db_path
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)

    def check_schema(self) -> dict:
        """
        Check current schema state.

        Returns:
            Dict with schema information
        """
        info = {
            "url_hash_exists": False,
            "content_hash_exists": False,
            "url_hash_indexed": False,
            "content_hash_indexed": False,
            "total_articles": 0,
            "articles_with_url_hash": 0,
            "articles_with_content_hash": 0,
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check columns
                cursor = conn.execute("PRAGMA table_info(articles)")
                columns = {row[1] for row in cursor.fetchall()}

                info["url_hash_exists"] = "url_hash" in columns
                info["content_hash_exists"] = "content_hash" in columns

                # Check indexes
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='articles'"
                )
                indexes = {row[0] for row in cursor.fetchall()}

                info["url_hash_indexed"] = "idx_articles_url_hash" in indexes
                info["content_hash_indexed"] = "idx_articles_content_hash" in indexes

                # Count articles
                cursor = conn.execute("SELECT COUNT(*) FROM articles")
                info["total_articles"] = cursor.fetchone()[0]

                # Count articles with hashes (if columns exist)
                if info["url_hash_exists"]:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM articles WHERE url_hash IS NOT NULL"
                    )
                    info["articles_with_url_hash"] = cursor.fetchone()[0]

                if info["content_hash_exists"]:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM articles WHERE content_hash IS NOT NULL"
                    )
                    info["articles_with_content_hash"] = cursor.fetchone()[0]

        except Exception as e:
            self.logger.error(f"Error checking schema: {e}")
            raise

        return info

    def migrate_schema(self) -> bool:
        """
        Migrate database schema to add deduplication columns.

        Returns:
            True if migration successful
        """
        self.logger.info("=" * 80)
        self.logger.info("DATABASE SCHEMA MIGRATION")
        self.logger.info("=" * 80)

        try:
            # Check current state
            info = self.check_schema()

            self.logger.info(f"\nCurrent schema state:")
            self.logger.info(f"  Total articles: {info['total_articles']}")
            self.logger.info(f"  url_hash column exists: {info['url_hash_exists']}")
            self.logger.info(f"  content_hash column exists: {info['content_hash_exists']}")
            self.logger.info(f"  url_hash indexed: {info['url_hash_indexed']}")
            self.logger.info(f"  content_hash indexed: {info['content_hash_indexed']}")

            changes_made = False

            with sqlite3.connect(self.db_path) as conn:
                # Add url_hash column if missing
                if not info["url_hash_exists"]:
                    self.logger.info("\nAdding url_hash column...")
                    conn.execute("""
                        ALTER TABLE articles
                        ADD COLUMN url_hash VARCHAR(64)
                    """)
                    self.logger.info("✓ url_hash column added")
                    changes_made = True
                else:
                    self.logger.info("\n✓ url_hash column already exists")

                # Add content_hash column if missing
                # Note: content_hash should already exist from DatabaseManager init
                if not info["content_hash_exists"]:
                    self.logger.info("\nAdding content_hash column...")
                    conn.execute("""
                        ALTER TABLE articles
                        ADD COLUMN content_hash VARCHAR(64)
                    """)
                    self.logger.info("✓ content_hash column added")
                    changes_made = True
                else:
                    self.logger.info("✓ content_hash column already exists")

                # Create url_hash index if missing
                if not info["url_hash_indexed"]:
                    self.logger.info("\nCreating url_hash index...")
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_articles_url_hash
                        ON articles (url_hash)
                    """)
                    self.logger.info("✓ url_hash index created")
                    changes_made = True
                else:
                    self.logger.info("✓ url_hash index already exists")

                # Create content_hash index if missing
                if not info["content_hash_indexed"]:
                    self.logger.info("\nCreating content_hash index...")
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_articles_content_hash
                        ON articles (content_hash)
                    """)
                    self.logger.info("✓ content_hash index created")
                    changes_made = True
                else:
                    self.logger.info("✓ content_hash index already exists")

                # Add unique constraint on url_hash (optional, can cause issues)
                # Commented out to allow flexibility
                # conn.execute("""
                #     CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_hash_unique
                #     ON articles (url_hash)
                # """)

                conn.commit()

            if changes_made:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("MIGRATION COMPLETED SUCCESSFULLY")
                self.logger.info("=" * 80)

                # Show next steps
                self.logger.info("\nNext steps:")
                self.logger.info("  1. Run backfill to populate hash columns:")
                self.logger.info("     python tools/check_duplicates.py --backfill")
                self.logger.info("  2. Analyze duplicates:")
                self.logger.info("     python tools/check_duplicates.py --analyze")
            else:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("NO MIGRATION NEEDED - Schema already up to date")
                self.logger.info("=" * 80)

            return True

        except Exception as e:
            self.logger.error(f"\n✗ Migration failed: {e}")
            return False

    def verify_migration(self) -> bool:
        """
        Verify migration was successful.

        Returns:
            True if schema is correct
        """
        self.logger.info("\nVerifying migration...")

        try:
            info = self.check_schema()

            required_checks = [
                ("url_hash column exists", info["url_hash_exists"]),
                ("content_hash column exists", info["content_hash_exists"]),
                ("url_hash indexed", info["url_hash_indexed"]),
                ("content_hash indexed", info["content_hash_indexed"]),
            ]

            all_passed = True
            for check_name, result in required_checks:
                status = "✓" if result else "✗"
                self.logger.info(f"  {status} {check_name}")
                if not result:
                    all_passed = False

            if all_passed:
                self.logger.info("\n✓ Migration verification PASSED")
            else:
                self.logger.warning("\n✗ Migration verification FAILED")

            return all_passed

        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            return False


def main():
    """Main entry point for schema migration."""
    parser = argparse.ArgumentParser(
        description="Migrate database schema for deduplication support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
  1. Add url_hash column (VARCHAR(64))
  2. Add content_hash column (VARCHAR(64)) if missing
  3. Create indexes on both columns for fast lookups
  4. Verify migration was successful

After running this migration, use check_duplicates.py to:
  - Backfill hash values for existing articles
  - Analyze and remove duplicates
        """
    )

    parser.add_argument(
        "--db-path",
        default="data/articles.db",
        help="Path to SQLite database (default: data/articles.db)"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify schema without making changes"
    )

    args = parser.parse_args()

    try:
        manager = SchemaManager(args.db_path)

        if args.verify_only:
            info = manager.check_schema()

            print("\nSchema Status:")
            print(f"  Total articles: {info['total_articles']}")
            print(f"  url_hash column: {'✓' if info['url_hash_exists'] else '✗'}")
            print(f"  content_hash column: {'✓' if info['content_hash_exists'] else '✗'}")
            print(f"  url_hash indexed: {'✓' if info['url_hash_indexed'] else '✗'}")
            print(f"  content_hash indexed: {'✓' if info['content_hash_indexed'] else '✗'}")

            if info['url_hash_exists']:
                print(f"  Articles with url_hash: {info['articles_with_url_hash']}")
            if info['content_hash_exists']:
                print(f"  Articles with content_hash: {info['articles_with_content_hash']}")

            return 0

        # Run migration
        success = manager.migrate_schema()
        if not success:
            return 1

        # Verify migration
        if not manager.verify_migration():
            return 1

        return 0

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
