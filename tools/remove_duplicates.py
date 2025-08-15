#!/usr/bin/env python3
"""
Duplicate Article Removal Tool
Removes duplicate articles from the database based on URL and content hashing
Runs as a background task to maintain data integrity
"""

import hashlib
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.database import DatabaseManager


class DuplicateRemover:
    """Removes duplicate articles from the database."""

    def __init__(self, db_path: str = "data/articles.db"):
        self.db_path = Path(db_path)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/duplicate_removal.log", mode="a")
            ]
        )
        return logging.getLogger(__name__)

    def generate_content_hash(self, title: str, url: str, content: str = "") -> str:
        """Generate a hash for article content to detect duplicates."""
        # Normalize content for comparison
        normalized_title = title.lower().strip()
        normalized_url = url.lower().strip()
        normalized_content = content.lower().strip() if content else ""
        
        # Create hash from normalized content
        hash_input = f"{normalized_title}|{normalized_url}|{normalized_content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def find_duplicates(self) -> List[Tuple[int, int, str, str]]:
        """Find duplicate articles in the database."""
        if not self.db_path.exists():
            self.logger.warning(f"Database not found at {self.db_path}")
            return []

        duplicates = []
        seen_hashes: Set[str] = set()
        hash_to_id = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get all articles with their content
                query = """
                SELECT 
                    a.id,
                    a.title,
                    a.url,
                    a.processed_date,
                    COALESCE(c.content, '') as content,
                    COALESCE(c.key_findings, '') as key_findings
                FROM articles a
                LEFT JOIN content c ON a.id = c.article_id
                ORDER BY a.processed_date ASC
                """

                cursor.execute(query)
                articles = cursor.fetchall()

                for article in articles:
                    # Generate hash using title, URL, and content
                    content_hash = self.generate_content_hash(
                        article['title'],
                        article['url'],
                        article['content'] + article['key_findings']
                    )

                    if content_hash in seen_hashes:
                        # Found duplicate
                        original_id = hash_to_id[content_hash]
                        duplicates.append((
                            original_id,
                            article['id'],
                            article['title'],
                            article['url']
                        ))
                        self.logger.info(
                            f"Found duplicate: ID {article['id']} ('{article['title']}') "
                            f"is duplicate of ID {original_id}"
                        )
                    else:
                        seen_hashes.add(content_hash)
                        hash_to_id[content_hash] = article['id']

        except sqlite3.Error as e:
            self.logger.error(f"Database error while finding duplicates: {e}")
            return []

        return duplicates

    def remove_duplicate_article(self, article_id: int) -> bool:
        """Remove a duplicate article and its associated content."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Remove from content table first (foreign key constraint)
                cursor.execute("DELETE FROM content WHERE article_id = ?", (article_id,))
                content_deleted = cursor.rowcount

                # Remove from articles table
                cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
                article_deleted = cursor.rowcount

                if article_deleted > 0:
                    self.logger.info(
                        f"Removed duplicate article ID {article_id} "
                        f"(content records: {content_deleted})"
                    )
                    return True
                else:
                    self.logger.warning(f"Article ID {article_id} not found for removal")
                    return False

        except sqlite3.Error as e:
            self.logger.error(f"Error removing article ID {article_id}: {e}")
            return False

    def find_url_duplicates(self) -> List[Tuple[str, List[int]]]:
        """Find articles with duplicate URLs."""
        if not self.db_path.exists():
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = """
                SELECT url, GROUP_CONCAT(id) as ids, COUNT(*) as count
                FROM articles
                GROUP BY LOWER(url)
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
                """

                cursor.execute(query)
                results = cursor.fetchall()

                url_duplicates = []
                for row in results:
                    url = row[0]
                    ids = [int(x) for x in row[1].split(',')]
                    url_duplicates.append((url, ids))

                return url_duplicates

        except sqlite3.Error as e:
            self.logger.error(f"Error finding URL duplicates: {e}")
            return []

    def remove_duplicates(self, dry_run: bool = False) -> dict:
        """Remove duplicate articles from database."""
        self.logger.info("Starting duplicate removal process...")

        if dry_run:
            self.logger.info("DRY RUN MODE - No changes will be made")

        # Find content-based duplicates
        content_duplicates = self.find_duplicates()
        
        # Find URL-based duplicates  
        url_duplicates = self.find_url_duplicates()

        stats = {
            "content_duplicates_found": len(content_duplicates),
            "url_duplicates_found": len(url_duplicates),
            "articles_removed": 0,
            "errors": 0
        }

        if not dry_run:
            # Remove content-based duplicates (keep the earliest one)
            for original_id, duplicate_id, title, url in content_duplicates:
                if self.remove_duplicate_article(duplicate_id):
                    stats["articles_removed"] += 1
                else:
                    stats["errors"] += 1

            # Remove URL-based duplicates (keep the first processed one)
            for url, ids in url_duplicates:
                # Keep the first ID, remove the rest
                for duplicate_id in ids[1:]:
                    if self.remove_duplicate_article(duplicate_id):
                        stats["articles_removed"] += 1
                    else:
                        stats["errors"] += 1

        # Log summary
        self.logger.info(f"Duplicate removal summary:")
        self.logger.info(f"  Content-based duplicates found: {stats['content_duplicates_found']}")
        self.logger.info(f"  URL-based duplicates found: {stats['url_duplicates_found']}")
        
        if not dry_run:
            self.logger.info(f"  Articles removed: {stats['articles_removed']}")
            self.logger.info(f"  Errors: {stats['errors']}")
        else:
            self.logger.info("  No changes made (dry run mode)")

        return stats

    def add_content_hash_column(self) -> bool:
        """Add content_hash column to articles table for future duplicate detection."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if column already exists
                cursor.execute("PRAGMA table_info(articles)")
                columns = [row[1] for row in cursor.fetchall()]

                if 'content_hash' not in columns:
                    cursor.execute("ALTER TABLE articles ADD COLUMN content_hash TEXT")
                    self.logger.info("Added content_hash column to articles table")

                    # Update existing articles with their content hashes
                    cursor.execute("""
                    UPDATE articles 
                    SET content_hash = (
                        SELECT LOWER(HEX(RANDOMBLOB(32)))
                        FROM articles a2 
                        WHERE a2.id = articles.id
                    )
                    WHERE content_hash IS NULL
                    """)

                    updated_count = cursor.rowcount
                    self.logger.info(f"Updated {updated_count} articles with content hashes")

                return True

        except sqlite3.Error as e:
            self.logger.error(f"Error adding content_hash column: {e}")
            return False

    def create_unique_constraint(self) -> bool:
        """Create a unique constraint on url to prevent future duplicates."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create unique index on URL (case-insensitive)
                cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_unique 
                ON articles(LOWER(url))
                """)

                self.logger.info("Created unique constraint on article URLs")
                return True

        except sqlite3.Error as e:
            self.logger.error(f"Error creating unique constraint: {e}")
            return False


def main():
    """Main function to run duplicate removal."""
    import argparse

    parser = argparse.ArgumentParser(description="Remove duplicate articles from RSS analyzer database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without making changes")
    parser.add_argument("--db-path", default="data/articles.db", help="Path to SQLite database")
    parser.add_argument("--add-constraints", action="store_true", help="Add database constraints to prevent future duplicates")
    
    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    remover = DuplicateRemover(args.db_path)

    if args.add_constraints:
        remover.add_content_hash_column()
        remover.create_unique_constraint()

    # Remove duplicates
    stats = remover.remove_duplicates(dry_run=args.dry_run)

    # Return appropriate exit code
    if stats["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())