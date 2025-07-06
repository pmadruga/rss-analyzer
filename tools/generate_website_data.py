#!/usr/bin/env python3
"""
RSS Article Analysis Dashboard - Data Generator
Professional Python Architecture

This module generates JSON data for the GitHub Pages website from the SQLite database.
It includes comprehensive error handling, logging, and data validation.

Author: RSS Analyzer Team
Version: 2.0.0
"""

import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from error_logger import get_error_logger
except ImportError:
    # Fallback if error_logger is not available
    def get_error_logger():
        class DummyLogger:
            def generate_website_status(self):
                return {
                    "system_status": "unknown",
                    "dates": {},
                    "recent_errors_by_date": {},
                }

        return DummyLogger()


import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/website_generator.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class LinkedArticle:
    """Data class for linked articles."""

    title: str
    url: str


@dataclass
class Article:
    """Data class for article data."""

    id: int
    title: str
    url: str
    processed_date: str
    status: str
    analysis: str
    ai_provider: str
    linked_articles: list[LinkedArticle]


@dataclass
class WebsiteData:
    """Data class for website data structure."""

    generated_at: str
    total_articles: int
    articles: list[Article]
    metadata: dict[str, Any]
    processing_status: dict[str, Any] | None = None


class DatabaseError(Exception):
    """Custom exception for database-related errors."""


class DataValidationError(Exception):
    """Custom exception for data validation errors."""


class WebsiteDataGenerator:
    """
    Professional website data generator with comprehensive error handling
    and data validation.
    """

    def __init__(self, db_path: str = "data/articles.db", output_dir: str = "docs"):
        """
        Initialize the data generator.

        Args:
            db_path: Path to SQLite database
            output_dir: Output directory for generated files
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.logger = logger

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)

    def validate_database(self) -> bool:
        """
        Validate database existence and structure.

        Returns:
            bool: True if database is valid

        Raises:
            DatabaseError: If database validation fails
        """
        if not self.db_path.exists():
            raise DatabaseError(f"Database not found at {self.db_path}")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if required tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('articles', 'content')
                """)
                tables = [row[0] for row in cursor.fetchall()]

                required_tables = {"articles", "content"}
                missing_tables = required_tables - set(tables)

                if missing_tables:
                    raise DatabaseError(f"Missing required tables: {missing_tables}")

                # Check articles table structure
                cursor.execute("PRAGMA table_info(articles)")
                article_columns = {row[1] for row in cursor.fetchall()}
                required_article_columns = {
                    "id",
                    "title",
                    "url",
                    "processed_date",
                    "status",
                }

                missing_columns = required_article_columns - article_columns
                if missing_columns:
                    raise DatabaseError(
                        f"Missing required columns in articles table: {missing_columns}"
                    )

                # Check content table structure
                cursor.execute("PRAGMA table_info(content)")
                content_columns = {row[1] for row in cursor.fetchall()}
                required_content_columns = {"article_id", "key_findings"}

                missing_columns = required_content_columns - content_columns
                if missing_columns:
                    raise DatabaseError(
                        f"Missing required columns in content table: {missing_columns}"
                    )

                self.logger.info("Database validation successful")
                return True

        except sqlite3.Error as e:
            raise DatabaseError(f"Database validation error: {e}")

    def get_articles_from_db(self) -> list[Article]:
        """
        Fetch and process articles from the database.

        Returns:
            List[Article]: List of processed articles

        Raises:
            DatabaseError: If database operation fails
            DataValidationError: If data validation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = """
                SELECT 
                    a.id,
                    a.title,
                    a.url,
                    a.processed_date,
                    a.status,
                    COALESCE(c.key_findings, '') as key_findings,
                    COALESCE(c.technical_approach, '') as technical_approach,
                    COALESCE(c.methodology_detailed, '') as methodology_detailed,
                    COALESCE(c.metadata, '{}') as metadata
                FROM articles a
                LEFT JOIN content c ON a.id = c.article_id
                WHERE a.status = 'completed'
                ORDER BY a.processed_date DESC
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                articles = []
                for row in rows:
                    try:
                        article = self._process_article_row(row)
                        articles.append(article)
                    except Exception as e:
                        self.logger.warning(
                            f"Skipping article {row['id']} due to processing error: {e}"
                        )
                        continue

                self.logger.info(
                    f"Successfully processed {len(articles)} articles from database"
                )
                return articles

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to fetch articles from database: {e}")

    def _process_article_row(self, row: sqlite3.Row) -> Article:
        """
        Process a single article row from the database.

        Args:
            row: Database row

        Returns:
            Article: Processed article object

        Raises:
            DataValidationError: If data validation fails
        """
        # Validate required fields
        required_fields = ["id", "title", "url", "processed_date", "status"]
        for field in required_fields:
            if not row[field]:
                raise DataValidationError(f"Missing required field: {field}")

        # Combine analysis fields
        analysis_parts = []
        if row["key_findings"]:
            analysis_parts.append(f"**Key Findings:** {row['key_findings']}")
        if row["technical_approach"]:
            analysis_parts.append(
                f"**Technical Approach:** {row['technical_approach']}"
            )
        if row["methodology_detailed"]:
            analysis_parts.append(f"**Methodology:** {row['methodology_detailed']}")

        analysis = (
            "\n\n".join(analysis_parts) if analysis_parts else "No analysis available"
        )

        # Parse metadata
        linked_articles = []
        ai_provider = "anthropic"  # Default

        try:
            if row["metadata"]:
                metadata = json.loads(row["metadata"])

                # Extract linked articles
                if "linked_articles" in metadata:
                    for la in metadata["linked_articles"]:
                        if isinstance(la, dict) and "title" in la and "url" in la:
                            linked_articles.append(
                                LinkedArticle(
                                    title=str(la["title"]), url=str(la["url"])
                                )
                            )

                # Extract AI provider
                if "ai_provider" in metadata:
                    ai_provider = str(metadata["ai_provider"])

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(
                f"Failed to parse metadata for article {row['id']}: {e}"
            )

        return Article(
            id=int(row["id"]),
            title=str(row["title"]),
            url=str(row["url"]),
            processed_date=str(row["processed_date"]),
            status=str(row["status"]),
            analysis=analysis,
            ai_provider=ai_provider,
            linked_articles=linked_articles,
        )

    def generate_metadata(self, articles: list[Article]) -> dict[str, Any]:
        """
        Generate metadata about the articles.

        Args:
            articles: List of articles

        Returns:
            Dict containing metadata
        """
        if not articles:
            return {"date_range": None, "ai_providers": {}, "status_counts": {}}

        # Date range
        dates = [
            datetime.fromisoformat(article.processed_date.replace("Z", "+00:00"))
            for article in articles
        ]
        date_range = {
            "earliest": min(dates).isoformat(),
            "latest": max(dates).isoformat(),
        }

        # AI provider counts
        ai_providers = {}
        for article in articles:
            provider = article.ai_provider
            ai_providers[provider] = ai_providers.get(provider, 0) + 1

        # Status counts
        status_counts = {}
        for article in articles:
            status = article.status
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "date_range": date_range,
            "ai_providers": ai_providers,
            "status_counts": status_counts,
        }

    def generate_website_data(self) -> WebsiteData:
        """
        Generate complete website data structure.

        Returns:
            WebsiteData: Complete website data

        Raises:
            DatabaseError: If database operations fail
            DataValidationError: If data validation fails
        """
        self.logger.info("Starting website data generation")

        # Validate database
        self.validate_database()

        # Fetch articles
        articles = self.get_articles_from_db()

        # Generate metadata
        metadata = self.generate_metadata(articles)

        # Get processing status from error logger
        try:
            error_logger = get_error_logger("output")
            processing_status = error_logger.generate_website_status()
        except Exception as e:
            self.logger.warning(f"Could not get processing status: {e}")
            processing_status = {
                "system_status": "unknown",
                "dates": {},
                "recent_errors_by_date": {},
            }

        # Create website data structure
        website_data = WebsiteData(
            generated_at=datetime.now(UTC).isoformat(),
            total_articles=len(articles),
            articles=articles,
            metadata=metadata,
            processing_status=processing_status,
        )

        self.logger.info(f"Generated website data with {len(articles)} articles")
        return website_data

    def save_data_json(self, website_data: WebsiteData) -> Path:
        """
        Save website data as JSON file.

        Args:
            website_data: Website data to save

        Returns:
            Path: Path to saved file
        """
        output_path = self.output_dir / "data.json"

        try:
            # Convert to dict and ensure JSON serializable
            data_dict = asdict(website_data)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Website data saved to {output_path}")
            return output_path

        except (OSError, json.JSONEncodeError) as e:
            raise DataValidationError(f"Failed to save JSON data: {e}")

    def print_summary(self, website_data: WebsiteData) -> None:
        """
        Print a summary of the generated data.

        Args:
            website_data: Website data to summarize
        """
        print(f"\n{'=' * 60}")
        print("RSS ANALYZER WEBSITE DATA GENERATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Generated at: {website_data.generated_at}")
        print(f"Total articles: {website_data.total_articles}")

        if website_data.metadata["date_range"]:
            earliest = datetime.fromisoformat(
                website_data.metadata["date_range"]["earliest"]
            )
            latest = datetime.fromisoformat(
                website_data.metadata["date_range"]["latest"]
            )
            print(f"Date range: {earliest.date()} to {latest.date()}")

        print("\nArticles by AI provider:")
        for provider, count in website_data.metadata["ai_providers"].items():
            print(f"  {provider}: {count}")

        print(f"{'=' * 60}\n")

    def run(self) -> bool:
        """
        Run the complete data generation process.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            website_data = self.generate_website_data()
            self.save_data_json(website_data)
            self.print_summary(website_data)
            return True

        except (DatabaseError, DataValidationError) as e:
            self.logger.error(f"Data generation failed: {e}")
            print(f"ERROR: {e}", file=sys.stderr)
            return False

        except Exception as e:
            self.logger.error(
                f"Unexpected error during data generation: {e}", exc_info=True
            )
            print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
            return False


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Generate JSON data for RSS Analyzer website",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="data/articles.db",
        help="Path to SQLite database",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs",
        help="Output directory for generated files",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run generator
    generator = WebsiteDataGenerator(db_path=args.db_path, output_dir=args.output_dir)

    success = generator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
