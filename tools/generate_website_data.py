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
log_handlers = [logging.StreamHandler(sys.stdout)]
try:
    # Try to create log file handler, but don't fail if we can't
    Path("logs").mkdir(exist_ok=True)
    log_handlers.append(logging.FileHandler("logs/website_generator.log", mode="a"))
except (PermissionError, OSError):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
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
    processed_date: str  # Now contains publication_date or falls back to processed_date
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
            raise DatabaseError(f"Database validation error: {e}") from e

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
                    COALESCE(a.publication_date, a.processed_date) as display_date,
                    a.processed_date,
                    a.status,
                    COALESCE(c.key_findings, '') as key_findings,
                    COALESCE(c.technical_approach, '') as technical_approach,
                    COALESCE(c.methodology_detailed, '') as methodology_detailed,
                    COALESCE(c.metadata, '{}') as metadata
                FROM articles a
                LEFT JOIN content c ON a.id = c.article_id
                WHERE a.status = 'completed'
                ORDER BY COALESCE(a.publication_date, a.processed_date) DESC
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
            raise DatabaseError(f"Failed to fetch articles from database: {e}") from e

    def _extract_better_title(
        self, analysis: str, url: str, original_title: str
    ) -> str:
        """Extract a better title from the analysis if possible."""
        import re

        # Skip if already has a good title (not a username)
        if "@" not in original_title and "bsky" not in original_title:
            return original_title

        # Avoid extracting generic analysis headers that shouldn't be titles
        generic_headers = [
            "In-Depth Analysis Using the Feynman Technique",
            "Analysis Using the Feynman Technique", 
            "In-Depth Analysis",
            "Feynman Technique Analysis",
            "Using the Feynman Technique"
        ]
        
        # Check for markdown headers at the beginning, but skip generic ones
        lines = analysis.strip().split("\n")
        for line in lines[:10]:
            line = line.strip()
            if line.startswith("# ") and len(line) > 2:
                title = line[2:].strip()
                title = title.replace("**", "").replace("*", "").strip()
                if (
                    len(title) > 10 
                    and len(title) < 150 
                    and not any(generic in title for generic in generic_headers)
                    and not title.lower().startswith(("the feynman", "feynman technique"))
                ):
                    return title

        # Look for specific title patterns in key findings section
        if "**Key Findings:" in analysis:
            key_findings_section = analysis.split("**Key Findings:")[1].split("**")[0]

            # Check for hashtag titles in key findings
            hashtag_match = re.search(r"#\s*([^#\n]{10,150})", key_findings_section)
            if hashtag_match:
                title = hashtag_match.group(1).strip()
                if not any(generic in title for generic in generic_headers):
                    return title

            # Check for quoted titles
            quote_patterns = [
                r'"([^"]{10,150})"',
                r"'([^']{10,150})'",
            ]
            for pattern in quote_patterns:
                match = re.search(pattern, key_findings_section)
                if match:
                    title = match.group(1).strip()
                    if (
                        not title.lower().startswith(("this", "the post", "this post"))
                        and not any(generic in title for generic in generic_headers)
                    ):
                        return title

        # Look for paper titles in the text with more specific patterns
        patterns = [
            r'paper titled ["\']([^"\']+)["\']',
            r'article titled ["\']([^"\']+)["\']',
            r'"([^"]+)"(?:\s+paper|\s+article)',
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis, re.IGNORECASE | re.MULTILINE)
            if match:
                title = match.group(1).strip()
                if (
                    len(title) > 10
                    and len(title) < 150
                    and not title.lower().startswith(("this", "the post", "another"))
                    and not any(generic in title for generic in generic_headers)
                ):
                    return title

        # Generate a descriptive title based on URL domain for social media posts
        if "bsky.app" in url:
            return f"Bluesky Post Analysis"
        elif "twitter.com" in url or "x.com" in url:
            return f"Twitter/X Post Analysis" 
        elif "arxiv.org" in url:
            return f"arXiv Paper Analysis"

        return original_title

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
        # Validate required fields (use display_date which is publication_date with fallback)
        required_fields = ["id", "title", "url", "display_date", "status"]
        for field in required_fields:
            if not row[field]:
                raise DataValidationError(f"Missing required field: {field}")

        # Use the methodology_detailed field as the primary analysis (this contains the full Feynman technique explanation)
        # If methodology_detailed is empty or too short, fall back to key_findings
        analysis = ""
        if row["methodology_detailed"] and len(row["methodology_detailed"]) > 500:
            analysis = row["methodology_detailed"]
        elif row["key_findings"] and len(row["key_findings"]) > 500:
            analysis = row["key_findings"]
        else:
            # Combine fields for shorter analyses
            analysis_parts = []
            if row["key_findings"]:
                analysis_parts.append(f"**Key Findings:** {row['key_findings']}")
            if (
                row["technical_approach"]
                and row["technical_approach"] != row["key_findings"]
            ):
                analysis_parts.append(
                    f"**Technical Approach:** {row['technical_approach']}"
                )
            if (
                row["methodology_detailed"]
                and row["methodology_detailed"] != row["key_findings"]
            ):
                analysis_parts.append(f"**Methodology:** {row['methodology_detailed']}")

            analysis = (
                "\n\n".join(analysis_parts)
                if analysis_parts
                else "No analysis available"
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

        # Extract better title if needed
        title = self._extract_better_title(analysis, str(row["url"]), str(row["title"]))

        # Manual title overrides for better readability
        title_overrides = {
            44: "CRUX: Enhanced Evaluation Metrics for Long-Form RAG Systems",
            43: "Text-to-LoRA: Instant Transformer Adaptation via Natural Language",  
            42: "LLM2Rec: Teaching Language Models Recommendation Systems",
            25: "Human-in-the-Loop LLM Annotation for Subjective Tasks",
            24: "InfoFlood: Academic Jargon Jailbreak for AI Safety Systems",
            23: "Statistical Rigor in Information Retrieval Testing",
            22: "FrugalRAG: Efficient AI Question-Answering",
            19: "LangChain Platform Update",
            17: "AI/ML Research Update by Sung Kim",
            15: "LlamaIndex Platform Development",
            14: "Machine Learning Research Update",
            13: "AI and Machine Learning Topics",
            12: "CRUX: Diagnostic Revolution for AI Information Retrieval",
            11: "Machine Learning Research Discussion",
            10: "LLM2Rec: Teaching Language Models Recommendation",
            9: "PentaRAG: Five-Lane Highway for Enterprise AI Queries",
            8: "HPC-ColPali: Powerful and Practical Document Understanding",
            7: "ARAG: Teaching AI Through Collaborative Intelligence",
            6: "VAT-KG: First True Multimodal Knowledge Encyclopedia",
            5: "IRanker: Teaching AI to Rank Like a Tournament Judge",
            1: "AT Protocol and Bluesky Social Platform",
        }

        if int(row["id"]) in title_overrides:
            title = title_overrides[int(row["id"])]

        return Article(
            id=int(row["id"]),
            title=title,
            url=str(row["url"]),
            processed_date=str(row["display_date"]),  # Use publication_date (with fallback)
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
        dates = []
        for article in articles:
            try:
                # Handle different datetime formats
                date_str = article.processed_date
                if date_str.endswith("Z"):
                    date_str = date_str.replace("Z", "+00:00")
                elif (
                    "+" not in date_str and "-" not in date_str[-6:] and "T" in date_str
                ):
                    # Add UTC timezone if none specified
                    date_str = date_str + "+00:00"

                parsed_date = datetime.fromisoformat(date_str)

                # Ensure all dates have timezone info (convert naive to UTC)
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=UTC)

                dates.append(parsed_date)
            except Exception as e:
                self.logger.warning(
                    f"Could not parse date {article.processed_date}: {e}"
                )
                continue
        if dates:
            date_range = {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat(),
            }
        else:
            date_range = None

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
            raise DataValidationError(f"Failed to save JSON data: {e}") from e

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
