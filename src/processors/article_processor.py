"""
Refactored Article Processor Module

Main application orchestrator with improved error handling,
better separation of concerns, and enhanced testability.
"""

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..clients import AIClientFactory
from ..config import CONFIG
from ..core import (
    DatabaseManager,
    ReportGenerator,
    RSSParser,
    WebScraper,
    format_timestamp,
)
from ..exceptions import (
    ConfigurationError,
    ContentProcessingError,
    RSSAnalyzerError,
    ScrapingError,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResults:
    """Data class for processing results"""

    start_time: float
    duration: float
    rss_entries_found: int
    new_articles: int
    scraped_articles: int
    analyzed_articles: int
    report_generated: bool
    errors: list[str]
    reports: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ProcessingConfig:
    """Configuration for processing run"""

    force_refresh: bool = False
    limit: int | None = None
    follow_links: bool = True
    max_linked_articles: int = 3

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ProcessingConfig":
        """Create from configuration dictionary"""
        return cls(
            force_refresh=config.get("force_refresh", False),
            limit=config.get("limit"),
            follow_links=config.get("follow_links", True),
            max_linked_articles=config.get("max_linked_articles", 3),
        )


class ArticleProcessor:
    """Main application orchestrator with improved architecture"""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize article processor

        Args:
            config: Application configuration dictionary

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self._initialize_components()
        logger.info("ArticleProcessor initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all components with error handling"""
        try:
            # Initialize database
            self.db = DatabaseManager(self.config["db_path"])

            # Initialize RSS parser
            user_agent = self.config.get("user_agent", CONFIG.scraping.USER_AGENT)
            self.rss_parser = RSSParser(user_agent)

            # Initialize web scraper
            scraper_delay = self.config.get(
                "scraper_delay", CONFIG.processing.SCRAPER_DELAY
            )
            self.scraper = WebScraper(scraper_delay)

            # Initialize AI client using factory
            self.ai_client = AIClientFactory.create_from_config(self.config)

            # Initialize report generator
            self.report_generator = ReportGenerator(self.config["output_dir"])

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")

    def run(
        self, processing_config: ProcessingConfig | None = None
    ) -> ProcessingResults:
        """
        Run the complete processing pipeline

        Args:
            processing_config: Processing configuration

        Returns:
            Processing results with detailed metrics
        """
        if processing_config is None:
            processing_config = ProcessingConfig()

        start_time = time.time()
        results = ProcessingResults(
            start_time=start_time,
            duration=0.0,
            rss_entries_found=0,
            new_articles=0,
            scraped_articles=0,
            analyzed_articles=0,
            report_generated=False,
            errors=[],
        )

        try:
            logger.info("Starting RSS article analysis pipeline")

            # Step 1: Test API connection
            self._test_api_connection()

            # Step 2: Fetch RSS feed
            rss_entries = self._fetch_rss_feed(results)
            if not rss_entries:
                return self._finalize_results(results, start_time)

            # Step 3: Filter new articles
            new_entries = self._filter_articles(rss_entries, processing_config, results)
            if not new_entries:
                return self._finalize_results(results, start_time)

            # Step 4: Process articles
            processed_articles = self._process_articles(
                new_entries, processing_config, results
            )

            # Step 5: Generate reports
            if processed_articles:
                self._generate_reports(processed_articles, results)

            # Step 6: Cleanup
            self._cleanup()

            logger.info("Pipeline completed successfully")

        except RSSAnalyzerError as e:
            logger.error(f"Pipeline failed with known error: {e}")
            results.errors.append(str(e))
            self.db.log_processing(None, "pipeline_failed", str(e))

        except Exception as e:
            logger.exception(f"Pipeline failed with unexpected error: {e}")
            results.errors.append(f"Unexpected error: {e}")
            self.db.log_processing(None, "pipeline_failed", str(e))

        return self._finalize_results(results, start_time)

    def _test_api_connection(self) -> None:
        """Test API connection"""
        if not self.ai_client.test_connection():
            raise ConfigurationError("API connection test failed")
        logger.info("API connection test passed")

    def _fetch_rss_feed(self, results: ProcessingResults) -> list[Any]:
        """Fetch RSS feed entries"""
        try:
            logger.info("Fetching RSS feed...")
            rss_entries = self.rss_parser.fetch_feed(self.config["rss_feed_url"])
            results.rss_entries_found = len(rss_entries)

            if not rss_entries:
                logger.warning("No entries found in RSS feed")

            return rss_entries

        except Exception as e:
            logger.error(f"Failed to fetch RSS feed: {e}")
            raise ContentProcessingError(f"RSS fetch failed: {e}")

    def _filter_articles(
        self,
        rss_entries: list[Any],
        processing_config: ProcessingConfig,
        results: ProcessingResults,
    ) -> list[Any]:
        """Filter articles based on processing configuration"""

        if not processing_config.force_refresh:
            # Only skip articles that have been fully analyzed
            analyzed_hashes = self.db.get_analyzed_content_hashes()
            new_entries = self.rss_parser.filter_new_entries(
                rss_entries, analyzed_hashes
            )

            skipped = len(rss_entries) - len(new_entries)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already analyzed articles")
        else:
            new_entries = rss_entries
            logger.info("Force refresh enabled - processing all articles")

        # Apply limit if specified
        if processing_config.limit and len(new_entries) > processing_config.limit:
            new_entries = new_entries[: processing_config.limit]
            logger.info(f"Limited processing to {processing_config.limit} articles")

        results.new_articles = len(new_entries)

        if not new_entries:
            logger.info("No new articles to process")

        return new_entries

    def _process_articles(
        self,
        entries: list[Any],
        processing_config: ProcessingConfig,
        results: ProcessingResults,
    ) -> list[dict[str, Any]]:
        """Process articles through scraping and analysis"""
        processed_articles = []

        logger.info(f"Processing {len(entries)} articles...")

        for i, entry in enumerate(entries):
            try:
                logger.info(
                    f"Processing article {i + 1}/{len(entries)}: {entry.title[:50]}..."
                )

                article_data = self._process_single_article(
                    entry, processing_config, results
                )
                if article_data:
                    processed_articles.append(article_data)

            except Exception as e:
                logger.error(f"Error processing article '{entry.title}': {e}")
                results.errors.append(f"Processing '{entry.title}': {e}")

        return processed_articles

    def _process_single_article(
        self,
        entry: Any,
        processing_config: ProcessingConfig,
        results: ProcessingResults,
    ) -> dict[str, Any] | None:
        """Process a single article"""
        article_id = None

        try:
            # Insert article into database
            article_id = self.db.insert_article(
                title=entry.title,
                url=entry.link,
                content_hash=entry.content_hash,
                rss_guid=entry.guid,
                publication_date=entry.publication_date,
            )

            self.db.log_processing(article_id, "started", processing_step="scraping")
            self.db.update_article_status(article_id, "processing")

            # Scrape article content
            scraped_content = self._scrape_article(entry, processing_config, article_id)
            if not scraped_content:
                return None

            results.scraped_articles += 1

            # Analyze with AI client
            analysis = self._analyze_article(entry, scraped_content, article_id)
            if not analysis:
                return None

            # Store content and analysis
            self.db.insert_content(article_id, scraped_content.content, analysis)
            results.analyzed_articles += 1

            self.db.update_article_status(article_id, "completed")

            # Prepare article data for reporting
            return self._prepare_article_data(article_id, entry, analysis)

        except Exception as e:
            if article_id:
                self.db.log_processing(article_id, "processing_failed", str(e))
                self.db.update_article_status(article_id, "failed")
            raise

    def _scrape_article(
        self, entry: Any, processing_config: ProcessingConfig, article_id: int
    ):
        """Scrape article content with error handling"""
        try:
            scrape_start = time.time()

            scraped_content = self.scraper.scrape_article(
                entry.link,
                follow_links=processing_config.follow_links,
                max_linked_articles=processing_config.max_linked_articles,
            )

            scrape_duration = time.time() - scrape_start

            if not scraped_content:
                self.db.log_processing(
                    article_id,
                    "scraping_failed",
                    processing_step="scraping",
                    duration_seconds=scrape_duration,
                )
                self.db.update_article_status(article_id, "scraping_failed")
                logger.warning(f"Failed to scrape article: {entry.title}")
                return None

            # Check if content with this hash has already been processed
            if not processing_config.force_refresh and self.db.is_content_already_processed(scraped_content.content_hash):
                logger.info(f"Content already processed, skipping: {scraped_content.title}")
                self.db.log_processing(
                    article_id,
                    "duplicate_content",
                    processing_step="scraping",
                    duration_seconds=scrape_duration,
                )
                self.db.update_article_status(article_id, "duplicate")
                return None

            # Update the article's content hash with the actual scraped content hash
            self.db.update_article_content_hash(article_id, scraped_content.content_hash)

            self.db.log_processing(
                article_id,
                "scraped",
                processing_step="scraping",
                duration_seconds=scrape_duration,
            )

            return scraped_content

        except Exception as e:
            logger.error(f"Scraping error for {entry.title}: {e}")
            raise ScrapingError(f"Scraping failed: {e}", entry.link)

    def _analyze_article(self, entry: Any, scraped_content: Any, article_id: int):
        """Analyze article with AI client"""
        try:
            analysis_start = time.time()

            analysis = self.ai_client.analyze_article(
                title=entry.title,
                content=scraped_content.content,
                url=entry.link,
            )

            analysis_duration = time.time() - analysis_start

            if not analysis:
                self.db.log_processing(
                    article_id,
                    "analysis_failed",
                    processing_step="analysis",
                    duration_seconds=analysis_duration,
                )
                self.db.update_article_status(article_id, "analysis_failed")
                logger.warning(f"Failed to analyze article: {entry.title}")
                return None

            self.db.log_processing(
                article_id,
                "completed",
                processing_step="analysis",
                duration_seconds=analysis_duration,
            )

            return analysis

        except Exception as e:
            logger.error(f"Analysis error for {entry.title}: {e}")
            raise ContentProcessingError(f"Analysis failed: {e}")

    def _prepare_article_data(
        self, article_id: int, entry: Any, analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare article data for reporting"""
        return {
            "id": article_id,
            "title": entry.title,
            "url": entry.link,
            "publication_date": entry.publication_date.isoformat()
            if entry.publication_date
            else None,
            "processed_date": format_timestamp(),
            **analysis,
        }

    def _generate_reports(
        self, articles: list[dict[str, Any]], results: ProcessingResults
    ) -> None:
        """Generate various reports"""
        try:
            logger.info("Generating reports...")

            report_files = {}

            # Generate main markdown report
            report_files["main_report"] = self.report_generator.generate_report(
                articles,
                self.config.get("report_filename", "article_analysis_report.md"),
                use_timestamp=True,
            )

            # Generate summary report
            report_files["summary_report"] = (
                self.report_generator.generate_summary_report(
                    articles, use_timestamp=True
                )
            )

            # Generate JSON export
            report_files["json_export"] = self.report_generator.generate_json_export(
                articles, use_timestamp=True
            )

            # Generate CSV export
            report_files["csv_export"] = self.report_generator.generate_csv_export(
                articles, use_timestamp=True
            )

            results.report_generated = True
            results.reports = report_files

            for report_type, path in report_files.items():
                logger.info(f"{report_type} generated: {path}")

        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            results.errors.append(f"Report generation: {e}")

    def _cleanup(self) -> None:
        """Perform cleanup operations"""
        try:
            self.db.cleanup_old_logs()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def _finalize_results(
        self, results: ProcessingResults, start_time: float
    ) -> ProcessingResults:
        """Finalize processing results"""
        results.duration = time.time() - start_time
        logger.info(f"Pipeline completed in {results.duration:.2f} seconds")
        return results

    def get_client_info(self) -> dict[str, Any]:
        """Get information about the AI client"""
        return self.ai_client.get_provider_info()

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics from database"""
        return self.db.get_processing_statistics()
