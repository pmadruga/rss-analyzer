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
from ..core.cache import ContentCache
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

            # Initialize two-tier cache (L1 memory + L2 disk)
            cache_db_path = self.config.get("cache_db_path", "data/cache.db")
            self.cache = ContentCache(cache_db_path)
            logger.info(f"Cache initialized: {cache_db_path}")

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

            # Step 1: Fetch RSS feed
            rss_entries = self._fetch_rss_feed(results)
            if not rss_entries:
                return self._finalize_results(results, start_time)

            # Step 2: Filter new articles
            new_entries = self._filter_articles(rss_entries, processing_config, results)
            if not new_entries:
                logger.info("No new articles to process - pipeline complete")
                return self._finalize_results(results, start_time)

            # Step 3: Process articles
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
        """Process articles through scraping and analysis with batch operations"""
        processed_articles = []

        logger.info(f"Processing {len(entries)} articles with batch operations...")

        # Phase 1: Batch insert all articles at once
        articles_to_insert = []
        for entry in entries:
            articles_to_insert.append({
                "title": entry.title,
                "url": entry.link,
                "content_hash": entry.content_hash,
                "rss_guid": entry.guid,
                "publication_date": entry.publication_date,
            })

        try:
            article_ids = self.db.insert_articles_batch(articles_to_insert)
            logger.info(f"Batch inserted {len(article_ids)} articles")
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            results.errors.append(f"Batch insert failed: {e}")
            return []

        # Phase 2: Process each article (scraping + analysis) and collect batch data
        status_updates = []
        content_records = []
        processing_logs = []
        title_updates = []  # Store title updates for batch execution
        title_map = {}  # Track final title for each article_id

        for i, (entry, article_id) in enumerate(zip(entries, article_ids)):
            try:
                logger.info(
                    f"Processing article {i + 1}/{len(entries)}: {entry.title[:50]}..."
                )

                # Initialize with original title
                title_map[article_id] = entry.title

                # Track that we started processing
                processing_logs.append({
                    "article_id": article_id,
                    "status": "started",
                    "processing_step": "scraping"
                })

                # Set status to processing
                status_updates.append((article_id, "processing"))

                # Scrape article content
                scraped_content = self._scrape_article(entry, processing_config, article_id)
                if not scraped_content:
                    continue

                results.scraped_articles += 1

                # Update title if scraped title is better
                if scraped_content.title and scraped_content.title != entry.title:
                    logger.info(f"Will update title from RSS to scraped: '{entry.title}' -> '{scraped_content.title}'")
                    title_updates.append((scraped_content.title, article_id))
                    title_map[article_id] = scraped_content.title

                # Analyze with AI client
                analysis = self._analyze_article(entry, scraped_content, article_id)
                if not analysis:
                    continue

                # Use AI-extracted title if available
                if analysis.get("extracted_title"):
                    ai_title = analysis["extracted_title"].strip()
                    current_title = title_map[article_id]

                    if ai_title != current_title and len(ai_title) > 5:
                        logger.info(f"Will update title with AI-extracted: '{current_title}' -> '{ai_title}'")
                        # Replace previous title update or add new one
                        title_updates = [(ai_title, article_id) if aid == article_id else (t, aid)
                                       for t, aid in title_updates]
                        if article_id not in [aid for _, aid in title_updates]:
                            title_updates.append((ai_title, article_id))
                        title_map[article_id] = ai_title

                # Store content record for batch insert
                content_records.append({
                    "article_id": article_id,
                    "original_content": scraped_content.content,
                    "analysis": analysis
                })

                results.analyzed_articles += 1

                # Mark as completed
                status_updates.append((article_id, "completed"))

                # Prepare article data for reporting with tracked title
                article_data = self._prepare_article_data(article_id, entry, analysis, title_map[article_id])
                processed_articles.append(article_data)

            except Exception as e:
                logger.error(f"Error processing article '{entry.title}': {e}")
                results.errors.append(f"Processing '{entry.title}': {e}")

                # Log failure
                processing_logs.append({
                    "article_id": article_id,
                    "status": "processing_failed",
                    "error_message": str(e)
                })
                status_updates.append((article_id, "failed"))

        # Phase 3: Execute all batch operations at once
        logger.info("Executing batch database operations...")

        # Batch update titles
        if title_updates:
            try:
                with self.db.get_connection() as conn:
                    conn.execute("BEGIN TRANSACTION")
                    for title, article_id in title_updates:
                        conn.execute(
                            "UPDATE articles SET title = ? WHERE id = ?",
                            (title, article_id)
                        )
                    conn.commit()
                logger.info(f"Batch updated {len(title_updates)} article titles")
            except Exception as e:
                logger.error(f"Batch title update failed: {e}")

        # Batch update statuses
        if status_updates:
            try:
                self.db.update_status_batch(status_updates)
            except Exception as e:
                logger.error(f"Batch status update failed: {e}")

        # Batch insert content
        if content_records:
            try:
                self.db.insert_content_batch(content_records)
            except Exception as e:
                logger.error(f"Batch content insert failed: {e}")

        # Batch log processing
        if processing_logs:
            try:
                self.db.log_processing_batch(processing_logs)
            except Exception as e:
                logger.error(f"Batch processing log failed: {e}")

        logger.info(f"Batch processing complete: {len(processed_articles)} articles successfully processed")
        return processed_articles


    def _scrape_article(
        self, entry: Any, processing_config: ProcessingConfig, article_id: int
    ):
        """Scrape article content with error handling and caching (batch-optimized)"""
        try:
            # Check cache first (unless force refresh)
            cache_key = ContentCache.generate_key(entry.link, "scraped_content")
            if not processing_config.force_refresh:
                cached_content = self.cache.get(cache_key)
                if cached_content:
                    logger.info(f"Cache hit for scraped content: {entry.link}")
                    return cached_content

            # Cache miss - scrape as usual
            scrape_start = time.time()

            scraped_content = self.scraper.scrape_article(
                entry.link,
                follow_links=processing_config.follow_links,
                max_linked_articles=processing_config.max_linked_articles,
            )

            scrape_duration = time.time() - scrape_start

            if not scraped_content:
                logger.warning(f"Failed to scrape article: {entry.title}")
                return None

            # Check if content with this hash has already been processed
            if not processing_config.force_refresh and self.db.is_content_already_processed(scraped_content.content_hash):
                logger.info(f"Content already processed, skipping: {scraped_content.title}")
                return None

            # Update the article's content hash with the actual scraped content hash
            self.db.update_article_content_hash(article_id, scraped_content.content_hash)

            # Store in cache for future use
            self.cache.set(
                cache_key,
                scraped_content,
                ttl=ContentCache.TTL_SCRAPED_CONTENT,
                content_type="scraped_content"
            )
            logger.debug(f"Cached scraped content: {entry.link}")

            return scraped_content

        except Exception as e:
            logger.error(f"Scraping error for {entry.title}: {e}")
            raise ScrapingError(f"Scraping failed: {e}", entry.link)

    def _analyze_article(self, entry: Any, scraped_content: Any, article_id: int):
        """Analyze article with AI client and caching (batch-optimized)"""
        try:
            # Check cache first (unless force refresh)
            # Include model in cache key to invalidate when model changes
            cache_key = ContentCache.generate_key(
                f"{entry.link}:{self.ai_client.model}",
                "ai_analysis"
            )

            # Get force_refresh from current processing config (if available)
            # This is a workaround since we don't pass processing_config to this method
            # In a future refactor, we should pass it as a parameter
            force_refresh = False  # Conservative default

            if not force_refresh:
                cached_analysis = self.cache.get(cache_key)
                if cached_analysis:
                    logger.info(f"Cache hit for AI analysis: {entry.link}")
                    return cached_analysis

            # Cache miss - analyze as usual
            analysis_start = time.time()

            analysis = self.ai_client.analyze_article(
                title=entry.title,
                content=scraped_content.content,
                url=entry.link,
            )

            analysis_duration = time.time() - analysis_start

            if not analysis:
                logger.warning(f"Failed to analyze article: {entry.title}")
                return None

            # Store in cache for future use
            self.cache.set(
                cache_key,
                analysis,
                ttl=ContentCache.TTL_API_RESPONSE,
                content_type="ai_analysis"
            )
            logger.debug(f"Cached AI analysis: {entry.link}")

            return analysis

        except Exception as e:
            logger.error(f"Analysis error for {entry.title}: {e}")
            raise ContentProcessingError(f"Analysis failed: {e}")

    def _prepare_article_data(
        self, article_id: int, entry: Any, analysis: dict[str, Any], title: str | None = None
    ) -> dict[str, Any]:
        """Prepare article data for reporting (batch-optimized)"""
        # Use provided title (from batch tracking) or fallback to entry title
        actual_title = title or entry.title

        return {
            "id": article_id,
            "title": actual_title,
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

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear all cached content"""
        self.cache.clear()
        logger.info("Cache cleared")

    def cleanup_expired_cache(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        return self.cache.cleanup_expired()
