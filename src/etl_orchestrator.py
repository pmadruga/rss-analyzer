"""
ETL Orchestrator

Lightweight orchestrator that delegates to existing core modules.
This module exists for backwards compatibility but delegates to the main ArticleProcessor.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from .core import DatabaseManager, RSSParser, WebScraper, ReportGenerator
from .clients import AIClientFactory

logger = logging.getLogger(__name__)


@dataclass
class ETLResults:
    """Results from ETL pipeline execution"""
    articles_fetched: int
    articles_analyzed: int
    articles_stored: int
    errors: List[str]
    processing_time: float


class ETLOrchestrator:
    """
    Lightweight orchestrator that delegates to core modules.

    Note: For new code, prefer using ArticleProcessor from src.processors
    which provides a more complete and tested implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize core components
        self.rss_parser = RSSParser()
        self.web_scraper = WebScraper(
            delay_between_requests=config.get("scraper_delay", 1.0)
        )
        self.database = DatabaseManager(
            config.get("database_path", config.get("db_path", "data/articles.db"))
        )
        self.report_generator = ReportGenerator(
            config.get("output_dir", "output")
        )

        # Initialize AI client
        self.ai_client = AIClientFactory.create_from_config(config)

    def run_full_pipeline(self, feed_urls: List[str], max_articles: int = None) -> ETLResults:
        """
        Run the complete ETL pipeline.

        Note: This is a simplified implementation. For production use,
        consider using ArticleProcessor from src.processors instead.
        """
        import time
        start_time = time.time()

        errors = []
        articles_fetched = 0
        articles_analyzed = 0
        articles_stored = 0

        try:
            logger.info("Starting ETL pipeline")

            # EXTRACT: Fetch content from RSS feeds
            logger.info("Phase 1: Extract - Fetching content")
            all_articles = []

            for feed_url in feed_urls:
                try:
                    # Get existing hashes to avoid duplicates
                    existing_hashes = set()
                    existing_articles = self.database.get_all_articles()
                    for article in existing_articles:
                        if article.get('content_hash'):
                            existing_hashes.add(article['content_hash'])

                    # Fetch RSS entries
                    entries = self.rss_parser.fetch_feed(feed_url)
                    new_entries = self.rss_parser.filter_new_entries(entries, existing_hashes)

                    if max_articles:
                        new_entries = new_entries[:max_articles]

                    # Scrape full content
                    for entry in new_entries:
                        try:
                            content = self.web_scraper.scrape_article(entry.link)
                            if content:
                                all_articles.append({
                                    'entry': entry,
                                    'content': content
                                })
                                articles_fetched += 1
                        except Exception as e:
                            logger.error(f"Failed to scrape {entry.link}: {e}")
                            errors.append(f"Scraping failed for {entry.link}: {e}")

                except Exception as e:
                    logger.error(f"Failed to process feed {feed_url}: {e}")
                    errors.append(f"Feed processing failed: {e}")
                    continue

            logger.info(f"Extracted {articles_fetched} articles")

            if not all_articles:
                logger.warning("No articles fetched, stopping pipeline")
                return ETLResults(0, 0, 0, ["No articles fetched"], time.time() - start_time)

            # TRANSFORM: Analyze content with AI
            logger.info("Phase 2: Transform - Analyzing content")
            for article_data in all_articles:
                try:
                    entry = article_data['entry']
                    content = article_data['content']

                    # Analyze with AI
                    analysis = self.ai_client.analyze_article(
                        title=content.title or entry.title,
                        content=content.content,
                        url=entry.link
                    )

                    if analysis:
                        article_data['analysis'] = analysis
                        articles_analyzed += 1
                    else:
                        errors.append(f"Analysis failed for {entry.link}")

                except Exception as e:
                    logger.error(f"Failed to analyze article: {e}")
                    errors.append(f"Analysis error: {e}")
                    continue

            logger.info(f"Analyzed {articles_analyzed} articles successfully")

            # LOAD: Store results
            logger.info("Phase 3: Load - Storing results")
            for article_data in all_articles:
                try:
                    entry = article_data['entry']
                    content = article_data.get('content')
                    analysis = article_data.get('analysis')

                    if not content:
                        continue

                    # Store in database
                    article_id = self.database.insert_article(
                        title=entry.title,
                        url=entry.link,
                        publication_date=entry.publication_date,
                        content_hash=entry.content_hash,
                        status='completed' if analysis else 'failed'
                    )

                    if content and analysis:
                        self.database.insert_content(
                            article_id=article_id,
                            content=content.content,
                            analysis=analysis
                        )

                    articles_stored += 1

                except Exception as e:
                    logger.error(f"Failed to store article: {e}")
                    errors.append(f"Storage error: {e}")
                    continue

            logger.info(f"Stored {articles_stored} articles")

            # Generate reports
            self._generate_outputs()

            processing_time = time.time() - start_time
            logger.info(f"ETL pipeline completed in {processing_time:.2f} seconds")

            return ETLResults(
                articles_fetched=articles_fetched,
                articles_analyzed=articles_analyzed,
                articles_stored=articles_stored,
                errors=errors,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            errors.append(str(e))
            return ETLResults(
                articles_fetched=articles_fetched,
                articles_analyzed=articles_analyzed,
                articles_stored=articles_stored,
                errors=errors,
                processing_time=time.time() - start_time
            )

    def _generate_outputs(self):
        """Generate all output formats"""
        try:
            # Generate reports
            articles = self.database.get_articles_with_analysis()
            if articles:
                self.report_generator.generate_markdown_report(articles)

        except Exception as e:
            logger.error(f"Failed to generate outputs: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed articles"""
        return self.database.get_processing_statistics()

    @property
    def content_fetcher(self):
        """Backwards compatibility property"""
        return self

    def get_feed_info(self, feed_url: str) -> Dict[str, Any]:
        """Get RSS feed information"""
        return self.rss_parser.get_feed_info(feed_url)
