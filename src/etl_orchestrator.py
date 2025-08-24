"""
ETL Orchestrator

Main coordination class that orchestrates the Extract, Transform, Load pipeline.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from .etl.extract.content_fetcher import ContentFetcher, FetchedArticle
from .etl.transform.analysis_engine import AnalysisEngine, AnalysisRequest, AnalysisResult
from .etl.load.database import DatabaseManager
from .etl.load.report_generator import ReportGenerator
from .etl.load.website_generator import WebsiteDataGenerator
from .etl.load.data_exporter import DataExporter

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
    """Orchestrates the complete ETL pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.content_fetcher = ContentFetcher()
        self.analysis_engine = AnalysisEngine(config)
        self.database = DatabaseManager(config.get("database_path", "data/articles.db"))
        self.report_generator = ReportGenerator(config.get("output_dir", "output"))
        self.website_generator = WebsiteDataGenerator(
            config.get("database_path", "data/articles.db"),
            config.get("docs_dir", "docs")
        )
        self.data_exporter = DataExporter(config.get("output_dir", "output"))
    
    def run_full_pipeline(self, feed_urls: List[str], max_articles: int = None) -> ETLResults:
        """Run the complete ETL pipeline"""
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
            fetched_articles = self._extract_content(feed_urls, max_articles)
            articles_fetched = len(fetched_articles)
            logger.info(f"Extracted {articles_fetched} articles")
            
            if not fetched_articles:
                logger.warning("No articles fetched, stopping pipeline")
                return ETLResults(0, 0, 0, ["No articles fetched"], time.time() - start_time)
            
            # TRANSFORM: Analyze content with AI
            logger.info("Phase 2: Transform - Analyzing content")
            analysis_results = self._transform_content(fetched_articles)
            articles_analyzed = sum(1 for r in analysis_results if r.success)
            logger.info(f"Analyzed {articles_analyzed} articles successfully")
            
            # LOAD: Store results and generate outputs
            logger.info("Phase 3: Load - Storing results and generating outputs")
            articles_stored = self._load_results(fetched_articles, analysis_results)
            logger.info(f"Stored {articles_stored} articles")
            
            # Generate reports and website
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
    
    def _extract_content(self, feed_urls: List[str], max_articles: int = None) -> List[FetchedArticle]:
        """Extract content from RSS feeds"""
        all_fetched = []
        
        for feed_url in feed_urls:
            try:
                # Get existing hashes to avoid duplicates
                existing_hashes = self.database.get_existing_content_hashes()
                
                # Fetch RSS entries
                rss_entries = self.content_fetcher.fetch_from_rss(feed_url, existing_hashes)
                
                if max_articles:
                    rss_entries = rss_entries[:max_articles]
                
                # Fetch full content for each entry
                fetched_articles = self.content_fetcher.fetch_articles_batch(rss_entries)
                all_fetched.extend(fetched_articles)
                
            except Exception as e:
                logger.error(f"Failed to fetch from {feed_url}: {e}")
                continue
        
        return all_fetched
    
    def _transform_content(self, fetched_articles: List[FetchedArticle]) -> List[AnalysisResult]:
        """Transform content through AI analysis"""
        analysis_requests = []
        
        for fetched in fetched_articles:
            if fetched.article_content:
                request = AnalysisRequest(
                    content=fetched.article_content.content,
                    title=fetched.article_content.title or fetched.rss_entry.title,
                    url=fetched.rss_entry.link
                )
                analysis_requests.append(request)
        
        return self.analysis_engine.batch_analyze(analysis_requests)
    
    def _load_results(self, fetched_articles: List[FetchedArticle], analysis_results: List[AnalysisResult]) -> int:
        """Load results into database"""
        stored_count = 0
        
        for fetched, analysis in zip(fetched_articles, analysis_results):
            try:
                # Store in database
                article_id = self.database.insert_article(
                    title=fetched.rss_entry.title,
                    url=fetched.rss_entry.link,
                    publication_date=fetched.rss_entry.publication_date,
                    content_hash=fetched.rss_entry.content_hash
                )
                
                if fetched.article_content and analysis.success:
                    self.database.insert_content(
                        article_id=article_id,
                        content=fetched.article_content.content,
                        analysis=analysis.analysis
                    )
                    
                    self.database.update_article_status(article_id, "completed")
                else:
                    self.database.update_article_status(article_id, "failed")
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to store article {fetched.rss_entry.link}: {e}")
                continue
        
        return stored_count
    
    def _generate_outputs(self):
        """Generate all output formats"""
        try:
            # Generate website data
            self.website_generator.generate_data()
            
            # Generate reports
            articles = self.database.get_articles_with_analysis()
            if articles:
                self.report_generator.generate_markdown_report(articles)
                self.data_exporter.export_to_json({"articles": articles})
                self.data_exporter.export_to_csv(articles)
                self.data_exporter.export_statistics(articles)
            
        except Exception as e:
            logger.error(f"Failed to generate outputs: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed articles"""
        return self.database.get_processing_statistics()