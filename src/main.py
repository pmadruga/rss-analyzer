"""
Main CLI Application

Entry point for the RSS article analyzer with command-line interface,
orchestrating the entire pipeline from RSS parsing to report generation.
"""

import logging
import os
import sys
import time

import click
from tqdm import tqdm

from .database import DatabaseManager
from .mistral_client import MistralClient
from .openai_client import OpenAIClient
from .report_generator import ReportGenerator
from .rss_parser import RSSParser
from .scraper import WebScraper

# Import our modules
from .utils import format_timestamp, load_config, setup_logging, validate_config

logger = logging.getLogger(__name__)


class ArticleProcessor:
    """Main application orchestrator"""

    def __init__(self, config: dict):
        self.config = config

        # Initialize components
        self.db = DatabaseManager(config['db_path'])
        self.rss_parser = RSSParser(config.get('user_agent', 'RSS-Article-Analyzer/1.0'))
        self.scraper = WebScraper(config.get('scraper_delay', 1.0))

        # Initialize API client based on provider
        api_provider = config.get('api_provider', 'mistral')

        if api_provider == 'mistral':
            self.api_client = MistralClient(
                api_key=config['mistral_api_key'],
                model=config.get('mistral_model', 'mistral-large-latest')
            )
            logger.info("Using Mistral API")
        elif api_provider == 'openai':
            self.api_client = OpenAIClient(
                api_key=config['openai_api_key'],
                model=config.get('openai_model', 'gpt-4')
            )
            logger.info("Using OpenAI API")
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}. Supported providers: mistral, openai")

        self.report_generator = ReportGenerator(config['output_dir'])

    def run(self, force_refresh: bool = False, limit: int | None = None) -> dict:
        """
        Run the complete processing pipeline
        
        Args:
            force_refresh: Reprocess all articles
            limit: Limit number of articles to process
            
        Returns:
            Processing results summary
        """
        start_time = time.time()
        results = {
            'start_time': start_time,
            'rss_entries_found': 0,
            'new_articles': 0,
            'scraped_articles': 0,
            'analyzed_articles': 0,
            'report_generated': False,
            'errors': []
        }

        try:
            logger.info("Starting RSS article analysis pipeline")

            # Step 1: Test API connection
            if not self.api_client.test_connection():
                raise Exception("API connection test failed")

            # Step 2: Fetch RSS feed
            logger.info("Fetching RSS feed...")
            rss_entries = self.rss_parser.fetch_feed(self.config['rss_feed_url'])
            results['rss_entries_found'] = len(rss_entries)

            if not rss_entries:
                logger.warning("No entries found in RSS feed")
                return results

            # Step 3: Filter new articles (unless force refresh)
            if not force_refresh:
                # Only skip articles that have been fully analyzed
                analyzed_hashes = self.db.get_analyzed_content_hashes()
                new_entries = self.rss_parser.filter_new_entries(rss_entries, analyzed_hashes)
                logger.info(f"Skipping {len(rss_entries) - len(new_entries)} already analyzed articles")
            else:
                new_entries = rss_entries
                logger.info("Force refresh enabled - processing all articles")

            # Apply limit if specified
            if limit and len(new_entries) > limit:
                new_entries = new_entries[:limit]
                logger.info(f"Limited processing to {limit} articles")

            results['new_articles'] = len(new_entries)

            if not new_entries:
                logger.info("No new articles to process")
                return results

            # Step 4: Process articles
            processed_articles = self._process_articles(new_entries, results)

            # Step 5: Generate reports
            if processed_articles:
                self._generate_reports(processed_articles, results)

            # Step 6: Cleanup
            self.db.cleanup_old_logs()

            results['duration'] = time.time() - start_time
            logger.info(f"Pipeline completed in {results['duration']:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['errors'].append(str(e))
            self.db.log_processing(None, 'pipeline_failed', str(e))
            return results

    def _process_articles(self, entries: list, results: dict) -> list[dict]:
        """Process articles through scraping and analysis"""
        processed_articles = []

        logger.info(f"Processing {len(entries)} articles...")

        with tqdm(total=len(entries), desc="Processing articles") as pbar:
            for entry in entries:
                try:
                    # Insert article into database
                    article_id = self.db.insert_article(
                        title=entry.title,
                        url=entry.link,
                        content_hash=entry.content_hash,
                        rss_guid=entry.guid,
                        publication_date=entry.publication_date
                    )

                    self.db.log_processing(article_id, 'started', processing_step='scraping')
                    self.db.update_article_status(article_id, 'processing')

                    # Scrape article content
                    scrape_start = time.time()
                    scraped_content = self.scraper.scrape_article(
                        entry.link,
                        follow_links=self.config.get('follow_links', True),
                        max_linked_articles=self.config.get('max_linked_articles', 3)
                    )
                    scrape_duration = time.time() - scrape_start

                    if not scraped_content:
                        logger.warning(f"Failed to scrape article: {entry.title}")
                        self.db.log_processing(article_id, 'scraping_failed',
                                             processing_step='scraping',
                                             duration_seconds=scrape_duration)
                        self.db.update_article_status(article_id, 'scraping_failed')
                        pbar.update(1)
                        continue

                    results['scraped_articles'] += 1
                    self.db.log_processing(article_id, 'scraped',
                                         processing_step='scraping',
                                         duration_seconds=scrape_duration)

                    # Analyze with API client
                    analysis_start = time.time()
                    analysis = self.api_client.analyze_article(
                        title=entry.title,
                        content=scraped_content.content,
                        url=entry.link
                    )
                    analysis_duration = time.time() - analysis_start

                    if not analysis:
                        logger.warning(f"Failed to analyze article: {entry.title}")
                        self.db.log_processing(article_id, 'analysis_failed',
                                             processing_step='analysis',
                                             duration_seconds=analysis_duration)
                        self.db.update_article_status(article_id, 'analysis_failed')
                        pbar.update(1)
                        continue

                    # Store content and analysis
                    self.db.insert_content(article_id, scraped_content.content, analysis)
                    results['analyzed_articles'] += 1

                    self.db.log_processing(article_id, 'completed',
                                         processing_step='analysis',
                                         duration_seconds=analysis_duration)
                    self.db.update_article_status(article_id, 'completed')

                    # Prepare for report generation
                    article_data = {
                        'id': article_id,
                        'title': entry.title,
                        'url': entry.link,
                        'publication_date': entry.publication_date.isoformat() if entry.publication_date else None,
                        'processed_date': format_timestamp(),
                        **analysis
                    }
                    processed_articles.append(article_data)

                    logger.info(f"Successfully processed: {entry.title}")

                except Exception as e:
                    logger.error(f"Error processing article '{entry.title}': {e}")
                    results['errors'].append(f"Processing '{entry.title}': {e}")
                    if 'article_id' in locals():
                        self.db.log_processing(article_id, 'processing_failed', str(e))
                        self.db.update_article_status(article_id, 'failed')

                pbar.update(1)

        return processed_articles

    def _generate_reports(self, articles: list[dict], results: dict):
        """Generate various reports"""
        try:
            logger.info("Generating reports...")

            # Generate main markdown report
            report_path = self.report_generator.generate_report(
                articles,
                self.config.get('report_filename', 'article_analysis_report.md')
            )
            logger.info(f"Main report generated: {report_path}")

            # Generate summary report
            summary_path = self.report_generator.generate_summary_report(articles)
            logger.info(f"Summary report generated: {summary_path}")

            # Generate JSON export
            json_path = self.report_generator.generate_json_export(articles)
            logger.info(f"JSON export generated: {json_path}")

            # Generate CSV export
            csv_path = self.report_generator.generate_csv_export(articles)
            logger.info(f"CSV export generated: {csv_path}")

            results['report_generated'] = True
            results['reports'] = {
                'main_report': report_path,
                'summary_report': summary_path,
                'json_export': json_path,
                'csv_export': csv_path
            }

        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            results['errors'].append(f"Report generation: {e}")


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', help='Log file path')
@click.pass_context
def cli(ctx, config, log_level, log_file):
    """RSS Article Analyzer - Fetch, scrape, and analyze articles using Claude"""

    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration
    app_config = load_config(config)

    # Override with CLI options
    if log_level:
        app_config['log_level'] = log_level
    if log_file:
        app_config['log_file'] = log_file

    # Setup logging
    setup_logging(app_config['log_level'], app_config.get('log_file'))

    # Validate configuration
    if not validate_config(app_config):
        click.echo("Configuration validation failed. Please check your settings.", err=True)
        sys.exit(1)

    ctx.obj['config'] = app_config


@cli.command()
@click.option('--force-refresh', is_flag=True, help='Reprocess all articles')
@click.option('--limit', '-l', type=int, help='Limit number of articles to process')
@click.option('--output-dir', help='Output directory for reports')
@click.pass_context
def run(ctx, force_refresh, limit, output_dir):
    """Run the complete article analysis pipeline"""

    config = ctx.obj['config']

    # Override configuration options
    if output_dir:
        config['output_dir'] = output_dir


    # Apply limits from config if not specified
    if not limit:
        limit = config.get('max_articles_per_run')

    try:
        processor = ArticleProcessor(config)
        results = processor.run(force_refresh=force_refresh, limit=limit)

        # Display results
        click.echo("\n" + "="*50)
        click.echo("PROCESSING RESULTS")
        click.echo("="*50)
        click.echo(f"RSS entries found: {results['rss_entries_found']}")
        click.echo(f"New articles: {results['new_articles']}")
        click.echo(f"Successfully scraped: {results['scraped_articles']}")
        click.echo(f"Successfully analyzed: {results['analyzed_articles']}")
        click.echo(f"Processing time: {results.get('duration', 0):.2f} seconds")

        if results.get('report_generated'):
            click.echo(f"\nReports generated in: {config['output_dir']}")
            if 'reports' in results:
                for report_type, path in results['reports'].items():
                    click.echo(f"  - {report_type}: {os.path.basename(path)}")

        if results.get('errors'):
            click.echo(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                click.echo(f"  - {error}")
            if len(results['errors']) > 5:
                click.echo(f"  ... and {len(results['errors']) - 5} more")

        click.echo("="*50)

    except KeyboardInterrupt:
        click.echo("\nProcess interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Fatal error: {e}", err=True)
        logger.exception("Fatal error in main pipeline")
        sys.exit(1)


@cli.command()
@click.pass_context
def test_api(ctx):
    """Test API connection"""
    config = ctx.obj['config']

    try:
        api_provider = config.get('api_provider', 'mistral')

        if api_provider == 'mistral':
            api_client = MistralClient(
                api_key=config['mistral_api_key'],
                model=config.get('mistral_model', 'mistral-large-latest')
            )
            provider_name = "Mistral"
        elif api_provider == 'openai':
            api_client = OpenAIClient(
                api_key=config['openai_api_key'],
                model=config.get('openai_model', 'gpt-4')
            )
            provider_name = "OpenAI"
        else:
            click.echo(f"❌ Unsupported API provider: {api_provider}. Supported: mistral, openai", err=True)
            sys.exit(1)

        if api_client.test_connection():
            click.echo(f"✅ {provider_name} connection successful")
        else:
            click.echo(f"❌ {provider_name} connection failed", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Connection test failed: {e}", err=True)
        sys.exit(1)




@cli.command()
@click.pass_context
def test_rss(ctx):
    """Test RSS feed parsing"""
    config = ctx.obj['config']

    try:
        rss_parser = RSSParser()
        feed_info = rss_parser.get_feed_info(config['rss_feed_url'])

        click.echo("✅ RSS feed information:")
        click.echo(f"  Title: {feed_info['title']}")
        click.echo(f"  Description: {feed_info['description'][:100]}...")
        click.echo(f"  Entries: {feed_info['entry_count']}")

    except Exception as e:
        click.echo(f"❌ RSS test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show processing statistics"""
    config = ctx.obj['config']

    try:
        db = DatabaseManager(config['db_path'])
        stats = db.get_processing_statistics()

        click.echo("Database Statistics:")
        click.echo(f"  Total articles: {stats.get('total_articles', 0)}")

        if 'by_status' in stats:
            for status, count in stats['by_status'].items():
                click.echo(f"  {status}: {count}")


        if 'recent_activity' in stats:
            click.echo("\nRecent activity:")
            for activity in stats['recent_activity'][:5]:
                click.echo(f"  {activity['date']}: {activity['count']} operations")

    except Exception as e:
        click.echo(f"❌ Failed to get statistics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_reports(ctx):
    """List generated reports"""
    config = ctx.obj['config']

    try:
        report_gen = ReportGenerator(config['output_dir'])
        reports = report_gen.list_reports()

        if not reports:
            click.echo("No reports found")
            return

        click.echo("Generated Reports:")
        for report in reports:
            size_mb = report['size_bytes'] / (1024 * 1024)
            click.echo(f"  {report['filename']}")
            click.echo(f"    Size: {size_mb:.1f} MB")
            click.echo(f"    Modified: {report['modified_time']}")
            click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to list reports: {e}", err=True)


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed API information')
@click.pass_context
def api_status(ctx, detailed):
    """Check API health status for all providers"""
    import subprocess
    import os
    from pathlib import Path
    
    try:
        # Run the quick API check script
        script_path = Path(__file__).parent.parent / "tools" / "quick_api_check.py"
        
        if not script_path.exists():
            click.echo("❌ API monitoring script not found", err=True)
            sys.exit(1)
        
        # Run the script
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True)
        
        click.echo(result.stdout)
        
        if result.stderr:
            click.echo("Errors:", err=True)
            click.echo(result.stderr, err=True)
        
        if detailed:
            click.echo("\n🔍 Detailed API Information:")
            click.echo("-" * 40)
            
            # Show configuration
            config = ctx.obj['config']
            api_config = config.get('api', {})
            
            for provider in ['anthropic', 'mistral', 'openai']:
                provider_config = api_config.get(provider, {})
                click.echo(f"\n{provider.title()}:")
                click.echo(f"  Model: {provider_config.get('model', 'Not configured')}")
                click.echo(f"  Max Retries: {provider_config.get('max_retries', 'Not configured')}")
                click.echo(f"  Timeout: {provider_config.get('timeout', 'Not configured')}s")
        
        sys.exit(result.returncode)
        
    except Exception as e:
        click.echo(f"❌ Failed to check API status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='logs/api_health_report.json', 
              help='Output file for detailed report')
@click.pass_context
def api_monitor(ctx, output):
    """Run comprehensive API monitoring and generate detailed report"""
    import subprocess
    import os
    from pathlib import Path
    
    try:
        # Run the comprehensive API monitor script
        script_path = Path(__file__).parent.parent / "tools" / "api_health_monitor.py"
        
        if not script_path.exists():
            click.echo("❌ API monitoring script not found", err=True)
            sys.exit(1)
        
        click.echo("🔍 Running comprehensive API health check...")
        
        # Set environment variable for output file
        env = os.environ.copy()
        env['API_MONITOR_OUTPUT'] = output
        
        # Run the script
        result = subprocess.run([sys.executable, str(script_path)], 
                              env=env, capture_output=True, text=True)
        
        click.echo(result.stdout)
        
        if result.stderr:
            click.echo("Errors:", err=True)
            click.echo(result.stderr, err=True)
        
        if result.returncode == 0:
            click.echo(f"\n📄 Detailed report saved to: {output}")
        
        sys.exit(result.returncode)
        
    except Exception as e:
        click.echo(f"❌ Failed to run API monitoring: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
