"""
New Main CLI Application using ETL Architecture

Clean entry point using the new Extract, Transform, Load architecture.
"""

import logging
import sys
from typing import List

import click

from .etl_orchestrator import ETLOrchestrator
from .config.settings import load_config, setup_logging, validate_config

logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.pass_context
def cli(ctx, config, log_level, log_file):
    """RSS Article Analyzer - Clean ETL Architecture"""
    
    ctx.ensure_object(dict)
    
    try:
        # Load configuration
        app_config = load_config(config)
        
        # Override with CLI options
        if log_level:
            app_config["log_level"] = log_level
        if log_file:
            app_config["log_file"] = log_file
        
        # Setup logging
        setup_logging(app_config["log_level"], app_config.get("log_file"))
        
        # Validate configuration
        if not validate_config(app_config):
            click.echo("Configuration validation failed. Please check your settings.", err=True)
            sys.exit(1)
        
        # Store config in context
        ctx.obj['config'] = app_config
        
    except Exception as e:
        click.echo(f"Initialization failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--limit", "-l", default=10, help="Maximum number of articles to process")
@click.option("--feeds", "-f", help="Comma-separated list of feed URLs")
@click.option("--force-refresh", is_flag=True, help="Force refresh of existing articles")
@click.pass_context
def run(ctx, limit, feeds, force_refresh):
    """Run the complete ETL pipeline"""
    
    config = ctx.obj['config']
    
    try:
        # Get feed URLs
        feed_urls = []
        if feeds:
            feed_urls = [url.strip() for url in feeds.split(',')]
        else:
            feed_urls = config.get("rss_feeds", [])
        
        if not feed_urls:
            click.echo("No RSS feeds configured. Use --feeds or configure in settings.", err=True)
            return
        
        click.echo(f"🚀 Starting ETL pipeline with {len(feed_urls)} feeds (limit: {limit})")
        
        # Initialize orchestrator
        orchestrator = ETLOrchestrator(config)
        
        # Run pipeline
        results = orchestrator.run_full_pipeline(feed_urls, limit)
        
        # Report results
        click.echo(f"\n📊 Pipeline Results:")
        click.echo(f"   📥 Articles fetched: {results.articles_fetched}")
        click.echo(f"   🔄 Articles analyzed: {results.articles_analyzed}")
        click.echo(f"   💾 Articles stored: {results.articles_stored}")
        click.echo(f"   ⏱️  Processing time: {results.processing_time:.2f}s")
        
        if results.errors:
            click.echo(f"   ❌ Errors: {len(results.errors)}")
            for error in results.errors[:3]:  # Show first 3 errors
                click.echo(f"      • {error}")
        
        click.echo(f"\n✅ ETL pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def extract(ctx):
    """Run only the Extract phase (fetch content)"""
    click.echo("🔍 Running Extract phase only...")
    # Implementation for extract-only mode
    

@cli.command() 
@click.pass_context
def transform(ctx):
    """Run only the Transform phase (analyze content)"""
    click.echo("🔄 Running Transform phase only...")
    # Implementation for transform-only mode


@cli.command()
@click.pass_context
def load(ctx):
    """Run only the Load phase (generate outputs)"""
    click.echo("💾 Running Load phase only...")
    # Implementation for load-only mode


@cli.command()
@click.pass_context
def stats(ctx):
    """Show processing statistics"""
    config = ctx.obj['config']
    
    try:
        orchestrator = ETLOrchestrator(config)
        stats = orchestrator.get_processing_statistics()
        
        click.echo("📊 Processing Statistics:")
        click.echo(f"   Total articles: {stats.get('total_articles', 0)}")
        click.echo(f"   Completed: {stats.get('completed', 0)}")
        click.echo(f"   Failed: {stats.get('failed', 0)}")
        click.echo(f"   Pending: {stats.get('pending', 0)}")
        
    except Exception as e:
        click.echo(f"Failed to get statistics: {e}", err=True)


@cli.command()
@click.option("--feed-url", "-u", required=True, help="RSS feed URL to test")
@click.pass_context
def test_rss(ctx, feed_url):
    """Test RSS feed connectivity"""
    config = ctx.obj['config']
    
    try:
        orchestrator = ETLOrchestrator(config)
        feed_info = orchestrator.content_fetcher.get_feed_info(feed_url)
        
        click.echo(f"📡 RSS Feed Information:")
        click.echo(f"   Title: {feed_info.get('title', 'Unknown')}")
        click.echo(f"   Description: {feed_info.get('description', 'N/A')}")
        click.echo(f"   Entries: {feed_info.get('entry_count', 0)}")
        click.echo(f"   Language: {feed_info.get('language', 'Unknown')}")
        
    except Exception as e:
        click.echo(f"❌ RSS test failed: {e}", err=True)


@cli.command()
@click.pass_context
def test_ai(ctx):
    """Test AI provider connectivity"""
    config = ctx.obj['config']
    
    try:
        from .etl.transform.ai_clients.factory import AIClientFactory
        
        provider = config.get("default_ai_provider", "anthropic")
        factory = AIClientFactory()
        client = factory.create_client(provider, config)
        
        # Test with simple content
        result = client.analyze_article(
            title="Test Article",
            content="This is a test article to verify AI connectivity.",
            url="https://example.com/test"
        )
        
        if result:
            click.echo(f"✅ AI provider '{provider}' is working correctly")
        else:
            click.echo(f"❌ AI provider '{provider}' returned empty result")
            
    except Exception as e:
        click.echo(f"❌ AI test failed: {e}", err=True)


if __name__ == "__main__":
    cli()