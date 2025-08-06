"""
Refactored Main CLI Application

Entry point for the RSS article analyzer with command-line interface,
using the new refactored architecture with improved error handling and modularity.
"""

import logging
import os
import sys

import click

from .clients import AIClientFactory
from .core import load_config, setup_logging, validate_config
from .exceptions import RSSAnalyzerError
from .processors import ArticleProcessor, ProcessingConfig

logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", help="Configuration file path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.pass_context
def cli(ctx, config, log_level, log_file):
    """RSS Article Analyzer - Fetch, scrape, and analyze articles using AI"""

    # Ensure context object exists
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
            click.echo(
                "Configuration validation failed. Please check your settings.", err=True
            )
            sys.exit(1)

        ctx.obj["config"] = app_config

    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--force-refresh", is_flag=True, help="Reprocess all articles")
@click.option("--limit", "-l", type=int, help="Limit number of articles to process")
@click.option("--output-dir", help="Output directory for reports")
@click.option("--no-follow-links", is_flag=True, help="Disable link following")
@click.option(
    "--max-linked", type=int, default=3, help="Maximum linked articles to analyze"
)
@click.pass_context
def run(ctx, force_refresh, limit, output_dir, no_follow_links, max_linked):
    """Run the complete article analysis pipeline"""

    config = ctx.obj["config"]

    # Override configuration options
    if output_dir:
        config["output_dir"] = output_dir

    # Apply limits from config if not specified
    if not limit:
        limit = config.get("max_articles_per_run")

    # Create processing configuration
    processing_config = ProcessingConfig(
        force_refresh=force_refresh,
        limit=limit,
        follow_links=not no_follow_links,
        max_linked_articles=max_linked,
    )

    try:
        processor = ArticleProcessor(config)

        click.echo("🚀 Starting RSS article analysis pipeline...")

        # Show configuration info
        client_info = processor.get_client_info()
        click.echo(
            f"📡 Using {client_info['provider']} with model {client_info['model']}"
        )

        if processing_config.limit:
            click.echo(f"📊 Processing limited to {processing_config.limit} articles")

        if processing_config.force_refresh:
            click.echo("🔄 Force refresh enabled - reprocessing all articles")

        results = processor.run(processing_config)

        # Display results
        _display_results(results, config)

    except KeyboardInterrupt:
        click.echo("\n❌ Process interrupted by user", err=True)
        sys.exit(1)
    except RSSAnalyzerError as e:
        click.echo(f"❌ Application error: {e}", err=True)
        logger.exception("Application error in main pipeline")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Fatal error: {e}", err=True)
        logger.exception("Fatal error in main pipeline")
        sys.exit(1)


def _display_results(results, config):
    """Display processing results to user"""
    click.echo("\n" + "=" * 60)
    click.echo("📈 PROCESSING RESULTS")
    click.echo("=" * 60)

    # Basic metrics
    click.echo(f"📰 RSS entries found: {results.rss_entries_found}")
    click.echo(f"🆕 New articles: {results.new_articles}")
    click.echo(f"🕷️  Successfully scraped: {results.scraped_articles}")
    click.echo(f"🧠 Successfully analyzed: {results.analyzed_articles}")
    click.echo(f"⏱️  Processing time: {results.duration:.2f} seconds")

    # Success rate
    if results.new_articles > 0:
        success_rate = (results.analyzed_articles / results.new_articles) * 100
        click.echo(f"✅ Success rate: {success_rate:.1f}%")

    # Reports
    if results.report_generated and results.reports:
        click.echo(f"\n📄 Reports generated in: {config['output_dir']}")
        for report_type, path in results.reports.items():
            click.echo(f"   📋 {report_type}: {os.path.basename(path)}")

    # Errors
    if results.errors:
        click.echo(f"\n⚠️  Errors encountered: {len(results.errors)}")
        for i, error in enumerate(results.errors[:3], 1):  # Show first 3 errors
            click.echo(f"   {i}. {error}")
        if len(results.errors) > 3:
            click.echo(f"   ... and {len(results.errors) - 3} more errors")

    click.echo("=" * 60)


@cli.command()
@click.pass_context
def test_api(ctx):
    """Test API connection for all configured providers"""
    config = ctx.obj["config"]

    try:
        click.echo("🔍 Testing API connections...")

        # Get current provider
        api_provider = config.get("api_provider", "anthropic")

        # Test current provider
        try:
            client = AIClientFactory.create_from_config(config)
            provider_info = client.get_provider_info()

            click.echo(
                f"\n📡 Testing {provider_info['provider']} ({provider_info['model']})..."
            )

            if client.test_connection():
                click.echo(f"✅ {provider_info['provider']} connection successful")
            else:
                click.echo(
                    f"❌ {provider_info['provider']} connection failed", err=True
                )
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ {api_provider} test failed: {e}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Connection test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def test_rss(ctx):
    """Test RSS feed parsing"""
    config = ctx.obj["config"]

    try:
        from .core import RSSParser

        click.echo("🔍 Testing RSS feed connection...")

        rss_parser = RSSParser()
        feed_info = rss_parser.get_feed_info(config["rss_feed_url"])

        click.echo("✅ RSS feed information:")
        click.echo(f"   📰 Title: {feed_info['title']}")
        click.echo(f"   📝 Description: {feed_info['description'][:100]}...")
        click.echo(f"   📊 Entries: {feed_info['entry_count']}")

    except Exception as e:
        click.echo(f"❌ RSS test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show processing statistics"""
    config = ctx.obj["config"]

    try:
        from .core import DatabaseManager

        db = DatabaseManager(config["db_path"])
        stats = db.get_processing_statistics()

        click.echo("📊 Database Statistics:")
        click.echo(f"   📰 Total articles: {stats.get('total_articles', 0)}")

        if "by_status" in stats:
            click.echo("   📈 By status:")
            for status, count in stats["by_status"].items():
                status_icon = (
                    "✅"
                    if status == "completed"
                    else "❌"
                    if status == "failed"
                    else "⏳"
                )
                click.echo(f"      {status_icon} {status}: {count}")

        if "recent_activity" in stats:
            click.echo("\n📅 Recent activity:")
            for activity in stats["recent_activity"][:5]:
                click.echo(f"   📅 {activity['date']}: {activity['count']} operations")

    except Exception as e:
        click.echo(f"❌ Failed to get statistics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_reports(ctx):
    """List generated reports"""
    config = ctx.obj["config"]

    try:
        from .core import ReportGenerator

        report_gen = ReportGenerator(config["output_dir"])
        reports = report_gen.list_reports()

        if not reports:
            click.echo("📄 No reports found")
            return

        click.echo("📄 Generated Reports:")
        for report in reports:
            size_mb = report["size_bytes"] / (1024 * 1024)
            click.echo(f"   📋 {report['filename']}")
            click.echo(f"      💾 Size: {size_mb:.1f} MB")
            click.echo(f"      📅 Modified: {report['modified_time']}")
            click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to list reports: {e}", err=True)


@cli.command()
@click.option("--provider", help="Test specific provider (anthropic, mistral, openai)")
@click.pass_context
def providers(ctx, provider):
    """List available AI providers and their status"""
    config = ctx.obj["config"]

    try:
        click.echo("🤖 Available AI Providers:")
        click.echo("=" * 40)

        available_providers = AIClientFactory.get_available_providers()
        current_provider = config.get("api_provider", "anthropic")

        for provider_name, default_model in available_providers.items():
            # Skip aliases
            if provider_name == "claude":
                continue

            status_icon = "🟢" if provider_name == current_provider else "⚪"
            click.echo(f"{status_icon} {provider_name.title()}")
            click.echo(f"   🎯 Default model: {default_model}")

            # Check if API key is configured
            key_field = f"{provider_name}_api_key"
            if provider_name == "anthropic":
                key_field = "anthropic_api_key"

            has_key = bool(config.get(key_field))
            key_status = "✅ Configured" if has_key else "❌ Not configured"
            click.echo(f"   🔑 API Key: {key_status}")

            # Test connection if requested and key is available
            if provider and provider.lower() == provider_name and has_key:
                try:
                    test_config = config.copy()
                    test_config["api_provider"] = provider_name
                    client = AIClientFactory.create_from_config(test_config)

                    click.echo("   🔍 Testing connection...")
                    if client.test_connection():
                        click.echo("   ✅ Connection successful")
                    else:
                        click.echo("   ❌ Connection failed")

                except Exception as e:
                    click.echo(f"   ❌ Test error: {e}")

            click.echo()

        click.echo(f"🎯 Current provider: {current_provider}")

    except Exception as e:
        click.echo(f"❌ Failed to list providers: {e}", err=True)


@cli.command()
@click.option(
    "--output", "-o", default="logs/health_report.json", help="Output file for report"
)
@click.pass_context
def health(ctx, output):
    """Run comprehensive system health check"""
    config = ctx.obj["config"]

    try:
        click.echo("🏥 Running system health check...")
        health_results = {}

        # Test database
        click.echo("📊 Checking database...")
        try:
            from .core import DatabaseManager

            db = DatabaseManager(config["db_path"])
            stats = db.get_processing_statistics()
            health_results["database"] = {
                "status": "healthy",
                "total_articles": stats.get("total_articles", 0),
            }
            click.echo("   ✅ Database connection successful")
        except Exception as e:
            health_results["database"] = {"status": "error", "error": str(e)}
            click.echo(f"   ❌ Database error: {e}")

        # Test RSS feed
        click.echo("📰 Checking RSS feed...")
        try:
            from .core import RSSParser

            parser = RSSParser()
            feed_info = parser.get_feed_info(config["rss_feed_url"])
            health_results["rss_feed"] = {
                "status": "healthy",
                "entries": feed_info["entry_count"],
            }
            click.echo(
                f"   ✅ RSS feed accessible ({feed_info['entry_count']} entries)"
            )
        except Exception as e:
            health_results["rss_feed"] = {"status": "error", "error": str(e)}
            click.echo(f"   ❌ RSS feed error: {e}")

        # Test AI provider
        click.echo("🧠 Checking AI provider...")
        try:
            client = AIClientFactory.create_from_config(config)
            if client.test_connection():
                provider_info = client.get_provider_info()
                health_results["ai_provider"] = {
                    "status": "healthy",
                    "provider": provider_info["provider"],
                    "model": provider_info["model"],
                }
                click.echo(f"   ✅ {provider_info['provider']} connection successful")
            else:
                health_results["ai_provider"] = {
                    "status": "error",
                    "error": "Connection test failed",
                }
                click.echo("   ❌ AI provider connection failed")
        except Exception as e:
            health_results["ai_provider"] = {"status": "error", "error": str(e)}
            click.echo(f"   ❌ AI provider error: {e}")

        # Save health report
        import json

        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(health_results, f, indent=2)

        click.echo(f"\n📄 Health report saved to: {output}")

        # Overall status
        all_healthy = all(
            result.get("status") == "healthy" for result in health_results.values()
        )
        overall_status = (
            "✅ All systems healthy" if all_healthy else "⚠️  Some issues detected"
        )
        click.echo(f"🏥 Overall status: {overall_status}")

    except Exception as e:
        click.echo(f"❌ Health check failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
