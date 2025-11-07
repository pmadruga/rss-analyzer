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
@click.option(
    "--async", "use_async", is_flag=True, help="Use async processing for 6-8x speedup"
)
@click.option(
    "--max-concurrent", type=int, default=5, help="Max concurrent articles (async mode)"
)
@click.pass_context
def run(ctx, force_refresh, limit, output_dir, no_follow_links, max_linked, use_async, max_concurrent):
    """Run the complete article analysis pipeline"""

    config = ctx.obj["config"]

    # Override configuration options
    if output_dir:
        config["output_dir"] = output_dir

    # Apply limits from config if not specified
    if not limit:
        limit = config.get("max_articles_per_run")

    # Add max_concurrent to config for async mode
    config["max_concurrent_articles"] = max_concurrent

    # Create processing configuration
    from .processors import ProcessingConfig
    processing_config = ProcessingConfig(
        force_refresh=force_refresh,
        limit=limit,
        follow_links=not no_follow_links,
        max_linked_articles=max_linked,
        max_concurrent=max_concurrent,
    )

    try:
        if use_async:
            # Use async processor for 6-8x throughput
            import asyncio
            from .processors import AsyncArticleProcessor

            processor = AsyncArticleProcessor(config)

            click.echo("🚀 Starting ASYNC RSS article analysis pipeline...")
            click.echo(f"⚡ Async mode enabled: {max_concurrent} concurrent articles")

            # Show configuration info
            client_info = processor.get_client_info()
            click.echo(
                f"📡 Using {client_info['provider']} with model {client_info['model']}"
            )

            if processing_config.limit:
                click.echo(f"📊 Processing limited to {processing_config.limit} articles")

            if processing_config.force_refresh:
                click.echo("🔄 Force refresh enabled - reprocessing all articles")

            # Run async pipeline
            results = asyncio.run(processor.run_async(processing_config))

            # Display results
            _display_results(results, config)
        else:
            # Use sync processor (original)
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
        from .core import get_monitor

        click.echo("🏥 Running system health check...")
        health_results = {}

        # Get system health from monitoring
        monitor = get_monitor()
        system_health = monitor.get_system_health()

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

        # Add system health to results
        health_results["system"] = system_health.to_dict()

        # Display system health
        click.echo("\n💻 System Health:")
        click.echo(f"   Status: {system_health.status}")
        click.echo(f"   Memory Available: {system_health.memory_available_mb:.1f} MB")
        click.echo(f"   Disk Space: {system_health.disk_space_available_mb:.1f} MB")
        click.echo(f"   CPU Count: {system_health.cpu_count}")
        if system_health.issues:
            click.echo("   ⚠️  Issues:")
            for issue in system_health.issues:
                click.echo(f"      - {issue}")

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


@cli.command()
@click.option("--format", "-f", default="json", type=click.Choice(["json", "csv"]), help="Output format")
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def metrics(ctx, format, output):
    """Show current performance metrics"""
    try:
        from .core import get_monitor

        click.echo("📊 Performance Metrics")
        click.echo("=" * 60)

        monitor = get_monitor()
        metrics = monitor.get_metrics()

        # Display key metrics
        click.echo("\n🔄 Processing:")
        click.echo(f"   Articles Processed: {metrics.articles_processed}")
        click.echo(f"   Success Rate: {metrics.success_rate:.1f}%")
        click.echo(f"   Total Time: {metrics.total_processing_time:.1f}s")
        click.echo(f"   Average Time: {metrics.average_processing_time:.2f}s")

        click.echo("\n🧠 API Calls:")
        click.echo(f"   Total Calls: {metrics.api_calls_made}")
        click.echo(f"   Total Time: {metrics.api_call_time:.1f}s")
        click.echo(f"   Average Time: {metrics.avg_api_call_time:.2f}s")
        click.echo(f"   Tokens Used: {metrics.api_tokens_used}")
        click.echo(f"   Estimated Cost: ${metrics.api_cost_estimate:.4f}")

        click.echo("\n🕷️  Web Scraping:")
        click.echo(f"   Pages Scraped: {metrics.pages_scraped}")
        click.echo(f"   Failed Scrapes: {metrics.failed_scrapes}")
        click.echo(f"   Total Time: {metrics.scraping_time:.1f}s")
        click.echo(f"   Average Time: {metrics.avg_scraping_time:.2f}s")
        click.echo(f"   Links Followed: {metrics.followed_links}")

        click.echo("\n💾 Database:")
        click.echo(f"   Queries Executed: {metrics.db_queries_executed}")
        click.echo(f"   Total Time: {metrics.db_query_time:.2f}s")
        click.echo(f"   Average Time: {metrics.avg_db_query_time:.4f}s")
        click.echo(f"   Cache Hit Rate: {metrics.cache_hit_rate:.1f}%")

        click.echo("\n💻 System:")
        click.echo(f"   Memory Usage: {metrics.memory_usage_mb:.1f} MB")
        click.echo(f"   CPU Usage: {metrics.cpu_usage_percent:.1f}%")
        click.echo(f"   Errors: {metrics.error_count}")
        click.echo(f"   Warnings: {metrics.warning_count}")

        # Export to file if requested
        if output:
            monitor.save_metrics(output, format)
            click.echo(f"\n📄 Metrics exported to: {output}")
        else:
            # Print export data
            click.echo(f"\n📋 {format.upper()} Export:")
            click.echo(monitor.export_metrics(format))

    except Exception as e:
        click.echo(f"❌ Failed to get metrics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def cache_stats(ctx):
    """Show cache statistics"""
    config = ctx.obj["config"]

    try:
        from .processors import ArticleProcessor

        processor = ArticleProcessor(config)
        stats = processor.get_cache_stats()

        click.echo("💾 Cache Statistics")
        click.echo("=" * 60)

        click.echo("\n📊 Performance:")
        click.echo(f"   Hit Rate: {stats['hit_rate']:.1f}%")
        click.echo(f"   Total Hits: {stats['total_hits']}")
        click.echo(f"   Total Misses: {stats['total_misses']}")

        click.echo("\n🔹 L1 Cache (Memory):")
        click.echo(f"   Hits: {stats['l1_hits']}")
        click.echo(f"   Misses: {stats['l1_misses']}")
        click.echo(f"   Hit Rate: {stats['l1_hit_rate']:.1f}%")
        click.echo(f"   Entries: {stats['l1_entries']}")
        click.echo(f"   Size: {stats['l1_size_mb']:.2f} MB")

        click.echo("\n🔸 L2 Cache (Disk):")
        click.echo(f"   Hits: {stats['l2_hits']}")
        click.echo(f"   Misses: {stats['l2_misses']}")
        click.echo(f"   Hit Rate: {stats['l2_hit_rate']:.1f}%")
        click.echo(f"   Entries: {stats['l2_entries']}")
        click.echo(f"   Size: {stats['l2_size_mb']:.2f} MB")

        click.echo("\n📈 Overall:")
        click.echo(f"   Total Size: {stats['total_size_mb']:.2f} MB")
        click.echo(f"   Evictions: {stats['evictions']}")
        click.echo(f"   Expirations: {stats['expirations']}")

    except Exception as e:
        click.echo(f"❌ Failed to get cache stats: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def clear_cache(ctx, confirm):
    """Clear all cached content"""
    config = ctx.obj["config"]

    try:
        from .processors import ArticleProcessor

        if not confirm:
            if not click.confirm("Are you sure you want to clear all cached content?"):
                click.echo("Cache clear cancelled.")
                return

        processor = ArticleProcessor(config)
        processor.clear_cache()

        click.echo("✅ Cache cleared successfully")

    except Exception as e:
        click.echo(f"❌ Failed to clear cache: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def cleanup_cache(ctx):
    """Remove expired cache entries"""
    config = ctx.obj["config"]

    try:
        from .processors import ArticleProcessor

        processor = ArticleProcessor(config)
        removed = processor.cleanup_expired_cache()

        if removed > 0:
            click.echo(f"✅ Removed {removed} expired cache entries")
        else:
            click.echo("✅ No expired cache entries found")

    except Exception as e:
        click.echo(f"❌ Failed to cleanup cache: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--iterations", "-i", default=10, type=int, help="Number of iterations")
@click.option("--output", "-o", help="Output file for results")
@click.pass_context
def benchmark(ctx, iterations, output):
    """Run performance benchmark"""
    config = ctx.obj["config"]

    try:
        import time
        from .core import DatabaseManager, RSSParser, get_monitor

        click.echo(f"🏃 Running performance benchmark ({iterations} iterations)...")
        click.echo("=" * 60)

        monitor = get_monitor()
        results = {
            "iterations": iterations,
            "start_time": time.time(),
            "tests": {}
        }

        # Benchmark: Database operations
        click.echo("\n📊 Benchmarking database operations...")
        db = DatabaseManager(config["db_path"])
        db_times = []
        for i in range(iterations):
            start = time.perf_counter()
            with monitor.track_db_query():
                stats = db.get_processing_statistics()
            db_times.append(time.perf_counter() - start)

        results["tests"]["database"] = {
            "avg_time": sum(db_times) / len(db_times),
            "min_time": min(db_times),
            "max_time": max(db_times),
        }
        click.echo(f"   Average: {results['tests']['database']['avg_time']:.4f}s")

        # Benchmark: RSS parsing
        click.echo("\n📰 Benchmarking RSS feed parsing...")
        parser = RSSParser()
        rss_times = []
        for i in range(min(iterations, 3)):  # Limit RSS tests to avoid hammering feed
            start = time.perf_counter()
            entries = parser.fetch_feed(config["rss_feed_url"])
            rss_times.append(time.perf_counter() - start)

        results["tests"]["rss_parsing"] = {
            "avg_time": sum(rss_times) / len(rss_times),
            "min_time": min(rss_times),
            "max_time": max(rss_times),
            "entries_fetched": len(entries) if entries else 0,
        }
        click.echo(f"   Average: {results['tests']['rss_parsing']['avg_time']:.2f}s")
        click.echo(f"   Entries: {results['tests']['rss_parsing']['entries_fetched']}")

        # Benchmark: Content hash generation
        click.echo("\n🔐 Benchmarking content hashing...")
        from .core import create_content_hash
        hash_times = []
        test_content = "Test article content " * 100
        for i in range(iterations * 10):  # More iterations for fast operation
            start = time.perf_counter()
            create_content_hash(test_content)
            hash_times.append(time.perf_counter() - start)

        results["tests"]["content_hashing"] = {
            "avg_time": sum(hash_times) / len(hash_times),
            "min_time": min(hash_times),
            "max_time": max(hash_times),
        }
        click.echo(f"   Average: {results['tests']['content_hashing']['avg_time']:.6f}s")

        results["end_time"] = time.time()
        results["total_duration"] = results["end_time"] - results["start_time"]

        # Display summary
        click.echo("\n" + "=" * 60)
        click.echo("📈 BENCHMARK SUMMARY")
        click.echo("=" * 60)
        click.echo(f"Total Duration: {results['total_duration']:.2f}s")
        click.echo(f"Iterations: {iterations}")

        # Save results if requested
        if output:
            import json
            os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            click.echo(f"\n📄 Results saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
