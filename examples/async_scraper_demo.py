#!/usr/bin/env python3
"""
Async Web Scraper Demo

Demonstrates concurrent article scraping with performance comparisons.
"""

import asyncio
import logging
import time
from typing import List

from src.core.async_scraper import AsyncWebScraper, ScrapedContent, scrape_articles_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Sample URLs for demonstration
DEMO_URLS = [
    # arXiv papers
    "https://arxiv.org/abs/2301.00001",
    "https://arxiv.org/abs/2302.00001",
    "https://arxiv.org/abs/2303.00001",

    # Tech blogs (these may or may not exist - demo purposes)
    "https://openai.com/blog/chatgpt",
    "https://anthropic.com/index/claude-2",
]


async def demo_single_scrape():
    """Demonstrate scraping a single article"""
    print("\n" + "="*70)
    print("DEMO 1: Single Article Scraping")
    print("="*70)

    scraper = AsyncWebScraper(
        delay_between_requests=1.0,
        max_concurrent=5,
        timeout=30
    )

    url = "https://arxiv.org/abs/2301.00001"
    print(f"\nScraping: {url}")

    result = await scraper.scrape_single(url, follow_links=False)

    if result:
        print(f"\n✓ Successfully scraped article")
        print(f"  Title: {result.title}")
        print(f"  Content length: {len(result.content)} characters")
        print(f"  Content hash: {result.content_hash}")
        print(f"  Metadata keys: {list(result.metadata.keys())}")
    else:
        print(f"\n✗ Failed to scrape article")


async def demo_batch_scrape():
    """Demonstrate concurrent batch scraping"""
    print("\n" + "="*70)
    print("DEMO 2: Concurrent Batch Scraping")
    print("="*70)

    scraper = AsyncWebScraper(
        delay_between_requests=1.0,
        max_concurrent=5,
        timeout=30
    )

    urls = DEMO_URLS[:3]  # Use first 3 URLs
    print(f"\nScraping {len(urls)} articles concurrently...")

    start_time = time.time()
    results = await scraper.scrape_articles_batch(urls)
    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed:.2f} seconds")
    print(f"  Success rate: {len(results)}/{len(urls)} ({len(results)/len(urls)*100:.0f}%)")

    for i, article in enumerate(results, 1):
        print(f"\n  Article {i}:")
        print(f"    Title: {article.title[:60]}...")
        print(f"    URL: {article.url}")
        print(f"    Content: {len(article.content)} chars")


async def demo_performance_comparison():
    """Compare sequential vs concurrent performance"""
    print("\n" + "="*70)
    print("DEMO 3: Performance Comparison")
    print("="*70)

    urls = DEMO_URLS[:3]

    # Simulate sequential scraping (max_concurrent=1)
    print("\n1. Sequential scraping (max_concurrent=1)...")
    scraper_seq = AsyncWebScraper(
        delay_between_requests=1.0,
        max_concurrent=1,
        timeout=30
    )

    start_time = time.time()
    results_seq = await scraper_seq.scrape_articles_batch(urls)
    seq_time = time.time() - start_time

    print(f"   Time: {seq_time:.2f}s")
    print(f"   Articles: {len(results_seq)}")

    # Concurrent scraping (max_concurrent=5)
    print("\n2. Concurrent scraping (max_concurrent=5)...")
    scraper_conc = AsyncWebScraper(
        delay_between_requests=1.0,
        max_concurrent=5,
        timeout=30
    )

    start_time = time.time()
    results_conc = await scraper_conc.scrape_articles_batch(urls)
    conc_time = time.time() - start_time

    print(f"   Time: {conc_time:.2f}s")
    print(f"   Articles: {len(results_conc)}")

    # Calculate speedup
    if seq_time > 0:
        speedup = seq_time / conc_time
        print(f"\n✓ Speedup: {speedup:.2f}x faster with concurrency")
        print(f"  Time saved: {seq_time - conc_time:.2f}s ({(1 - conc_time/seq_time)*100:.0f}%)")


async def demo_link_following():
    """Demonstrate link following and recursive scraping"""
    print("\n" + "="*70)
    print("DEMO 4: Link Following")
    print("="*70)

    scraper = AsyncWebScraper(
        delay_between_requests=1.0,
        max_concurrent=3,
        timeout=30
    )

    # Use a URL that likely has links (blog post or article)
    url = "https://arxiv.org/abs/2301.00001"
    print(f"\nScraping with link following: {url}")

    result = await scraper.scrape_single(
        url,
        follow_links=True,
        max_linked_articles=3
    )

    if result:
        print(f"\n✓ Main article scraped: {result.title[:60]}...")

        # Check for linked articles
        if "linked_articles" in result.metadata:
            linked = result.metadata["linked_articles"]
            print(f"\n  Found {len(linked)} linked articles:")
            for i, link_url in enumerate(linked, 1):
                print(f"    {i}. {link_url}")
        else:
            print("\n  No linked articles found")
    else:
        print(f"\n✗ Failed to scrape article")


async def demo_error_handling():
    """Demonstrate graceful error handling"""
    print("\n" + "="*70)
    print("DEMO 5: Error Handling")
    print("="*70)

    scraper = AsyncWebScraper(
        delay_between_requests=0.5,
        max_concurrent=3,
        timeout=10
    )

    # Mix of valid and invalid URLs
    urls = [
        "https://arxiv.org/abs/2301.00001",  # Valid
        "https://invalid-domain-xyz123.com/article",  # Invalid domain
        "https://arxiv.org/abs/9999.99999",  # Valid domain, invalid paper
        "https://example.com/404-page-not-found",  # Valid domain, 404
    ]

    print(f"\nScraping {len(urls)} URLs (some will fail)...")

    results = await scraper.scrape_articles_batch(urls)

    print(f"\n✓ Completed with graceful error handling")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(urls) - len(results)}")
    print(f"  Success rate: {len(results)/len(urls)*100:.0f}%")

    print("\nSuccessfully scraped articles:")
    for i, article in enumerate(results, 1):
        print(f"  {i}. {article.title[:60]}... ({article.url})")


async def demo_convenience_function():
    """Demonstrate convenience function usage"""
    print("\n" + "="*70)
    print("DEMO 6: Convenience Function")
    print("="*70)

    urls = DEMO_URLS[:3]
    print(f"\nUsing scrape_articles_async() convenience function...")
    print(f"URLs: {len(urls)}")

    start_time = time.time()
    results = await scrape_articles_async(
        urls,
        max_concurrent=5,
        delay=1.0
    )
    elapsed = time.time() - start_time

    print(f"\n✓ Scraped {len(results)} articles in {elapsed:.2f}s")

    for article in results:
        print(f"\n  • {article.title[:60]}...")
        print(f"    URL: {article.url}")
        print(f"    Hash: {article.content_hash}")


async def demo_academic_papers():
    """Demonstrate scraping academic papers"""
    print("\n" + "="*70)
    print("DEMO 7: Academic Paper Scraping")
    print("="*70)

    scraper = AsyncWebScraper(
        delay_between_requests=1.0,
        max_concurrent=3,
        timeout=30
    )

    # arXiv paper URLs (abstract and PDF)
    arxiv_urls = [
        "https://arxiv.org/abs/2301.00001",
        "https://arxiv.org/pdf/2302.00001.pdf",  # PDF URL (auto-converts)
        "https://arxiv.org/abs/2303.00001",
    ]

    print(f"\nScraping {len(arxiv_urls)} arXiv papers...")

    papers = await scraper.scrape_articles_batch(arxiv_urls)

    print(f"\n✓ Scraped {len(papers)} papers")

    for i, paper in enumerate(papers, 1):
        print(f"\n  Paper {i}:")
        print(f"    Title: {paper.title[:60]}...")
        print(f"    Source: {paper.metadata.get('source', 'Unknown')}")
        print(f"    Authors: {paper.metadata.get('authors', 'N/A')[:60]}...")
        print(f"    Subjects: {paper.metadata.get('subjects', 'N/A')[:60]}...")


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("ASYNC WEB SCRAPER DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases the AsyncWebScraper capabilities:")
    print("  • Concurrent scraping (5-10x faster)")
    print("  • Connection pooling")
    print("  • Rate limiting")
    print("  • Error handling")
    print("  • Academic paper support")
    print("  • Link following")

    demos = [
        ("Single Article Scraping", demo_single_scrape),
        ("Batch Scraping", demo_batch_scrape),
        ("Performance Comparison", demo_performance_comparison),
        ("Link Following", demo_link_following),
        ("Error Handling", demo_error_handling),
        ("Convenience Function", demo_convenience_function),
        ("Academic Papers", demo_academic_papers),
    ]

    # Run each demo
    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n✗ Demo '{name}' failed: {e}")
            logger.exception(f"Error in demo: {name}")

        # Pause between demos
        await asyncio.sleep(2)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nFor more information, see docs/async_scraper_usage.md")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
