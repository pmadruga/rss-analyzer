#!/usr/bin/env python3
"""
Test script for link following functionality
"""

import os
import sys

sys.path.append("/app" if os.path.exists("/app") else ".")

import logging

from src.scraper import WebScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_link_following():
    """Test the enhanced scraper with link following"""
    scraper = WebScraper(delay_between_requests=0.5)  # Faster for testing

    # Test URLs that likely contain interesting links
    test_urls = [
        "https://jina.ai/news/quantization-aware-training-of-jina-embeddings-v4/",
        # Add more test URLs as needed
    ]

    for url in test_urls:
        logger.info(f"Testing link following for: {url}")

        try:
            # Test with link following enabled
            result = scraper.scrape_article(
                url, follow_links=True, max_linked_articles=2
            )

            if result:
                logger.info(f"✅ Successfully scraped: {result.title}")
                logger.info(f"Content length: {len(result.content)} characters")

                # Check if links were followed
                linked_articles = result.metadata.get("linked_articles", [])
                if linked_articles:
                    logger.info(f"🔗 Found {len(linked_articles)} linked articles:")
                    for i, link in enumerate(linked_articles, 1):
                        logger.info(f"  {i}. {link}")
                else:
                    logger.info("ℹ️ No interesting links found to follow")

                # Check if content was enhanced
                if "Referenced Articles and Links" in result.content:
                    logger.info("✅ Content was enhanced with linked article summaries")
                else:
                    logger.info("ℹ️ No linked content summaries added")

            else:
                logger.error(f"❌ Failed to scrape: {url}")

        except Exception as e:
            logger.error(f"❌ Error testing {url}: {e}")

        logger.info("-" * 50)


if __name__ == "__main__":
    print("Testing enhanced link following functionality...")
    test_link_following()
    print("Test completed!")
