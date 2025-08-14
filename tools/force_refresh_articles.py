#!/usr/bin/env python3
"""
Force refresh all existing articles with improved title extraction and markdown formatting.

This script will:
1. Re-scrape all existing articles using the improved title extraction
2. Update their titles and content with better formatting
3. Re-analyze articles if needed
4. Update the website data
"""

import logging
import os
import sys
import sqlite3
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import CONFIG
from core.scraper import WebScraper
from core.database import DatabaseManager
from core.rss_parser import RSSParser
from processors.article_processor import ArticleProcessor, ProcessingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_existing_articles(db_path: str) -> list[dict]:
    """Get all existing articles from database"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute("""
            SELECT id, title, url, content_hash, status, processed_date
            FROM articles 
            WHERE status IN ('completed', 'duplicate')
            ORDER BY processed_date DESC
        """)
        
        articles = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        logger.info(f"Found {len(articles)} existing articles to refresh")
        return articles
        
    except Exception as e:
        logger.error(f"Failed to get existing articles: {e}")
        return []

def refresh_article_titles_and_content(db_path: str) -> int:
    """Re-scrape articles to get proper titles and improved markdown"""
    
    db = DatabaseManager(db_path)
    scraper = WebScraper(delay_between_requests=0.5)  # Faster for updates
    
    articles = get_existing_articles(db_path)
    updated_count = 0
    
    logger.info(f"Starting refresh of {len(articles)} articles...")
    
    for i, article in enumerate(articles, 1):
        try:
            logger.info(f"Refreshing article {i}/{len(articles)}: {article['title'][:50]}...")
            
            # Re-scrape the article with improved extraction
            scraped_content = scraper.scrape_article(
                article['url'], 
                follow_links=False,  # Don't follow links during refresh to save time
                max_linked_articles=0
            )
            
            if scraped_content:
                # Check if title has actually improved
                old_title = article['title']
                new_title = scraped_content.title
                
                if new_title != old_title:
                    logger.info(f"Title improved:")
                    logger.info(f"  Old: '{old_title}'")
                    logger.info(f"  New: '{new_title}'")
                    
                    # Update article title and content in database
                    with db.get_connection() as conn:
                        conn.execute("""
                            UPDATE articles 
                            SET title = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (new_title, scraped_content.content_hash, article['id']))
                        
                        # Update content table with improved markdown
                        conn.execute("""
                            UPDATE content 
                            SET original_content = ?
                            WHERE article_id = ?
                        """, (scraped_content.content, article['id']))
                    
                    updated_count += 1
                else:
                    logger.debug(f"Title unchanged for: {old_title}")
                    
                    # Still update content for better markdown formatting
                    with db.get_connection() as conn:
                        conn.execute("""
                            UPDATE content 
                            SET original_content = ?
                            WHERE article_id = ?
                        """, (scraped_content.content, article['id']))
            
            else:
                logger.warning(f"Failed to re-scrape article: {article['url']}")
                
        except Exception as e:
            logger.error(f"Error refreshing article {article['id']}: {e}")
            continue
            
        # Small delay between requests
        time.sleep(0.1)
    
    logger.info(f"Refresh complete: Updated {updated_count} titles out of {len(articles)} articles")
    return updated_count

def main():
    """Main function"""
    # Use same database path as main application
    db_path = CONFIG.database.DB_PATH
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}. Run the analyzer first to create articles.")
        return
    
    logger.info("Starting force refresh of articles with improved titles and formatting...")
    
    # Refresh articles
    updated_count = refresh_article_titles_and_content(db_path)
    
    if updated_count > 0:
        logger.info(f"Successfully updated {updated_count} articles")
        logger.info("Next step: Update website data with improved titles")
    else:
        logger.info("No articles were updated - titles may already be optimal")

if __name__ == "__main__":
    main()