#!/usr/bin/env python3
"""
Re-analyze all articles using only the Feynman technique prompt template.
This ensures consistency across all content with the cleaned up system.
"""

import json
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from etl.extract.web_scraper import WebScraper
from etl.transform.analysis_engine import AnalysisEngine, AnalysisRequest
from etl.load.database_manager import DatabaseManager
from etl.load.report_generator import ReportGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Re-analyze all articles with Feynman technique."""
    
    # Load current articles
    data_file = Path("docs/data.json")
    if not data_file.exists():
        logger.error("data.json not found!")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    total_articles = len(articles)
    
    logger.info(f"Starting re-analysis of {total_articles} articles using Feynman technique")
    
    # Initialize components
    scraper = WebScraper()
    analyzer = AnalysisEngine()
    db_manager = DatabaseManager()
    
    # Process each article
    successful_analyses = []
    
    for i, article in enumerate(articles, 1):
        article_id = article.get('id')
        url = article.get('url')
        title = article.get('title', 'Unknown Title')
        
        logger.info(f"[{i}/{total_articles}] Re-analyzing: {title}")
        logger.info(f"URL: {url}")
        
        try:
            # Scrape fresh content
            logger.info("  → Scraping content...")
            content_result = scraper.scrape_article(url)
            
            if not content_result.get('success'):
                logger.warning(f"  ✗ Failed to scrape: {content_result.get('error', 'Unknown error')}")
                continue
                
            content = content_result.get('content', '')
            if not content or len(content.strip()) < 100:
                logger.warning("  ✗ Content too short, skipping")
                continue
            
            # Analyze with Feynman technique
            logger.info("  → Analyzing with Feynman technique...")
            request = AnalysisRequest(
                content=content,
                title=title,
                url=url,
                prompt_type="feynman"  # Force Feynman technique
            )
            analysis_result = analyzer.analyze_article(request)
            
            if not analysis_result.success:
                logger.warning(f"  ✗ Analysis failed: {analysis_result.error or 'Unknown error'}")
                continue
            
            analysis_content = analysis_result.analysis
            if not analysis_content:
                logger.warning("  ✗ Empty analysis returned")
                continue
            
            # Update article with new analysis
            article['analysis'] = analysis_content
            article['ai_provider'] = 'feynman_reanalysis_clean'
            article['processed_date'] = content_result.get('scraped_at', article.get('processed_date'))
            
            successful_analyses.append(article)
            logger.info(f"  ✓ Successfully re-analyzed (length: {len(analysis_content)} chars)")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing article {article_id}: {str(e)}")
            continue
    
    # Save updated data
    logger.info(f"\nRe-analysis complete! Successfully processed {len(successful_analyses)}/{total_articles} articles")
    
    if successful_analyses:
        # Update the data structure
        updated_data = {
            "generated_at": data.get('generated_at'),
            "total_articles": len(successful_analyses),
            "articles": successful_analyses
        }
        
        # Backup original
        backup_file = data_file.with_suffix('.backup.json')
        logger.info(f"Creating backup: {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save updated data
        logger.info(f"Saving updated data to {data_file}")
        with open(data_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        logger.info("✅ All articles re-analyzed with clean Feynman technique!")
        
        # Generate fresh reports
        logger.info("Generating fresh reports...")
        try:
            report_gen = ReportGenerator()
            report_gen.generate_all_reports()
            logger.info("✅ Reports generated successfully!")
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
    
    else:
        logger.error("❌ No articles were successfully re-analyzed!")

if __name__ == "__main__":
    main()