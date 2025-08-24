#!/usr/bin/env python3
"""
Re-analyze arXiv papers that have HTML versions for full content analysis.
This will give us much better Feynman technique explanations based on complete papers.
"""

import json
from pathlib import Path
from datetime import datetime

def main():
    """Extract arXiv HTML URLs for focused re-analysis."""
    
    # Load current articles
    data_file = Path("docs/data.json")
    if not data_file.exists():
        print("❌ data.json not found!")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    
    # Find arXiv HTML articles (full papers)
    html_articles = [
        article for article in articles 
        if 'arxiv.org/html' in article.get('url', '')
    ]
    
    print(f"🔍 Found {len(html_articles)} arXiv papers with HTML versions (full content):")
    print()
    
    for i, article in enumerate(html_articles, 1):
        article_id = article.get('id')
        title = article.get('title', 'Unknown Title')
        url = article.get('url')
        content_length = article.get('analysis', '')
        
        print(f"[{i}] Article {article_id}: {title}")
        print(f"    URL: {url}")
        print(f"    Current analysis length: {len(content_length)} chars")
        print()
    
    # Extract URLs for re-analysis
    html_urls = [article.get('url') for article in html_articles]
    
    # Save URLs for Docker re-analysis
    urls_file = Path("arxiv_full_papers.txt")
    with open(urls_file, 'w') as f:
        for url in html_urls:
            f.write(f"{url}\n")
    
    print("📝 Full paper URLs saved to: arxiv_full_papers.txt")
    print()
    print("🎯 THESE PAPERS WILL GET FULL CONTENT ANALYSIS:")
    print("   Instead of just abstracts, we'll analyze the complete papers!")
    print()
    print("🚀 NEXT STEPS:")
    print("1. Run the analyzer with these full paper URLs")
    print("2. Get much richer Feynman technique explanations")
    print("3. Compare the difference in analysis quality!")
    print()
    print("📋 URLs to re-analyze:")
    for i, url in enumerate(html_urls, 1):
        print(f"   {i}. {url}")
    
    # Mark articles for re-analysis
    for article in html_articles:
        article['needs_full_analysis'] = True
        article['analysis_type'] = 'abstract_only'  # Mark current analysis as limited
    
    # Save updated data
    backup_file = data_file.with_suffix('.pre_full_analysis.json')
    print(f"💾 Creating backup: {backup_file}")
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saving updated data to {data_file}")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ arXiv papers marked for full content re-analysis!")

if __name__ == "__main__":
    main()