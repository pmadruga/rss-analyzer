#!/usr/bin/env python3
"""
Convert arXiv PDF/abstract links to HTML versions for full paper analysis.
This ensures we analyze the complete papers rather than just abstracts.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

def convert_arxiv_url_to_html(url):
    """
    Convert arXiv PDF/abstract URLs to HTML versions.
    
    Examples:
    - https://arxiv.org/pdf/2502.09356 -> https://arxiv.org/html/2502.09356v1
    - https://arxiv.org/abs/2507.21110 -> https://arxiv.org/html/2507.21110v1
    - https://arxiv.org/html/2311.09476v2 -> https://arxiv.org/html/2311.09476v2 (already HTML)
    """
    
    if not url or 'arxiv.org' not in url:
        return url, False
    
    # Already HTML version
    if '/html/' in url:
        return url, False
    
    # Extract paper ID from different arXiv URL formats
    patterns = [
        r'arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?(?:\.pdf)?',  # PDF links
        r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?',           # Abstract links
    ]
    
    paper_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            paper_id = match.group(1)
            break
    
    if not paper_id:
        print(f"  ⚠️  Could not extract paper ID from: {url}")
        return url, False
    
    # Create HTML version URL (try v1 first, which is most common)
    html_url = f"https://arxiv.org/html/{paper_id}v1"
    
    return html_url, True

def main():
    """Convert arXiv papers to HTML versions and update data.json."""
    
    # Load current articles
    data_file = Path("docs/data.json")
    if not data_file.exists():
        print("❌ data.json not found!")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    
    # Find arXiv articles
    arxiv_articles = [article for article in articles if 'arxiv.org' in article.get('url', '')]
    
    print(f"🔍 Found {len(arxiv_articles)} arXiv articles to convert to HTML:")
    print()
    
    conversions = []
    
    for article in arxiv_articles:
        article_id = article.get('id')
        title = article.get('title', 'Unknown Title')
        original_url = article.get('url')
        
        print(f"📄 Article {article_id}: {title[:60]}{'...' if len(title) > 60 else ''}")
        print(f"   Original: {original_url}")
        
        html_url, was_converted = convert_arxiv_url_to_html(original_url)
        
        if was_converted:
            print(f"   HTML:     {html_url}")
            print(f"   ✅ Converted to HTML version")
            
            # Update the article
            article['url'] = html_url
            article['original_url'] = original_url  # Keep track of original
            article['converted_to_html'] = True
            
            conversions.append({
                'id': article_id,
                'title': title,
                'original_url': original_url,
                'html_url': html_url
            })
        else:
            print(f"   ℹ️  Already HTML or could not convert")
        
        print()
    
    # Summary
    print("=" * 70)
    print(f"📊 CONVERSION SUMMARY")
    print(f"🔄 Converted to HTML: {len(conversions)} articles")
    print(f"📋 Total arXiv articles: {len(arxiv_articles)}")
    
    if conversions:
        print()
        print("🔄 CONVERTED ARTICLES:")
        for conv in conversions:
            print(f"   • ID {conv['id']}: {conv['title'][:50]}{'...' if len(conv['title']) > 50 else ''}")
        
        # Save updated data
        backup_file = data_file.with_suffix('.arxiv_backup.json')
        print(f"💾 Creating backup: {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saving updated data to {data_file}")
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("✅ arXiv URLs updated to HTML versions!")
        print()
        print("🔄 NEXT STEPS:")
        print("Now run the analyzer again to get full paper content:")
        
        # Create list of HTML URLs to re-analyze
        html_urls = [conv['html_url'] for conv in conversions]
        urls_file = Path("arxiv_html_urls.txt")
        with open(urls_file, 'w') as f:
            for url in html_urls:
                f.write(f"{url}\n")
        
        print(f"📝 HTML URLs saved to: {urls_file}")
        print("🚀 Run: docker compose run rss-analyzer run --force-refresh --limit 10")
        print("   This will re-analyze with full paper content instead of just abstracts!")
        
    else:
        print()
        print("ℹ️  No conversions needed - all arXiv articles already use HTML or could not be converted")

if __name__ == "__main__":
    main()