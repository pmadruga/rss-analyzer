#!/usr/bin/env python3
"""
Validate which arXiv HTML versions actually exist and fall back to PDF if needed.
"""

import json
import requests
from pathlib import Path
from urllib.parse import urlparse

def check_url_exists(url):
    """Check if a URL exists by making a HEAD request."""
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except:
        return False

def get_arxiv_alternatives(paper_id):
    """Get alternative arXiv URLs for a paper ID."""
    base_id = paper_id.split('v')[0]  # Remove version if present
    
    alternatives = [
        f"https://arxiv.org/html/{base_id}v1",
        f"https://arxiv.org/html/{base_id}v2", 
        f"https://arxiv.org/html/{base_id}",
        f"https://arxiv.org/abs/{base_id}",  # Abstract as fallback
        f"https://arxiv.org/pdf/{base_id}.pdf"  # PDF as final fallback
    ]
    
    return alternatives

def extract_paper_id(url):
    """Extract paper ID from arXiv URL."""
    if 'arxiv.org' not in url:
        return None
    
    # Extract from various formats
    if '/html/' in url:
        return url.split('/html/')[-1]
    elif '/abs/' in url:
        return url.split('/abs/')[-1]
    elif '/pdf/' in url:
        return url.split('/pdf/')[-1].replace('.pdf', '')
    
    return None

def main():
    """Validate arXiv HTML URLs and fix broken ones."""
    
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
    
    print(f"🔍 Validating {len(arxiv_articles)} arXiv URLs...")
    print()
    
    fixes = []
    
    for article in arxiv_articles:
        article_id = article.get('id')
        title = article.get('title', 'Unknown Title')
        current_url = article.get('url')
        
        print(f"📄 Article {article_id}: {title[:50]}{'...' if len(title) > 50 else ''}")
        print(f"   Current URL: {current_url}")
        
        # Check if current URL works
        if check_url_exists(current_url):
            print(f"   ✅ URL works!")
            print()
            continue
        
        print(f"   ❌ URL broken, finding alternative...")
        
        # Extract paper ID and try alternatives
        paper_id = extract_paper_id(current_url)
        if not paper_id:
            print(f"   ⚠️  Could not extract paper ID")
            print()
            continue
        
        # Try alternatives
        alternatives = get_arxiv_alternatives(paper_id)
        working_url = None
        
        for alt_url in alternatives:
            print(f"   🔍 Trying: {alt_url}")
            if check_url_exists(alt_url):
                working_url = alt_url
                print(f"   ✅ Found working URL!")
                break
            else:
                print(f"   ❌ Not available")
        
        if working_url:
            article['url'] = working_url
            article['original_broken_url'] = current_url
            fixes.append({
                'id': article_id,
                'title': title,
                'old_url': current_url,
                'new_url': working_url
            })
            print(f"   🔧 Fixed: {working_url}")
        else:
            print(f"   ⚠️  No working alternative found!")
        
        print()
    
    # Summary
    print("=" * 70)
    print(f"📊 VALIDATION SUMMARY")
    print(f"🔧 Fixed URLs: {len(fixes)}")
    print(f"📋 Total arXiv articles: {len(arxiv_articles)}")
    
    if fixes:
        print()
        print("🔧 FIXED ARTICLES:")
        for fix in fixes:
            print(f"   • ID {fix['id']}: {fix['title'][:40]}{'...' if len(fix['title']) > 40 else ''}")
            print(f"     {fix['old_url']} → {fix['new_url']}")
        
        # Save updated data
        backup_file = data_file.with_suffix('.url_fix_backup.json')
        print(f"💾 Creating backup: {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saving fixed data to {data_file}")
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("✅ arXiv URLs fixed!")
        
        # Create list of working URLs for re-analysis
        working_urls = [fix['new_url'] for fix in fixes]
        urls_file = Path("working_arxiv_urls.txt")
        with open(urls_file, 'w') as f:
            for url in working_urls:
                f.write(f"{url}\n")
        
        print(f"📝 Working URLs saved to: {urls_file}")
        print("🚀 Now these articles need re-analysis with full content!")
        
    else:
        print()
        print("✅ All arXiv URLs are working correctly!")

if __name__ == "__main__":
    main()