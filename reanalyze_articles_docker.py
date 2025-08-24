#!/usr/bin/env python3
"""
Simple script to extract URLs from data.json for re-analysis via Docker.
"""

import json
from pathlib import Path

def main():
    """Extract URLs from data.json for re-analysis."""
    
    # Load current articles
    data_file = Path("docs/data.json")
    if not data_file.exists():
        print("data.json not found!")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    print(f"Found {len(articles)} articles to re-analyze")
    
    # Extract unique URLs
    urls = []
    for article in articles:
        url = article.get('url')
        if url and url not in urls:
            urls.append(url)
    
    print(f"Extracted {len(urls)} unique URLs")
    
    # Save URLs to a file for Docker processing
    urls_file = Path("urls_to_reanalyze.txt")
    with open(urls_file, 'w') as f:
        for url in urls:
            f.write(f"{url}\n")
    
    print(f"URLs saved to {urls_file}")
    print("\nTo re-analyze all articles, run:")
    print("docker compose run rss-analyzer run --file urls_to_reanalyze.txt")

if __name__ == "__main__":
    main()