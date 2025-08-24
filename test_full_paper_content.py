#!/usr/bin/env python3
"""
Test scraping full arXiv HTML content to see the difference in content quality.
"""

import requests
from bs4 import BeautifulSoup
import re

def scrape_arxiv_html(url):
    """Simple scraper to test arXiv HTML content extraction."""
    
    try:
        print(f"🔍 Fetching: {url}")
        
        headers = {
            'User-Agent': 'RSS-Article-Analyzer/1.0'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1', class_='ltx_title') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Unknown Title"
        
        # Extract main content
        content_parts = []
        
        # Look for main content containers
        main_content = soup.find('div', class_='ltx_page_main') or soup.find('main') or soup.find('article')
        
        if main_content:
            # Get sections
            sections = main_content.find_all(['section', 'div'], class_=re.compile(r'ltx_section|section'))
            
            for section in sections[:10]:  # Limit to first 10 sections
                section_title = section.find(['h2', 'h3', 'h4'], class_=re.compile(r'ltx_title|title'))
                if section_title:
                    content_parts.append(f"\n## {section_title.get_text().strip()}\n")
                
                paragraphs = section.find_all('p')
                for p in paragraphs[:5]:  # Limit paragraphs per section
                    text = p.get_text().strip()
                    if text and len(text) > 20:
                        content_parts.append(text)
        
        content = '\n\n'.join(content_parts)
        
        return {
            'title': title,
            'content': content,
            'content_length': len(content),
            'success': True
        }
        
    except Exception as e:
        return {
            'title': '',
            'content': '',
            'content_length': 0,
            'success': False,
            'error': str(e)
        }

def main():
    """Test one arXiv HTML paper to see content quality."""
    
    # Test with the first paper
    test_url = "https://arxiv.org/html/2507.21110v1"
    
    print("🧪 TESTING ARXIV HTML CONTENT EXTRACTION")
    print("=" * 50)
    
    result = scrape_arxiv_html(test_url)
    
    if result['success']:
        print(f"✅ Successfully scraped!")
        print(f"📄 Title: {result['title']}")
        print(f"📊 Content length: {result['content_length']:,} characters")
        print()
        print("📖 CONTENT PREVIEW (first 1000 chars):")
        print("-" * 40)
        print(result['content'][:1000] + "..." if len(result['content']) > 1000 else result['content'])
        print("-" * 40)
        print()
        
        if result['content_length'] > 5000:
            print("🎉 EXCELLENT! This is full paper content!")
            print("📈 Much richer than abstract-only analysis")
            print("✨ This will give us much better Feynman explanations!")
        else:
            print("⚠️  Content seems short, might still be abstract only")
            
    else:
        print(f"❌ Failed to scrape: {result['error']}")
    
    print()
    print("🚀 If this looks good, proceed to re-analyze all 6 papers!")

if __name__ == "__main__":
    main()