#!/usr/bin/env python3
"""
Validate that all articles have proper Feynman technique format.
Since we cleaned up the system to only use Feynman technique,
this ensures all existing analyses follow the correct structure.
"""

import json
from pathlib import Path
from datetime import datetime

def validate_feynman_structure(analysis_text):
    """
    Validate that analysis follows Feynman technique JSON structure.
    Returns (is_valid, issues_found)
    """
    issues = []
    
    if not analysis_text or not isinstance(analysis_text, str):
        return False, ["No analysis content"]
    
    # Check if it's wrapped in ```json blocks
    if not (analysis_text.strip().startswith('```json\n') and analysis_text.strip().endswith('\n```')):
        issues.append("Not wrapped in ```json code blocks")
    
    try:
        # Extract JSON content
        if analysis_text.strip().startswith('```json\n'):
            json_content = analysis_text.strip()[8:-4]  # Remove ```json\n and \n```
        else:
            json_content = analysis_text.strip()
        
        # Parse JSON
        data = json.loads(json_content)
        
        # Check required structure
        if 'extracted_title' not in data:
            issues.append("Missing 'extracted_title' field")
        
        if 'analysis' not in data:
            issues.append("Missing 'analysis' field")
        elif 'feynman_technique_breakdown' not in data['analysis']:
            issues.append("Missing 'analysis.feynman_technique_breakdown' field")
        else:
            breakdown = data['analysis']['feynman_technique_breakdown']
            
            # Check required Feynman sections
            required_sections = [
                'simple_explanation',
                'key_components', 
                'step_by_step_process',
                'real_world_applications',
                'analogies_and_examples',
                'significance_and_impact'
            ]
            
            for section in required_sections:
                if section not in breakdown:
                    issues.append(f"Missing Feynman section: {section}")
        
        if 'metadata' not in data:
            issues.append("Missing 'metadata' field")
        
    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON: {str(e)}")
    except Exception as e:
        issues.append(f"Parsing error: {str(e)}")
    
    return len(issues) == 0, issues

def main():
    """Validate all articles have proper Feynman technique format."""
    
    # Load current articles
    data_file = Path("docs/data.json")
    if not data_file.exists():
        print("❌ data.json not found!")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    total_articles = len(articles)
    
    print(f"🔍 Validating {total_articles} articles for Feynman technique format...")
    print()
    
    valid_articles = []
    invalid_articles = []
    
    for i, article in enumerate(articles, 1):
        article_id = article.get('id')
        title = article.get('title', 'Unknown Title')
        analysis = article.get('analysis', '')
        
        print(f"[{i:2}/{total_articles}] {title[:60]}{'...' if len(title) > 60 else ''}")
        
        is_valid, issues = validate_feynman_structure(analysis)
        
        if is_valid:
            print(f"    ✅ Valid Feynman format")
            valid_articles.append(article)
        else:
            print(f"    ❌ Invalid format:")
            for issue in issues:
                print(f"       • {issue}")
            invalid_articles.append((article, issues))
        
        print()
    
    # Summary
    print("=" * 70)
    print(f"📊 VALIDATION SUMMARY")
    print(f"✅ Valid articles: {len(valid_articles)}/{total_articles}")
    print(f"❌ Invalid articles: {len(invalid_articles)}/{total_articles}")
    
    if invalid_articles:
        print()
        print("🔧 INVALID ARTICLES NEED RE-ANALYSIS:")
        for article, issues in invalid_articles:
            print(f"   • ID {article.get('id')}: {article.get('title', 'Unknown')[:50]}")
    
    # Check if all are valid
    if len(valid_articles) == total_articles:
        print()
        print("🎉 ALL ARTICLES HAVE PROPER FEYNMAN FORMAT!")
        print("✨ System is ready with clean Feynman technique implementation!")
    else:
        print()
        print("⚠️  Some articles need to be re-analyzed with proper Feynman technique.")
        print("   Run the RSS analyzer again to fix invalid articles.")
    
    # Update metadata for valid articles
    if valid_articles:
        # Mark all valid articles as using clean Feynman technique
        for article in valid_articles:
            if article.get('ai_provider') != 'feynman_clean':
                article['ai_provider'] = 'feynman_clean'
        
        # Update data
        updated_data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_articles": len(valid_articles),
            "articles": valid_articles
        }
        
        # Backup and save
        backup_file = data_file.with_suffix('.validation_backup.json')
        print(f"💾 Creating backup: {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saving validated data to {data_file}")
        with open(data_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        print("✅ Data updated with Feynman validation markers!")

if __name__ == "__main__":
    main()