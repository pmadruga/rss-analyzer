"""
Data Export Module

Handles exporting data to various formats (JSON, CSV, etc.)
"""

import json
import csv
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DataExporter:
    """Exports data to various formats"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_to_json(self, data: Dict[str, Any], filename: str = "articles_export.json") -> Path:
        """Export data to JSON format"""
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported JSON data to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise
    
    def export_to_csv(self, articles: List[Dict[str, Any]], filename: str = "articles_export.csv") -> Path:
        """Export articles to CSV format"""
        output_path = self.output_dir / filename
        
        if not articles:
            logger.warning("No articles to export to CSV")
            return output_path
        
        try:
            # Define CSV columns
            columns = [
                'id', 'title', 'url', 'processed_date', 'created_at', 
                'status', 'ai_provider', 'content_length', 'analysis_preview'
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                
                for article in articles:
                    # Extract analysis preview
                    analysis = article.get('analysis', '')
                    analysis_preview = analysis[:200] + '...' if len(analysis) > 200 else analysis
                    
                    row = {
                        'id': article.get('id', ''),
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'processed_date': article.get('processed_date', ''),
                        'created_at': article.get('created_at', ''),
                        'status': article.get('status', ''),
                        'ai_provider': article.get('ai_provider', ''),
                        'content_length': len(analysis),
                        'analysis_preview': analysis_preview.replace('\n', ' ')
                    }
                    writer.writerow(row)
            
            logger.info(f"Exported CSV data to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def export_analyses_only(self, articles: List[Dict[str, Any]], filename: str = "analyses_only.json") -> Path:
        """Export only the analysis content in a clean format"""
        output_path = self.output_dir / filename
        
        try:
            analyses = {}
            
            for article in articles:
                article_id = article.get('id')
                title = article.get('title', 'Unknown Title')
                analysis = article.get('analysis', '')
                
                # Try to extract title from JSON analysis if available
                extracted_title = title
                if analysis.startswith('```json'):
                    try:
                        json_content = analysis[8:-4]  # Remove ```json and ```
                        parsed = json.loads(json_content)
                        if 'extracted_title' in parsed:
                            extracted_title = parsed['extracted_title'].replace('**', '').strip()
                    except:
                        pass  # Use original title if parsing fails
                
                analyses[str(article_id)] = {
                    'original_title': title,
                    'extracted_title': extracted_title,
                    'url': article.get('url', ''),
                    'analysis': analysis,
                    'ai_provider': article.get('ai_provider', ''),
                    'processed_date': article.get('processed_date', '')
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analyses, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported analyses to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export analyses: {e}")
            raise
    
    def export_statistics(self, articles: List[Dict[str, Any]], filename: str = "statistics.json") -> Path:
        """Export statistics about the articles"""
        output_path = self.output_dir / filename
        
        try:
            total_articles = len(articles)
            
            # Count by status
            status_counts = {}
            provider_counts = {}
            
            for article in articles:
                status = article.get('status', 'unknown')
                provider = article.get('ai_provider', 'unknown')
                
                status_counts[status] = status_counts.get(status, 0) + 1
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            # Calculate content statistics
            analysis_lengths = [len(article.get('analysis', '')) for article in articles]
            avg_analysis_length = sum(analysis_lengths) / len(analysis_lengths) if analysis_lengths else 0
            
            stats = {
                'total_articles': total_articles,
                'status_breakdown': status_counts,
                'ai_provider_breakdown': provider_counts,
                'average_analysis_length': int(avg_analysis_length),
                'longest_analysis': max(analysis_lengths) if analysis_lengths else 0,
                'shortest_analysis': min(analysis_lengths) if analysis_lengths else 0
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported statistics to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
            raise