"""
Report Generator Module

Generates comprehensive markdown reports from analyzed articles
with structured formatting and table of contents.
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from .utils import format_timestamp, sanitize_filename, ensure_directory

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive markdown reports from analyzed articles"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        ensure_directory(output_dir)
    
    def generate_report(self, articles: List[Dict[str, Any]], 
                       report_filename: str = "article_analysis_report.md",
                       include_stats: bool = True,
                       include_toc: bool = True) -> str:
        """
        Generate comprehensive markdown report
        
        Args:
            articles: List of analyzed articles
            report_filename: Output filename
            include_stats: Whether to include processing statistics
            include_toc: Whether to include table of contents
            
        Returns:
            Path to generated report file
        """
        try:
            report_path = os.path.join(self.output_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # Write header
                self._write_header(f, len(articles))
                
                # Write statistics if requested
                if include_stats:
                    self._write_statistics(f, articles)
                
                # Write table of contents if requested
                if include_toc:
                    self._write_table_of_contents(f, articles)
                
                # Write article summaries
                self._write_article_summaries(f, articles)
                
                # Write footer
                self._write_footer(f)
            
            logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def _write_header(self, f, article_count: int):
        """Write report header"""
        f.write("# RSS Feed Article Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Articles Analyzed:** {article_count}\n\n")
        f.write("---\n\n")
    
    def _write_statistics(self, f, articles: List[Dict[str, Any]]):
        """Write processing statistics"""
        if not articles:
            return
        
        f.write("## Processing Statistics\n\n")
        
        # Calculate statistics
        total_articles = len(articles)
        avg_confidence = sum(article.get('confidence_score', 0) for article in articles) / total_articles
        
        # Count by confidence ranges
        high_confidence = sum(1 for article in articles if article.get('confidence_score', 0) >= 8)
        medium_confidence = sum(1 for article in articles if 5 <= article.get('confidence_score', 0) < 8)
        low_confidence = sum(1 for article in articles if article.get('confidence_score', 0) < 5)
        
        # Count by domain
        domains = {}
        for article in articles:
            domain = article.get('metadata', {}).get('domain', 'Unknown')
            domains[domain] = domains.get(domain, 0) + 1
        
        # Write statistics
        f.write(f"- **Total Articles:** {total_articles}\n")
        f.write(f"- **Average Confidence Score:** {avg_confidence:.1f}/10\n")
        f.write(f"- **High Confidence (8-10):** {high_confidence} articles\n")
        f.write(f"- **Medium Confidence (5-7):** {medium_confidence} articles\n")
        f.write(f"- **Low Confidence (1-4):** {low_confidence} articles\n\n")
        
        # Domain breakdown
        if domains:
            f.write("### Articles by Domain\n\n")
            for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{domain}:** {count} articles\n")
            f.write("\n")
        
        f.write("---\n\n")
    
    def _write_table_of_contents(self, f, articles: List[Dict[str, Any]]):
        """Write table of contents"""
        if not articles:
            return
        
        f.write("## Table of Contents\n\n")
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'Untitled')
            # Create anchor link
            anchor = self._create_anchor(title, i)
            f.write(f"{i}. [{title}](#{anchor})\n")
        
        f.write("\n---\n\n")
    
    def _write_article_summaries(self, f, articles: List[Dict[str, Any]]):
        """Write detailed article summaries"""
        f.write("## Article Summaries\n\n")
        
        for i, article in enumerate(articles, 1):
            self._write_single_article(f, article, i)
            
            # Add separator between articles (except for the last one)
            if i < len(articles):
                f.write("\n---\n\n")
    
    def _write_single_article(self, f, article: Dict[str, Any], index: int):
        """Write summary for a single article"""
        title = article.get('title', 'Untitled')
        url = article.get('url', '')
        
        # Create anchor for table of contents
        anchor = self._create_anchor(title, index)
        
        # Article header
        f.write(f"### {index}. {title} {{#{anchor}}}\n\n")
        
        # Metadata
        f.write("#### Article Information\n\n")
        if url:
            f.write(f"**Source:** [{url}]({url})\n\n")
        
        # Publication date
        pub_date = article.get('publication_date')
        if pub_date:
            f.write(f"**Publication Date:** {pub_date}\n\n")
        
        # Processing metadata
        if article.get('processed_date'):
            f.write(f"**Processed:** {article.get('processed_date')}\n\n")
        
        confidence = article.get('confidence_score', 0)
        if confidence > 0:
            f.write(f"**Analysis Confidence:** {confidence}/10\n\n")
        
        # Methodology Section
        methodology = article.get('methodology_detailed', '').strip()
        if methodology and methodology != "Not clearly specified in the content":
            f.write("#### Methodology\n\n")
            f.write(f"{methodology}\n\n")
        
        # Key Findings Section
        key_findings = article.get('key_findings', '').strip()
        if key_findings and key_findings != "Not clearly specified in the content":
            f.write("#### Key Findings\n\n")
            f.write(f"{key_findings}\n\n")
        
        # Technical Approach Section
        technical_approach = article.get('technical_approach', '').strip()
        if technical_approach and technical_approach != "Not clearly specified in the content":
            f.write("#### Technical Approach\n\n")
            f.write(f"{technical_approach}\n\n")
        
        # Research Design Section
        research_design = article.get('research_design', '').strip()
        if research_design and research_design != "Not clearly specified in the content":
            f.write("#### Research Design\n\n")
            f.write(f"{research_design}\n\n")
        
        # Additional metadata
        metadata = article.get('metadata', {})
        if metadata:
            interesting_fields = ['author', 'subjects', 'paper_type', 'source']
            metadata_to_show = {k: v for k, v in metadata.items() if k in interesting_fields and v}
            
            if metadata_to_show:
                f.write("#### Additional Information\n\n")
                for key, value in metadata_to_show.items():
                    formatted_key = key.replace('_', ' ').title()
                    f.write(f"**{formatted_key}:** {value}\n\n")
    
    def _write_footer(self, f):
        """Write report footer"""
        f.write("\n---\n\n")
        f.write("*This report was generated automatically by the RSS Article Analyzer using Claude Sonnet.*\n")
        f.write(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")
    
    def _create_anchor(self, title: str, index: int) -> str:
        """Create anchor link for table of contents"""
        # Sanitize title for anchor
        anchor = title.lower()
        anchor = anchor.replace(' ', '-')
        anchor = ''.join(c for c in anchor if c.isalnum() or c == '-')
        anchor = f"article-{index}-{anchor}"
        return anchor[:50]  # Limit length
    
    def generate_summary_report(self, articles: List[Dict[str, Any]], 
                              filename: str = "summary_report.md") -> str:
        """
        Generate a shorter summary report
        
        Args:
            articles: List of analyzed articles
            filename: Output filename
            
        Returns:
            Path to generated summary report
        """
        try:
            report_path = os.path.join(self.output_dir, filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Article Analysis Summary\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Articles Analyzed:** {len(articles)}\n\n")
                
                for i, article in enumerate(articles, 1):
                    title = article.get('title', 'Untitled')
                    url = article.get('url', '')
                    key_findings = article.get('key_findings', '').strip()
                    
                    f.write(f"## {i}. {title}\n\n")
                    if url:
                        f.write(f"**Source:** [{url}]({url})\n\n")
                    
                    if key_findings and key_findings != "Not clearly specified in the content":
                        f.write(f"**Key Findings:** {key_findings}\n\n")
                    
                    f.write("---\n\n")
            
            logger.info(f"Summary report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            raise
    
    def generate_json_export(self, articles: List[Dict[str, Any]], 
                           filename: str = "articles_export.json") -> str:
        """
        Generate JSON export of analyzed articles
        
        Args:
            articles: List of analyzed articles
            filename: Output filename
            
        Returns:
            Path to generated JSON file
        """
        try:
            import json
            
            export_path = os.path.join(self.output_dir, filename)
            
            # Prepare export data
            export_data = {
                'generated_at': datetime.now().isoformat(),
                'total_articles': len(articles),
                'articles': articles
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON export generated: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to generate JSON export: {e}")
            raise
    
    def generate_csv_export(self, articles: List[Dict[str, Any]], 
                          filename: str = "articles_export.csv") -> str:
        """
        Generate CSV export of analyzed articles
        
        Args:
            articles: List of analyzed articles
            filename: Output filename
            
        Returns:
            Path to generated CSV file
        """
        try:
            import csv
            
            export_path = os.path.join(self.output_dir, filename)
            
            if not articles:
                logger.warning("No articles to export to CSV")
                return export_path
            
            # Define CSV columns
            columns = [
                'title', 'url', 'publication_date', 'processed_date',
                'confidence_score', 'methodology_detailed', 'key_findings',
                'technical_approach', 'research_design'
            ]
            
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                
                for article in articles:
                    # Prepare row data
                    row = {}
                    for col in columns:
                        value = article.get(col, '')
                        # Truncate long text fields for CSV
                        if col in ['methodology_detailed', 'key_findings', 'technical_approach', 
                                 'research_design']:
                            if len(str(value)) > 500:
                                value = str(value)[:497] + "..."
                        row[col] = value
                    
                    writer.writerow(row)
            
            logger.info(f"CSV export generated: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to generate CSV export: {e}")
            raise
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """
        List all generated reports in the output directory
        
        Returns:
            List of report information dictionaries
        """
        try:
            if not os.path.exists(self.output_dir):
                return []
            
            reports = []
            for filename in os.listdir(self.output_dir):
                filepath = os.path.join(self.output_dir, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    reports.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size_bytes': stat.st_size,
                        'modified_time': datetime.fromtimestamp(stat.st_mtime),
                        'extension': os.path.splitext(filename)[1]
                    })
            
            # Sort by modification time (newest first)
            reports.sort(key=lambda x: x['modified_time'], reverse=True)
            return reports
            
        except Exception as e:
            logger.error(f"Failed to list reports: {e}")
            return []