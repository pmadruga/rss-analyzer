"""
Content Processing Module

Handles text cleaning, preprocessing, and preparation for AI analysis.
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Processes and cleans content for analysis"""
    
    def __init__(self):
        self.max_content_length = 50000  # Max chars to send to AI
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML entities that might have been missed
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Clean up quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        
        return text.strip()
    
    def extract_metadata(self, content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        metadata = {
            "content_length": len(content),
            "word_count": len(content.split()) if content else 0,
            "url": url,
            "content_type": self._detect_content_type(url, content)
        }
        
        # Extract potential author information
        author_patterns = [
            r'[Bb]y\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'[Aa]uthor[s]?:\s*([A-Z][a-z\s,]+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content[:1000])  # Search in first 1000 chars
            if match:
                metadata["potential_author"] = match.group(1).strip()
                break
        
        return metadata
    
    def _detect_content_type(self, url: str, content: str) -> str:
        """Detect the type of content based on URL and content patterns"""
        url_lower = url.lower()
        
        if "arxiv.org" in url_lower:
            return "academic_paper"
        elif "blog" in url_lower or "medium.com" in url_lower:
            return "blog_post"
        elif "github.com" in url_lower:
            return "code_repository"
        elif "news" in url_lower:
            return "news_article"
        elif any(keyword in content.lower()[:500] for keyword in ["abstract", "introduction", "methodology", "results", "conclusion"]):
            return "academic_paper"
        else:
            return "article"
    
    def prepare_for_analysis(self, content: str, title: str, url: str) -> Dict[str, Any]:
        """Prepare content for AI analysis"""
        # Clean the content
        cleaned_content = self.clean_text(content)
        cleaned_title = self.clean_text(title)
        
        # Truncate if too long
        if len(cleaned_content) > self.max_content_length:
            cleaned_content = cleaned_content[:self.max_content_length]
            logger.warning(f"Content truncated to {self.max_content_length} characters for {url}")
        
        # Extract metadata
        metadata = self.extract_metadata(cleaned_content, url)
        
        # Combine title and content for analysis
        analysis_text = f"Title: {cleaned_title}\n\nContent: {cleaned_content}"
        
        return {
            "analysis_text": analysis_text,
            "metadata": metadata,
            "original_length": len(content),
            "processed_length": len(cleaned_content)
        }
    
    def validate_content_quality(self, content: str) -> Dict[str, Any]:
        """Validate content quality and provide warnings"""
        issues = []
        
        if not content or len(content.strip()) < 100:
            issues.append("Content too short for meaningful analysis")
        
        if len(content) > 100000:
            issues.append("Content very long, may be truncated")
        
        # Check for common scraping issues
        error_indicators = [
            "403 forbidden", "404 not found", "access denied",
            "subscription required", "paywall", "login required"
        ]
        
        content_lower = content.lower()
        for indicator in error_indicators:
            if indicator in content_lower:
                issues.append(f"Potential scraping issue detected: {indicator}")
        
        # Check content diversity (not just repeated text)
        words = content.split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            issues.append("Content appears repetitive, may indicate scraping issues")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0, 1.0 - len(issues) * 0.2)  # Simple scoring
        }