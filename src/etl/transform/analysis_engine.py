"""
Analysis Engine

Coordinates AI analysis using different providers and prompt templates.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .ai_clients.factory import AIClientFactory
from .prompts import select_prompt_for_url, get_prompt_template
from .content_processor import ContentProcessor

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRequest:
    """Represents a request for AI analysis"""
    content: str
    title: str
    url: str
    prompt_type: Optional[str] = None
    ai_provider: Optional[str] = None


@dataclass
class AnalysisResult:
    """Represents the result of AI analysis"""
    analysis: str
    provider: str
    prompt_type: str
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class AnalysisEngine:
    """Orchestrates AI analysis of content"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.content_processor = ContentProcessor()
        self.ai_factory = AIClientFactory()
        
    def analyze_article(self, request: AnalysisRequest) -> AnalysisResult:
        """Analyze an article using AI"""
        try:
            # Process the content
            processed = self.content_processor.prepare_for_analysis(
                request.content, request.title, request.url
            )
            
            # Validate content quality
            quality = self.content_processor.validate_content_quality(request.content)
            if not quality["is_valid"]:
                logger.warning(f"Content quality issues for {request.url}: {quality['issues']}")
            
            # Select prompt
            prompt_type = request.prompt_type or self._select_prompt_type(request.url)
            prompt = get_prompt_template(prompt_type)
            
            # Select AI provider
            provider = request.ai_provider or self.config.get("default_ai_provider", "anthropic")
            
            # Get AI client
            ai_client = self.ai_factory.create_client(provider, self.config)
            
            # Prepare the full prompt
            full_prompt = f"{prompt}\n\nTitle: {request.title}\nURL: {request.url}\n\nContent:\n{processed['analysis_text']}"
            
            # Perform analysis
            logger.info(f"Analyzing {request.url} with {provider} using {prompt_type} prompt")
            analysis = ai_client.analyze_article(
                title=request.title,
                content=processed['analysis_text'],
                url=request.url,
                custom_prompt=prompt
            )
            
            # Prepare result
            result_metadata = {
                **processed['metadata'],
                "quality_assessment": quality,
                "prompt_type": prompt_type,
                "analysis_length": len(analysis) if analysis else 0
            }
            
            return AnalysisResult(
                analysis=analysis,
                provider=provider,
                prompt_type=prompt_type,
                metadata=result_metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {request.url}: {e}")
            return AnalysisResult(
                analysis="",
                provider=request.ai_provider or "unknown",
                prompt_type=request.prompt_type or "unknown",
                metadata={"error_details": str(e)},
                success=False,
                error=str(e)
            )
    
    def _select_prompt_type(self, url: str) -> str:
        """Select appropriate prompt type based on URL"""
        return select_prompt_for_url(url).split('\n')[0]  # Get first line as type indicator
    
    def batch_analyze(self, requests: list[AnalysisRequest]) -> list[AnalysisResult]:
        """Analyze multiple articles"""
        results = []
        
        for request in requests:
            result = self.analyze_article(request)
            results.append(result)
            
            # Add delay between requests to be respectful to AI APIs
            import time
            time.sleep(self.config.get("analysis_delay", 1.0))
        
        return results
    
    def get_analysis_stats(self, results: list[AnalysisResult]) -> Dict[str, Any]:
        """Get statistics about analysis results"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        
        providers = {}
        prompt_types = {}
        
        for result in results:
            providers[result.provider] = providers.get(result.provider, 0) + 1
            prompt_types[result.prompt_type] = prompt_types.get(result.prompt_type, 0) + 1
        
        return {
            "total_analyses": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "providers_used": providers,
            "prompt_types_used": prompt_types
        }