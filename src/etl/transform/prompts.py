"""
Centralized Prompt Templates

All AI analysis prompts are defined here for easy management and consistency.
This module uses only the Feynman technique for educational, accessible explanations.
"""


# Feynman Technique Analysis Prompt - The ONLY prompt used
FEYNMAN_ANALYSIS_PROMPT = """Do an in-depth analysis of this article using the Feynman technique.

Structure your response as a valid JSON object with the following format:

{
    "extracted_title": "Clear, descriptive title of the work",
    "analysis": {
        "feynman_technique_breakdown": {
            "simple_explanation": {
                "core_concept": "Break down the core concept into simple terms that anyone could understand",
                "analogy": "Use analogies to make complex concepts more accessible"
            },
            "key_components": {
                "methodology": "Explain the main methodology or approach",
                "validation": "How the work was validated or tested",
                "contributions": "What new insights or tools this provides"
            },
            "step_by_step_process": {
                "problem_identification": "What problem was being solved",
                "solution_development": "How the solution was developed",
                "experimental_validation": "How it was tested and validated",
                "analysis_and_conclusions": "What conclusions were drawn"
            },
            "real_world_applications": {
                "immediate_applications": "Direct practical applications",
                "future_potential": "Long-term possibilities",
                "industry_impact": "How this could change the field"
            },
            "analogies_and_examples": {
                "conceptual_analogy": "A conceptual analogy that makes the work relatable",
                "practical_example": "A concrete example of how this applies"
            },
            "significance_and_impact": {
                "scientific_contribution": "What this adds to scientific knowledge",
                "practical_value": "Real-world utility",
                "future_research": "What research directions this opens"
            }
        }
    },
    "metadata": {
        "analysis_approach": "Feynman technique applied to break down complex concepts into accessible explanations",
        "content_source": "Article content analysis",
        "analysis_focus": "Educational explanation emphasizing practical understanding and applications"
    }
}

Make it educational and accessible while maintaining technical accuracy. Focus on helping people understand WHY this work matters and HOW it works, not just WHAT it does."""


def get_prompt_template() -> str:
    """
    Get the Feynman technique prompt template.
    
    Returns:
        The Feynman analysis prompt template string.
    """
    return FEYNMAN_ANALYSIS_PROMPT


def select_prompt_for_url(url: str) -> str:
    """
    Return the Feynman prompt for all URLs.
    
    Args:
        url: The URL (ignored, always returns Feynman prompt).
        
    Returns:
        The Feynman analysis prompt template string.
    """
    return FEYNMAN_ANALYSIS_PROMPT


# Export main functions and constants
__all__ = [
    "FEYNMAN_ANALYSIS_PROMPT",
    "get_prompt_template",
    "select_prompt_for_url",
]