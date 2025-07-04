"""
Claude API Client Module

Handles integration with Claude Sonnet API for article summarization
with specialized prompts for methodology explanation and key insights.
"""

import json
import logging
import time

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Claude API client with specialized prompts for academic article analysis"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = 3
        self.base_delay = 1.0

        # System prompt for article analysis
        self.system_prompt = """You are an expert technical communicator who specializes in explaining complex research methodologies and technical approaches in simple, understandable terms. Your primary task is to analyze academic papers and articles, focusing on making technical concepts accessible to a general audience.

Your analysis should prioritize:
1. Simple, clear explanations of research methodologies - break down complex processes into easy-to-understand steps
2. Detailed technical approach explanations - explain algorithms, frameworks, tools, and implementation details in plain language
3. How the technical components work together and why they were chosen
4. Step-by-step breakdowns of the research process and technical implementation

Focus heavily on methodology and technical approach explanations. Provide less detail on research design specifics and key findings - just summarize these briefly. Always explain technical concepts in simple terms that a non-expert could understand."""

    def analyze_article(self, title: str, content: str, url: str = "") -> dict | None:
        """
        Analyze article content and generate structured summary
        
        Args:
            title: Article title
            content: Article content
            url: Article URL (optional)
            
        Returns:
            Dictionary with structured analysis or None if failed
        """
        try:
            # Create the analysis prompt
            prompt = self._create_analysis_prompt(title, content, url)

            # Make API call with retries
            response = self._make_api_call(prompt)

            if not response:
                return None

            # Parse the structured response
            analysis = self._parse_analysis_response(response, title, url)

            logger.info(f"Successfully analyzed article: {title}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing article '{title}': {e}")
            return None

    def _create_analysis_prompt(self, title: str, content: str, url: str = "") -> str:
        """Create a detailed prompt for article analysis"""

        # Truncate content if too long (Claude has token limits)
        max_content_length = 50000  # Adjust based on token limits
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[Content truncated due to length]"

        url_section = f"\n**Original URL:** {url}" if url else ""

        prompt = f"""Please analyze the following academic article/paper and provide a structured summary. Focus heavily on explaining the methodology and technical approach in simple, clear terms.

**Article Title:** {title}{url_section}

**Content:**
{content}

Please provide your analysis in the following JSON structure:

{{
    "methodology_detailed": "A comprehensive, step-by-step explanation of the research methodology in simple terms. Break down the complete research process into easy-to-understand steps. Explain HOW the research was conducted as if explaining to someone without technical background. Focus extensively on the methodology and make it accessible.",
    
    "technical_approach": "A detailed, in-depth explanation of the technical methods, tools, algorithms, frameworks, software, or systems used. Explain all technical components in plain language, how they work together, why they were chosen, and their implementation details. This should be the most detailed section - explain every technical aspect clearly.",
    
    "key_findings": "A brief summary of the main discoveries or results from the research. Keep this concise.",
    
    "research_design": "A brief overview of the experimental setup or study design. Keep this short and focused.",
    
}}

Make sure your response is valid JSON. Prioritize methodology and technical approach explanations - these should be much more detailed than key findings and research design. If certain sections cannot be determined from the content, use "Not clearly specified in the content" or similar phrases."""

        return prompt

    def _make_api_call(self, prompt: str) -> str | None:
        """Make API call to Claude with retry logic"""

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making Claude API call (attempt {attempt + 1}/{self.max_retries})")

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.3,
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                if response.content and len(response.content) > 0:
                    return response.content[0].text
                else:
                    logger.warning("Empty response from Claude API")
                    return None

            except Exception as e:
                logger.warning(f"Claude API call attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All Claude API call attempts failed for article")
                    return None

        return None

    def _parse_analysis_response(self, response: str, title: str, url: str) -> dict:
        """Parse Claude's response into structured format"""

        try:
            # Try to extract JSON from the response
            # Claude might wrap JSON in markdown code blocks
            response_clean = response.strip()

            # Remove markdown code block formatting if present
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:]

            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]

            response_clean = response_clean.strip()

            # Parse JSON
            analysis_data = json.loads(response_clean)

            # Add metadata
            analysis_data['title'] = title
            analysis_data['url'] = url
            analysis_data['analyzed_at'] = time.time()
            analysis_data['model_used'] = self.model

            # Validate required fields
            required_fields = [
                'methodology_detailed', 'technical_approach', 'key_findings', 'research_design'
            ]

            for field in required_fields:
                if field not in analysis_data:
                    analysis_data[field] = "Not specified in analysis"

            return analysis_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Raw response: {response}")

            # Fallback: create basic structure with raw response
            return {
                'title': title,
                'url': url,
                'methodology_detailed': "Analysis parsing failed",
                'technical_approach': "Analysis parsing failed",
                'key_findings': "Analysis parsing failed",
                'research_design': "Analysis parsing failed",
                'raw_response': response,
                'parsing_error': str(e),
                'analyzed_at': time.time(),
                'model_used': self.model,
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing Claude response: {e}")
            return {
                'title': title,
                'url': url,
                'error': str(e),
                'analyzed_at': time.time(),
                'model_used': self.model,
            }

    def batch_analyze(self, articles: list[dict], progress_callback=None) -> list[dict]:
        """
        Analyze multiple articles with progress tracking
        
        Args:
            articles: List of article dictionaries with 'title', 'content', 'url'
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of analysis results
        """
        results = []
        total = len(articles)

        logger.info(f"Starting batch analysis of {total} articles")

        for i, article in enumerate(articles, 1):
            try:
                if progress_callback:
                    progress_callback(i, total, article.get('title', 'Unknown'))

                analysis = self.analyze_article(
                    title=article.get('title', 'Untitled'),
                    content=article.get('content', ''),
                    url=article.get('url', '')
                )

                if analysis:
                    results.append(analysis)
                    logger.info(f"Analyzed article {i}/{total}: {article.get('title', 'Unknown')}")
                else:
                    logger.warning(f"Failed to analyze article {i}/{total}: {article.get('title', 'Unknown')}")

            except Exception as e:
                logger.error(f"Error in batch analysis for article {i}/{total}: {e}")
                continue

        success_rate = len(results) / total * 100 if total > 0 else 0
        logger.info(f"Batch analysis completed: {len(results)}/{total} articles analyzed ({success_rate:.1f}% success rate)")

        return results

    def test_connection(self) -> bool:
        """Test connection to Claude API with detailed logging"""
        import time
        
        start_time = time.time()
        
        try:
            logger.info("Testing Claude API connection...")
            logger.info(f"Model: {self.model}")
            logger.info(f"Max tokens: 50")
            
            # Check if API key is available
            if not hasattr(self.client, '_api_key') or not self.client._api_key:
                logger.error("No API key found for Claude")
                return False

            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, please respond with 'API connection successful'"
                    }
                ]
            )
            
            response_time = (time.time() - start_time) * 1000

            if response.content and len(response.content) > 0:
                response_text = response.content[0].text
                logger.info(f"✅ Claude API test successful")
                logger.info(f"Response time: {response_time:.0f}ms")
                logger.info(f"Response: {response_text}")
                logger.info(f"Usage: {getattr(response, 'usage', 'Not available')}")
                return True
            else:
                logger.error("❌ Empty response from Claude API test")
                return False

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Enhanced error logging with categorization
            logger.error(f"❌ Claude API connection test failed after {response_time:.0f}ms")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {error_msg}")
            
            # Specific error analysis
            if "credit balance" in error_msg.lower():
                logger.error("🔍 Diagnosis: Insufficient API credits")
                logger.error("💡 Solution: Add credits to your Anthropic account")
            elif "rate limit" in error_msg.lower():
                logger.error("🔍 Diagnosis: Rate limit exceeded")
                logger.error("💡 Solution: Wait before retrying or upgrade plan")
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                logger.error("🔍 Diagnosis: Invalid or missing API key")
                logger.error("💡 Solution: Check ANTHROPIC_API_KEY environment variable")
            elif "400" in error_msg:
                logger.error("🔍 Diagnosis: Bad request")
                logger.error("💡 Solution: Check request parameters")
            elif "500" in error_msg:
                logger.error("🔍 Diagnosis: Server error")
                logger.error("💡 Solution: Retry later or contact Anthropic support")
            else:
                logger.error("🔍 Diagnosis: Unknown error")
                logger.error("💡 Solution: Check network connectivity and API status")
            
            return False
