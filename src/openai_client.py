"""
OpenAI API Client Module

Handles integration with OpenAI API for article summarization
with specialized prompts for methodology explanation and key insights.
"""

import json
import logging
import time

import requests

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI API client with specialized prompts for academic article analysis"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
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

        # Truncate content if too long
        max_content_length = 50000
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
        """Make API call to OpenAI with retry logic"""

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making OpenAI API call (attempt {attempt + 1}/{self.max_retries})")

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.3
                }

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("choices") and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                    else:
                        logger.warning("Empty response from OpenAI API")
                        return None
                else:
                    logger.warning(f"OpenAI API returned status {response.status_code}: {response.text}")
                    return None

            except Exception as e:
                logger.warning(f"OpenAI API call attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All OpenAI API call attempts failed for article")
                    return None

        return None

    def _parse_analysis_response(self, response: str, title: str, url: str) -> dict:
        """Parse OpenAI's response into structured format"""

        try:
            # Try to extract JSON from the response
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
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
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
            logger.error(f"Unexpected error parsing OpenAI response: {e}")
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
        """Test connection to OpenAI API"""
        try:
            logger.info("Testing OpenAI API connection...")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, please respond with 'API connection successful'"
                    }
                ],
                "max_tokens": 50
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("choices") and len(data["choices"]) > 0:
                    response_text = data["choices"][0]["message"]["content"]
                    logger.info(f"OpenAI API test response: {response_text}")
                    return True
                else:
                    logger.error("Empty response from OpenAI API test")
                    return False
            else:
                logger.error(f"OpenAI API connection test failed with status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False
