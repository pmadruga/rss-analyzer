"""
Anthropic Claude API Client Module

Handles integration with Anthropic Claude API for article summarization
with specialized prompts for methodology explanation and key insights.
"""

import json
import logging
import time

import anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Anthropic Claude API client with specialized prompts for academic article analysis"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries = 3
        self.base_delay = 1.0

        # System prompt for article analysis
        self.system_prompt = """You are the author of the paper being analyzed. Use the Feynman technique to explain your research in depth, as if you were teaching it to someone who has never encountered this topic before. Break down complex concepts into simple, fundamental principles and use clear analogies where helpful.

As the author explaining your own work, you should:
1. Explain the core concepts from first principles, using simple language and analogies
2. Walk through your methodology step-by-step, explaining why each step was necessary
3. Describe your technical approach as if teaching someone who needs to understand the fundamentals
4. Share your thought process and decision-making throughout the research
5. Explain complex technical concepts by breaking them down into simpler components

Your explanation should be comprehensive yet accessible, demonstrating deep understanding through simplicity. If you can't explain it simply, you don't understand it well enough."""

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
            content = (
                content[:max_content_length] + "\n\n[Content truncated due to length]"
            )

        url_section = f"\n**Original URL:** {url}" if url else ""

        prompt = f"""Explain this paper to me in depth using the Feynman technique, as if you were its author. Take on the persona of the researcher who conducted this work and explain it as if teaching someone who has never encountered this topic before.

**Article Title:** {title}{url_section}

**Content:**
{content}

As the author of this paper, please provide your explanation in the following JSON structure:

{{
    "methodology_detailed": "As the author, explain your research methodology using the Feynman technique. Start with the fundamental problem and walk through your approach step-by-step. Use simple language and analogies to make complex processes understandable. Explain why you chose each methodological step and how it contributes to solving the problem. Think of this as teaching a curious student who needs to understand the 'why' behind every decision.",

    "technical_approach": "As the author, explain your technical implementation using first principles. Break down complex algorithms, frameworks, and tools into their fundamental components. Use analogies and simple explanations to make technical concepts accessible. Explain your thought process behind technical choices and how different components work together. Imagine you're teaching someone who needs to understand not just what you did, but why it works.",

    "key_findings": "As the author, share your main discoveries and results. Explain what you found and why it's significant, using simple language that anyone can understand. Connect your findings back to the original problem you were trying to solve.",

    "research_design": "As the author, explain how you designed your study or experiment. Walk through your reasoning for the experimental setup, using simple terms and explaining why each design choice was important for answering your research question.",

}}

Make sure your response is valid JSON. Write in first person as the author, using the Feynman technique to make complex concepts simple and accessible. If certain sections cannot be determined from the content, explain what information would be needed to provide a complete explanation."""

        return prompt

    def _make_api_call(self, prompt: str) -> str | None:
        """Make API call to Claude with retry logic"""

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Making Claude API call (attempt {attempt + 1}/{self.max_retries})"
                )

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.3,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
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
                    delay = self.base_delay * (2**attempt)
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
            response_clean = response.strip()

            # Remove markdown code block formatting if present
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            elif response_clean.startswith("```"):
                response_clean = response_clean[3:]

            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]

            response_clean = response_clean.strip()

            # Parse JSON
            analysis_data = json.loads(response_clean)

            # Add metadata
            analysis_data["title"] = title
            analysis_data["url"] = url
            analysis_data["analyzed_at"] = time.time()
            analysis_data["model_used"] = self.model

            # Validate required fields
            required_fields = [
                "methodology_detailed",
                "technical_approach",
                "key_findings",
                "research_design",
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
                "title": title,
                "url": url,
                "methodology_detailed": "Analysis parsing failed",
                "technical_approach": "Analysis parsing failed",
                "key_findings": "Analysis parsing failed",
                "research_design": "Analysis parsing failed",
                "raw_response": response,
                "parsing_error": str(e),
                "analyzed_at": time.time(),
                "model_used": self.model,
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing Claude response: {e}")
            return {
                "title": title,
                "url": url,
                "error": str(e),
                "analyzed_at": time.time(),
                "model_used": self.model,
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
                    progress_callback(i, total, article.get("title", "Unknown"))

                analysis = self.analyze_article(
                    title=article.get("title", "Untitled"),
                    content=article.get("content", ""),
                    url=article.get("url", ""),
                )

                if analysis:
                    results.append(analysis)
                    logger.info(
                        f"Analyzed article {i}/{total}: {article.get('title', 'Unknown')}"
                    )
                else:
                    logger.warning(
                        f"Failed to analyze article {i}/{total}: {article.get('title', 'Unknown')}"
                    )

            except Exception as e:
                logger.error(f"Error in batch analysis for article {i}/{total}: {e}")
                continue

        success_rate = len(results) / total * 100 if total > 0 else 0
        logger.info(
            f"Batch analysis completed: {len(results)}/{total} articles analyzed ({success_rate:.1f}% success rate)"
        )

        return results

    def test_connection(self) -> bool:
        """Test connection to Claude API"""
        try:
            logger.info("Testing Claude API connection...")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, please respond with 'API connection successful'",
                    }
                ],
            )

            if response.content and len(response.content) > 0:
                response_text = response.content[0].text
                logger.info(f"Claude API test response: {response_text}")
                return True
            else:
                logger.error("Empty response from Claude API test")
                return False

        except Exception as e:
            logger.error(f"Claude API connection test failed: {e}")
            return False
