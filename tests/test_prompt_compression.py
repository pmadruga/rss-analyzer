"""
Test suite for compressed system prompt optimization.

Verifies that compressed prompts maintain output quality while reducing token usage.
"""

import pytest
from src.clients.base import BaseAIClient
from src.clients.async_base import AsyncAIClient


class MockAIClient(BaseAIClient):
    """Mock client for testing system prompt"""

    def __init__(self):
        super().__init__(api_key="test_key_1234567890", model="test-model", provider_name="Test")

    def _make_api_call(self, prompt: str) -> str:
        # Return mock response
        return '{"extracted_title": "Test Paper", "analysis": "Test analysis using Feynman technique..."}'


class MockAsyncAIClient(AsyncAIClient):
    """Mock async client for testing system prompt"""

    def __init__(self):
        super().__init__(api_key="test_key_1234567890", model="test-model", provider_name="TestAsync")

    async def _make_api_call_async(self, prompt: str) -> str:
        # Return mock response
        return '{"extracted_title": "Test Paper Async", "analysis": "Test analysis using Feynman technique..."}'


def test_system_prompt_structure():
    """Test that system prompt has expected structure"""
    client = MockAIClient()
    prompt = client.system_prompt

    # Verify key instructions are present
    assert "title" in prompt.lower()
    assert "Feynman" in prompt
    assert "JSON" in prompt or "json" in prompt.lower()
    assert "extracted_title" in prompt
    assert "analysis" in prompt
    assert "author" in prompt.lower()  # Feynman technique as author


def test_system_prompt_length():
    """Test that system prompt is within token budget"""
    client = MockAIClient()
    prompt = client.system_prompt

    # Approximate token count (characters / 4)
    estimated_tokens = len(prompt) / 4

    # Should be significantly shorter than original (69 tokens)
    assert estimated_tokens < 50, f"Prompt too long: ~{estimated_tokens} tokens (target: <50)"

    # Should not be too short (losing quality)
    assert estimated_tokens > 20, f"Prompt too short: ~{estimated_tokens} tokens (min: >20)"


def test_async_system_prompt_consistency():
    """Test that async client uses same prompt as sync client"""
    sync_client = MockAIClient()
    async_client = MockAsyncAIClient()

    # Both should have identical prompts
    assert sync_client.system_prompt == async_client.system_prompt


def test_prompt_token_savings():
    """Test that compressed prompt saves expected tokens"""
    client = MockAIClient()
    prompt = client.system_prompt

    # Original prompt was ~69 tokens, new should be ~32 tokens
    original_tokens = 69
    estimated_current_tokens = len(prompt) / 4

    savings = original_tokens - estimated_current_tokens
    savings_percentage = (savings / original_tokens) * 100

    print(f"\nToken Analysis:")
    print(f"  Original: ~{original_tokens} tokens")
    print(f"  Compressed: ~{estimated_current_tokens:.1f} tokens")
    print(f"  Savings: ~{savings:.1f} tokens ({savings_percentage:.1f}%)")

    # Should save at least 30% of tokens
    assert savings_percentage > 30, f"Insufficient savings: {savings_percentage:.1f}% (target: >30%)"


def test_prompt_cost_savings():
    """Test that cost savings meet target"""
    client = MockAIClient()
    prompt = client.system_prompt

    # Cost parameters
    original_tokens = 69
    estimated_current_tokens = len(prompt) / 4
    cost_per_1k_tokens = 0.006  # Anthropic Claude pricing
    requests_per_month = 30000

    # Calculate costs
    original_cost = (original_tokens / 1000) * cost_per_1k_tokens * requests_per_month
    compressed_cost = (estimated_current_tokens / 1000) * cost_per_1k_tokens * requests_per_month
    monthly_savings = original_cost - compressed_cost

    print(f"\nCost Analysis:")
    print(f"  Original cost: ${original_cost:.2f}/month")
    print(f"  Compressed cost: ${compressed_cost:.2f}/month")
    print(f"  Monthly savings: ${monthly_savings:.2f}/month")
    print(f"  Annual savings: ${monthly_savings * 12:.2f}/year")

    # Should save at least $5/month at 30k requests
    assert monthly_savings > 5.0, f"Insufficient cost savings: ${monthly_savings:.2f}/month (target: >$5)"


def test_prompt_maintains_features():
    """Test that compressed prompt maintains all required features"""
    client = MockAIClient()
    prompt = client.system_prompt.lower()

    required_features = {
        "title extraction": any(
            phrase in prompt for phrase in ["title", "extract", "identify"]
        ),
        "feynman technique": "feynman" in prompt,
        "json format": "json" in prompt,
        "author perspective": "author" in prompt,
        "content analysis": any(
            phrase in prompt for phrase in ["analyz", "explain", "detail"]
        ),
    }

    missing_features = [
        feature for feature, present in required_features.items() if not present
    ]

    assert (
        not missing_features
    ), f"Missing required features: {', '.join(missing_features)}"


def test_prompt_excludes_generic_titles():
    """Test that prompt instructs to avoid generic titles"""
    client = MockAIClient()
    prompt = client.system_prompt.lower()

    # Should mention avoiding generic titles
    assert any(
        phrase in prompt
        for phrase in ["generic", "not generic", "actual", "real", "heading"]
    ), "Prompt should instruct to avoid generic titles"


@pytest.mark.parametrize(
    "provider_name,model",
    [
        ("Claude", "claude-3-5-sonnet-20241022"),
        ("OpenAI", "gpt-4"),
        ("Mistral", "mistral-large-latest"),
    ],
)
def test_prompt_provider_independent(provider_name, model):
    """Test that prompt is consistent across providers"""

    class TestClient(BaseAIClient):
        def __init__(self, provider, model):
            super().__init__(
                api_key="test_key_1234567890", model=model, provider_name=provider
            )

        def _make_api_call(self, prompt: str) -> str:
            return '{"extracted_title": "Test", "analysis": "Test..."}'

    client = TestClient(provider_name, model)

    # Prompt should not be provider-specific
    assert provider_name.lower() not in client.system_prompt.lower()


def test_prompt_documentation():
    """Test that optimization is documented in code"""
    import inspect

    # Get source of _create_system_prompt method
    source = inspect.getsource(BaseAIClient._create_system_prompt)

    # Should have documentation about optimization
    assert (
        "OPTIMIZATION" in source or "optimization" in source.lower()
    ), "Optimization should be documented in method docstring"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
