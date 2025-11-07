"""
Tests for Token Utilities Module

Comprehensive tests for token-aware truncation functionality.
"""

import pytest

from src.clients.token_utils import (
    count_tokens,
    estimate_cost_savings,
    get_encoding_for_model,
    truncate_by_tokens,
)


class TestGetEncodingForModel:
    """Tests for get_encoding_for_model function"""

    def test_claude_model_encoding(self):
        """Test that Claude models use cl100k_base encoding"""
        encoding = get_encoding_for_model("claude-3-5-sonnet-20241022")
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_gpt4_model_encoding(self):
        """Test that GPT-4 models use cl100k_base encoding"""
        encoding = get_encoding_for_model("gpt-4")
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_mistral_model_encoding(self):
        """Test that Mistral models use cl100k_base encoding"""
        encoding = get_encoding_for_model("mistral-large-latest")
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_encoding_caching(self):
        """Test that encodings are cached for performance"""
        encoding1 = get_encoding_for_model("gpt-4")
        encoding2 = get_encoding_for_model("gpt-4")
        # Should return the same cached instance
        assert encoding1 is encoding2

    def test_unknown_model_defaults_to_cl100k(self):
        """Test that unknown models default to cl100k_base"""
        encoding = get_encoding_for_model("unknown-model-xyz")
        assert encoding is not None
        assert encoding.name == "cl100k_base"


class TestCountTokens:
    """Tests for count_tokens function"""

    def test_empty_string(self):
        """Test token counting for empty string"""
        assert count_tokens("", "gpt-4") == 0

    def test_simple_text(self):
        """Test token counting for simple text"""
        text = "Hello, world!"
        tokens = count_tokens(text, "gpt-4")
        # "Hello, world!" is approximately 4 tokens
        assert 3 <= tokens <= 5

    def test_long_text(self):
        """Test token counting for longer text"""
        text = "This is a longer piece of text that should have more tokens. " * 10
        tokens = count_tokens(text, "gpt-4")
        # Should be significantly more than a few tokens
        assert tokens > 50

    def test_consistent_across_models(self):
        """Test that token count is consistent for models using same encoding"""
        text = "Consistent token counting test"
        claude_tokens = count_tokens(text, "claude-3-5-sonnet-20241022")
        gpt4_tokens = count_tokens(text, "gpt-4")
        # Both use cl100k_base, should be identical
        assert claude_tokens == gpt4_tokens

    def test_special_characters(self):
        """Test token counting with special characters"""
        text = "Special chars: 你好 🌟 ñ é ü"
        tokens = count_tokens(text, "gpt-4")
        assert tokens > 0


class TestTruncateByTokens:
    """Tests for truncate_by_tokens function"""

    def test_no_truncation_needed(self):
        """Test that short text is not truncated"""
        text = "Short text"
        result = truncate_by_tokens(text, 1000, "gpt-4")
        assert result == text

    def test_truncation_occurs(self):
        """Test that long text is truncated"""
        text = "This is a long piece of text. " * 1000  # Very long text
        max_tokens = 100
        result = truncate_by_tokens(text, max_tokens, "gpt-4")

        # Result should be shorter than original
        assert len(result) < len(text)

        # Result should fit within token limit
        result_tokens = count_tokens(result, "gpt-4")
        assert result_tokens <= max_tokens

    def test_truncation_with_suffix(self):
        """Test that truncation includes suffix"""
        text = "Long text " * 1000
        max_tokens = 50
        suffix = "\n\n[Truncated]"

        result = truncate_by_tokens(text, max_tokens, "gpt-4", suffix=suffix)

        # Should end with suffix
        assert result.endswith(suffix)

        # Total tokens should still be within limit
        result_tokens = count_tokens(result, "gpt-4")
        assert result_tokens <= max_tokens

    def test_empty_text(self):
        """Test truncation of empty text"""
        result = truncate_by_tokens("", 100, "gpt-4")
        assert result == ""

    def test_zero_max_tokens(self):
        """Test that zero max_tokens returns empty string"""
        text = "Some text"
        result = truncate_by_tokens(text, 0, "gpt-4")
        assert result == ""

    def test_negative_max_tokens(self):
        """Test that negative max_tokens returns empty string"""
        text = "Some text"
        result = truncate_by_tokens(text, -10, "gpt-4")
        assert result == ""

    def test_exact_boundary(self):
        """Test truncation at exact token boundary"""
        # Create text with known token count
        text = "word " * 100  # Approximately 100 tokens
        actual_tokens = count_tokens(text, "gpt-4")

        # Truncate to exact count
        result = truncate_by_tokens(text, actual_tokens, "gpt-4")
        result_tokens = count_tokens(result, "gpt-4")

        # Should be at or just below the limit
        assert result_tokens <= actual_tokens

    def test_different_models(self):
        """Test truncation works for different models"""
        text = "Test text " * 500

        claude_result = truncate_by_tokens(text, 100, "claude-3-5-sonnet-20241022")
        gpt4_result = truncate_by_tokens(text, 100, "gpt-4")
        mistral_result = truncate_by_tokens(text, 100, "mistral-large-latest")

        # All should be truncated
        assert len(claude_result) < len(text)
        assert len(gpt4_result) < len(text)
        assert len(mistral_result) < len(text)

        # Since they all use cl100k_base, results should be similar
        assert count_tokens(claude_result, "claude-3-5-sonnet-20241022") <= 100
        assert count_tokens(gpt4_result, "gpt-4") <= 100
        assert count_tokens(mistral_result, "mistral-large-latest") <= 100

    def test_realistic_article(self):
        """Test with realistic article-length content"""
        # Simulate a 5000-word article (approximately 6500 tokens)
        article = """
        This is a realistic article with multiple paragraphs discussing
        various technical topics related to AI and machine learning.
        """ * 500

        max_tokens = 10000
        result = truncate_by_tokens(article, max_tokens, "gpt-4")

        result_tokens = count_tokens(result, "gpt-4")
        assert result_tokens <= max_tokens


class TestEstimateCostSavings:
    """Tests for estimate_cost_savings function"""

    def test_basic_cost_calculation(self):
        """Test basic cost savings calculation"""
        # 50k chars, 10k token limit, $0.003 per 1k tokens
        savings = estimate_cost_savings(50000, 10000, 0.003)

        assert savings["naive_tokens"] > 0
        assert savings["optimal_tokens"] > 0
        assert savings["savings"] >= 0
        assert savings["savings_percent"] >= 0

    def test_no_truncation_needed(self):
        """Test when content is already within limit"""
        # Small content that doesn't need truncation
        savings = estimate_cost_savings(1000, 10000, 0.003)

        # Optimal should equal naive (no savings)
        assert savings["savings"] == 0
        assert savings["savings_percent"] == 0

    def test_significant_savings(self):
        """Test calculation with significant savings"""
        # 200k chars (~50k tokens) truncated to 10k tokens
        savings = estimate_cost_savings(200000, 10000, 0.003)

        # Should show substantial savings
        assert savings["savings"] > 0
        assert savings["savings_percent"] > 10  # At least 10% savings
        assert savings["tokens_saved"] > 0

    def test_different_cost_rates(self):
        """Test with different API cost rates"""
        content_length = 100000
        max_tokens = 10000

        # Test with different costs
        claude_savings = estimate_cost_savings(content_length, max_tokens, 0.003)
        gpt4_savings = estimate_cost_savings(content_length, max_tokens, 0.03)

        # GPT-4 (higher cost) should show more dollar savings
        assert gpt4_savings["savings"] > claude_savings["savings"]

        # But percentage should be the same
        assert abs(
            gpt4_savings["savings_percent"] - claude_savings["savings_percent"]
        ) < 0.1

    def test_return_structure(self):
        """Test that return dict has correct structure"""
        savings = estimate_cost_savings(50000, 10000)

        # Check all expected keys are present
        expected_keys = [
            "naive_tokens",
            "optimal_tokens",
            "naive_cost",
            "optimal_cost",
            "savings",
            "savings_percent",
            "tokens_saved",
        ]

        for key in expected_keys:
            assert key in savings
            assert isinstance(savings[key], (int, float))


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_truncation_workflow(self):
        """Test complete workflow: count, truncate, verify"""
        # Create test content
        article = "This is a test article. " * 2000  # ~6000 tokens

        model = "gpt-4"
        max_tokens = 1000

        # Count original tokens
        original_tokens = count_tokens(article, model)
        assert original_tokens > max_tokens

        # Truncate
        truncated = truncate_by_tokens(article, max_tokens, model)

        # Verify truncation
        truncated_tokens = count_tokens(truncated, model)
        assert truncated_tokens <= max_tokens

        # Calculate savings
        savings = estimate_cost_savings(len(article), max_tokens)
        assert savings["tokens_saved"] > 0

    def test_edge_case_very_small_limit(self):
        """Test with very small token limit"""
        text = "A long article with many words and sentences."
        result = truncate_by_tokens(text, 5, "gpt-4")

        tokens = count_tokens(result, "gpt-4")
        assert tokens <= 5

    def test_unicode_handling(self):
        """Test with unicode and special characters"""
        text = "Testing unicode: 你好世界 🌍 ñoño café résumé " * 100
        result = truncate_by_tokens(text, 50, "gpt-4")

        # Should not raise any encoding errors
        tokens = count_tokens(result, "gpt-4")
        assert tokens <= 50
        assert len(result) > 0


# Performance benchmarks (optional, can be slow)
@pytest.mark.slow
class TestPerformance:
    """Performance tests for token utilities"""

    def test_large_document_performance(self):
        """Test performance with very large document"""
        import time

        # Create 1MB document
        large_text = "Test content. " * 100000

        start = time.time()
        result = truncate_by_tokens(large_text, 10000, "gpt-4")
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert count_tokens(result, "gpt-4") <= 10000

    def test_encoding_cache_performance(self):
        """Test that encoding caching improves performance"""
        import time

        text = "Test " * 1000

        # First call (with cache miss)
        start = time.time()
        get_encoding_for_model("gpt-4")
        first_call = time.time() - start

        # Second call (with cache hit)
        start = time.time()
        get_encoding_for_model("gpt-4")
        second_call = time.time() - start

        # Cached call should be faster
        assert second_call < first_call
