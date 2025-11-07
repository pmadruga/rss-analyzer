"""
Integration test configuration and fixtures

Provides common fixtures and markers for integration tests.
"""

import pytest


def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests"
    )


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        "api_provider": "anthropic",
        "test_api_key": "test-key-123456",
        "max_concurrent": 5,
        "pool_size": 5,
    }


@pytest.fixture
def sample_articles():
    """Provide sample articles for testing"""
    return [
        {
            "title": f"Test Article {i}",
            "content": f"Content for article {i} " * 50,
            "url": f"https://example.com/article{i}",
        }
        for i in range(10)
    ]


@pytest.fixture
def mock_analysis():
    """Provide mock analysis result"""
    return {
        "methodology_detailed": "Detailed analysis using Feynman technique",
        "technical_approach": "Technical details of the methodology",
        "key_findings": "Key findings from the research",
        "research_design": "Research approach and design",
        "extracted_title": "Test Title",
        "metadata": {
            "ai_provider": "test",
            "model": "test-model",
            "processed_at": 1234567890,
        }
    }
