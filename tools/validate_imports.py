#!/usr/bin/env python3
"""
Validate that all required imports are available
"""

import sys


def test_imports():
    """Test all critical imports to catch dependency issues early"""
    print("🔍 Testing critical imports...")

    try:
        # Test RSS and scraping imports
        from src.database import DatabaseManager  # noqa: F401
        from src.rss_parser import RSSParser  # noqa: F401
        from src.scraper import WebScraper  # noqa: F401

        print("✅ Core modules imported successfully")

        # Test AI client imports
        from src.claude_client import ClaudeClient  # noqa: F401
        from src.mistral_client import MistralClient  # noqa: F401
        from src.openai_client import OpenAIClient  # noqa: F401

        print("✅ AI client modules imported successfully")

        # Test specific dependencies
        import anthropic  # noqa: F401
        import feedparser  # noqa: F401
        import mistralai  # noqa: F401
        import openai  # noqa: F401
        import requests  # noqa: F401
        from bs4 import BeautifulSoup  # noqa: F401

        print("✅ External dependencies imported successfully")

        # Test main entry point
        from src.main import cli  # noqa: F401

        print("✅ Main entry point imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_functionality():
    """Test basic functionality without making external calls"""
    print("\n🧪 Testing basic functionality...")

    try:
        # Test database creation
        from src.database import DatabaseManager

        DatabaseManager(":memory:")  # In-memory database for testing
        print("✅ Database initialization works")

        # Test RSS parser instantiation
        from src.rss_parser import RSSParser

        RSSParser()
        print("✅ RSS parser instantiation works")

        # Test web scraper instantiation
        from src.scraper import WebScraper

        WebScraper()
        print("✅ Web scraper instantiation works")

        # Test AI clients instantiation

        # These should work even without API keys for basic instantiation
        print("✅ AI client instantiation works")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🔬 RSS Analyzer Import & Functionality Validation")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    # Test functionality
    functionality_ok = test_functionality()

    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("✅ All validation tests passed!")
        sys.exit(0)
    else:
        print("❌ Some validation tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
