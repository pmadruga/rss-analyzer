#!/usr/bin/env python3
"""
Quick API Status Checker

A simple script to quickly test which APIs are working.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_env_vars():
    """Check which API keys are set"""
    print("🔑 Environment Variables Check:")
    print("-" * 40)

    apis = {
        "ANTHROPIC_API_KEY": "Anthropic Claude",
        "MISTRAL_API_KEY": "Mistral AI",
        "OPENAI_API_KEY": "OpenAI",
    }

    for env_var, name in apis.items():
        key = os.getenv(env_var)
        if key:
            # Show first/last 4 chars for security
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
            print(f"✅ {name}: {masked_key}")
        else:
            print(f"❌ {name}: Not set")
    print()


def quick_test_anthropic():
    """Quick test of Anthropic API"""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "❌", "No API key"

        import anthropic

        start_time = time.time()
        client = anthropic.Anthropic(api_key=api_key)

        # Simple test message
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )

        response_time = (time.time() - start_time) * 1000
        return "✅", f"OK ({response_time:.0f}ms)"

    except Exception as e:
        error_msg = str(e)
        if "credit balance" in error_msg.lower():
            return "💰", "Insufficient credits"
        elif "rate limit" in error_msg.lower():
            return "⏱️", "Rate limited"
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            return "🔐", "Invalid API key"
        else:
            return "❌", f"Error: {error_msg[:50]}..."


def quick_test_mistral():
    """Quick test of Mistral API"""
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return "❌", "No API key"

        import requests

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
        }

        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10,
        )

        response_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            return "✅", f"OK ({response_time:.0f}ms)"
        elif response.status_code == 429:
            error_data = response.json()
            if "capacity exceeded" in error_data.get("message", "").lower():
                return "🏭", "Capacity exceeded"
            else:
                return "⏱️", "Rate limited"
        elif response.status_code == 401:
            return "🔐", "Invalid API key"
        else:
            return "❌", f"HTTP {response.status_code}"

    except Exception as e:
        error_msg = str(e)
        return "❌", f"Error: {error_msg[:50]}..."


def quick_test_openai():
    """Quick test of OpenAI API"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌", "No API key"

        import openai

        start_time = time.time()
        client = openai.OpenAI(api_key=api_key)

        client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hi"}], max_tokens=10
        )

        response_time = (time.time() - start_time) * 1000
        return "✅", f"OK ({response_time:.0f}ms)"

    except Exception as e:
        error_msg = str(e)
        if "insufficient" in error_msg.lower() and "quota" in error_msg.lower():
            return "💰", "Insufficient quota"
        elif "rate limit" in error_msg.lower():
            return "⏱️", "Rate limited"
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            return "🔐", "Invalid API key"
        else:
            return "❌", f"Error: {error_msg[:50]}..."


def main():
    """Main function"""
    print("⚡ Quick API Status Check")
    print("=" * 50)
    print()

    # Check environment variables
    check_env_vars()

    # Test APIs
    print("🧪 API Connection Tests:")
    print("-" * 40)

    apis = [
        ("Anthropic Claude", quick_test_anthropic),
        ("Mistral AI", quick_test_mistral),
        ("OpenAI", quick_test_openai),
    ]

    working_count = 0

    for name, test_func in apis:
        print(f"Testing {name}... ", end="", flush=True)
        try:
            status, message = test_func()
            print(f"{status} {message}")
            if status == "✅":
                working_count += 1
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    print()
    print("📊 Summary:")
    print("-" * 40)
    print(f"Working APIs: {working_count}/{len(apis)}")

    if working_count == 0:
        print("⚠️  No APIs are currently working!")
        print("💡 Recommendations:")
        print("   - Check API key validity")
        print("   - Verify account credits/quota")
        print("   - Wait for rate limits to reset")
        return 1
    else:
        print(f"✅ {working_count} API(s) available for processing")
        return 0


if __name__ == "__main__":
    sys.exit(main())
