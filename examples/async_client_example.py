#!/usr/bin/env python3
"""
Async AI Client Example

Demonstrates how to use async AI clients for concurrent article processing.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients import (
    AsyncArticleProcessor,
    AsyncClaudeClient,
    process_articles_async,
    run_async_processing,
)


# Sample articles for testing
SAMPLE_ARTICLES = [
    {
        "title": "Understanding Neural Networks",
        "content": """Neural networks are computational models inspired by biological neural networks.
        They consist of interconnected nodes (neurons) organized in layers. The input layer receives data,
        hidden layers process information, and the output layer produces results. Through training with
        backpropagation, neural networks learn to recognize patterns and make predictions.""",
        "url": "https://example.com/neural-networks",
    },
    {
        "title": "Introduction to Machine Learning",
        "content": """Machine learning is a subset of artificial intelligence that enables systems to learn
        and improve from experience without being explicitly programmed. It uses algorithms to identify
        patterns in data and make decisions with minimal human intervention. Common applications include
        image recognition, natural language processing, and predictive analytics.""",
        "url": "https://example.com/machine-learning",
    },
    {
        "title": "Deep Learning Fundamentals",
        "content": """Deep learning uses artificial neural networks with multiple layers to progressively
        extract higher-level features from raw input. It excels at processing unstructured data like
        images, audio, and text. Deep learning has revolutionized fields such as computer vision,
        speech recognition, and autonomous systems.""",
        "url": "https://example.com/deep-learning",
    },
    {
        "title": "Natural Language Processing",
        "content": """Natural Language Processing (NLP) enables computers to understand, interpret, and
        generate human language. It combines computational linguistics with machine learning and deep
        learning. Applications include sentiment analysis, machine translation, chatbots, and text
        summarization.""",
        "url": "https://example.com/nlp",
    },
    {
        "title": "Computer Vision Applications",
        "content": """Computer vision enables machines to interpret and understand visual information from
        the world. Using deep learning techniques like convolutional neural networks, systems can perform
        tasks such as object detection, facial recognition, image segmentation, and autonomous navigation.
        Applications span healthcare, security, automotive, and entertainment industries.""",
        "url": "https://example.com/computer-vision",
    },
]


async def example_1_basic_async():
    """Example 1: Basic async article analysis"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Async Article Analysis")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping example.")
        return

    # Create async client
    async with AsyncClaudeClient(api_key=api_key) as client:
        print(f"✓ Created async {client.provider_name} client")
        print(f"  Model: {client.model}")
        print(f"  Max concurrent: {client.max_concurrent_requests}")

        # Analyze single article
        print("\nAnalyzing single article...")
        result = await client.analyze_article_async(
            title=SAMPLE_ARTICLES[0]["title"],
            content=SAMPLE_ARTICLES[0]["content"],
            url=SAMPLE_ARTICLES[0]["url"],
        )

        if result:
            print(f"✓ Analysis complete")
            print(f"  Extracted title: {result.get('extracted_title', 'N/A')}")
            print(
                f"  Analysis length: {len(result['methodology_detailed'])} characters"
            )
        else:
            print("✗ Analysis failed")


async def example_2_batch_processing():
    """Example 2: Batch processing with concurrent requests"""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing (Concurrent)")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping example.")
        return

    async with AsyncClaudeClient(api_key=api_key) as client:
        print(f"Processing {len(SAMPLE_ARTICLES)} articles concurrently...")
        print(f"Max concurrent requests: {client.max_concurrent_requests}")

        start_time = time.time()

        # Process all articles concurrently
        results = await client.batch_analyze_async(
            SAMPLE_ARTICLES, max_concurrent=3  # Limit to 3 concurrent for demo
        )

        elapsed = time.time() - start_time

        # Show results
        successful = [r for r in results if r is not None]
        print(f"\n✓ Batch processing complete in {elapsed:.2f} seconds")
        print(f"  Successful: {len(successful)}/{len(SAMPLE_ARTICLES)}")
        print(f"  Average time per article: {elapsed/len(SAMPLE_ARTICLES):.2f}s")

        # Show sample result
        if successful:
            print(f"\nSample result:")
            print(f"  Title: {SAMPLE_ARTICLES[0]['title']}")
            print(
                f"  Analysis: {successful[0]['methodology_detailed'][:150]}..."
            )


async def example_3_processor():
    """Example 3: Using AsyncArticleProcessor"""
    print("\n" + "=" * 60)
    print("Example 3: AsyncArticleProcessor with Progress")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping example.")
        return

    processor = AsyncArticleProcessor(
        provider="anthropic", api_key=api_key, max_concurrent=3
    )

    print(f"Processing {len(SAMPLE_ARTICLES)} articles with progress tracking...")

    start_time = time.time()

    # Process with progress tracking
    results = await processor.process_articles(SAMPLE_ARTICLES, show_progress=True)

    elapsed = time.time() - start_time

    await processor.close()

    print(f"\n✓ Processing complete in {elapsed:.2f} seconds")
    print(f"  Results: {len([r for r in results if r])}/{len(results)} successful")


async def example_4_batches_with_delay():
    """Example 4: Processing in batches with delays"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing with Delays")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping example.")
        return

    processor = AsyncArticleProcessor(
        provider="anthropic", api_key=api_key, max_concurrent=2
    )

    print(f"Processing {len(SAMPLE_ARTICLES)} articles in batches...")
    print("  Batch size: 2")
    print("  Delay between batches: 1.0s")

    results = await processor.process_in_batches(
        SAMPLE_ARTICLES, batch_size=2, delay_between_batches=1.0
    )

    await processor.close()

    print(f"\n✓ Batch processing complete")
    print(f"  Results: {len([r for r in results if r])}/{len(results)} successful")


def example_5_sync_wrapper():
    """Example 5: Using synchronous wrapper"""
    print("\n" + "=" * 60)
    print("Example 5: Synchronous Wrapper (run_async_processing)")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping example.")
        return

    print("Using synchronous wrapper for async processing...")

    # This function handles asyncio.run() for you
    results = run_async_processing(
        articles=SAMPLE_ARTICLES[:3],  # Use first 3 articles
        provider="anthropic",
        api_key=api_key,
        max_concurrent=3,
    )

    print(f"\n✓ Processing complete")
    print(f"  Results: {len([r for r in results if r])}/{len(results)} successful")


async def example_6_performance_comparison():
    """Example 6: Compare sync vs async performance"""
    print("\n" + "=" * 60)
    print("Example 6: Performance Comparison (Sync vs Async)")
    print("=" * 60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping example.")
        return

    # Note: This is a simulation since we need real API calls for accurate comparison
    print("Simulated performance comparison:")
    print(f"  Articles: {len(SAMPLE_ARTICLES)}")
    print(f"  Average API latency: ~5 seconds per request")
    print()

    # Sequential (sync) simulation
    print("Sequential Processing (Sync):")
    print(f"  Time: {len(SAMPLE_ARTICLES) * 5} seconds")
    print(f"  Total API calls: {len(SAMPLE_ARTICLES)} (sequential)")
    print()

    # Concurrent (async) simulation
    concurrent = 3
    batches = (len(SAMPLE_ARTICLES) + concurrent - 1) // concurrent
    print(f"Concurrent Processing (Async, max_concurrent={concurrent}):")
    print(f"  Time: ~{batches * 5} seconds")
    print(f"  Total API calls: {len(SAMPLE_ARTICLES)} ({concurrent} concurrent)")
    print(
        f"  Speedup: {(len(SAMPLE_ARTICLES) * 5) / (batches * 5):.1f}x faster"
    )


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("ASYNC AI CLIENTS - EXAMPLES")
    print("=" * 60)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY environment variable not set.")
        print("   Some examples will be skipped.")
        print("\n   To run all examples:")
        print('   export ANTHROPIC_API_KEY="your-api-key"')

    # Run examples
    await example_1_basic_async()
    await example_2_batch_processing()
    await example_3_processor()
    await example_4_batches_with_delay()
    example_5_sync_wrapper()
    await example_6_performance_comparison()

    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())
