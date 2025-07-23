#!/usr/bin/env python3
"""
Reanalyze all articles using Claude directly with Feynman technique
"""

import json
import sqlite3
from datetime import UTC, datetime


def analyze_with_claude(title, content):
    """
    Analyze an article using Claude with the Feynman technique as if the author
    """
    prompt = f"""Explain this paper to me in depth using the Feynman technique, as if you were its author.

Title: {title}

Content:
{content}

Please provide a comprehensive analysis that breaks down the complex concepts into simple, understandable terms. Imagine you are the author explaining your own work to someone who is intelligent but not an expert in your field.

Focus on:
1. The core problem you were trying to solve
2. Your approach and methodology in simple terms
3. The key insights and findings
4. Why this matters and what it means for the field
5. Any limitations or future directions

Explain as if you're teaching someone who wants to truly understand, not just memorize."""

    # This would normally call Claude API, but since we're using Claude Code directly,
    # we'll return the prompt for manual processing
    return prompt


def main():
    # Load articles to reanalyze
    with open("articles_to_reanalyze.json") as f:
        articles = json.load(f)

    print(f"Loaded {len(articles)} articles to reanalyze")

    # Process each article
    for i, article in enumerate(articles, 1):
        print(f"\n{'=' * 60}")
        print(f"ARTICLE {i}/{len(articles)}: {article['title'][:50]}...")
        print(f"{'=' * 60}")

        # Generate Claude prompt
        prompt = analyze_with_claude(article["title"], article["content"])

        print(f"Prompt generated for article ID {article['id']}")
        print(f"Title: {article['title']}")
        print(f"URL: {article['url']}")
        print("\nPrompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)

        # Ask user for Claude's response
        print("\nPlease provide Claude's analysis for this article:")
        print("(Press Enter on an empty line when done, or 'SKIP' to skip)")

        analysis_lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == "SKIP":
                    print(f"Skipping article {article['id']}")
                    break
                if line.strip() == "" and analysis_lines:
                    break
                analysis_lines.append(line)
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return

        if analysis_lines:
            analysis = "\n".join(analysis_lines)

            # Update database
            conn = sqlite3.connect("data/articles.db")
            cursor = conn.cursor()

            # Update content table with new analysis
            cursor.execute(
                """
                UPDATE content
                SET key_findings = ?,
                    technical_approach = ?,
                    methodology_detailed = ?,
                    metadata = ?
                WHERE article_id = ?
            """,
                (
                    analysis,  # Store full analysis in key_findings
                    "Claude Code analysis using Feynman technique",
                    analysis,  # Also store in methodology_detailed
                    json.dumps(
                        {
                            "ai_provider": "claude",
                            "analysis_method": "feynman_technique",
                            "analyzed_at": datetime.now(UTC).isoformat(),
                        }
                    ),
                    article["id"],
                ),
            )

            # Update articles table status and processed_date
            cursor.execute(
                """
                UPDATE articles
                SET processed_date = ?, updated_at = ?
                WHERE id = ?
            """,
                (
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                    article["id"],
                ),
            )

            conn.commit()
            conn.close()

            print(f"✅ Updated article {article['id']} in database")

        # Ask if user wants to continue
        if i < len(articles):
            continue_response = (
                input("\nContinue with next article? (y/n/q): ").strip().lower()
            )
            if continue_response in ["n", "q", "quit"]:
                print("Stopping analysis process")
                break

    print("\n🎉 Analysis complete! Processed articles.")


if __name__ == "__main__":
    main()
