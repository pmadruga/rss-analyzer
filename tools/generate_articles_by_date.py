#!/usr/bin/env python3
"""
Generate articles by date markdown file from database
"""

import os
import sqlite3
from collections import defaultdict
from datetime import datetime


def get_articles_from_db(db_path="data/articles.db"):
    """Get all completed articles from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT
        a.title,
        a.url,
        a.processed_date,
        c.methodology_detailed,
        c.technical_approach,
        c.key_findings
    FROM articles a
    LEFT JOIN content c ON a.id = c.article_id
    WHERE a.status = 'completed'
    ORDER BY a.processed_date;
    """

    cursor.execute(query)
    articles = cursor.fetchall()
    conn.close()

    return articles


def format_article_data(articles):
    """Format articles by date"""
    articles_by_date = defaultdict(list)

    for article in articles:
        title, url, processed_date, methodology, technical_approach, key_findings = (
            article
        )

        # Parse date
        date_obj = datetime.fromisoformat(processed_date.replace("Z", "+00:00"))
        date_key = date_obj.strftime("%B %d, %Y")

        articles_by_date[date_key].append(
            {
                "title": title,
                "url": url,
                "processed_date": processed_date,
                "methodology": methodology or "Not specified",
                "technical_approach": technical_approach or "Not specified",
                "key_findings": key_findings or "Not specified",
            }
        )

    return articles_by_date


def generate_markdown(articles_by_date, output_path="output/articles_by_date.md"):
    """Generate markdown file"""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_articles = sum(len(articles) for articles in articles_by_date.values())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Articles Analysis by Date\n\n")
        f.write(
            "This document contains all analyzed articles organized by their processing date.\n\n"
        )

        for date, articles in articles_by_date.items():
            f.write(f"## {date}\n\n")

            for article in articles:
                f.write(f"### {article['title']}\n")
                f.write(f"**Source:** {article['url']}  \n")
                f.write(f"**Processed:** {article['processed_date']}  \n")

                f.write("**Methodology:**\n")
                f.write(f"{article['methodology']}\n\n")

                f.write("**Technical Approach:**\n")
                f.write(f"{article['technical_approach']}\n\n")

                f.write("**Key Findings:**\n")
                f.write(f"{article['key_findings']}\n\n")

                f.write("---\n\n")

        f.write("## Summary Statistics\n")
        f.write(f"- **Total Articles Analyzed:** {total_articles}\n")
        f.write("- **Sources:** ArXiv papers, Jina.ai articles, Bluesky posts\n")
        f.write(
            "- **Topics:** AI/ML, Embeddings, Quantization, LLM Routing, Knowledge Graphs, Document Retrieval, Recommendation Systems\n"
        )


def cleanup_other_reports():
    """Remove other report files, keep only articles_by_date.md"""

    files_to_remove = [
        "output/article_analysis_report.md",
        "output/summary_report.md",
        "output/articles_export.json",
        "output/articles_export.csv",
    ]

    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")
        except Exception as e:
            print(f"Warning: Could not remove {file_path}: {e}")


if __name__ == "__main__":
    print("Generating articles by date file...")
    articles = get_articles_from_db()
    articles_by_date = format_article_data(articles)
    generate_markdown(articles_by_date)

    # Clean up other report files
    cleanup_other_reports()

    total_articles = sum(len(a) for a in articles_by_date.values())
    print(f"Generated articles_by_date.md with {total_articles} articles")
    print("Removed other report files - keeping only articles_by_date.md")
