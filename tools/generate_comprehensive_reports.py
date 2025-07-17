#!/usr/bin/env python3
"""
Generate comprehensive reports organized by paper titles
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any


def extract_paper_title_from_url(url: str) -> str:
    """Extract paper title from various URL patterns"""
    if "arxiv.org/abs/" in url:
        arxiv_id = url.split("/abs/")[-1]
        return f"ArXiv Paper {arxiv_id}"
    elif "bsky.app" in url and "reachsumit.com" in url:
        if "3lssbir3mk222" in url:
            return "IRanker: A Ranking Foundation Model"
        elif "3lssbxtzylc22" in url:
            return "VAT-KG: A Multimodal Knowledge Graph"
        elif "3lsi5qzveoc2x" in url:
            return "CRUX: Evaluating Retrieval-Augmented Contexts"
        else:
            return "Research Paper (Sumit)"
    elif "bsky.app" in url and "arxiv-cs-ir" in url:
        if "3lssft2zuof25" in url:
            return "ARAG: Agentic Retrieval Augmented Generation"
        elif "3lssineizm42c" in url:
            return "HPC-ColPali: High-Performance Document Retrieval"
        elif "3lssiq54mri2x" in url:
            return "PentaRAG: Multi-layered Enterprise RAG System"
        elif "3lsskaxcsh52p" in url:
            return "LLM2Rec: Sequential Recommendation with LLMs"
        else:
            return "IR Research Paper (ArXiv CS.IR)"
    elif "bsky.app" in url and "paper.bsky.social" in url:
        return "Text-to-LoRA: Instant Transformer Adaption"
    elif "bsky.app" in url and "sungkim.bsky.social" in url:
        if "3lrs76hb3tk2p" in url:
            return "Deep Research Systems Survey"
        elif "3lrlxhzbtsk26" in url:
            return "Web Agent Research Paradigm"
        else:
            return "Research Paper (Sung Kim)"
    else:
        return f"Research Paper ({url.split('/')[-2] if '/' in url else 'Unknown'})"


def get_all_articles(db_path: str) -> list[dict[str, Any]]:
    """Get all completed articles from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT a.id, a.title, a.url, a.publication_date, a.processed_date,
           c.methodology_detailed, c.technical_approach, c.key_findings,
           c.research_design, c.metadata
    FROM articles a
    JOIN content c ON a.id = c.article_id
    WHERE a.status = 'completed'
    ORDER BY a.processed_date ASC
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    articles = []
    for row in rows:
        metadata = json.loads(row[10]) if row[10] else {}
        paper_title = extract_paper_title_from_url(row[2])

        article = {
            "id": row[0],
            "title": row[1],
            "paper_title": paper_title,
            "url": row[2],
            "publication_date": row[3],
            "processed_date": row[4],
            "methodology_detailed": row[5],
            "technical_approach": row[6],
            "key_findings": row[7],
            "research_design": row[8],
            "analyzed_at": metadata.get("analyzed_at"),
            "model_used": metadata.get("model_used", "Unknown"),
        }
        articles.append(article)

    return articles


def generate_comprehensive_json(
    articles: list[dict[str, Any]], output_dir: str, timestamp: str
):
    """Generate comprehensive JSON export with all articles"""
    # Group by paper title
    papers_by_title = {}
    for article in articles:
        title = article["paper_title"]
        if title not in papers_by_title:
            papers_by_title[title] = []
        papers_by_title[title].append(article)

    # Create comprehensive export
    export_data = {
        "generated_at": datetime.now().isoformat(),
        "total_papers": len(papers_by_title),
        "total_articles": len(articles),
        "papers": {},
    }

    for paper_title, paper_articles in papers_by_title.items():
        export_data["papers"][paper_title] = {
            "article_count": len(paper_articles),
            "articles": paper_articles,
        }

    # Write comprehensive JSON with timestamp
    json_path = os.path.join(
        output_dir, f"comprehensive_articles_export_{timestamp}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Generated comprehensive JSON: {json_path}")


def generate_paper_specific_files(
    articles: list[dict[str, Any]], output_dir: str, timestamp: str
):
    """Generate individual files for each paper"""
    # Group by paper title
    papers_by_title = {}
    for article in articles:
        title = article["paper_title"]
        if title not in papers_by_title:
            papers_by_title[title] = []
        papers_by_title[title].append(article)

    for paper_title, paper_articles in papers_by_title.items():
        # Sanitize filename
        safe_filename = "".join(
            c for c in paper_title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_filename = safe_filename.replace(" ", "_")

        # Generate JSON for this paper
        paper_data = {
            "paper_title": paper_title,
            "generated_at": datetime.now().isoformat(),
            "article_count": len(paper_articles),
            "articles": paper_articles,
        }

        json_path = os.path.join(output_dir, f"{safe_filename}_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(paper_data, f, indent=2, ensure_ascii=False)

        # Generate Markdown for this paper
        md_content = f"# {paper_title}\n\n"
        md_content += (
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        md_content += f"**Total Articles:** {len(paper_articles)}\n\n"

        for i, article in enumerate(paper_articles, 1):
            md_content += f"## Analysis {i}\n\n"
            md_content += f"**Source:** [{article['url']}]({article['url']})\n\n"
            md_content += f"**Publication Date:** {article['publication_date']}\n\n"
            md_content += f"**Processed:** {article['processed_date']}\n\n"
            md_content += f"**Model Used:** {article['model_used']}\n\n"

            md_content += "### Methodology\n\n"
            md_content += f"{article['methodology_detailed']}\n\n"

            md_content += "### Key Findings\n\n"
            md_content += f"{article['key_findings']}\n\n"

            md_content += "### Technical Approach\n\n"
            md_content += f"{article['technical_approach']}\n\n"

            md_content += "### Research Design\n\n"
            md_content += f"{article['research_design']}\n\n"

            md_content += "---\n\n"

        md_path = os.path.join(output_dir, f"{safe_filename}_{timestamp}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(
            f"Generated files for '{paper_title}': {safe_filename}_{timestamp}.json, {safe_filename}_{timestamp}.md"
        )


def generate_comprehensive_markdown(
    articles: list[dict[str, Any]], output_dir: str, timestamp: str
):
    """Generate comprehensive markdown report"""
    # Group by paper title
    papers_by_title = {}
    for article in articles:
        title = article["paper_title"]
        if title not in papers_by_title:
            papers_by_title[title] = []
        papers_by_title[title].append(article)

    md_content = "# Comprehensive RSS Feed Article Analysis Report\n\n"
    md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Total Papers:** {len(papers_by_title)}\n\n"
    md_content += f"**Total Articles:** {len(articles)}\n\n"

    # Table of Contents
    md_content += "## Table of Contents\n\n"
    for i, paper_title in enumerate(papers_by_title.keys(), 1):
        safe_anchor = (
            paper_title.lower().replace(" ", "-").replace(":", "").replace(",", "")
        )
        md_content += f"{i}. [{paper_title}](#{safe_anchor})\n"
    md_content += "\n---\n\n"

    # Paper sections
    for paper_title, paper_articles in papers_by_title.items():
        safe_anchor = (
            paper_title.lower().replace(" ", "-").replace(":", "").replace(",", "")
        )
        md_content += f"## {paper_title} {{#{safe_anchor}}}\n\n"
        md_content += f"**Articles Analyzed:** {len(paper_articles)}\n\n"

        for i, article in enumerate(paper_articles, 1):
            md_content += f"### Analysis {i}\n\n"
            md_content += f"**Source:** [{article['url']}]({article['url']})\n\n"
            md_content += f"**Publication Date:** {article['publication_date']}\n\n"
            md_content += f"**Processed:** {article['processed_date']}\n\n"
            md_content += f"**Model Used:** {article['model_used']}\n\n"

            md_content += "#### Methodology\n\n"
            md_content += f"{article['methodology_detailed']}\n\n"

            md_content += "#### Key Findings\n\n"
            md_content += f"{article['key_findings']}\n\n"

            md_content += "#### Technical Approach\n\n"
            md_content += f"{article['technical_approach']}\n\n"

            md_content += "#### Research Design\n\n"
            md_content += f"{article['research_design']}\n\n"

            if i < len(paper_articles):
                md_content += "---\n\n"

        md_content += "\n---\n\n"

    md_content += "*This comprehensive report was generated automatically by the RSS Article Analyzer.*\n"
    md_content += (
        f"*Report generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n"
    )

    # Write comprehensive markdown with timestamp
    md_path = os.path.join(output_dir, f"comprehensive_analysis_report_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated comprehensive markdown: {md_path}")


def main():
    db_path = "/app/data/articles.db"
    output_dir = "/app/output"

    # Get all articles
    articles = get_all_articles(db_path)
    print(f"Found {len(articles)} completed articles")

    # Generate single timestamp for all files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate comprehensive reports
    generate_comprehensive_json(articles, output_dir, timestamp)
    generate_comprehensive_markdown(articles, output_dir, timestamp)
    generate_paper_specific_files(articles, output_dir, timestamp)

    print("All comprehensive reports generated successfully!")


if __name__ == "__main__":
    main()
