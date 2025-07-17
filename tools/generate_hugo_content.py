#!/usr/bin/env python3
"""
Generate Hugo content from RSS article analysis data
"""

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Paths
DATA_JSON = Path("/Users/pedromadruga/dev/claude/rss-analyzer/docs/data.json")
HUGO_CONTENT_DIR = Path("/Users/pedromadruga/dev/claude/rss-analyzer/website/content")


def sanitize_filename(text):
    """Sanitize text for use as filename"""
    # Remove special characters and replace spaces with hyphens
    sanitized = re.sub(r"[^\w\s-]", "", text)
    sanitized = re.sub(r"[-\s]+", "-", sanitized)
    return sanitized.lower().strip("-")


def extract_domain(url):
    """Extract domain from URL"""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except (ValueError, AttributeError):
        return "unknown"


def create_article_content(article):
    """Create Hugo content for an article"""

    # Extract metadata
    title = article["title"]
    url = article["url"]
    date = article["processed_date"]
    analysis = article["analysis"]
    domain = extract_domain(url)

    # Parse date
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        hugo_date = parsed_date.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        hugo_date = "2025-01-17"

    # Create filename
    filename = sanitize_filename(title)
    if len(filename) > 50:
        filename = filename[:50]

    # Format analysis content - convert Feynman style to structured markdown
    formatted_analysis = format_analysis_content(analysis)

    # Generate front matter
    front_matter = f"""---
title: "{title}"
date: {hugo_date}
draft: false
weight: {article["id"]}
url: "/articles/{filename}/"
tags:
  - "{domain}"
  - "research"
  - "ai-analysis"
summary: "{title}"
params:
  original_url: "{url}"
  article_id: {article["id"]}
  domain: "{domain}"
---

# {title}

{{{{< callout type="info" >}}}}
**Original Source:** [{domain}]({url})
**Analyzed:** {hugo_date}
**AI Provider:** {article.get("ai_provider", "claude")}
{{{{< /callout >}}}}

{formatted_analysis}

---

{{{{< callout type="note" >}}}}
This analysis was generated using the Feynman technique, where the AI takes on the role of the paper's author and explains the research using simple language and analogies to make complex concepts accessible.
{{{{< /callout >}}}}
"""

    return front_matter, filename


def format_analysis_content(analysis):
    """Format the analysis content with proper markdown structure"""

    # Split analysis into sections based on bold headers
    sections = re.split(r"\*\*(.*?)\*\*:", analysis)

    formatted = []

    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            header = sections[i].strip()
            content = sections[i + 1].strip()

            # Create proper markdown section
            formatted.append(f"## {header}\n\n{content}\n")

    # If no sections found, return original with some formatting
    if not formatted:
        return analysis.replace("**", "").replace("\n\n", "\n\n")

    return "\n".join(formatted)


def create_articles_index():
    """Create the articles index page"""

    with open(DATA_JSON) as f:
        data = json.load(f)

    articles = data["articles"]
    total_articles = len(articles)

    # Group articles by domain
    by_domain = {}
    for article in articles:
        domain = extract_domain(article["url"])
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(article)

    # Create index content
    index_content = f"""---
title: "Analyzed Articles"
layout: docs
---

# Analyzed Articles

This section contains {total_articles} articles analyzed using AI with the Feynman technique. Each article is explained as if the author were teaching the concepts to someone encountering the topic for the first time.

## Browse by Source

"""

    # Add domain sections
    for domain, domain_articles in sorted(by_domain.items()):
        index_content += f"### {domain.title()} ({len(domain_articles)} articles)\n\n"

        for article in domain_articles[:5]:  # Show first 5 articles per domain
            filename = sanitize_filename(article["title"])
            if len(filename) > 50:
                filename = filename[:50]

            index_content += f"- [{article['title']}](/articles/{filename}/)\n"

        if len(domain_articles) > 5:
            index_content += f"- *...and {len(domain_articles) - 5} more*\n"

        index_content += "\n"

    return index_content


def generate_sidebar_data():
    """Generate sidebar navigation data for Hugo"""

    with open(DATA_JSON) as f:
        data = json.load(f)

    articles = data["articles"]

    # Create sidebar structure
    sidebar_items = []

    # Group by domain for better organization
    by_domain = {}
    for article in articles:
        domain = extract_domain(article["url"])
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(article)

    for domain, domain_articles in sorted(by_domain.items()):
        domain_section = {"name": domain.title(), "pages": []}

        for article in domain_articles:
            filename = sanitize_filename(article["title"])
            if len(filename) > 50:
                filename = filename[:50]

            domain_section["pages"].append(
                {
                    "name": article["title"][:60]
                    + ("..." if len(article["title"]) > 60 else ""),
                    "pageRef": f"/articles/{filename}/",
                    "weight": article["id"],
                }
            )

        sidebar_items.append(domain_section)

    return sidebar_items


def main():
    """Generate all Hugo content"""

    print("Loading article data...")
    with open(DATA_JSON) as f:
        data = json.load(f)

    articles = data["articles"]

    # Create articles directory
    articles_dir = HUGO_CONTENT_DIR / "articles"
    articles_dir.mkdir(exist_ok=True)

    print(f"Generating content for {len(articles)} articles...")

    # Generate individual article pages
    for article in articles:
        content, filename = create_article_content(article)

        # Write article file
        article_file = articles_dir / f"{filename}.md"
        with open(article_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Created: {article_file}")

    # Generate articles index
    print("Creating articles index...")
    index_content = create_articles_index()
    index_file = articles_dir / "_index.md"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(index_content)

    print(f"Created: {index_file}")

    # Generate about page
    about_content = """---
title: "About"
date: 2025-01-17
draft: false
---

# About RSS Article Analysis

This project automatically fetches and analyzes academic papers from RSS feeds using AI APIs (Anthropic Claude, Mistral, or OpenAI). The goal is to make complex research accessible through the Feynman technique.

## How It Works

1. **RSS Feed Parsing**: Automatically fetches articles from configured RSS feeds
2. **Content Scraping**: Extracts full article content from academic publisher websites
3. **AI Analysis**: Uses the Feynman technique to generate educational explanations
4. **Report Generation**: Creates comprehensive reports in multiple formats

## The Feynman Technique

All analyses use the Feynman technique, where:
- The AI takes on the role of the paper's author
- Complex concepts are explained using simple language and analogies
- Technical details are broken down to fundamental components
- Research is explained step-by-step with clear reasoning

## Supported Sources

- **Academic**: arXiv, IEEE Xplore, ACM Digital Library, Nature, PubMed
- **Tech Companies**: OpenAI, Anthropic, DeepMind, Google AI Research
- **Tech Blogs**: Medium, Substack, TechCrunch, Wired
- **Social Media**: Bluesky posts with embedded arXiv links

## Technology Stack

- **Backend**: Python with SQLite database
- **AI Providers**: Anthropic Claude, Mistral AI, OpenAI
- **Website**: Hugo with Hextra theme
- **Deployment**: GitHub Pages
"""

    about_file = HUGO_CONTENT_DIR / "about.md"
    with open(about_file, "w", encoding="utf-8") as f:
        f.write(about_content)

    print(f"Created: {about_file}")

    print("\nGenerated Hugo content successfully!")
    print(f"Total files created: {len(articles) + 2}")


if __name__ == "__main__":
    main()
