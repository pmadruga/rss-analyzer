#!/usr/bin/env python3
"""
Update website data.json with new Feynman-style summaries
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

# Database path
DB_PATH = Path("/Users/pedromadruga/dev/claude/rss-analyzer/data/articles.db")
OUTPUT_PATH = Path("/Users/pedromadruga/dev/claude/rss-analyzer/docs/data.json")

# Feynman summaries mapping
FEYNMAN_SUMMARIES = {
    24: {
        "title": "Jailbreaking LLMs with InfoFlood Method",
        "analysis": """**The Core Idea:** I've discovered a way to trick AI systems by speaking in a way that sounds incredibly academic and sophisticated, but is actually just nonsense designed to confuse the AI's safety systems. Think of it like this: AI systems have guards at the door (safety filters) that check if someone is trying to make them do something harmful. But what if instead of walking up to the guard directly, you dressed up in a professor's outfit and started using incredibly complex academic language?

**How It Works:** The InfoFlood method transforms simple, potentially harmful requests into elaborate academic prose filled with complex words, fake citations, and technical jargon. It's like wrapping a simple request in so many layers of academic packaging that the AI gets confused about what's actually being asked.

**Why It Works:** Large Language Models rely on pattern recognition. When you bury the actual request under mountains of academic-sounding text, the model's attention gets diluted. The safety filters are looking for obvious red flags, but academic language rarely triggers these filters.

**The Implications:** This reveals a fundamental weakness in how we currently implement AI safety: we're too focused on surface-level patterns rather than deep understanding of intent.""",
    },
    23: {
        "title": "Measuring Hypothesis Testing Errors in Information Retrieval",
        "analysis": """**The Problem I'm Solving:** When we test whether one search system is better than another, we typically make mistakes - but we've only been counting half of them! We've been obsessed with avoiding Type I errors (false positives) but completely ignoring Type II errors (false negatives). That's like a doctor who's so worried about misdiagnosing healthy people that they miss actual sick patients!

**My Solution:** I propose that we need to measure BOTH types of errors to truly understand how good our evaluation methods are. I introduce balanced accuracy as a single metric that captures both how often you correctly identify differences and how often you correctly identify no difference.

**The Key Insight:** Different evaluation methods have different "discriminative power" - their ability to correctly identify when one system is truly better than another. By only measuring Type I errors, we've been flying half-blind.

**What This Means:** We need to rethink how we evaluate our evaluation methods. We've been so conservative about avoiding false positives that we may have been using evaluation approaches that miss real improvements.""",
    },
    22: {
        "title": "FrugalRAG: Efficient Multi-hop Question Answering",
        "analysis": """**The Core Innovation:** I've discovered that large language models don't need massive amounts of training to become better at retrieval-augmented generation (RAG). With just 1,000 carefully chosen examples, we can teach them to be nearly twice as efficient while maintaining the same accuracy!

**The Two-Stage Magic:** My approach works in two clever stages. Stage 1 teaches the model to recognize when it actually needs more information. Stage 2 trains it to reason through documents efficiently, connecting pieces of information without redundant searches.

**Why This Matters:** Current RAG systems are like students who run to the library every time they need to answer any part of a question. My system reduces retrieval calls by nearly 50% while maintaining competitive accuracy.

**The Surprising Discovery:** You don't need millions of examples to achieve this. With just 1,000 well-chosen training examples, the model learns the PATTERN of when retrieval is useful, not just memorizing specific cases.""",
    },
    21: {
        "title": "Context Engineering for LLM Agents",
        "analysis": """**The Central Metaphor:** Think of Large Language Models as a new kind of operating system, where the LLM is like the CPU and its context window is like RAM. Just as your computer slows down when RAM is full, LLMs struggle when their context windows are overloaded.

**The Four Pillars of Context Engineering:**
1. **Write Context:** Just as you take notes while solving problems, agents need scratchpads and memories
2. **Select Context:** Not everything in your notes is relevant - context selection is like having a smart assistant who knows which files to pull
3. **Compress Context:** Sometimes you need to summarize War and Peace into a paragraph
4. **Isolate Context:** Complex tasks benefit from splitting context across specialized sub-agents

**Why This Matters Now:** As we build agents that can work for hours or days on complex tasks, context management becomes THE critical bottleneck. It's not about having the smartest model - it's about using its intelligence efficiently.

**The Key Insight:** Context engineering isn't just an optimization - it's fundamental to agent capability. We're moving from "prompt engineering" (what to say) to "context engineering" (what to remember and when).""",
    },
    20: {
        "title": "Harnessing Multiple LLMs: A Survey on LLM Ensemble",
        "analysis": """**The Big Idea:** Instead of relying on a single AI model, what if we could orchestrate multiple models to work together, each contributing their unique strengths? I'm proposing a comprehensive framework for "LLM Ensemble" - making multiple large language models collaborate like musicians in an orchestra.

**Three Ways to Ensemble:**
1. **Ensemble-Before-Inference:** Like having a pre-meeting where experts discuss strategy
2. **Ensemble-During-Inference:** Models work together in real-time, like a surgical team
3. **Ensemble-After-Inference:** Combining outputs after generation, like synthesizing multiple expert reports

**The Challenge of Coordination:** The hardest part isn't getting models to work - it's getting them to work TOGETHER effectively. How do you resolve disagreements? Prevent redundant work? Ensure models complement rather than interfere?

**Why This Changes Everything:** Single models have inherent biases and blind spots. By combining multiple models, we can compensate for individual weaknesses, achieve more reliable outputs, and handle more complex tasks.""",
    },
    19: {
        "title": "LangChain Context Engineering Deep Dive",
        "analysis": """**The Memory Revolution:** I'm proposing a fundamental shift in how we think about AI agent development. Instead of focusing on making models smarter, we need to make them better at managing their own memories and attention.

**The Technical Implementation:**
- **State Management as Memory:** Every agent needs a state object - think of it as the agent's desk
- **Multi-Agent Architecture:** For complex tasks, split work across specialized agents like running a newspaper
- **Sandboxing for Safety:** Isolate operations that generate massive data in separate environments

**The Practical Impact:** With proper context engineering, we're seeing agents handle tasks 10x longer without degrading, 50% reduction in token usage, and more reliable performance.

**The Future Vision:** We're moving toward agents that can work on problems for days or weeks, maintaining context across sessions and managing their own cognitive resources.""",
    },
    17: {
        "title": "Multi-Agent Systems and Context Management",
        "analysis": """**Advanced Context Engineering:** This explores how multiple AI agents can work together effectively by managing their individual and shared contexts. Think of it as organizing a team where each member has their own workspace but can share important information when needed.

**Key Technical Components:** The system uses specialized routing, memory management, and coordination protocols to ensure agents don't step on each other's toes while maximizing their collective capabilities.""",
    },
    16: {
        "title": "GlórIA: Portuguese Language Model",
        "analysis": """**Breaking Language Barriers:** A significant development in making AI accessible to Portuguese speakers worldwide, addressing the linguistic diversity gap in current LLM technology. This represents an important step toward democratizing AI access across different languages and cultures.

**Technical Achievement:** The model demonstrates strong understanding of Portuguese language nuances, handling various tasks with coherent and contextually relevant text generation.""",
    },
    15: {
        "title": "LlamaIndex Integration Patterns",
        "analysis": """**Building Better RAG Systems:** LlamaIndex provides powerful tools for creating retrieval-augmented generation systems. This explores integration patterns and best practices for building efficient, scalable RAG applications.

**Key Focus Areas:** The emphasis is on modular design, efficient indexing strategies, and seamless integration with various data sources and LLM providers.""",
    },
    14: {
        "title": "Web Agent Paradigm Shift",
        "analysis": """**Rethinking Web Interaction:** The advocates propose a paradigm shift: rather than forcing web agents to adapt to interfaces designed for humans, we should develop a new interaction paradigm specifically optimized for agents.

**The Vision:** "Build the web for agents, not agents for the web" - this fundamental rethinking could lead to more efficient and capable web automation systems.""",
    },
    13: {
        "title": "Deep Research Survey: Systems and Applications",
        "analysis": """**Comprehensive Analysis:** A thorough survey of more than 80 commercial and non-commercial deep research implementations that have emerged since 2023, including offerings from OpenAI, Gemini, Perplexity, and others.

**Key Insights:** The survey reveals common patterns, architectural choices, and implementation strategies across different deep research systems, providing valuable guidance for future development.""",
    },
    12: {
        "title": "Controlled RAG Context Evaluation",
        "analysis": """**Better RAG Evaluation:** Introduces a framework for evaluating retrieval context in long-form RAG using human-written summaries to control information scope. This addresses a critical gap in how we measure RAG system effectiveness.

**The CRUX Framework:** Uses question-based evaluation to assess RAG's retrieval in a fine-grained manner, offering more reflective and diagnostic evaluation than traditional metrics.""",
    },
    11: {
        "title": "Text-to-LoRA: Instant Transformer Adaptation",
        "analysis": """**Democratizing Model Specialization:** Text-to-LoRA enables adapting large language models on the fly solely based on natural language descriptions. It's a hypernetwork that constructs LoRAs in a single inexpensive forward pass.

**The Innovation:** After training on just 9 pre-trained LoRA adapters, the system can match task-specific adapter performance and even generalize to entirely unseen tasks with minimal compute requirements.""",
    },
    10: {
        "title": "Advanced Information Retrieval Research",
        "analysis": """**Pushing IR Boundaries:** Multiple papers explore cutting-edge techniques in information retrieval, from multi-vector document retrieval to ranking foundation models.

**Key Themes:** Efficiency improvements through compression and quantization, better evaluation metrics, and novel architectures for handling complex retrieval tasks at scale.""",
    },
    9: {
        "title": "PentaRAG: Enterprise-Scale Knowledge Retrieval",
        "analysis": """**Five-Layer Intelligence:** PentaRAG introduces a five-layer module that routes queries through instant caches, memory-recall mode, adaptive session memory, and conventional RAG. This achieves sub-second latency while maintaining freshness.

**Enterprise Impact:** The system cuts average GPU time to 0.248 seconds per query and sustains ~100,000 queries per second, demonstrating production-grade efficiency.""",
    },
    8: {
        "title": "ColPali Hierarchical Patch Compression",
        "analysis": """**Efficient Multi-Vector Retrieval:** Addresses the storage and computational costs of multi-vector document retrieval systems through K-Means quantization, attention-guided pruning, and optional binary encoding.

**Real-World Results:** Achieves 30-50% lower query latency while maintaining high retrieval precision, with up to 32x storage reduction.""",
    },
    7: {
        "title": "ARAG: Agentic RAG for Personalization",
        "analysis": """**Multi-Agent Personalization:** ARAG integrates four specialized LLM-based agents working together to understand user preferences, evaluate semantic alignment, summarize findings, and rank recommendations.

**Performance Gains:** Achieves up to 42.1% improvement in NDCG@5 and 35.5% in Hit@5 over standard RAG baselines, highlighting the effectiveness of agentic reasoning.""",
    },
    6: {
        "title": "VAT-KG: Multimodal Knowledge Graphs",
        "analysis": """**Beyond Text:** VAT-KG is the first concept-centric knowledge graph covering visual, audio, and text information. Each triplet is linked to multimodal data and enriched with detailed concept descriptions.

**Enabling Multimodal RAG:** The system enables retrieval and reasoning across different modalities, supporting MLLMs in tasks that require understanding of images, sounds, and text together.""",
    },
    5: {
        "title": "IRanker: Ranking Foundation Model",
        "analysis": """**Universal Ranking:** IRanker unifies diverse ranking tasks using a single model through reinforcement learning and iterative decoding. It decomposes complex ranking into step-by-step candidate elimination.

**Broad Impact:** A single IRanker-3B achieves state-of-the-art results across recommendation, routing, and passage ranking, even surpassing larger models on certain datasets.""",
    },
    4: {
        "title": "Text-to-LoRA Implementation Details",
        "analysis": """**Instant Adaptation:** T2L can adapt LLMs in a single forward pass based on natural language task descriptions. After training on just 9 LoRA adapters, it matches task-specific performance and generalizes to unseen tasks.

**Democratization:** This approach provides language-based adaptation with minimal compute requirements, making model specialization accessible to a broader audience.""",
    },
    3: {
        "title": "Arch-Router: Human-Aligned LLM Routing",
        "analysis": """**Preference-Aligned Routing:** Arch-Router guides model selection by matching queries to user-defined domains or action types, offering a practical mechanism to encode preferences in routing decisions.

**The Innovation:** This 1.5B model outperforms top proprietary models in matching queries with human preferences, making routing decisions more transparent and flexible.""",
    },
    2: {
        "title": "Quantization-Aware Training at Jina",
        "analysis": """**Lossless Compression:** Jina demonstrates how quantization-aware training can make embeddings 64x smaller while maintaining performance. This is crucial for deploying AI at scale with limited resources.

**Technical Excellence:** The approach combines output QAT with careful scaling strategies, achieving the best of both worlds: smaller embeddings without sacrificing quality.""",
    },
    1: {
        "title": "Advanced Embedding Research",
        "analysis": """**Pushing Embedding Boundaries:** Research in embedding technology continues to advance, focusing on efficiency, quality, and practical deployment considerations.

**Key Innovations:** From quantization techniques to novel training approaches, the field is making embeddings more accessible and efficient for real-world applications.""",
    },
}


def update_website_data():
    """Update website data.json with new Feynman summaries"""

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all articles with their content
    query = """
    SELECT a.id, a.title, a.url, a.processed_date, a.status,
           c.original_content
    FROM articles a
    LEFT JOIN content c ON a.id = c.article_id
    WHERE a.status = 'completed'
    ORDER BY a.processed_date DESC
    """

    cursor.execute(query)
    articles = cursor.fetchall()

    # Process articles
    processed_articles = []
    for article in articles:
        article_id, title, url, processed_date, status, original_content = article

        # Linked articles - placeholder for now since table doesn't exist
        linked_articles = []

        # Use Feynman summary if available, otherwise use a default message
        if article_id in FEYNMAN_SUMMARIES:
            analysis = FEYNMAN_SUMMARIES[article_id]["analysis"]
            # Override title if provided in Feynman summaries
            if "title" in FEYNMAN_SUMMARIES[article_id]:
                title = FEYNMAN_SUMMARIES[article_id]["title"]
        else:
            analysis = (
                "**Analysis pending:** This article is awaiting Feynman-style analysis."
            )

        processed_articles.append(
            {
                "id": article_id,
                "title": title,
                "url": url,
                "processed_date": processed_date,
                "status": status,
                "analysis": analysis,
                "ai_provider": "claude",
                "linked_articles": linked_articles,
            }
        )

    # Get processing status
    status_query = """
    SELECT
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
        COUNT(*) as total
    FROM articles
    """
    cursor.execute(status_query)
    stats = cursor.fetchone()

    # Determine system status
    if stats[1] == 0:
        system_status = "success"
    elif stats[1] < stats[0]:
        system_status = "partial"
    else:
        system_status = "failed"

    # Create the output data structure
    output_data = {
        "generated_at": datetime.utcnow().isoformat() + "+00:00",
        "total_articles": len(processed_articles),
        "articles": processed_articles,
        "metadata": {
            "date_range": {
                "earliest": min(a["processed_date"] for a in processed_articles)
                if processed_articles
                else None,
                "latest": max(a["processed_date"] for a in processed_articles)
                if processed_articles
                else None,
            },
            "sources": list({a["url"].split("/")[2] for a in processed_articles}),
            "ai_provider": "claude",
        },
        "processing_status": {
            "successful_articles": stats[0],
            "failed_articles": stats[1],
            "total_processed": stats[2],
            "system_status": system_status,
            "recent_errors_by_date": {},
        },
    }

    # Write to file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Updated {OUTPUT_PATH} with {len(processed_articles)} articles")
    print(f"System status: {system_status}")

    conn.close()


if __name__ == "__main__":
    update_website_data()
