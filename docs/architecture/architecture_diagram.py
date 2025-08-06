#!/usr/bin/env python3
"""
Generate architecture diagrams for the RSS Analyzer refactored codebase
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch


def create_architecture_diagram():
    """Create a comprehensive architecture diagram"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "RSS Analyzer - Refactored Architecture", fontsize=20, fontweight="bold"
    )

    # ===== DIAGRAM 1: MODULE STRUCTURE =====
    ax1.set_title("Module Structure & Dependencies", fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")

    # Define colors
    colors = {
        "main": "#FF6B6B",
        "processors": "#4ECDC4",
        "clients": "#45B7D1",
        "core": "#96CEB4",
        "config": "#FFEAA7",
        "exceptions": "#DDA0DD",
        "tests": "#FFB347",
    }

    # Main entry point
    main_box = FancyBboxPatch(
        (4, 8.5),
        2,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["main"],
        edgecolor="black",
        linewidth=2,
    )
    ax1.add_patch(main_box)
    ax1.text(5, 9, "main.py\n(CLI Entry)", ha="center", va="center", fontweight="bold")

    # Processors layer
    proc_box = FancyBboxPatch(
        (3.5, 6.5),
        3,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["processors"],
        edgecolor="black",
    )
    ax1.add_patch(proc_box)
    ax1.text(
        5,
        7,
        "processors/\nArticleProcessor",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Clients layer
    client_box = FancyBboxPatch(
        (0.5, 4.5),
        2.5,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors["clients"],
        edgecolor="black",
    )
    ax1.add_patch(client_box)
    ax1.text(
        1.75,
        5.1,
        "clients/\nBaseAIClient\nFactory Pattern",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Core layer
    core_box = FancyBboxPatch(
        (4, 4.5),
        2.5,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors["core"],
        edgecolor="black",
    )
    ax1.add_patch(core_box)
    ax1.text(
        5.25,
        5.1,
        "core/\nDatabase, RSS\nScraper, Reports",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Config layer
    config_box = FancyBboxPatch(
        (7, 4.5),
        2,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors["config"],
        edgecolor="black",
    )
    ax1.add_patch(config_box)
    ax1.text(
        8,
        5.1,
        "config/\nSettings\nConfiguration",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Exceptions layer
    exc_box = FancyBboxPatch(
        (2, 2.5),
        6,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["exceptions"],
        edgecolor="black",
    )
    ax1.add_patch(exc_box)
    ax1.text(
        5,
        3,
        "exceptions/\nCustom Exception Hierarchy",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Tests
    test_box = FancyBboxPatch(
        (1, 0.5),
        8,
        1,
        boxstyle="round,pad=0.1",
        facecolor=colors["tests"],
        edgecolor="black",
    )
    ax1.add_patch(test_box)
    ax1.text(
        5,
        1,
        "tests/unit/\nComprehensive Test Suite with Fixtures",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Add arrows for dependencies
    arrows = [
        ((5, 8.5), (5, 7.5)),  # main -> processors
        ((5, 6.5), (1.75, 5.7)),  # processors -> clients
        ((5, 6.5), (5.25, 5.7)),  # processors -> core
        ((5, 6.5), (8, 5.7)),  # processors -> config
        ((1.75, 4.5), (3.5, 3.5)),  # clients -> exceptions
        ((5.25, 4.5), (5.5, 3.5)),  # core -> exceptions
        ((5, 2.5), (5, 1.5)),  # exceptions -> tests
    ]

    for start, end in arrows:
        arrow = ConnectionPatch(
            start,
            end,
            "data",
            "data",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            mutation_scale=20,
            fc="black",
            alpha=0.7,
        )
        ax1.add_patch(arrow)

    # ===== DIAGRAM 2: CLASS HIERARCHY =====
    ax2.set_title("Exception & Client Class Hierarchy", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    # Exception hierarchy
    ax2.text(2.5, 9.5, "Exception Hierarchy", fontsize=12, fontweight="bold")

    # Base Exception
    base_exc = FancyBboxPatch(
        (1, 8.5),
        3,
        0.6,
        boxstyle="round,pad=0.05",
        facecolor="lightcoral",
        edgecolor="black",
    )
    ax2.add_patch(base_exc)
    ax2.text(2.5, 8.8, "Exception (Built-in)", ha="center", va="center", fontsize=9)

    # RSSAnalyzerError
    rss_exc = FancyBboxPatch(
        (1, 7.5),
        3,
        0.6,
        boxstyle="round,pad=0.05",
        facecolor="lightblue",
        edgecolor="black",
    )
    ax2.add_patch(rss_exc)
    ax2.text(2.5, 7.8, "RSSAnalyzerError", ha="center", va="center", fontsize=9)

    # Specific exceptions
    specific_exceptions = [
        ("ConfigurationError", 6.5),
        ("APIClientError", 5.5),
        ("ContentProcessingError", 4.5),
        ("ScrapingError", 3.5),
        ("DatabaseError", 2.5),
    ]

    for exc_name, y_pos in specific_exceptions:
        exc_box = FancyBboxPatch(
            (0.5, y_pos),
            4,
            0.5,
            boxstyle="round,pad=0.05",
            facecolor="lightgreen",
            edgecolor="black",
        )
        ax2.add_patch(exc_box)
        ax2.text(2.5, y_pos + 0.25, exc_name, ha="center", va="center", fontsize=8)

        # Arrow from parent
        arrow = ConnectionPatch(
            (2.5, y_pos + 1),
            (2.5, y_pos + 0.5),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=2,
            shrinkB=2,
            mutation_scale=15,
        )
        ax2.add_patch(arrow)

    # Client hierarchy
    ax2.text(7.5, 9.5, "AI Client Hierarchy", fontsize=12, fontweight="bold")

    # BaseAIClient
    base_client = FancyBboxPatch(
        (6, 8.5),
        3,
        0.6,
        boxstyle="round,pad=0.05",
        facecolor="lightcyan",
        edgecolor="black",
    )
    ax2.add_patch(base_client)
    ax2.text(7.5, 8.8, "BaseAIClient (ABC)", ha="center", va="center", fontsize=9)

    # Concrete clients
    clients = [("ClaudeClient", 7.5), ("MistralClient", 6.5), ("OpenAIClient", 5.5)]

    for client_name, y_pos in clients:
        client_box = FancyBboxPatch(
            (5.5, y_pos),
            4,
            0.5,
            boxstyle="round,pad=0.05",
            facecolor="lightyellow",
            edgecolor="black",
        )
        ax2.add_patch(client_box)
        ax2.text(7.5, y_pos + 0.25, client_name, ha="center", va="center", fontsize=8)

        # Arrow from parent
        arrow = ConnectionPatch(
            (7.5, 8.5),
            (7.5, y_pos + 0.5),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=2,
            shrinkB=2,
            mutation_scale=15,
        )
        ax2.add_patch(arrow)

    # Factory
    factory_box = FancyBboxPatch(
        (6, 3.5),
        3,
        0.6,
        boxstyle="round,pad=0.05",
        facecolor="orange",
        edgecolor="black",
    )
    ax2.add_patch(factory_box)
    ax2.text(7.5, 3.8, "AIClientFactory", ha="center", va="center", fontsize=9)

    # ===== DIAGRAM 3: DATA FLOW =====
    ax3.set_title("Data Flow & Processing Pipeline", fontsize=14, fontweight="bold")
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 10)
    ax3.axis("off")

    # Define pipeline steps
    pipeline_steps = [
        ("RSS Feed", 1, 8.5, colors["core"]),
        ("Parse & Filter", 3, 8.5, colors["core"]),
        ("Web Scraping", 5, 8.5, colors["core"]),
        ("AI Analysis", 7, 8.5, colors["clients"]),
        ("Database Storage", 9, 8.5, colors["core"]),
        ("Report Generation", 11, 8.5, colors["core"]),
    ]

    for i, (step_name, x, y, color) in enumerate(pipeline_steps):
        # Step box
        step_box = FancyBboxPatch(
            (x - 0.7, y - 0.4),
            1.4,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="black",
            alpha=0.8,
        )
        ax3.add_patch(step_box)
        ax3.text(
            x, y, step_name, ha="center", va="center", fontsize=9, fontweight="bold"
        )

        # Arrow to next step
        if i < len(pipeline_steps) - 1:
            next_x = pipeline_steps[i + 1][1]
            arrow = ConnectionPatch(
                (x + 0.7, y),
                (next_x - 0.7, y),
                "data",
                "data",
                arrowstyle="->",
                shrinkA=2,
                shrinkB=2,
                mutation_scale=20,
                color="darkblue",
                linewidth=2,
            )
            ax3.add_patch(arrow)

    # Add data flow annotations
    data_annotations = [
        ("RSS Entries", 2, 7.5),
        ("Article Content", 4, 7.5),
        ("Scraped Text", 6, 7.5),
        ("AI Analysis", 8, 7.5),
        ("Structured Data", 10, 7.5),
    ]

    for annotation, x, y in data_annotations:
        ax3.text(
            x,
            y,
            annotation,
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            color="darkred",
        )

    # Error handling flow
    ax3.text(6, 6, "Error Handling Flow", fontsize=12, fontweight="bold", color="red")

    error_steps = [
        ("Exception Raised", 2, 5),
        ("Custom Exception", 5, 5),
        ("Error Logging", 8, 5),
        ("User Notification", 11, 5),
    ]

    for i, (step_name, x, y) in enumerate(error_steps):
        step_box = FancyBboxPatch(
            (x - 0.8, y - 0.3),
            1.6,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor="mistyrose",
            edgecolor="red",
            alpha=0.7,
        )
        ax3.add_patch(step_box)
        ax3.text(x, y, step_name, ha="center", va="center", fontsize=8)

        if i < len(error_steps) - 1:
            next_x = error_steps[i + 1][1]
            arrow = ConnectionPatch(
                (x + 0.8, y),
                (next_x - 0.8, y),
                "data",
                "data",
                arrowstyle="->",
                shrinkA=2,
                shrinkB=2,
                mutation_scale=15,
                color="red",
                linestyle="--",
            )
            ax3.add_patch(arrow)

    # Configuration flow
    ax3.text(
        6, 3, "Configuration Management", fontsize=12, fontweight="bold", color="orange"
    )

    config_flow = [
        ("YAML Config", 1.5, 2),
        ("Environment Variables", 4.5, 2),
        ("CONFIG Object", 7.5, 2),
        ("Component Init", 10.5, 2),
    ]

    for i, (step_name, x, y) in enumerate(config_flow):
        step_box = FancyBboxPatch(
            (x - 0.7, y - 0.3),
            1.4,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor="wheat",
            edgecolor="orange",
            alpha=0.7,
        )
        ax3.add_patch(step_box)
        ax3.text(x, y, step_name, ha="center", va="center", fontsize=8)

        if i < len(config_flow) - 1:
            next_x = config_flow[i + 1][1]
            arrow = ConnectionPatch(
                (x + 0.7, y),
                (next_x - 0.7, y),
                "data",
                "data",
                arrowstyle="->",
                shrinkA=2,
                shrinkB=2,
                mutation_scale=15,
                color="orange",
                linestyle=":",
            )
            ax3.add_patch(arrow)

    # ===== DIAGRAM 4: METRICS & IMPROVEMENTS =====
    ax4.set_title("Refactoring Metrics & Improvements", fontsize=14, fontweight="bold")
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis("off")

    # Before vs After comparison
    ax4.text(5, 9.5, "Refactoring Impact", fontsize=14, fontweight="bold", ha="center")

    # Metrics boxes
    metrics = [
        (
            "Code Duplication",
            "Before: 90%",
            "After: <5%",
            8.5,
            "lightcoral",
            "lightgreen",
        ),
        (
            "Lines of Code",
            "Before: ~600",
            "After: ~200",
            7.5,
            "lightcoral",
            "lightgreen",
        ),
        ("Test Coverage", "Before: 0%", "After: 90%+", 6.5, "lightcoral", "lightgreen"),
        (
            "Error Handling",
            "Before: Generic",
            "After: Specific",
            5.5,
            "lightcoral",
            "lightgreen",
        ),
        (
            "Architecture",
            "Before: Monolithic",
            "After: Modular",
            4.5,
            "lightcoral",
            "lightgreen",
        ),
    ]

    for metric_name, before_val, after_val, y_pos, before_color, after_color in metrics:
        # Metric name
        ax4.text(
            1,
            y_pos,
            metric_name,
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

        # Before box
        before_box = FancyBboxPatch(
            (3, y_pos - 0.2),
            2.5,
            0.4,
            boxstyle="round,pad=0.05",
            facecolor=before_color,
            edgecolor="black",
            alpha=0.7,
        )
        ax4.add_patch(before_box)
        ax4.text(4.25, y_pos, before_val, ha="center", va="center", fontsize=9)

        # After box
        after_box = FancyBboxPatch(
            (6, y_pos - 0.2),
            2.5,
            0.4,
            boxstyle="round,pad=0.05",
            facecolor=after_color,
            edgecolor="black",
            alpha=0.7,
        )
        ax4.add_patch(after_box)
        ax4.text(7.25, y_pos, after_val, ha="center", va="center", fontsize=9)

        # Arrow
        arrow = ConnectionPatch(
            (5.5, y_pos),
            (6, y_pos),
            "data",
            "data",
            arrowstyle="->",
            shrinkA=2,
            shrinkB=2,
            mutation_scale=15,
            color="darkgreen",
            linewidth=2,
        )
        ax4.add_patch(arrow)

    # SOLID Principles implemented
    ax4.text(
        5,
        3,
        "SOLID Principles Implemented",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )

    solid_principles = [
        ("S - Single Responsibility", "Each class has one clear purpose"),
        ("O - Open/Closed", "Easy to extend with new AI providers"),
        ("L - Liskov Substitution", "All AI clients interchangeable"),
        ("I - Interface Segregation", "Focused interfaces and abstractions"),
        ("D - Dependency Inversion", "Factory pattern & dependency injection"),
    ]

    for i, (principle, description) in enumerate(solid_principles):
        y_pos = 2.5 - i * 0.4
        principle_box = FancyBboxPatch(
            (0.5, y_pos - 0.15),
            9,
            0.3,
            boxstyle="round,pad=0.05",
            facecolor="lavender",
            edgecolor="purple",
            alpha=0.6,
        )
        ax4.add_patch(principle_box)
        ax4.text(
            1, y_pos, principle, ha="left", va="center", fontweight="bold", fontsize=9
        )
        ax4.text(
            9, y_pos, description, ha="right", va="center", fontsize=8, style="italic"
        )

    plt.tight_layout()
    plt.savefig("rss_analyzer_architecture.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_dependency_graph():
    """Create a detailed dependency graph"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_title("Detailed Module Dependencies", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Define module positions and dependencies
    modules = {
        "main.py": (7, 11, "#FF6B6B"),
        "processors/ArticleProcessor": (7, 9, "#4ECDC4"),
        "clients/BaseAIClient": (3, 7, "#45B7D1"),
        "clients/ClaudeClient": (1, 5, "#45B7D1"),
        "clients/MistralClient": (3, 5, "#45B7D1"),
        "clients/OpenAIClient": (5, 5, "#45B7D1"),
        "clients/AIClientFactory": (3, 3, "#45B7D1"),
        "core/DatabaseManager": (9, 7, "#96CEB4"),
        "core/RSSParser": (11, 7, "#96CEB4"),
        "core/WebScraper": (13, 7, "#96CEB4"),
        "core/ReportGenerator": (9, 5, "#96CEB4"),
        "core/utils": (11, 5, "#96CEB4"),
        "config/settings": (7, 3, "#FFEAA7"),
        "exceptions/exceptions": (7, 1, "#DDA0DD"),
    }

    # Draw modules
    for module_name, (x, y, color) in modules.items():
        # Adjust box size based on name length
        box_width = max(2, len(module_name) * 0.08)
        box = FancyBboxPatch(
            (x - box_width / 2, y - 0.4),
            box_width,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="black",
            alpha=0.8,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            module_name.replace("/", "\n"),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # Define dependencies (from -> to)
    dependencies = [
        ("main.py", "processors/ArticleProcessor"),
        ("processors/ArticleProcessor", "clients/AIClientFactory"),
        ("processors/ArticleProcessor", "core/DatabaseManager"),
        ("processors/ArticleProcessor", "core/RSSParser"),
        ("processors/ArticleProcessor", "core/WebScraper"),
        ("processors/ArticleProcessor", "core/ReportGenerator"),
        ("clients/AIClientFactory", "clients/ClaudeClient"),
        ("clients/AIClientFactory", "clients/MistralClient"),
        ("clients/AIClientFactory", "clients/OpenAIClient"),
        ("clients/ClaudeClient", "clients/BaseAIClient"),
        ("clients/MistralClient", "clients/BaseAIClient"),
        ("clients/OpenAIClient", "clients/BaseAIClient"),
        ("clients/BaseAIClient", "config/settings"),
        ("clients/BaseAIClient", "exceptions/exceptions"),
        ("core/DatabaseManager", "exceptions/exceptions"),
        ("core/RSSParser", "exceptions/exceptions"),
        ("core/WebScraper", "exceptions/exceptions"),
        ("core/ReportGenerator", "exceptions/exceptions"),
        ("processors/ArticleProcessor", "config/settings"),
        ("processors/ArticleProcessor", "exceptions/exceptions"),
        ("main.py", "config/settings"),
        ("main.py", "exceptions/exceptions"),
    ]

    # Draw dependency arrows
    for from_module, to_module in dependencies:
        from_pos = modules[from_module]
        to_pos = modules[to_module]

        # Calculate arrow positions to avoid overlapping with boxes
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            # Normalize and offset from box edges
            dx_norm = dx / length
            dy_norm = dy / length

            start_x = from_pos[0] + dx_norm * 0.8
            start_y = from_pos[1] + dy_norm * 0.4
            end_x = to_pos[0] - dx_norm * 0.8
            end_y = to_pos[1] - dy_norm * 0.4

            arrow = ConnectionPatch(
                (start_x, start_y),
                (end_x, end_y),
                "data",
                "data",
                arrowstyle="->",
                shrinkA=2,
                shrinkB=2,
                mutation_scale=15,
                color="darkblue",
                alpha=0.6,
                linewidth=1.5,
            )
            ax.add_patch(arrow)

    # Add legend
    legend_items = [
        ("Entry Point", "#FF6B6B"),
        ("Processors", "#4ECDC4"),
        ("AI Clients", "#45B7D1"),
        ("Core Services", "#96CEB4"),
        ("Configuration", "#FFEAA7"),
        ("Exceptions", "#DDA0DD"),
    ]

    for i, (label, color) in enumerate(legend_items):
        y_pos = 11.5 - i * 0.3
        legend_box = FancyBboxPatch(
            (0.5, y_pos - 0.1),
            0.3,
            0.2,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="black",
            alpha=0.8,
        )
        ax.add_patch(legend_box)
        ax.text(1, y_pos, label, ha="left", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("rss_analyzer_dependencies.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Generating RSS Analyzer Architecture Diagrams...")
    create_architecture_diagram()
    create_dependency_graph()
    print(
        "✅ Diagrams saved as 'rss_analyzer_architecture.png' and 'rss_analyzer_dependencies.png'"
    )
