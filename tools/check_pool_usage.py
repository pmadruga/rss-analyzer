#!/usr/bin/env python3
"""
Check that database operations use connection pool.

This script verifies that all database operations use the connection pool
rather than creating direct connections.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


# Patterns that indicate direct connection creation (anti-patterns)
ANTI_PATTERNS = [
    (r'sqlite3\.connect\s*\(', 'Direct sqlite3 connection'),
    (r'psycopg2\.connect\s*\(', 'Direct psycopg2 connection'),
    (r'MySQLdb\.connect\s*\(', 'Direct MySQLdb connection'),
    (r'mysql\.connector\.connect\s*\(', 'Direct mysql.connector connection'),
    (r'pymongo\.MongoClient\s*\(', 'Direct MongoDB client (use motor)'),
]

# Files that are exempt from this check
EXEMPT_FILES = [
    'database.py',  # Connection pool implementation
    'migration',  # Migration scripts may need direct access
]


def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Check a single Python file for direct database connections.

    Args:
        filepath: Path to Python file to check

    Returns:
        List of issues found (line_number, pattern, message)
    """
    # Check if file is exempt
    for exempt in EXEMPT_FILES:
        if exempt in str(filepath):
            return []

    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        issues: List[Tuple[int, str, str]] = []

        for pattern, message in ANTI_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                matched_text = match.group(0)
                issues.append((line_num, matched_text, message))

        return issues

    except Exception as e:
        print(f"Error checking {filepath}: {e}", file=sys.stderr)
        return []


def get_context_lines(filepath: Path, lineno: int, context: int = 2) -> str:
    """
    Get surrounding lines for context.

    Args:
        filepath: Path to file
        lineno: Line number (1-indexed)
        context: Number of lines before/after to include

    Returns:
        Context lines as string
    """
    try:
        with open(filepath, encoding='utf-8') as f:
            lines = f.readlines()

        start = max(0, lineno - context - 1)
        end = min(len(lines), lineno + context)

        context_lines = []
        for i in range(start, end):
            prefix = "→" if i == lineno - 1 else " "
            context_lines.append(f"  {prefix} {i+1:4d} | {lines[i].rstrip()}")

        return "\n".join(context_lines)

    except Exception:
        return "  (context unavailable)"


def format_issue(
    filepath: Path,
    lineno: int,
    matched_text: str,
    message: str
) -> str:
    """Format issue for display."""
    context = get_context_lines(filepath, lineno)

    return f"""
{filepath}:{lineno}
  ❌ {message}
  Found: {matched_text}

{context}

  💡 Suggested fix:
  Replace with: async with db_manager.get_connection() as conn:
               This uses the connection pool for better performance.

  📊 Performance impact:
  Connection pooling provides 2.78x faster database operations.
"""


def main():
    """Main entry point."""
    src_path = Path("src")

    if not src_path.exists():
        print(f"Error: Source directory '{src_path}' not found", file=sys.stderr)
        sys.exit(1)

    all_issues: List[Tuple[Path, int, str, str]] = []

    # Check all Python files in src/
    for py_file in src_path.rglob("*.py"):
        issues = check_file(py_file)
        if issues:
            all_issues.extend([(py_file, *issue) for issue in issues])

    # Report results
    if all_issues:
        print("=" * 80)
        print("❌ DIRECT DATABASE CONNECTIONS DETECTED")
        print("=" * 80)

        issue_count = len(all_issues)
        file_count = len(set(filepath for filepath, *_ in all_issues))

        print(f"\nFound {issue_count} issue(s) in {file_count} file(s):")
        print("\nAll database operations should use the connection pool for:")
        print("  • Better performance (2.78x faster)")
        print("  • Connection reuse")
        print("  • Automatic cleanup")
        print("  • Resource management")

        for filepath, lineno, matched_text, message in all_issues:
            print(format_issue(filepath, lineno, matched_text, message))

        print("=" * 80)
        print("SUMMARY:")
        print(f"  Total issues: {issue_count}")
        print(f"  Files affected: {file_count}")
        print("=" * 80)
        print("\n📚 Resources:")
        print("   - Connection Pooling Guide: docs/CONNECTION_POOLING.md")
        print("   - Optimization Results: docs/OPTIMIZATION_RESULTS.md")
        print("   - Review Checklist: docs/code-review/REVIEW_CHECKLIST.md")
        print("=" * 80)

        sys.exit(1)
    else:
        print("✅ All database operations use connection pool")
        sys.exit(0)


if __name__ == "__main__":
    main()
