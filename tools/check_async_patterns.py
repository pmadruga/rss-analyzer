#!/usr/bin/env python3
"""
Check for blocking I/O in async functions.

This script performs static analysis to detect common async anti-patterns
that can block the event loop and degrade performance.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict


BLOCKING_PATTERNS = {
    'requests.get': 'Use aiohttp.ClientSession.get() instead',
    'requests.post': 'Use aiohttp.ClientSession.post() instead',
    'requests.put': 'Use aiohttp.ClientSession.put() instead',
    'requests.delete': 'Use aiohttp.ClientSession.delete() instead',
    'requests.patch': 'Use aiohttp.ClientSession.patch() instead',
    'requests.head': 'Use aiohttp.ClientSession.head() instead',
    'requests.options': 'Use aiohttp.ClientSession.options() instead',
    'time.sleep': 'Use asyncio.sleep() instead',
    'open(': 'Use aiofiles.open() for async file I/O',
    'sqlite3.connect': 'Use async database client with connection pool',
    'psycopg2.connect': 'Use asyncpg instead',
    'MySQLdb.connect': 'Use aiomysql instead',
    'urllib.request': 'Use aiohttp instead of urllib',
    'http.client': 'Use aiohttp instead of http.client',
}


class AsyncBlockingChecker(ast.NodeVisitor):
    """AST visitor to detect blocking I/O in async functions."""

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Tuple[int, str, str, str]] = []
        self.in_async_function = False
        self.current_function = None

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        previous_state = self.in_async_function
        previous_function = self.current_function

        self.in_async_function = True
        self.current_function = node.name

        self.generic_visit(node)

        self.in_async_function = previous_state
        self.current_function = previous_function

    def visit_Call(self, node):
        """Visit function call node."""
        if self.in_async_function:
            call_name = self._get_call_name(node)

            for pattern, suggestion in BLOCKING_PATTERNS.items():
                if pattern in call_name:
                    self.issues.append((
                        node.lineno,
                        self.current_function,
                        f"Blocking call '{call_name}' in async function",
                        suggestion
                    ))

        self.generic_visit(node)

    def _get_call_name(self, node) -> str:
        """Extract full call name from AST node."""
        if isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func

            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.insert(0, current.id)

            return '.'.join(parts)

        elif isinstance(node.func, ast.Name):
            return node.func.id

        return ''


def check_file(filepath: Path) -> List[Tuple[int, str, str, str]]:
    """
    Check a single Python file for blocking I/O in async functions.

    Args:
        filepath: Path to Python file to check

    Returns:
        List of issues found (line_number, function_name, message, suggestion)
    """
    try:
        with open(filepath, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        checker = AsyncBlockingChecker(str(filepath))
        checker.visit(tree)
        return checker.issues

    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error checking {filepath}: {e}", file=sys.stderr)
        return []


def format_issue(
    filepath: Path,
    lineno: int,
    function_name: str,
    message: str,
    suggestion: str
) -> str:
    """Format issue for display."""
    return f"""
{filepath}:{lineno} in {function_name}()
  ❌ {message}
  💡 {suggestion}
"""


def main():
    """Main entry point."""
    src_path = Path("src")

    if not src_path.exists():
        print(f"Error: Source directory '{src_path}' not found", file=sys.stderr)
        sys.exit(1)

    all_issues: List[Tuple[Path, int, str, str, str]] = []

    # Check all Python files in src/
    for py_file in src_path.rglob("*.py"):
        issues = check_file(py_file)
        if issues:
            all_issues.extend([(py_file, *issue) for issue in issues])

    # Report results
    if all_issues:
        print("=" * 80)
        print("❌ BLOCKING I/O DETECTED IN ASYNC FUNCTIONS")
        print("=" * 80)

        issue_count = len(all_issues)
        file_count = len(set(filepath for filepath, *_ in all_issues))

        print(f"\nFound {issue_count} issue(s) in {file_count} file(s):\n")

        for filepath, lineno, function_name, message, suggestion in all_issues:
            print(format_issue(filepath, lineno, function_name, message, suggestion))

        print("=" * 80)
        print("SUMMARY:")
        print(f"  Total issues: {issue_count}")
        print(f"  Files affected: {file_count}")
        print("=" * 80)
        print("\nℹ️  These issues can severely impact async performance by blocking")
        print("   the event loop and preventing concurrent execution.")
        print("\n📚 Resources:")
        print("   - Async Best Practices: docs/ASYNC_GUIDE.md")
        print("   - Review Checklist: docs/code-review/REVIEW_CHECKLIST.md")
        print("=" * 80)

        sys.exit(1)
    else:
        print("✅ No blocking I/O detected in async functions")
        sys.exit(0)


if __name__ == "__main__":
    main()
