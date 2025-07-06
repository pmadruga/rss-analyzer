#!/usr/bin/env python3
"""
Claude Code Non-Interactive Integration Script

This script provides non-interactive access to Claude Code for automated
analysis and code assistance within the RSS analyzer workflow.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ClaudeCodeClient:
    """Non-interactive Claude Code client for automation"""

    def __init__(self, timeout: int = 300):
        """
        Initialize Claude Code client

        Args:
            timeout: Command timeout in seconds
        """
        self.timeout = timeout
        self.claude_command = "claude"

    def is_available(self) -> bool:
        """Check if Claude Code is available on the system"""
        try:
            result = subprocess.run(
                [self.claude_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def analyze_code(
        self, prompt: str, file_paths: list[str] | None = None
    ) -> str | None:
        """
        Analyze code using Claude Code in non-interactive mode

        Args:
            prompt: Analysis prompt
            file_paths: Optional list of file paths to include in analysis

        Returns:
            Claude's analysis response or None if failed
        """
        try:
            if file_paths:
                # Create a temporary file with file contents
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as tmp:
                    tmp.write("Files for analysis:\n\n")
                    for file_path in file_paths:
                        if os.path.exists(file_path):
                            tmp.write(f"=== {file_path} ===\n")
                            with open(
                                file_path, encoding="utf-8", errors="ignore"
                            ) as f:
                                tmp.write(f.read())
                            tmp.write("\n\n")

                    tmp_path = tmp.name

                try:
                    # Use cat to pipe file contents to Claude
                    cmd = f"cat {tmp_path} | {self.claude_command} -p '{prompt}'"
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_path)
            else:
                # Direct prompt without files
                result = subprocess.run(
                    [self.claude_command, "-p", prompt],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"Claude Code error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Claude Code command timed out after {self.timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Claude Code execution failed: {e}")
            return None

    def review_code_changes(self, changed_files: list[str]) -> str | None:
        """Review code changes before commit"""
        prompt = """
        Please review these code changes for:
        1. Potential bugs or issues
        2. Code quality and best practices
        3. Security vulnerabilities
        4. Performance considerations
        5. Suggestions for improvement
        
        Provide a concise summary of findings.
        """
        return self.analyze_code(prompt, changed_files)

    def analyze_errors(self, error_log: str) -> str | None:
        """Analyze error logs and suggest fixes"""
        prompt = """
        Analyze this error log and provide:
        1. Root cause analysis
        2. Specific steps to fix the issues
        3. Prevention strategies
        4. Code examples if applicable
        
        Focus on actionable solutions.
        """

        # Write error log to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tmp:
            tmp.write(error_log)
            tmp_path = tmp.name

        try:
            return self.analyze_code(prompt, [tmp_path])
        finally:
            os.unlink(tmp_path)

    def optimize_code(self, file_paths: list[str]) -> str | None:
        """Get optimization suggestions for code files"""
        prompt = """
        Analyze these code files for optimization opportunities:
        1. Performance improvements
        2. Code efficiency
        3. Memory usage optimization
        4. Better algorithms or data structures
        5. Refactoring suggestions
        
        Provide specific, actionable recommendations.
        """
        return self.analyze_code(prompt, file_paths)

    def generate_documentation(self, file_paths: list[str]) -> str | None:
        """Generate documentation for code files"""
        prompt = """
        Generate comprehensive documentation for these code files:
        1. Module/class/function descriptions
        2. Usage examples
        3. Parameter explanations
        4. Return value documentation
        5. Dependencies and requirements
        
        Format as markdown documentation.
        """
        return self.analyze_code(prompt, file_paths)


class RSSAnalyzerClaudeIntegration:
    """Integration between RSS Analyzer and Claude Code"""

    def __init__(self):
        self.claude = ClaudeCodeClient()
        self.project_root = Path(__file__).parent.parent

    def pre_commit_review(self) -> dict[str, Any]:
        """Review changes before commit"""
        try:
            # Get changed files
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                return {"success": False, "error": "Failed to get changed files"}

            changed_files = [
                str(self.project_root / f.strip())
                for f in result.stdout.split("\n")
                if f.strip()
                and f.strip().endswith((".py", ".js", ".html", ".css", ".md"))
            ]

            if not changed_files:
                return {"success": True, "message": "No relevant files changed"}

            # Review changes with Claude
            review = self.claude.review_code_changes(changed_files)

            return {
                "success": True,
                "review": review,
                "files_reviewed": len(changed_files),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_api_errors(self, error_log_path: str) -> dict[str, Any]:
        """Analyze API errors from log files"""
        try:
            if not os.path.exists(error_log_path):
                return {"success": False, "error": "Log file not found"}

            with open(error_log_path) as f:
                error_log = f.read()

            analysis = self.claude.analyze_errors(error_log)

            return {"success": True, "analysis": analysis, "log_file": error_log_path}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def optimize_performance(self) -> dict[str, Any]:
        """Get performance optimization suggestions"""
        try:
            # Key files for optimization
            key_files = [
                "src/main.py",
                "src/rss_parser.py",
                "src/scraper.py",
                "src/database.py",
                "src/mistral_client.py",
            ]

            existing_files = [
                str(self.project_root / f)
                for f in key_files
                if (self.project_root / f).exists()
            ]

            if not existing_files:
                return {"success": False, "error": "No key files found"}

            suggestions = self.claude.optimize_code(existing_files)

            return {
                "success": True,
                "suggestions": suggestions,
                "files_analyzed": len(existing_files),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    """Main entry point for Claude Code integration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Claude Code integration for RSS Analyzer"
    )
    parser.add_argument(
        "command", choices=["check", "review", "analyze-errors", "optimize", "help"]
    )
    parser.add_argument("--log-file", help="Path to error log file for analysis")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    integration = RSSAnalyzerClaudeIntegration()

    if not integration.claude.is_available():
        print("❌ Claude Code is not available. Please install it first.")
        print("Visit: https://claude.ai/code")
        print()
        print("📝 Demo Mode: Showing what Claude Code integration can do:")
        print("=" * 50)

        if args.command == "help":
            # Still show help in demo mode
            pass  # Continue to help section
        else:
            print("This integration would:")
            print()

            if args.command == "review":
                print("🔍 Review code changes for:")
                print("  • Potential bugs and security issues")
                print("  • Code quality and best practices")
                print("  • Performance optimizations")
                print("  • Suggestions for improvement")

            elif args.command == "analyze-errors":
                print("🐛 Analyze error logs to provide:")
                print("  • Root cause analysis")
                print("  • Specific steps to fix issues")
                print("  • Prevention strategies")
                print("  • Code examples for solutions")

            elif args.command == "optimize":
                print("⚡ Analyze code for:")
                print("  • Performance improvements")
                print("  • Memory usage optimization")
                print("  • Better algorithms and data structures")
                print("  • Refactoring opportunities")

            print()
            print("To use these features, install Claude Code first.")
            return 0

    if args.command == "check":
        print("✅ Claude Code is available and ready for use")
        return 0

    elif args.command == "review":
        print("🔍 Reviewing code changes...")
        result = integration.pre_commit_review()

        if result["success"]:
            if "review" in result:
                print("📋 Code Review Results:")
                print("=" * 50)
                print(result["review"])
            else:
                print(result["message"])
        else:
            print(f"❌ Review failed: {result['error']}")
            return 1

    elif args.command == "analyze-errors":
        if not args.log_file:
            print("❌ Please provide --log-file argument")
            return 1

        print(f"🔍 Analyzing errors in {args.log_file}...")
        result = integration.analyze_api_errors(args.log_file)

        if result["success"]:
            print("🔍 Error Analysis Results:")
            print("=" * 50)
            print(result["analysis"])
        else:
            print(f"❌ Analysis failed: {result['error']}")
            return 1

    elif args.command == "optimize":
        print("⚡ Analyzing code for optimization opportunities...")
        result = integration.optimize_performance()

        if result["success"]:
            print("⚡ Optimization Suggestions:")
            print("=" * 50)
            print(result["suggestions"])
        else:
            print(f"❌ Optimization analysis failed: {result['error']}")
            return 1

    elif args.command == "help":
        print("Claude Code Integration for RSS Analyzer")
        print("=" * 50)
        print("Available commands:")
        print("  check           - Check if Claude Code is available")
        print("  review          - Review current code changes")
        print("  analyze-errors  - Analyze error logs (requires --log-file)")
        print("  optimize        - Get performance optimization suggestions")
        print("  help            - Show this help message")
        print()
        print("Examples:")
        print("  python tools/claude_code_integration.py check")
        print("  python tools/claude_code_integration.py review")
        print(
            "  python tools/claude_code_integration.py analyze-errors --log-file logs/analyzer.log"
        )
        print("  python tools/claude_code_integration.py optimize")

    # Save output if requested
    if args.output and "result" in locals():
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"📄 Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
