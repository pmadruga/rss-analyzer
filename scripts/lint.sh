#!/bin/bash

# Python linting script using Ruff

echo "🚀 Running Python code quality checks..."
echo "======================================"

# Check if ruff is installed
if ! command -v ruff &> /dev/null; then
    echo "❌ Ruff not found. Installing..."
    pip install ruff
fi

echo ""
echo "🔍 Running Ruff linter..."
echo "-------------------------"
if ruff check .; then
    echo "✅ Linting passed!"
else
    echo "❌ Linting issues found. Run 'ruff check . --fix' to auto-fix some issues."
    LINT_FAILED=true
fi

echo ""
echo "📝 Checking code formatting..."
echo "-----------------------------"
if ruff format --check .; then
    echo "✅ Code formatting is correct!"
else
    echo "❌ Code formatting issues found. Run 'ruff format .' to fix."
    FORMAT_FAILED=true
fi

echo ""
echo "📊 Summary"
echo "=========="

if [ "$LINT_FAILED" = true ] || [ "$FORMAT_FAILED" = true ]; then
    echo "❌ Code quality checks failed"
    echo ""
    echo "🔧 To fix issues automatically:"
    echo "   ruff check . --fix    # Fix linting issues"
    echo "   ruff format .         # Fix formatting"
    echo ""
    exit 1
else
    echo "✅ All code quality checks passed!"
    exit 0
fi