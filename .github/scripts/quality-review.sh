#!/bin/bash
set -e

# Code Quality Review Script for Code Review Swarm
# Usage: ./quality-review.sh <PR_NUMBER> <CHANGED_FILES>

PR_NUM=$1
CHANGED_FILES=$2

echo "📋 Running code quality review for PR #$PR_NUM..."

# Split comma-separated files into array
IFS=',' read -ra FILES <<< "$CHANGED_FILES"

# Filter Python files
PY_FILES=""
for file in "${FILES[@]}"; do
    if [[ "$file" == *.py ]] && [ -f "$file" ]; then
        PY_FILES="$PY_FILES $file"
    fi
done

if [ -z "$PY_FILES" ]; then
    cat > review-output.md << EOF
## 📋 Code Quality Review

**Status**: ⏭️ **Skipped** - No Python files changed

---

🤖 *Automated Code Quality Review by Code Review Swarm*
EOF
    exit 0
fi

echo "  Analyzing Python files: $PY_FILES"

# Run pylint
echo "  Running pylint..."
PYLINT_OUTPUT=$(pylint $PY_FILES --score=y --output-format=text 2>&1 || true)
PYLINT_SCORE=$(echo "$PYLINT_OUTPUT" | grep "Your code has been rated" | grep -oP '\d+\.\d+(?=/10)' || echo "0.0")

# Run flake8
echo "  Running flake8..."
FLAKE8_OUTPUT=$(flake8 $PY_FILES --count --statistics --max-line-length=88 2>&1 || true)
FLAKE8_ERRORS=$(echo "$FLAKE8_OUTPUT" | tail -1 | grep -oP '^\d+' || echo "0")

# Run radon complexity
echo "  Running complexity analysis..."
COMPLEXITY_OUTPUT=$(radon cc $PY_FILES -a -s 2>&1 || true)
AVG_COMPLEXITY=$(echo "$COMPLEXITY_OUTPUT" | grep "Average complexity:" | grep -oP '\d+\.\d+' || echo "N/A")

# Determine status
STATUS="✅ Good"
EMOJI="🟢"

if (( $(echo "$PYLINT_SCORE < 7.0" | bc -l) )); then
    STATUS="⚠️ Needs Improvement"
    EMOJI="🟡"
fi

if [ "$FLAKE8_ERRORS" -gt 10 ]; then
    STATUS="⚠️ Needs Improvement"
    EMOJI="🟡"
fi

if (( $(echo "$PYLINT_SCORE < 5.0" | bc -l) )) || [ "$FLAKE8_ERRORS" -gt 50 ]; then
    STATUS="❌ Poor Quality"
    EMOJI="🔴"
fi

# Generate review
cat > review-output.md << EOF
## 📋 Code Quality Review Results

**Status**: $EMOJI **$STATUS**

---

### 📊 Quality Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Pylint Score** | $PYLINT_SCORE/10 | ≥ 8.0 | $([ $(echo "$PYLINT_SCORE >= 8.0" | bc -l) -eq 1 ] && echo "✅" || echo "⚠️") |
| **Flake8 Issues** | $FLAKE8_ERRORS | 0 | $([ "$FLAKE8_ERRORS" -eq 0 ] && echo "✅" || echo "⚠️") |
| **Avg Complexity** | $AVG_COMPLEXITY | < 10 | $([ "$AVG_COMPLEXITY" != "N/A" ] && ([ $(echo "$AVG_COMPLEXITY < 10" | bc -l) -eq 1 ] && echo "✅" || echo "⚠️") || echo "ℹ️") |

---

### 🔍 Detailed Analysis

#### Pylint Report
\`\`\`
$(echo "$PYLINT_OUTPUT" | head -50)
...
\`\`\`

#### Flake8 Issues
\`\`\`
$(echo "$FLAKE8_OUTPUT" | head -30)
\`\`\`

#### Complexity Analysis
\`\`\`
$COMPLEXITY_OUTPUT
\`\`\`

---

### 💡 Recommendations

1. **Code Quality Goals**:
   - Aim for Pylint score ≥ 8.0
   - Resolve all Flake8 issues
   - Keep cyclomatic complexity < 10 per function

2. **Common Improvements**:
   - Add docstrings to functions and classes
   - Follow PEP 8 style guidelines
   - Break complex functions into smaller units
   - Remove unused imports and variables

3. **Tools to Use**:
   \`\`\`bash
   # Auto-format code
   uv run ruff format .

   # Auto-fix issues
   uv run ruff check --fix .

   # Check types
   uv run mypy src/
   \`\`\`

---

### 📚 Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Pylint Documentation](https://pylint.readthedocs.io/)
- [Flake8 Rules](https://flake8.pycqa.org/en/latest/user/error-codes.html)
- [Code Complexity](https://radon.readthedocs.io/en/latest/intro.html)

---

🤖 *Automated Code Quality Review by Code Review Swarm*
EOF

echo "✅ Code quality review complete"
echo "📊 Pylint Score: $PYLINT_SCORE/10"
echo "📊 Flake8 Issues: $FLAKE8_ERRORS"
echo "📊 Avg Complexity: $AVG_COMPLEXITY"

exit 0
