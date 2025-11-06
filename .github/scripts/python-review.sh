#!/bin/bash
set -e

# Python Best Practices Review Script for Code Review Swarm
# Usage: ./python-review.sh <PR_NUMBER> <CHANGED_FILES>

PR_NUM=$1
CHANGED_FILES=$2

echo "🐍 Running Python best practices review for PR #$PR_NUM..."

SUGGESTIONS=""
SUGGESTION_COUNT=0

# Split comma-separated files
IFS=',' read -ra FILES <<< "$CHANGED_FILES"

for file in "${FILES[@]}"; do
    if [[ "$file" == *.py ]] && [ -f "$file" ]; then
        echo "  Analyzing: $file"

        # Check 1: Type hints
        if ! grep -q " -> " "$file"; then
            SUGGESTIONS="$SUGGESTIONS\n- 📝 **Type Hints**: Consider adding type hints to functions in \`$file\`"
            ((SUGGESTION_COUNT++))
        fi

        # Check 2: Pathlib usage
        if grep -q "os\.path\." "$file"; then
            SUGGESTIONS="$SUGGESTIONS\n- 🗂️ **Pathlib**: Consider using \`pathlib.Path\` instead of \`os.path\` in \`$file\`"
            ((SUGGESTION_COUNT++))
        fi

        # Check 3: F-strings
        if grep -q "\.format\(" "$file" || grep -q "%" "$file"; then
            SUGGESTIONS="$SUGGESTIONS\n- 💬 **F-strings**: Consider using f-strings instead of \`.format()\` or \`%\` in \`$file\`"
            ((SUGGESTION_COUNT++))
        fi

        # Check 4: Context managers
        if grep -q "open(" "$file" && ! grep -q "with open" "$file"; then
            SUGGESTIONS="$SUGGESTIONS\n- 🔓 **Context Manager**: Use \`with\` statement for file operations in \`$file\`"
            ((SUGGESTION_COUNT++))
        fi

        # Check 5: List comprehensions
        if grep -E "for .+ in .+:" "$file" | grep -q "\.append"; then
            SUGGESTIONS="$SUGGESTIONS\n- 🔄 **Comprehensions**: Consider using list comprehensions in \`$file\`"
            ((SUGGESTION_COUNT++))
        fi

        # Check 6: Dataclasses
        if grep -q "class.*:" "$file" && grep -q "def __init__" "$file"; then
            if ! grep -q "@dataclass" "$file"; then
                SUGGESTIONS="$SUGGESTIONS\n- 📦 **Dataclasses**: Consider using \`@dataclass\` for simple data classes in \`$file\`"
                ((SUGGESTION_COUNT++))
            fi
        fi

        # Check 7: Named tuples
        if grep -E "return .+,.+" "$file" && ! grep -q "NamedTuple\|dataclass" "$file"; then
            SUGGESTIONS="$SUGGESTIONS\n- 🏷️ **Named Returns**: Consider using \`NamedTuple\` for multiple return values in \`$file\`"
            ((SUGGESTION_COUNT++))
        fi

        # Check 8: Enum usage
        if grep -E "^[A-Z_]+ = ['\"]" "$file" | grep -c "" | grep -q "[3-9]"; then
            if ! grep -q "from enum import" "$file"; then
                SUGGESTIONS="$SUGGESTIONS\n- 🔢 **Enums**: Consider using \`Enum\` for related constants in \`$file\`"
                ((SUGGESTION_COUNT++))
            fi
        fi
    fi
done

# Generate review
if [ -n "$SUGGESTIONS" ]; then
    cat > review-output.md << EOF
## 🐍 Python Best Practices Review

**Status**: 💡 **Suggestions Available**
**Suggestions**: $SUGGESTION_COUNT improvement(s) recommended

---

### 💡 Suggestions for Improvement

$SUGGESTIONS

---

### 🎯 Modern Python Patterns

#### 1. Type Hints (PEP 484)
\`\`\`python
# Before
def process_data(data, limit):
    return [x for x in data if x > limit]

# After
def process_data(data: list[int], limit: int) -> list[int]:
    return [x for x in data if x > limit]
\`\`\`

#### 2. Pathlib (PEP 428)
\`\`\`python
# Before
import os
path = os.path.join('data', 'articles.db')
if os.path.exists(path):
    os.remove(path)

# After
from pathlib import Path
path = Path('data') / 'articles.db'
if path.exists():
    path.unlink()
\`\`\`

#### 3. F-strings (PEP 498)
\`\`\`python
# Before
message = "Found {} articles in {}".format(count, category)

# After
message = f"Found {count} articles in {category}"
\`\`\`

#### 4. Context Managers
\`\`\`python
# Before
file = open('data.txt', 'r')
data = file.read()
file.close()

# After
with open('data.txt', 'r') as file:
    data = file.read()
\`\`\`

#### 5. Dataclasses (PEP 557)
\`\`\`python
# Before
class Article:
    def __init__(self, title: str, url: str):
        self.title = title
        self.url = url

# After
from dataclasses import dataclass

@dataclass
class Article:
    title: str
    url: str
\`\`\`

#### 6. Named Tuples
\`\`\`python
# Before
def get_stats():
    return 42, 15, 0.95

total, errors, success_rate = get_stats()

# After
from typing import NamedTuple

class Stats(NamedTuple):
    total: int
    errors: int
    success_rate: float

def get_stats() -> Stats:
    return Stats(42, 15, 0.95)

stats = get_stats()
print(stats.total)  # Named access!
\`\`\`

#### 7. Enums (PEP 435)
\`\`\`python
# Before
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"

# After
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
\`\`\`

---

### 📚 Resources

- [Type Hints - PEP 484](https://www.python.org/dev/peps/pep-0484/)
- [Pathlib - PEP 428](https://www.python.org/dev/peps/pep-0428/)
- [F-strings - PEP 498](https://www.python.org/dev/peps/pep-0498/)
- [Dataclasses - PEP 557](https://www.python.org/dev/peps/pep-0557/)
- [Python Best Practices](https://docs.python-guide.org/)
- [Real Python Tutorials](https://realpython.com/)

---

### ⚡ Quick Refactoring Commands

\`\`\`bash
# Auto-upgrade to modern Python syntax
uv run pyupgrade --py311-plus **/*.py

# Sort and organize imports
uv run isort src/

# Format code
uv run ruff format .

# Type check
uv run mypy src/
\`\`\`

---

🤖 *Automated Python Best Practices Review by Code Review Swarm*
EOF
else
    cat > review-output.md << EOF
## 🐍 Python Best Practices Review

**Status**: ✅ **Excellent!**
**Suggestions**: No improvements needed

---

### ✨ Code Quality Highlights

Your code follows modern Python best practices:

- ✅ Proper type hints
- ✅ Modern pathlib usage
- ✅ F-string formatting
- ✅ Context managers for resources
- ✅ Clean, pythonic patterns

Keep up the great work!

---

🤖 *Automated Python Best Practices Review by Code Review Swarm*
EOF
fi

echo "✅ Python best practices review complete"
echo "📊 Suggestions: $SUGGESTION_COUNT"

exit 0
