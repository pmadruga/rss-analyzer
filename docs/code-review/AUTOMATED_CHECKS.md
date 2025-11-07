# Automated Code Review Checks

## Comprehensive automated checking system for RSS Analyzer

This document outlines the automated checks that run on every pull request to ensure code quality, security, and performance.

---

## 1. Static Analysis Tools

### Ruff (Primary Linter)

Fast Python linter and formatter that combines multiple tools:

**Configuration** (from `pyproject.toml`):
```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "SIM",    # flake8-simplify
    "I",      # isort
    "N",      # pep8-naming
    "C4",     # flake8-comprehensions
    "PIE",    # flake8-pie
    "T20",    # flake8-print
    "RUF",    # Ruff-specific rules
]

ignore = [
    "E501",   # Line too long (handled by formatter)
    "T201",   # Print statements allowed
    "B008",   # Do not perform function calls in argument defaults
]

[tool.ruff.lint.mccabe]
max-complexity = 10  # Cyclomatic complexity threshold
```

**Checks Performed**:
- Code style violations (PEP 8)
- Import organization
- Unused imports and variables
- Complexity analysis (max 10)
- Naming conventions
- Simplification opportunities
- Bug-prone patterns

**CI/CD Integration**:
```yaml
- name: Run Ruff linter
  run: |
    pip install ruff==0.12.2
    ruff check . --output-format=github

- name: Run Ruff formatter
  run: ruff format --check .
```

**Bot Comments**:
```markdown
## Ruff Issues Found

**E501** (line-too-long) at line 42:
Line length exceeds 88 characters (current: 105)

**F401** (unused-import) at line 5:
`typing.Optional` imported but unused

**Suggestion**: Run `ruff format .` to auto-fix formatting issues
```

---

### Mypy (Type Checker)

Static type checker for Python code:

**Configuration**:
```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
strict_equality = True
check_untyped_defs = True

[mypy-feedparser.*]
ignore_missing_imports = True

[mypy-aiohttp.*]
ignore_missing_imports = True
```

**Checks Performed**:
- Type annotation correctness
- Return type validation
- Argument type matching
- Generic type consistency
- Optional type usage
- Protocol compliance

**CI/CD Integration**:
```yaml
- name: Run Mypy type checker
  run: |
    pip install mypy types-requests types-aiohttp
    mypy src/ --config-file=mypy.ini
```

**Bot Comments**:
```markdown
## Type Check Issues

**src/core/async_scraper.py:45**: error: Incompatible return value type
Expected `Optional[Dict[str, Any]]`, got `str`

**src/ai_clients/base.py:23**: error: Missing type annotation for parameter
Function `analyze` parameter `content` is missing type hint

**Suggestion**: Add type hints: `async def analyze(self, content: str) -> Dict[str, Any]:`
```

---

### Pylint (Advanced Linter)

Comprehensive Python linter:

**Configuration** (`.pylintrc`):
```ini
[MASTER]
disable=
    C0111,  # missing-docstring (covered by ruff)
    C0301,  # line-too-long (covered by ruff)
    R0913,  # too-many-arguments
    R0914,  # too-many-locals

[DESIGN]
max-args=7
max-locals=20
max-branches=15
max-statements=60

[FORMAT]
max-line-length=88

[MESSAGES CONTROL]
enable=
    useless-suppression,
    use-symbolic-message-instead
```

**Checks Performed**:
- Code quality issues
- Design problems
- Refactoring opportunities
- Duplicate code detection
- Complexity metrics
- Convention violations

**CI/CD Integration**:
```yaml
- name: Run Pylint
  run: |
    pip install pylint
    pylint src/ --rcfile=.pylintrc --score=y --exit-zero
```

**Quality Thresholds**:
- Minimum score: 8.0/10.0
- Critical issues: 0
- Blocking issues: < 5

---

## 2. Security Scanning

### Bandit (Security Linter)

Scans for common security issues:

**Configuration** (`.bandit`):
```yaml
tests:
  - B201  # flask_debug_true
  - B301  # pickle
  - B302  # marshal
  - B303  # md5 (allowed for non-cryptographic use)
  - B304  # insecure cipher
  - B305  # insecure cipher mode
  - B306  # tempfile.mktemp
  - B307  # eval
  - B308  # mark_safe
  - B309  # httpsconnection
  - B310  # urllib_urlopen
  - B311  # random (cryptographic)
  - B312  # telnetlib
  - B313  # xml_bad_cElementTree
  - B314  # xml_bad_ElementTree
  - B315  # xml_bad_expatreader
  - B316  # xml_bad_expatbuilder
  - B317  # xml_bad_sax
  - B318  # xml_bad_minidom
  - B319  # xml_bad_pulldom
  - B320  # xml_bad_etree
  - B321  # ftplib
  - B323  # unverified_context
  - B324  # hashlib_insecure_functions
  - B401  # import_telnetlib
  - B402  # import_ftplib
  - B403  # import_pickle
  - B404  # import_subprocess
  - B405  # import_xml_etree
  - B406  # import_xml_sax
  - B407  # import_xml_expat
  - B408  # import_xml_minidom
  - B409  # import_xml_pulldom
  - B410  # import_lxml
  - B411  # import_xmlrpclib
  - B412  # import_httpoxy
  - B501  # request_with_no_cert_validation
  - B502  # ssl_with_bad_version
  - B503  # ssl_with_bad_defaults
  - B504  # ssl_with_no_version
  - B505  # weak_cryptographic_key
  - B506  # yaml_load
  - B507  # ssh_no_host_key_verification
  - B601  # paramiko_calls
  - B602  # shell_injection
  - B603  # subprocess_without_shell_equals_true
  - B604  # any_other_function_with_shell_equals_true
  - B605  # start_process_with_a_shell
  - B606  # start_process_with_no_shell
  - B607  # start_process_with_partial_path
  - B608  # hardcoded_sql_expressions
  - B609  # linux_commands_wildcard_injection

exclude_dirs:
  - tests/
  - venv/
  - .venv/

skips:
  - B303  # MD5 used for non-cryptographic hashing (content deduplication)
```

**Checks Performed**:
- Hardcoded passwords/API keys
- SQL injection risks
- Shell injection vulnerabilities
- Unsafe deserialization
- Weak cryptography
- XML vulnerabilities
- Dangerous function usage (eval, exec)

**CI/CD Integration**:
```yaml
- name: Run Bandit security scan
  run: |
    pip install bandit
    bandit -r src/ -c .bandit -f json -o bandit-report.json

- name: Check for critical security issues
  run: |
    CRITICAL=$(jq '.results[] | select(.issue_severity=="HIGH" or .issue_severity=="CRITICAL")' bandit-report.json | wc -l)
    if [ $CRITICAL -gt 0 ]; then
      echo "❌ Critical security issues found!"
      exit 1
    fi
```

**Bot Comments**:
```markdown
## 🔒 Security Issues Found

**HIGH Severity** at `src/database.py:45`:
Possible SQL injection vulnerability detected

```python
cursor.execute(f"SELECT * FROM articles WHERE id = {article_id}")
```

**Recommendation**: Use parameterized queries:
```python
cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
```

**References**:
- [OWASP SQL Injection](https://owasp.org/www-community/attacks/SQL_Injection)
- [Python DB-API Parameterized Queries](https://peps.python.org/pep-0249/)
```

---

### Safety (Dependency Vulnerability Scanner)

Checks for known security vulnerabilities in dependencies:

**CI/CD Integration**:
```yaml
- name: Run Safety dependency check
  run: |
    pip install safety
    safety check --json --output safety-report.json

- name: Check for vulnerable dependencies
  run: |
    VULNS=$(jq '.vulnerabilities | length' safety-report.json)
    if [ $VULNS -gt 0 ]; then
      echo "⚠️ Vulnerable dependencies found!"
      jq '.vulnerabilities' safety-report.json
      exit 1
    fi
```

**Bot Comments**:
```markdown
## ⚠️ Dependency Vulnerabilities

**requests 2.25.1** has known vulnerability:
- CVE-2023-32681: Unintended leak of Proxy-Authorization header
- CVSS Score: 6.1 (Medium)
- Fixed in: 2.31.0+

**Action Required**: Update `requirements.txt`:
```diff
- requests==2.25.1
+ requests==2.31.0
```

Run: `uv pip install --upgrade requests`
```

---

## 3. Async Pattern Validation

Custom checks for async/await best practices:

### Blocking I/O Detection

**Script**: `tools/check_async_patterns.py`

```python
#!/usr/bin/env python3
"""
Check for blocking I/O in async functions.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

BLOCKING_PATTERNS = {
    'requests.get': 'Use aiohttp.ClientSession.get() instead',
    'requests.post': 'Use aiohttp.ClientSession.post() instead',
    'time.sleep': 'Use asyncio.sleep() instead',
    'open(': 'Use aiofiles.open() for async file I/O',
    'sqlite3.connect': 'Use async database client',
    'urllib.request': 'Use aiohttp instead of urllib',
}

class AsyncBlockingChecker(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Tuple[int, str, str]] = []
        self.in_async_function = False

    def visit_AsyncFunctionDef(self, node):
        self.in_async_function = True
        self.generic_visit(node)
        self.in_async_function = False

    def visit_Call(self, node):
        if self.in_async_function:
            call_name = self._get_call_name(node)
            for pattern, suggestion in BLOCKING_PATTERNS.items():
                if pattern in call_name:
                    self.issues.append((
                        node.lineno,
                        f"Blocking call '{call_name}' in async function",
                        suggestion
                    ))
        self.generic_visit(node)

    def _get_call_name(self, node) -> str:
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

def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check a single Python file for blocking I/O in async functions."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=str(filepath))
        checker = AsyncBlockingChecker(str(filepath))
        checker.visit(tree)
        return checker.issues
    except Exception as e:
        print(f"Error checking {filepath}: {e}", file=sys.stderr)
        return []

def main():
    src_path = Path("src")
    all_issues = []

    for py_file in src_path.rglob("*.py"):
        issues = check_file(py_file)
        if issues:
            all_issues.extend([(py_file, *issue) for issue in issues])

    if all_issues:
        print("❌ Blocking I/O detected in async functions:\n")
        for filepath, lineno, message, suggestion in all_issues:
            print(f"{filepath}:{lineno}: {message}")
            print(f"  💡 {suggestion}\n")
        sys.exit(1)
    else:
        print("✅ No blocking I/O detected in async functions")

if __name__ == "__main__":
    main()
```

**CI/CD Integration**:
```yaml
- name: Check async patterns
  run: python tools/check_async_patterns.py
```

---

### Connection Pool Usage Check

**Script**: `tools/check_pool_usage.py`

```python
#!/usr/bin/env python3
"""
Check that database operations use connection pool.
"""

import re
import sys
from pathlib import Path

ANTI_PATTERNS = [
    (r'sqlite3\.connect\(', 'Direct connection creation detected'),
    (r'psycopg2\.connect\(', 'Direct connection creation detected'),
    (r'MySQLdb\.connect\(', 'Direct connection creation detected'),
]

REQUIRED_PATTERN = r'db_manager\.get_connection\(\)|connection_pool\.acquire\(\)'

def check_file(filepath: Path) -> bool:
    """Check if file uses connection pool properly."""
    with open(filepath) as f:
        content = f.read()

    issues = []
    for pattern, message in ANTI_PATTERNS:
        matches = re.finditer(pattern, content)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            issues.append((line_num, message))

    if issues:
        print(f"\n❌ {filepath}:")
        for line_num, message in issues:
            print(f"  Line {line_num}: {message}")
            print(f"  💡 Use: db_manager.get_connection() instead")
        return False
    return True

def main():
    src_path = Path("src")
    all_ok = True

    for py_file in src_path.rglob("*.py"):
        if 'database.py' in str(py_file):  # Skip database manager itself
            continue
        if not check_file(py_file):
            all_ok = False

    if all_ok:
        print("✅ All database operations use connection pool")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 4. Test Coverage

### Pytest with Coverage

**Configuration** (`pyproject.toml`):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "asyncio: mark test as an asyncio test",
    "slow: mark test as slow running",
    "integration: mark test as integration test",
    "unit: mark test as unit test",
]
addopts = "-v --tb=short --strict-markers --cov=src --cov-report=html --cov-report=term"

[tool.coverage.run]
branch = true
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false

[tool.coverage.html]
directory = "htmlcov"
```

**CI/CD Integration**:
```yaml
- name: Run tests with coverage
  run: |
    pip install pytest pytest-asyncio pytest-cov
    pytest --cov=src --cov-report=json --cov-report=term

- name: Check coverage threshold
  run: |
    COVERAGE=$(jq '.totals.percent_covered' coverage.json)
    if (( $(echo "$COVERAGE < 80" | bc -l) )); then
      echo "❌ Coverage $COVERAGE% is below 80% threshold"
      exit 1
    fi
    echo "✅ Coverage: $COVERAGE%"

- name: Upload coverage report
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.json
    flags: unittests
```

**Coverage Thresholds**:
- Minimum: 80% for new code
- Critical paths: 100% (database, API clients, cache)
- Ideal: 90%+ overall

**Bot Comments**:
```markdown
## 📊 Test Coverage Report

**Overall Coverage**: 87.5% (target: 80%)

### Coverage by Module:
- ✅ `src/core/database.py`: 95.2%
- ✅ `src/core/cache.py`: 92.1%
- ⚠️ `src/async_scraper.py`: 78.3% (below target)
- ❌ `src/etl_orchestrator.py`: 65.4% (below target)

### Missing Coverage:
- `src/async_scraper.py` lines 145-152 (error handling)
- `src/etl_orchestrator.py` lines 78-85, 112-120 (edge cases)

**Action Required**: Add tests for uncovered code paths before merging
```

---

## 5. Pre-commit Hooks

Automated checks that run before each commit:

**Configuration** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-aiohttp]

  - repo: local
    hooks:
      - id: async-pattern-check
        name: Check async patterns
        entry: python tools/check_async_patterns.py
        language: system
        types: [python]
        pass_filenames: false

      - id: pool-usage-check
        name: Check connection pool usage
        entry: python tools/check_pool_usage.py
        language: system
        types: [python]
        pass_filenames: false
```

**Installation**:
```bash
# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

## 6. CI/CD Pipeline Integration

Complete automated check workflow:

**File**: `.github/workflows/automated-checks.yml`

```yaml
name: Automated Code Checks

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

jobs:
  lint-and-type-check:
    name: Linting and Type Checking
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install ruff mypy pylint types-requests types-aiohttp

      - name: Run Ruff linter
        run: ruff check . --output-format=github

      - name: Run Ruff formatter check
        run: ruff format --check .

      - name: Run Mypy type checker
        run: mypy src/ --config-file=mypy.ini

      - name: Run Pylint
        run: |
          SCORE=$(pylint src/ --rcfile=.pylintrc --score=y --exit-zero | grep "Your code has been rated" | grep -oP '\d+\.\d+')
          echo "Pylint score: $SCORE/10"
          if (( $(echo "$SCORE < 8.0" | bc -l) )); then
            echo "❌ Pylint score below 8.0 threshold"
            exit 1
          fi

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install security tools
        run: pip install bandit safety

      - name: Run Bandit security scan
        run: |
          bandit -r src/ -c .bandit -f json -o bandit-report.json
          CRITICAL=$(jq '[.results[] | select(.issue_severity=="HIGH" or .issue_severity=="CRITICAL")] | length' bandit-report.json)
          if [ $CRITICAL -gt 0 ]; then
            echo "❌ Critical security issues found!"
            jq '.results[] | select(.issue_severity=="HIGH" or .issue_severity=="CRITICAL")' bandit-report.json
            exit 1
          fi

      - name: Run Safety dependency check
        run: safety check --json

  async-patterns:
    name: Async Pattern Validation
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Check for blocking I/O in async functions
        run: python tools/check_async_patterns.py

      - name: Check connection pool usage
        run: python tools/check_pool_usage.py

  test-coverage:
    name: Test Coverage
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest pytest-asyncio pytest-cov
          pip install -r requirements.txt

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=json --cov-report=term

      - name: Check coverage threshold
        run: |
          COVERAGE=$(jq '.totals.percent_covered' coverage.json)
          echo "Coverage: $COVERAGE%"
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "❌ Coverage below 80% threshold"
            exit 1
          fi

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.json
```

---

## Quick Reference: Check Summary

| Check | Tool | Threshold | Blocking |
|-------|------|-----------|----------|
| Code style | Ruff | Must pass | Yes |
| Type checking | Mypy | Must pass | Yes |
| Code quality | Pylint | ≥ 8.0/10 | Yes |
| Security | Bandit | 0 critical | Yes |
| Dependencies | Safety | 0 critical | Yes |
| Async patterns | Custom | Must pass | Yes |
| Connection pool | Custom | Must pass | Yes |
| Test coverage | Pytest-cov | ≥ 80% | Yes |

All checks must pass before PR can be merged.

---

## Troubleshooting

### Common Issues

**Issue**: Ruff format check fails
```bash
# Solution: Auto-format code
ruff format .
```

**Issue**: Mypy type errors
```bash
# Solution: Add type hints
async def process_article(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ...
```

**Issue**: Coverage below threshold
```bash
# Solution: Add tests for uncovered code
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to see uncovered lines
```

**Issue**: Security scan fails
```bash
# Solution: Review and fix security issues
bandit -r src/ -v
# Address each HIGH/CRITICAL issue
```

---

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
