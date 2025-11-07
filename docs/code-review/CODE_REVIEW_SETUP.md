# Code Review Setup - Complete Configuration

## Overview

This document provides a complete overview of the automated code review system for the RSS Analyzer project, including all configurations, tools, and workflows.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Automated Code Review System                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐ │
│  │  Pre-commit    │→ │  GitHub        │→ │  Review Bot     │ │
│  │  Hooks         │  │  Actions       │  │  Comments       │ │
│  │  (Local)       │  │  (CI/CD)       │  │  (Automated)    │ │
│  └────────────────┘  └────────────────┘  └─────────────────┘ │
│         │                    │                     │          │
│         ▼                    ▼                     ▼          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │             Quality Gates (7 Total)                      │ │
│  │  1. Code Quality    (Ruff, Mypy, Pylint)                │ │
│  │  2. Security        (Bandit, Safety)                    │ │
│  │  3. Async Patterns  (Custom validators)                 │ │
│  │  4. Test Coverage   (Pytest ≥80%)                       │ │
│  │  5. Documentation   (Docstrings, Type hints)            │ │
│  │  6. Performance     (Regression detection)              │ │
│  │  7. Code Review     (Human approval)                    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install Pre-commit Hooks (Local Development)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks on all files
pre-commit run --all-files
```

### 2. Run Quality Gate Checks Manually

```bash
# Run all quality gates
./tools/check_quality_gate.sh all

# Run specific gate
./tools/check_quality_gate.sh code-quality
./tools/check_quality_gate.sh security
./tools/check_quality_gate.sh async-patterns
./tools/check_quality_gate.sh test-coverage
```

### 3. GitHub Actions (Automatic)

GitHub Actions automatically run on every pull request. No setup required.

---

## Configuration Files

### Pre-commit Hooks
- **File**: `.pre-commit-config.yaml`
- **Purpose**: Local checks before commit
- **Tools**: Ruff, Mypy, Bandit, custom validators
- **Execution**: On `git commit`

### GitHub Actions Workflow
- **File**: `.github/workflows/automated-code-review.yml`
- **Purpose**: CI/CD quality gates
- **Tools**: All quality checks + summary generation
- **Execution**: On pull request creation/update

### Quality Gate Checker
- **File**: `tools/check_quality_gate.sh`
- **Purpose**: Manual quality gate execution
- **Usage**: `./tools/check_quality_gate.sh [gate]`

### Async Pattern Validator
- **File**: `tools/check_async_patterns.py`
- **Purpose**: Detect blocking I/O in async functions
- **Checks**: requests, time.sleep, sqlite3.connect, etc.

### Connection Pool Validator
- **File**: `tools/check_pool_usage.py`
- **Purpose**: Ensure database operations use pool
- **Checks**: Direct connection creation

---

## Quality Gates Details

### Gate 1: Code Quality

**Tools**: Ruff, Mypy, Pylint

**Requirements**:
- Ruff: 0 errors
- Mypy: 0 type errors
- Pylint: Score ≥ 8.0/10

**Run Manually**:
```bash
ruff check src/
mypy src/ --ignore-missing-imports
pylint src/ --score=y
```

**Fix Common Issues**:
```bash
# Auto-fix Ruff issues
ruff check src/ --fix

# Format code
ruff format src/
```

---

### Gate 2: Security

**Tools**: Bandit, Safety

**Requirements**:
- Bandit: 0 critical/high issues
- Safety: 0 critical vulnerabilities

**Run Manually**:
```bash
bandit -r src/ --skip B303
safety check --json
```

**Common Issues**:
- Hardcoded secrets → Use environment variables
- SQL injection → Use parameterized queries
- Vulnerable dependencies → Update packages

---

### Gate 3: Async Patterns

**Tools**: Custom Python validators

**Requirements**:
- 0 blocking I/O in async functions
- 100% connection pool usage

**Run Manually**:
```bash
python tools/check_async_patterns.py
python tools/check_pool_usage.py
```

**Common Issues**:
- `requests.get()` → Use `aiohttp`
- `time.sleep()` → Use `asyncio.sleep()`
- `sqlite3.connect()` → Use connection pool

---

### Gate 4: Test Coverage

**Tools**: Pytest with coverage

**Requirements**:
- Overall: ≥ 80%
- New code: ≥ 90%
- Critical paths: 100%

**Run Manually**:
```bash
pytest --cov=src --cov-report=html --cov-report=term
open htmlcov/index.html
```

**Add Tests**:
```python
# tests/test_module.py
import pytest

@pytest.mark.asyncio
async def test_function():
    result = await function()
    assert result is not None
```

---

### Gate 5: Documentation

**Requirements**:
- 100% docstrings on public functions
- 100% type hints on function signatures
- README updates for API changes

**Check Manually**:
```bash
# Find functions without docstrings
grep -r "^def \|^async def " src/ --include="*.py"

# Find functions without type hints
grep -r "^def " src/ --include="*.py" | grep -v " -> "
```

**Add Documentation**:
```python
async def process_article(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single article with AI analysis.

    Args:
        article: Article dictionary with 'url' and 'title' keys

    Returns:
        Analysis result or None if processing fails

    Raises:
        ValueError: If article data is invalid
    """
    # Function body
```

---

### Gate 6: Performance

**Requirements**:
- ≤ 5% regression
- Memory: ≤ +10%

**Status**: Manual benchmarking required

**Run Benchmarks**:
```bash
# Future: Automated performance benchmarks
python tools/performance_benchmark.py --compare-baseline
```

---

### Gate 7: Code Review

**Requirements**:
- 1+ maintainer approval
- All automated gates passed
- All conversations resolved

**Process**:
1. Automated gates pass
2. Request human review
3. Address feedback
4. Get approval
5. Merge

---

## Review Bot Comments

The automated review bot posts helpful comments on PRs for common issues:

### Comment Categories

1. **Async/Await Issues**
   - Blocking I/O detected
   - Missing await
   - Wrong sleep function

2. **Database Issues**
   - Direct connection creation
   - Missing context manager

3. **Cache Issues**
   - Missing cache check
   - Incorrect TTL

4. **Security Issues**
   - Hardcoded secrets (CRITICAL)
   - SQL injection risks (HIGH)
   - Missing input validation (MEDIUM)

5. **Type Hint Issues**
   - Missing type hints
   - Incorrect return types

6. **Documentation Issues**
   - Missing docstrings
   - Incomplete documentation

7. **Performance Issues**
   - Unbounded concurrency
   - Missing rate limiting

8. **Test Coverage Issues**
   - Missing tests
   - Insufficient error testing

See [REVIEW_BOT_TEMPLATES.md](./REVIEW_BOT_TEMPLATES.md) for all comment templates.

---

## Troubleshooting

### Pre-commit Hook Fails

```bash
# Skip hooks temporarily (not recommended)
git commit --no-verify

# Fix issues and retry
pre-commit run --all-files
git add .
git commit
```

### GitHub Actions Fail

1. Check the Actions tab for detailed logs
2. Run the same checks locally:
   ```bash
   ./tools/check_quality_gate.sh all
   ```
3. Fix issues and push changes

### Coverage Below Threshold

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Open report in browser
open htmlcov/index.html

# Add tests for uncovered lines
vim tests/test_module.py
```

### Security Issues Found

```bash
# View Bandit report
bandit -r src/ -v

# View Safety report
safety check --full-report

# Fix issues and re-run
./tools/check_quality_gate.sh security
```

---

## CI/CD Integration

### GitHub Branch Protection

Recommended settings for `main` branch:

```yaml
Require pull request before merging: true
Require approvals: 1
Require status checks to pass: true
Required status checks:
  - code-quality
  - security
  - async-patterns
  - test-coverage
  - documentation
Require conversation resolution: true
Require signed commits: false (optional)
```

### Auto-merge Rules

Can enable auto-merge when:
- All quality gates pass
- All conversations resolved
- Required approvals obtained

---

## Best Practices

### For Developers

1. **Run pre-commit hooks before pushing**
   ```bash
   pre-commit run --all-files
   ```

2. **Write tests for new code**
   - Aim for 90%+ coverage on new code
   - Include error scenarios

3. **Document as you code**
   - Add docstrings to all public functions
   - Include type hints

4. **Follow async patterns**
   - Use `aiohttp` instead of `requests`
   - Use connection pool for database
   - Check cache before expensive operations

5. **Review security checklist**
   - No hardcoded secrets
   - Use parameterized queries
   - Validate all inputs

### For Reviewers

1. **Check automated results first**
   - Review bot comments
   - Quality gate status

2. **Focus on logic and architecture**
   - Automated tools handle style/syntax
   - Review algorithmic correctness
   - Check for edge cases

3. **Verify documentation**
   - Clear and accurate
   - Examples where helpful
   - Updated for changes

4. **Consider performance**
   - No obvious bottlenecks
   - Appropriate use of async
   - Efficient data structures

---

## Maintenance

### Update Dependencies

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Update Python packages
pip install --upgrade ruff mypy pylint bandit safety
```

### Review Thresholds Quarterly

- Are quality gates still appropriate?
- Should thresholds be adjusted?
- Are tools up to date?

### Monitor Metrics

Track:
- False positive rate
- Time to merge
- Defect escape rate
- Developer satisfaction

---

## Resources

### Documentation
- [Review Checklist](./REVIEW_CHECKLIST.md) - Comprehensive review guidelines
- [Automated Checks](./AUTOMATED_CHECKS.md) - Tool configuration details
- [Quality Gates](./QUALITY_GATES.md) - Gate definitions and thresholds
- [Review Bot Templates](./REVIEW_BOT_TEMPLATES.md) - Comment templates

### Project Documentation
- [Async Guide](../ASYNC_GUIDE.md) - Async/await best practices
- [Connection Pooling](../CONNECTION_POOLING.md) - Database optimization
- [Cache Usage](../CACHE_USAGE.md) - Caching strategies
- [Optimization Results](../OPTIMIZATION_RESULTS.md) - Performance benchmarks

### External Resources
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Documentation](https://pre-commit.com/)

---

## Summary

The automated code review system for RSS Analyzer includes:

### Files Created
1. `docs/code-review/REVIEW_CHECKLIST.md` - Complete review checklist
2. `docs/code-review/AUTOMATED_CHECKS.md` - Tool configuration
3. `docs/code-review/QUALITY_GATES.md` - Gate definitions
4. `docs/code-review/REVIEW_BOT_TEMPLATES.md` - Comment templates
5. `docs/code-review/CODE_REVIEW_SETUP.md` - This file
6. `tools/check_async_patterns.py` - Async validator
7. `tools/check_pool_usage.py` - Pool validator
8. `tools/check_quality_gate.sh` - Gate runner
9. `.pre-commit-config.yaml` - Pre-commit hooks
10. `.github/workflows/automated-code-review.yml` - CI/CD workflow

### Quality Gates (7 Total)
1. Code Quality (Ruff, Mypy, Pylint)
2. Security (Bandit, Safety)
3. Async Patterns (Custom validators)
4. Test Coverage (Pytest ≥80%)
5. Documentation (Docstrings, Type hints)
6. Performance (Regression detection)
7. Code Review (Human approval)

### Key Features
- Pre-commit hooks for fast local checks
- GitHub Actions for comprehensive CI/CD
- Automated review bot with helpful comments
- Custom async pattern validators
- Connection pool usage enforcement
- Security vulnerability scanning
- Test coverage enforcement

### Getting Started
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run all checks
./tools/check_quality_gate.sh all

# Create PR - automated review runs automatically
```

---

## Contact

For questions or issues with the code review system:
- Create an issue in the repository
- Tag maintainers in PR comments
- Review the documentation in `docs/code-review/`
