# Pull Request Review Guide

## Overview

This guide provides comprehensive checklists and guidelines for reviewing pull requests in the RSS Analyzer project.

## Quick Reference

- **PR Template**: `.github/pull_request_template.md`
- **Code Review Workflow**: `.github/workflows/code-review-swarm.yml`
- **PR Automation**: `.github/workflows/pr-automation.yml`
- **Quality Checks**: `.github/workflows/pr-checks.yml`

## Review Process

### 1. Automated Review (AI Swarm)

All PRs automatically trigger the AI Code Review Swarm which performs:

- 🔒 **Security Review**: Scans for hardcoded secrets, SQL injection, dangerous functions
- 📋 **Code Quality**: Runs pylint, flake8, complexity analysis
- 🐍 **Python Best Practices**: Checks type hints, pathlib usage, f-strings, context managers
- ⚡ **Performance Analysis**: Reviews database queries, caching, async patterns

**Wait for automated review to complete before human review.**

### 2. Human Review Checklist

#### Code Quality

- [ ] Code follows PEP 8 style guide
- [ ] Functions are well-named and self-documenting
- [ ] Complex logic has explanatory comments
- [ ] No obvious code duplication (DRY principle)
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate and informative

#### Python Best Practices

- [ ] Type hints present on all functions
  ```python
  def process_article(article: Article) -> ProcessedArticle:
  ```

- [ ] Using `pathlib.Path` instead of `os.path`
  ```python
  # Good
  from pathlib import Path
  path = Path('data') / 'articles.db'

  # Avoid
  import os
  path = os.path.join('data', 'articles.db')
  ```

- [ ] F-strings instead of `.format()` or `%`
  ```python
  # Good
  message = f"Found {count} articles"

  # Avoid
  message = "Found {} articles".format(count)
  ```

- [ ] Context managers for resources
  ```python
  # Good
  with open('file.txt') as f:
      data = f.read()

  # Avoid
  f = open('file.txt')
  data = f.read()
  f.close()
  ```

- [ ] Specific exceptions instead of bare `except:`
  ```python
  # Good
  try:
      process()
  except ValueError as e:
      logger.error(f"Invalid value: {e}")

  # Avoid
  try:
      process()
  except:
      pass
  ```

#### Async/Await Patterns

- [ ] All I/O operations use async where applicable
- [ ] Proper use of `await` for async functions
- [ ] No blocking calls in async code (use `asyncio.to_thread()` if needed)
- [ ] Rate limiting implemented for concurrent operations
- [ ] Connection pooling used for database and API operations

```python
# Good async pattern
async def fetch_articles(urls: list[str]) -> list[Article]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_article(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Avoid blocking in async
async def bad_example():
    time.sleep(5)  # ❌ Blocks event loop

async def good_example():
    await asyncio.sleep(5)  # ✅ Non-blocking
```

#### Database Operations

- [ ] Using connection pooling (not creating new connections each time)
- [ ] Transactions used for multi-statement operations
- [ ] No SQL injection vulnerabilities (parameterized queries)
- [ ] Efficient queries (proper indexing, no N+1 queries)
- [ ] Hash-based deduplication used where applicable

```python
# Good - parameterized query with pool
async with pool.get_connection() as conn:
    await conn.execute(
        "INSERT INTO articles VALUES (?, ?)",
        (title, url)
    )

# Avoid - string formatting in SQL
await conn.execute(
    f"INSERT INTO articles VALUES ('{title}', '{url}')"  # ❌ SQL injection risk
)
```

#### Security

- [ ] No hardcoded secrets or API keys
- [ ] Environment variables used for sensitive data
- [ ] Input validation on all user-provided data
- [ ] No use of dangerous functions (eval, exec, os.system)
- [ ] Proper authentication/authorization checks
- [ ] Dependencies reviewed for known vulnerabilities

```python
# Good - environment variables
import os
api_key = os.getenv('ANTHROPIC_API_KEY')

# Avoid - hardcoded secrets
api_key = "sk-ant-12345..."  # ❌ Never commit secrets
```

#### Performance

- [ ] Efficient algorithms (consider time/space complexity)
- [ ] Caching used appropriately (L1/L2 cache system)
- [ ] Database queries optimized (proper indices, batch operations)
- [ ] No unnecessary loops or redundant operations
- [ ] Async operations used for I/O-bound tasks
- [ ] Rate limiting prevents API abuse

#### Testing

- [ ] New functionality has corresponding tests
- [ ] Tests cover normal cases and edge cases
- [ ] Tests cover error conditions
- [ ] Integration tests for API changes
- [ ] Test coverage maintained at 80%+
- [ ] Tests are readable and maintainable

```python
# Good test structure
@pytest.mark.asyncio
async def test_article_processing():
    """Test article processing with valid input."""
    # Arrange
    article = create_test_article()
    processor = ArticleProcessor()

    # Act
    result = await processor.process(article)

    # Assert
    assert result.status == "success"
    assert result.content is not None
```

#### Documentation

- [ ] Docstrings present for all public functions/classes
- [ ] Complex algorithms explained with comments
- [ ] README.md updated if functionality changed
- [ ] CHANGELOG.md updated for notable changes
- [ ] API documentation updated if endpoints changed
- [ ] Migration guide provided for breaking changes

```python
def process_article(article: Article, options: ProcessingOptions) -> ProcessedArticle:
    """
    Process an article with AI analysis.

    Args:
        article: The article to process
        options: Processing configuration options

    Returns:
        Processed article with AI analysis

    Raises:
        APIError: If API call fails
        ValidationError: If article data is invalid
    """
```

#### Breaking Changes

- [ ] Breaking changes clearly documented
- [ ] Migration path provided
- [ ] Backward compatibility maintained where possible
- [ ] Deprecation warnings added for phased migration

### 3. Review Focus Areas by PR Type

#### Bug Fixes

- [ ] Bug is clearly described
- [ ] Root cause identified
- [ ] Fix addresses root cause (not just symptoms)
- [ ] Test added to prevent regression
- [ ] No side effects or unintended changes

#### New Features

- [ ] Feature is well-documented
- [ ] Design is sound and maintainable
- [ ] Fits with existing architecture
- [ ] Tests cover happy path and edge cases
- [ ] Performance impact considered

#### Performance Improvements

- [ ] Benchmarks provided showing improvement
- [ ] No regressions in other areas
- [ ] Memory usage considered
- [ ] Scalability implications understood

#### Refactoring

- [ ] Behavior unchanged (verified by tests)
- [ ] Code is clearer/more maintainable
- [ ] No unnecessary changes mixed in
- [ ] Tests still pass

#### Documentation

- [ ] Documentation is clear and accurate
- [ ] Examples are helpful and correct
- [ ] Links are valid
- [ ] Formatting is consistent

## Review Workflow

### Step 1: Initial Triage (Automated)

The PR automation workflow automatically:

1. Labels PR based on changed files
2. Assigns size label (XS/S/M/L/XL)
3. Sets priority based on critical paths
4. Checks PR description completeness
5. Links related issues
6. Suggests reviewers from CODEOWNERS

### Step 2: Quality Checks (Automated)

The PR checks workflow runs:

1. Unit tests
2. Type checking with mypy
3. Linting with ruff
4. Security scanning with bandit/safety
5. Code coverage analysis
6. Import validation
7. Docker build test

**All checks must pass before merge.**

### Step 3: AI Code Review (Automated)

The AI Review Swarm performs:

1. **Security Review**
   - Scans for hardcoded secrets
   - Checks for SQL injection vulnerabilities
   - Identifies dangerous function usage
   - Validates secure patterns

2. **Code Quality Review**
   - Runs pylint (target: 8.0+/10)
   - Runs flake8 (target: 0 issues)
   - Analyzes cyclomatic complexity (target: <10)

3. **Python Best Practices Review**
   - Type hints coverage
   - Modern Python patterns
   - Proper resource management
   - Pythonic code patterns

**Address critical issues before human review.**

### Step 4: Human Review

Reviewer responsibilities:

1. Review automated feedback
2. Verify fixes address root causes
3. Check architecture and design decisions
4. Evaluate maintainability
5. Consider long-term implications
6. Test locally if needed
7. Provide constructive feedback

## Providing Feedback

### Good Feedback

- **Specific**: Point to exact lines/files
- **Constructive**: Suggest improvements
- **Educational**: Explain reasoning
- **Respectful**: Be kind and professional

Example:
```
Consider using a list comprehension here for better readability:

# Current
results = []
for item in items:
    if item.valid:
        results.append(item.value)

# Suggested
results = [item.value for item in items if item.valid]

This is more pythonic and often more performant.
```

### Avoid

- Vague comments ("This looks wrong")
- Nitpicking without value
- Personal attacks or negativity
- Blocking on personal preferences

## Common Issues and Solutions

### Issue: Tests Failing

**Solution:**
1. Check test logs in GitHub Actions
2. Run tests locally: `uv run pytest tests/ -v`
3. Fix failing tests or update if behavior changed intentionally

### Issue: Coverage Below Threshold

**Solution:**
1. Add tests for uncovered code paths
2. Or justify why coverage is acceptable

### Issue: Linting Errors

**Solution:**
1. Run locally: `uv run ruff check src/`
2. Auto-fix where possible: `uv run ruff check --fix src/`
3. Format code: `uv run ruff format .`

### Issue: Type Checking Errors

**Solution:**
1. Run locally: `uv run mypy src/`
2. Add missing type hints
3. Use `# type: ignore` with comment only when necessary

### Issue: Security Issues

**Solution:**
1. Never ignore security issues
2. Use environment variables for secrets
3. Use parameterized queries for SQL
4. Avoid dangerous functions

## Approval Criteria

PRs must meet ALL criteria before approval:

### Required
- [ ] All automated checks pass
- [ ] No critical security issues
- [ ] Tests pass and coverage ≥ 80%
- [ ] Code quality meets standards
- [ ] Documentation updated
- [ ] No unresolved review comments

### Recommended
- [ ] Pylint score ≥ 8.0
- [ ] No flake8 issues
- [ ] Cyclomatic complexity < 10
- [ ] Type hints on all functions
- [ ] Follows Python best practices

## Merge Process

1. ✅ All checks pass
2. ✅ Human approval received
3. ✅ All comments resolved
4. 🔄 Squash merge to main
5. 🗑️ Delete feature branch
6. 📝 Update CHANGELOG if applicable

## Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Async Programming](https://docs.python.org/3/library/asyncio.html)
- [OWASP Security](https://owasp.org/www-project-top-ten/)
- [Project Documentation](../docs/)

## Questions?

If you have questions about the review process:

1. Check this guide
2. Review existing PRs for examples
3. Ask in PR comments
4. Reach out to maintainers

---

*This guide is part of the RSS Analyzer project documentation.*
