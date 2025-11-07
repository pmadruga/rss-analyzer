# Pull Request Description

## Summary

Brief description of what this PR does and why.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] CI/CD or workflow changes
- [ ] Dependency update

## Related Issues

Closes #(issue number)

## Motivation and Context

Why is this change required? What problem does it solve?

## Changes Made

Detailed list of changes:

- Change 1
- Change 2
- Change 3

## Testing Strategy

Describe the tests you ran to verify your changes:

- [ ] Unit tests pass locally
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Test coverage maintained or improved

### Test Commands Run

```bash
# Commands used for testing
uv run pytest tests/
docker compose run rss-analyzer test-pipeline
```

## Performance Impact

- [ ] No performance impact
- [ ] Performance improved (describe below)
- [ ] Performance degraded (justify below)

**Performance notes:**

## Breaking Changes

- [ ] No breaking changes
- [ ] Breaking changes (describe migration path below)

**Migration path:**

## Documentation

- [ ] Documentation updated in `/docs`
- [ ] README.md updated
- [ ] CHANGELOG.md updated
- [ ] API documentation updated
- [ ] Inline code comments added/updated
- [ ] No documentation needed

## Code Quality Checklist

### Python Best Practices

- [ ] Type hints added to all functions
- [ ] Docstrings present for classes and functions
- [ ] Using `pathlib.Path` instead of `os.path`
- [ ] Using f-strings instead of `.format()` or `%`
- [ ] Context managers (`with` statements) for file/resource handling
- [ ] Proper exception handling with specific exceptions
- [ ] No bare `except:` clauses
- [ ] PEP 8 compliant (line length, naming conventions)

### Async/Await Patterns

- [ ] All I/O operations are async where applicable
- [ ] Proper use of `await` for async functions
- [ ] No blocking calls in async code
- [ ] Rate limiting implemented for concurrent operations
- [ ] Connection pooling used for database operations

### Error Handling

- [ ] All errors have descriptive messages
- [ ] Critical errors logged with proper context
- [ ] Error recovery mechanisms in place
- [ ] No silent failures
- [ ] User-facing errors are clear and actionable

### Database Operations

- [ ] Using connection pooling
- [ ] Transactions used where appropriate
- [ ] No SQL injection vulnerabilities
- [ ] Proper indexing for queries
- [ ] Migration scripts included if schema changed

### Security

- [ ] No hardcoded secrets or API keys
- [ ] Environment variables used for sensitive data
- [ ] Input validation implemented
- [ ] No SQL injection vulnerabilities
- [ ] No use of dangerous functions (eval, exec, os.system)
- [ ] Proper authentication/authorization checks
- [ ] Dependencies scanned for vulnerabilities

### Testing

- [ ] New tests added for new functionality
- [ ] Existing tests pass
- [ ] Test coverage maintained at 80%+
- [ ] Edge cases tested
- [ ] Error paths tested
- [ ] Integration tests included for API changes

## Dependencies

List any new dependencies added:

- Package 1: reason for addition
- Package 2: reason for addition

## Deployment Notes

Special considerations for deployment:

- [ ] Database migrations required
- [ ] Configuration changes needed
- [ ] Environment variables to add/update
- [ ] Backward compatibility maintained
- [ ] Deployment order matters (describe below)

## Screenshots

If applicable, add screenshots to help explain your changes.

## Additional Context

Any other context or information reviewers should know.

## Reviewer Checklist

For reviewers - ensure the following before approving:

- [ ] Code follows project style guidelines
- [ ] Changes are well-documented
- [ ] Tests adequately cover changes
- [ ] No obvious security issues
- [ ] Performance impact is acceptable
- [ ] Breaking changes are properly documented
- [ ] CI/CD checks pass

---

## AI Review Summary

The AI Code Review Swarm will automatically analyze this PR and provide:

- 🔒 Security analysis
- 📋 Code quality metrics
- 🐍 Python best practices review
- ⚡ Performance analysis (if applicable)
- 📚 Documentation completeness check

Please address any critical issues raised by the automated review before requesting human review.
