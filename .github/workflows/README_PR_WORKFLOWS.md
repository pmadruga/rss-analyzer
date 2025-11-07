# Pull Request Workflows Documentation

## Overview

The RSS Analyzer project has comprehensive automated PR workflows that handle review, quality checks, and automation. This document provides an overview of all PR-related automation.

## Workflow Files

### 1. PR Automation (`pr-automation.yml`)

**Triggers**: PR opened, synchronized, reopened, edited

**Functions**:
- **Auto-labeling**: Applies labels based on changed files
- **Size labeling**: Adds size labels (XS/S/M/L/XL)
- **Priority labeling**: Sets priority based on critical paths
- **Completeness check**: Validates PR description against template
- **Issue linking**: Automatically links referenced issues
- **Reviewer suggestion**: Suggests reviewers from CODEOWNERS

**Labels Applied**:
- Component: `core`, `processors`, `database`, `api-integration`, `async`, `performance`
- File type: `tests`, `documentation`, `ci-cd`, `docker`, `dependencies`
- Size: `size/XS`, `size/S`, `size/M`, `size/L`, `size/XL`
- Priority: `priority: critical`, `priority: high`, `priority: medium`

### 2. PR Quality Checks (`pr-checks.yml`)

**Triggers**: PR to main/develop with Python file changes

**Checks**:
1. **Tests**: Runs pytest with coverage reporting
2. **Type Checking**: Runs mypy on source code
3. **Linting**: Runs ruff for code style
4. **Security Scan**: Runs bandit and safety
5. **Coverage Check**: Ensures 80%+ coverage
6. **Import Validation**: Validates all imports work
7. **Docker Build**: Tests container builds

**Artifacts**:
- Coverage reports (XML + HTML)
- Security scan reports (JSON)

**Required for Merge**: All checks must pass

### 3. AI Code Review Swarm (`code-review-swarm.yml`)

**Triggers**: PR opened, synchronized, reopened

**Review Agents**:

1. **Review Triage**
   - Analyzes changed files
   - Determines review depth (standard/thorough/comprehensive)
   - Assigns appropriate agents
   - Sets priority level

2. **Security Review** (🔒)
   - Scans for hardcoded secrets
   - Checks for SQL injection risks
   - Identifies dangerous function usage
   - Validates secure patterns
   - **Severity levels**: critical, high, medium, low

3. **Code Quality Review** (📋)
   - Runs pylint (target: 8.0/10)
   - Runs flake8 (target: 0 issues)
   - Analyzes cyclomatic complexity (target: <10)
   - Generates detailed metrics

4. **Python Best Practices** (🐍)
   - Checks type hints coverage
   - Validates modern Python patterns
   - Suggests improvements
   - Provides code examples

5. **Review Summary**
   - Aggregates all agent results
   - Determines overall status
   - Auto-approves if all checks pass
   - Blocks merge for critical issues

## PR Template

Location: `.github/pull_request_template.md`

**Sections**:
- Summary
- Type of Change
- Related Issues
- Motivation and Context
- Changes Made
- Testing Strategy
- Performance Impact
- Breaking Changes
- Documentation
- Code Quality Checklist (comprehensive)
- Dependencies
- Deployment Notes

**Key Checklists**:
- Python best practices (type hints, pathlib, f-strings, context managers)
- Async/await patterns (non-blocking I/O, rate limiting, connection pooling)
- Error handling (descriptive messages, proper logging, recovery)
- Database operations (pooling, transactions, no SQL injection)
- Security (no hardcoded secrets, input validation, safe functions)
- Testing (coverage, edge cases, error paths)

## Review Guide

Location: `.github/PR_REVIEW_GUIDE.md`

**Contents**:
- Review process overview
- Automated review details
- Human review checklists
- Code examples (good vs bad patterns)
- Focus areas by PR type
- Feedback guidelines
- Common issues and solutions
- Approval criteria
- Merge process

## Review Scripts

### Python Review (`python-review.sh`)

Checks for:
- Type hints
- Pathlib usage
- F-strings
- Context managers
- List comprehensions
- Dataclasses
- Named tuples
- Enum usage

### Quality Review (`quality-review.sh`)

Runs:
- Pylint with scoring
- Flake8 with statistics
- Radon complexity analysis
- Generates detailed metrics report

### Security Review (`security-review.sh`)

Scans for:
- Hardcoded secrets
- SQL injection risks
- Dangerous functions (eval, exec)
- Shell injection risks
- Insecure random usage
- Missing error handling
- Insecure deserialization

## Workflow Configuration

Location: `.github/review-swarm.yml`

**Key Settings**:
```yaml
review:
  auto-trigger: true
  required-agents: [security, code-quality, python-best-practices]
  optional-agents: [performance, database, api-integration, documentation]

thresholds:
  security: block              # Block PR on security issues
  performance: warn            # Warn on performance issues
  code-quality: suggest        # Suggest improvements
  database: warn               # Warn on database issues

depth-rules:
  critical-paths:
    - "**/database.py"
    - "**/auth/**"
    - "**/*_client.py"
    depth: comprehensive
    agents: ["security", "performance", "database"]
```

## Quality Thresholds

**Code Quality**:
- Pylint score: ≥ 8.0/10
- Flake8 issues: 0
- Cyclomatic complexity: < 10 per function
- Code coverage: ≥ 80%

**Security**:
- No critical security issues
- No hardcoded secrets
- No SQL injection vulnerabilities
- No dangerous function usage

**Performance**:
- No significant regressions
- Efficient database queries
- Proper caching used
- Rate limiting for APIs

## Merge Requirements

All PRs must meet:

1. ✅ All automated checks pass
2. ✅ No critical security issues
3. ✅ Test coverage ≥ 80%
4. ✅ Code quality thresholds met
5. ✅ Documentation updated
6. ✅ Human approval received
7. ✅ All review comments resolved

## Common Workflows

### Standard Feature PR

1. Create feature branch
2. Make changes
3. Open PR (auto-labels applied)
4. Automated checks run
5. AI review swarm analyzes
6. Address feedback
7. Request human review
8. Approval and merge

### Hotfix PR

1. Create hotfix branch
2. Make critical fix
3. Open PR with "hotfix" label
4. Priority: critical (auto-applied)
5. Fast-tracked review
6. Merge and backport if needed

### Large Refactoring PR

1. Create draft PR early
2. Label as "size/XL"
3. Comprehensive review assigned
4. Phased review by agents
5. Multiple review iterations
6. Approval and merge

## Monitoring and Metrics

**Tracked Metrics**:
- Review time per PR
- Issues found per agent
- False positive rate
- Fix rate
- Agent accuracy

**Reporting**:
- Weekly summary reports
- Trend analysis
- Top issues identified
- Improvement suggestions

## Best Practices

### For PR Authors

1. Use the PR template
2. Fill out all sections completely
3. Run checks locally before pushing
4. Address automated feedback first
5. Respond to review comments promptly
6. Keep PRs focused and reasonably sized

### For Reviewers

1. Wait for automated review to complete
2. Review automated feedback first
3. Focus on design and architecture
4. Provide specific, constructive feedback
5. Test locally for complex changes
6. Approve only when all criteria met

## Troubleshooting

### Checks Failing

**Problem**: Tests or quality checks fail

**Solution**:
1. Check GitHub Actions logs
2. Run checks locally
3. Fix issues
4. Push changes to re-run

### AI Review Not Triggered

**Problem**: Code review swarm doesn't run

**Solution**:
1. Check workflow triggers
2. Ensure PR targets main/develop
3. Re-run workflow manually if needed

### Coverage Below Threshold

**Problem**: Test coverage < 80%

**Solution**:
1. Add tests for uncovered code
2. Or justify why coverage is acceptable
3. Update coverage reports

## Integration with Other Tools

**CODEOWNERS**:
- Auto-suggests reviewers
- Routes PRs to domain experts

**GitHub Labels**:
- Auto-applied based on content
- Used for filtering and reporting

**Branch Protection**:
- Requires status checks
- Requires human approval
- Prevents force push

## Future Enhancements

Planned improvements:
- Automated dependency updates
- Performance regression detection
- Automatic changelog generation
- PR size recommendations
- Enhanced AI agent coordination

## Resources

- [Pull Request Template](.github/pull_request_template.md)
- [Review Guide](.github/PR_REVIEW_GUIDE.md)
- [Workflow Configuration](.github/review-swarm.yml)
- [Python Review Script](.github/scripts/python-review.sh)
- [Quality Review Script](.github/scripts/quality-review.sh)
- [Security Review Script](.github/scripts/security-review.sh)

## Support

For questions or issues:

1. Check this documentation
2. Review existing PRs for examples
3. Ask in PR comments
4. Contact maintainers

---

**Last Updated**: 2025-11-07
**Version**: 1.0.0
