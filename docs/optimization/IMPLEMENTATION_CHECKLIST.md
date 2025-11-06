# GitHub Actions Optimization - Implementation Checklist

## Overview

Use this checklist to track progress implementing workflow optimizations.

**Expected Results**:
- ✅ 45-60% faster workflow execution
- ✅ 30-40% reduction in GitHub Actions minutes
- ✅ 50% reduction in redundant code
- ✅ Better reliability and maintainability

---

## Phase 1: Quick Wins (Week 1)

### 1.1 Dependency Caching

**Target**: All 10 workflow files
**Time**: 30 minutes
**Impact**: 10-15% cost reduction

- [ ] Update `rss-complete-pipeline.yml`
  - [ ] Add cache after Python setup
  - [ ] Test with workflow_dispatch
  - [ ] Verify cache hit in logs

- [ ] Update `deduplication-check.yml`
  - [ ] Add cache after Python setup
  - [ ] Test with workflow_dispatch
  - [ ] Verify cache hit in logs

- [ ] Update `refresh-titles.yml`
  - [ ] Add cache after Python setup
  - [ ] Test with workflow_dispatch
  - [ ] Verify cache hit in logs

- [ ] Update `force-refresh-now.yml`
  - [ ] Add cache after Python setup
  - [ ] Test with workflow_dispatch
  - [ ] Verify cache hit in logs

- [ ] Update `code-review-swarm.yml`
  - [ ] Add cache after Python setup
  - [ ] Test on a test PR
  - [ ] Verify cache hit in logs

**Validation**:
```bash
# Check cache hit rate
gh run view <run-id> --log | grep "Cache hit"
```

---

### 1.2 Workflow Timeouts

**Target**: All jobs in all workflows
**Time**: 15 minutes
**Impact**: Prevents runaway workflows

- [ ] Add timeouts to `rss-complete-pipeline.yml`
  - [ ] Main job: 20 minutes
  - [ ] Validation jobs: 5 minutes each

- [ ] Add timeouts to `deduplication-check.yml`
  - [ ] Main job: 10 minutes

- [ ] Add timeouts to `refresh-titles.yml`
  - [ ] Main job: 15 minutes

- [ ] Add timeouts to `force-refresh-now.yml`
  - [ ] Main job: 15 minutes

- [ ] Add timeouts to `code-review-swarm.yml`
  - [ ] Triage job: 5 minutes
  - [ ] Review jobs: 10 minutes each

**Validation**:
```yaml
# Verify timeout is set
cat .github/workflows/rss-complete-pipeline.yml | grep "timeout-minutes"
```

---

### 1.3 Fixed Artifact Naming

**Target**: All artifact uploads
**Time**: 15 minutes
**Impact**: Reduces storage costs, easier retrieval

- [ ] Update `rss-complete-pipeline.yml`
  - [ ] Change `database-backup-${{ github.run_number }}` → `database-backup-latest`
  - [ ] Add `overwrite: true`
  - [ ] Test artifact upload/download

- [ ] Update `deduplication-check.yml`
  - [ ] Change artifact naming scheme
  - [ ] Add `overwrite: true`
  - [ ] Test artifact upload/download

- [ ] Update `refresh-titles.yml`
  - [ ] Change artifact naming scheme
  - [ ] Add `overwrite: true`
  - [ ] Test artifact upload/download

**Validation**:
```bash
# Check artifact count (should decrease over time)
gh api repos/:owner/:repo/actions/artifacts | jq '.total_count'
```

---

### 1.4 Shallow Clones for Validation

**Target**: Validation-only jobs
**Time**: 5 minutes
**Impact**: 5-10 seconds per checkout

- [ ] Update validation jobs to use `fetch-depth: 1`
- [ ] Keep `fetch-depth: 0` for jobs that need git history
- [ ] Test validation passes with shallow clone

**Example**:
```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 1  # For validation jobs only
```

---

### 1.5 Phase 1 Testing

- [ ] Run `rss-complete-pipeline.yml` manually
  - [ ] Verify all steps pass
  - [ ] Check runtime improvement
  - [ ] Confirm cache is working

- [ ] Run `deduplication-check.yml` manually
  - [ ] Verify all steps pass
  - [ ] Check runtime improvement
  - [ ] Confirm artifacts created

- [ ] Trigger code review on test PR
  - [ ] Verify reviews post correctly
  - [ ] Check parallel execution
  - [ ] Confirm no regressions

**Success Criteria**:
- ✅ All workflows pass tests
- ✅ Cache hit rate > 80%
- ✅ No workflow failures
- ✅ Artifacts created correctly

---

## Phase 2: Parallel Execution (Week 2)

### 2.1 Parallel Validation in RSS Pipeline

**Target**: `rss-complete-pipeline.yml`
**Time**: 1 hour
**Impact**: 1.5-2 minutes saved per run

- [ ] Create validation matrix
  ```yaml
  strategy:
    fail-fast: false
    matrix:
      validation: [imports, rss, scraper, database]
  ```

- [ ] Move validation steps to matrix job
- [ ] Add case statement for validation types
- [ ] Test all validation types pass
- [ ] Verify parallel execution in Actions UI

**Validation**:
```bash
# Check that jobs run in parallel
gh run view <run-id> --json jobs | jq '.jobs[] | select(.name | contains("Validation")) | .name'
```

---

### 2.2 Parallel Reporting

**Target**: `rss-complete-pipeline.yml`
**Time**: 30 minutes
**Impact**: 20-30 seconds saved

- [ ] Identify independent reporting tasks
  - [ ] Articles by date generation
  - [ ] Duplicate report
  - [ ] Analysis summary

- [ ] Wrap tasks in background jobs
  ```bash
  { task1 } &
  { task2 } &
  wait
  ```

- [ ] Test all reports generated correctly
- [ ] Verify runtime improvement

---

### 2.3 Phase 2 Testing

- [ ] Test parallel validation on test branch
- [ ] Monitor for flaky tests
- [ ] Compare runtime before/after
- [ ] Check for any race conditions

**Success Criteria**:
- ✅ All validations pass in parallel
- ✅ 1.5-2 min time savings achieved
- ✅ No increase in failure rate
- ✅ Reports generated correctly

---

## Phase 3: Reusable Workflows (Week 3)

### 3.1 Implement Reusable Setup

**Target**: All workflows
**Time**: 2 hours
**Impact**: 50-60 lines eliminated per workflow

- [ ] Verify `_reusable-setup.yml` exists
- [ ] Test reusable workflow in isolation

- [ ] Refactor `rss-complete-pipeline.yml`
  - [ ] Replace setup steps with reusable call
  - [ ] Test workflow passes
  - [ ] Verify cache still works

- [ ] Refactor `deduplication-check.yml`
  - [ ] Replace setup steps with reusable call
  - [ ] Test workflow passes
  - [ ] Verify cache still works

- [ ] Refactor remaining workflows (3-5 more)
  - [ ] Replace setup steps
  - [ ] Test each workflow
  - [ ] Document any workflow-specific needs

**Validation**:
```bash
# Count lines of code before/after
wc -l .github/workflows/*.yml
```

---

### 3.2 Extract Review Scripts

**Target**: `code-review-swarm.yml`
**Time**: 1 hour
**Impact**: Better maintainability

- [ ] Verify scripts exist:
  - [ ] `.github/scripts/security-review.sh`
  - [ ] `.github/scripts/quality-review.sh`
  - [ ] `.github/scripts/python-review.sh`

- [ ] Make scripts executable
  ```bash
  chmod +x .github/scripts/*.sh
  ```

- [ ] Update workflow to call scripts
- [ ] Test each review agent
- [ ] Verify review output format

---

### 3.3 Phase 3 Testing

- [ ] Test all refactored workflows
- [ ] Verify no functionality lost
- [ ] Check that setup is consistent
- [ ] Monitor first week of scheduled runs

**Success Criteria**:
- ✅ All workflows use reusable setup
- ✅ Setup is consistent across workflows
- ✅ No increase in failure rate
- ✅ Easier to maintain

---

## Phase 4: Advanced Optimizations (Week 4)

### 4.1 Path Filters

**Target**: All PR-triggered workflows
**Time**: 30 minutes
**Impact**: 40-50% fewer unnecessary runs

- [ ] Add path filters to code review workflow
  ```yaml
  on:
    pull_request:
      paths:
        - 'src/**/*.py'
        - 'tests/**/*.py'
      paths-ignore:
        - 'docs/**'
        - '**.md'
  ```

- [ ] Test with documentation-only PR
- [ ] Test with code change PR
- [ ] Verify correct behavior

---

### 4.2 Artifact Cleanup Automation

**Target**: New workflow or existing cleanup job
**Time**: 1 hour
**Impact**: Storage cost reduction

- [ ] Create artifact cleanup job
- [ ] Set cleanup policy (30 days)
- [ ] Keep 'latest' artifacts
- [ ] Test cleanup logic
- [ ] Schedule weekly execution

**Example**:
```yaml
- name: Cleanup old artifacts
  uses: actions/github-script@v7
  with:
    script: |
      const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
      // Delete artifacts older than cutoff
```

---

### 4.3 Cost Tracking Workflow

**Target**: New workflow
**Time**: 1 hour
**Impact**: Cost visibility

- [ ] Create `workflow-cost-analysis.yml`
- [ ] Schedule weekly execution
- [ ] Generate usage report
- [ ] Post to GitHub Summary
- [ ] Track metrics over time

---

### 4.4 Phase 4 Testing

- [ ] Test path filters with various PR types
- [ ] Verify artifact cleanup doesn't delete current artifacts
- [ ] Review cost tracking report accuracy
- [ ] Monitor for any unintended side effects

**Success Criteria**:
- ✅ Workflows skip when appropriate
- ✅ Old artifacts cleaned up automatically
- ✅ Cost tracking provides useful insights
- ✅ No loss of critical artifacts

---

## Performance Tracking

### Baseline Metrics (Before Optimization)

Record these before starting:

```markdown
| Workflow | Avg Runtime | Runs/Week | Minutes/Week |
|----------|-------------|-----------|--------------|
| RSS Pipeline | 5-7 min | 7 | 35-49 min |
| Code Review | 3-4 min | 5 | 15-20 min |
| Deduplication | 1.5-2 min | 1 | 1.5-2 min |
| Refresh Titles | 4-5 min | 1 | 4-5 min |
| Force Refresh | 5-6 min | 2 | 10-12 min |
| **TOTAL** | | | **65-88 min/week** |
```

### Track Progress

Update after each phase:

```markdown
| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 | Final |
|--------|----------|---------|---------|---------|-------|
| Avg Runtime | 5-7 min | ? | ? | ? | ? |
| Cache Hit % | 0% | ? | ? | ? | ? |
| Weekly Minutes | 65-88 | ? | ? | ? | ? |
| Est. Cost | $7-9 | ? | ? | ? | ? |
```

---

## Monitoring Dashboard

Create this in your repository README or Wiki:

```markdown
## 📊 GitHub Actions Performance

**Last Updated**: 2025-10-29

### Current Metrics
- **Average Runtime**: X minutes (↓ Y% from baseline)
- **Cache Hit Rate**: Z%
- **Weekly Minutes**: A minutes (↓ B% from baseline)
- **Estimated Monthly Cost**: $C (↓ D% from baseline)

### Recent Optimizations
- ✅ Added dependency caching (10-15% improvement)
- ✅ Implemented parallel validation (15-20% improvement)
- ✅ Fixed artifact naming (storage cost reduction)
- 🚧 In Progress: Reusable workflows
- ⏳ Planned: Path filters

### Workflow Health
| Workflow | Status | Last Run | Success Rate |
|----------|--------|----------|--------------|
| RSS Pipeline | ✅ | 2h ago | 98% |
| Code Review | ✅ | 4h ago | 95% |
| Deduplication | ✅ | 1d ago | 100% |
```

---

## Issue Tracking

Use this template to track any issues:

```markdown
### Issue: [Description]
**Date**: YYYY-MM-DD
**Workflow**: [workflow name]
**Phase**: [1-4]

**Problem**:
[Describe the issue]

**Impact**:
- [ ] Workflow fails
- [ ] Slower runtime
- [ ] Increased costs
- [ ] Other: [specify]

**Solution**:
[What was done to fix it]

**Verification**:
[How to verify it's fixed]

**Status**: [ ] Open / [ ] Resolved
```

---

## Sign-Off Checklist

### Phase 1 Sign-Off
- [ ] All workflows have dependency caching
- [ ] All jobs have timeouts
- [ ] Artifact naming is fixed
- [ ] 10-15% cost reduction achieved
- [ ] No increase in failure rate

**Signed**: _________________ Date: _________

---

### Phase 2 Sign-Off
- [ ] Parallel validation implemented
- [ ] Parallel reporting implemented
- [ ] 1.5-2 min time savings achieved
- [ ] No flaky tests introduced

**Signed**: _________________ Date: _________

---

### Phase 3 Sign-Off
- [ ] Reusable workflow implemented
- [ ] All workflows refactored
- [ ] Review scripts extracted
- [ ] 50+ lines removed per workflow
- [ ] Maintenance is easier

**Signed**: _________________ Date: _________

---

### Phase 4 Sign-Off
- [ ] Path filters implemented
- [ ] Artifact cleanup automated
- [ ] Cost tracking operational
- [ ] 40-50% total improvement achieved
- [ ] Documentation complete

**Signed**: _________________ Date: _________

---

## Rollback Procedures

If needed, follow these steps to rollback:

### Rollback Phase 4
1. Remove path filters from workflows
2. Disable artifact cleanup job
3. Remove cost tracking workflow

### Rollback Phase 3
1. Restore original setup steps in workflows
2. Keep extracted scripts (they don't hurt)
3. Revert reusable workflow usage

### Rollback Phase 2
1. Replace matrix validation with sequential steps
2. Remove background job parallelization
3. Restore original structure

### Rollback Phase 1
1. Remove cache steps from workflows
2. Remove timeout-minutes settings
3. Restore original artifact naming
4. Remove fetch-depth settings

**Rollback Command**:
```bash
# Revert specific workflow to previous version
git checkout <commit-hash> -- .github/workflows/<workflow-name>.yml
git commit -m "Rollback: <workflow-name> optimization"
git push
```

---

## Resources

- **Full Documentation**: `docs/optimization/GITHUB_ACTIONS_OPTIMIZATION.md`
- **Quick Reference**: `docs/optimization/GITHUB_ACTIONS_QUICK_REFERENCE.md`
- **Reusable Workflow**: `.github/workflows/_reusable-setup.yml`
- **Review Scripts**: `.github/scripts/*.sh`

---

## Support

For questions or issues:
1. Check documentation in `docs/optimization/`
2. Review workflow logs in GitHub Actions
3. Test changes in feature branch first
4. Create GitHub issue if problem persists

---

**Project**: RSS Analyzer
**Document Version**: 1.0
**Last Updated**: 2025-10-29
**Owner**: DevOps Team
