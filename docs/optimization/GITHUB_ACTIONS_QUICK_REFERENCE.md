# GitHub Actions Optimization - Quick Reference

## Overview

This document provides quick access to the most impactful optimizations for the RSS Analyzer GitHub Actions workflows.

**Quick Stats**:
- **Time Savings**: 45-60% reduction in workflow runtime
- **Cost Savings**: 30-40% reduction in GitHub Actions minutes
- **Effort**: Low to medium implementation complexity

---

## 🚀 Quick Wins (Implement First)

### 1. Add Dependency Caching (5 minutes)

**Impact**: 30-45 seconds per workflow run, ~10-15% cost reduction

Add this to **every workflow** after the Python setup step:

```yaml
- name: 📦 Cache dependencies
  uses: actions/cache@v4
  with:
    path: |
      .venv
      ~/.cache/uv
    key: ${{ runner.os }}-uv-v1-${{ hashFiles('**/pyproject.toml', '**/uv.lock') }}
    restore-keys: |
      ${{ runner.os }}-uv-v1-
```

**Files to update**:
- ✅ `.github/workflows/rss-complete-pipeline.yml`
- ✅ `.github/workflows/deduplication-check.yml`
- ✅ `.github/workflows/refresh-titles.yml`
- ✅ `.github/workflows/force-refresh-now.yml`
- ✅ `.github/workflows/code-review-swarm.yml`

---

### 2. Use Reusable Setup Workflow (10 minutes)

**Impact**: Eliminates 50-60 lines per workflow, ensures consistency

Replace setup steps with:

```yaml
jobs:
  setup:
    uses: ./.github/workflows/_reusable-setup.yml
    with:
      python-version: '3.11'
      fetch-depth: 0
      install-dependencies: true

  actual-work:
    needs: setup
    runs-on: ubuntu-latest
    steps:
      # Your workflow steps here
```

**Reusable workflow already created**: `.github/workflows/_reusable-setup.yml`

---

### 3. Parallelize Validation Steps (15 minutes)

**Impact**: 1.5-2 minutes saved per run, ~15-20% reduction

In `rss-complete-pipeline.yml`, replace sequential tests with:

```yaml
jobs:
  validate:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        validation: [imports, rss, scraper, database]

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Run validation - ${{ matrix.validation }}
      run: |
        case "${{ matrix.validation }}" in
          imports) uv run python tools/validate_imports.py ;;
          rss) uv run python -c "from src.core.rss_parser import RSSParser; ..." ;;
          scraper) uv run python -c "from src.core.scraper import WebScraper; ..." ;;
          database) uv run python -c "from src.core.database import DatabaseManager; ..." ;;
        esac
```

---

### 4. Fix Artifact Naming (2 minutes)

**Impact**: Prevents artifact proliferation, reduces storage costs

Replace:
```yaml
name: database-backup-${{ github.run_number }}
```

With:
```yaml
name: database-backup-latest
overwrite: true
```

**Files to update**:
- ✅ All workflows that upload artifacts

---

### 5. Add Workflow Timeouts (2 minutes)

**Impact**: Prevents runaway workflows from consuming excess minutes

Add to every job:

```yaml
jobs:
  my-job:
    runs-on: ubuntu-latest
    timeout-minutes: 20  # Adjust based on expected runtime
```

**Recommended timeouts**:
- RSS Complete Pipeline: 20 minutes
- Deduplication: 10 minutes
- Code Review: 15 minutes
- Refresh Titles: 15 minutes

---

## 📊 Impact Summary

| Optimization | Time Saved | Cost Reduction | Effort |
|--------------|------------|----------------|--------|
| Dependency caching | 30-45 sec | 10-15% | Low |
| Reusable workflows | N/A | Maintenance | Low |
| Parallel validation | 1.5-2 min | 15-20% | Medium |
| Fixed artifact names | N/A | Storage costs | Low |
| Workflow timeouts | Prevents runaway | Variable | Low |
| **Total** | **2-3 min/run** | **30-40%** | |

---

## 🔄 Migration Guide

### Step 1: Update One Workflow (15 minutes)

Start with `rss-complete-pipeline.yml`:

1. **Add dependency caching** (after Python setup)
2. **Add workflow timeout** (at job level)
3. **Fix artifact names** (replace run_number with 'latest')
4. **Test the workflow** (run manually via workflow_dispatch)

### Step 2: Verify Results (5 minutes)

Check that:
- ✅ Cache is being restored (check workflow logs)
- ✅ Workflow completes faster
- ✅ Artifacts are created with correct names

### Step 3: Roll Out to Other Workflows (30 minutes)

Apply same changes to:
- ✅ `deduplication-check.yml`
- ✅ `refresh-titles.yml`
- ✅ `force-refresh-now.yml`
- ✅ `code-review-swarm.yml`

### Step 4: Implement Parallel Validation (30 minutes)

1. Replace sequential validation with matrix strategy
2. Test thoroughly
3. Monitor for any issues

### Step 5: Implement Reusable Workflow (1 hour)

1. Use `_reusable-setup.yml` (already created)
2. Refactor workflows one by one
3. Test each refactored workflow

---

## 🧪 Testing Checklist

Before deploying to production:

- [ ] Test with workflow_dispatch (manual trigger)
- [ ] Verify cache hit/miss in logs
- [ ] Check that all validations pass
- [ ] Confirm artifacts are created correctly
- [ ] Monitor first 3 scheduled runs
- [ ] Compare runtime before/after
- [ ] Check Actions usage page for cost impact

---

## 📈 Monitoring After Deployment

Track these metrics weekly:

```yaml
# Workflow Performance Dashboard
| Workflow | Before | After | Improvement |
|----------|--------|-------|-------------|
| RSS Pipeline | 5-7 min | ? min | ?% |
| Code Review | 3-4 min | ? min | ?% |
| Deduplication | 1.5-2 min | ? min | ?% |
```

**How to measure**:
1. Go to Actions tab in GitHub
2. Click on workflow
3. Check "Duration" column for recent runs
4. Compare with baseline (current average)

---

## 🚨 Rollback Plan

If issues occur:

### Issue: Cache Corruption
```bash
# Clear cache manually
gh cache delete <cache-key>
```

### Issue: Parallel Jobs Failing
```yaml
# Revert to sequential
# Remove matrix strategy, restore sequential steps
```

### Issue: Artifact Not Found
```yaml
# Revert artifact naming
name: database-backup-${{ github.run_number }}
# Remove: overwrite: true
```

---

## 💡 Advanced Optimizations (Phase 2)

Once quick wins are deployed and stable:

### 1. Parallel Reporting (Medium Effort)
Run report generation tasks in parallel using background jobs:

```bash
{
  uv run python tools/generate_articles_by_date.py
} &

{
  uv run python tools/check_duplicates.py --report > output/report.txt
} &

wait  # Wait for all background jobs
```

**Impact**: 20-30 seconds saved

### 2. Conditional Path Filters (Low Effort)
Skip workflows when only docs are changed:

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

**Impact**: 40-50% fewer unnecessary runs

### 3. Artifact Cleanup Automation (Medium Effort)
See full script in optimization documentation:

```yaml
- name: Cleanup old artifacts
  uses: actions/github-script@v7
  with:
    script: |
      # Delete artifacts older than 30 days
```

**Impact**: Storage cost reduction

---

## 📚 Resources

- **Full Documentation**: `docs/optimization/GITHUB_ACTIONS_OPTIMIZATION.md`
- **Reusable Workflow**: `.github/workflows/_reusable-setup.yml`
- **Review Scripts**: `.github/scripts/*.sh`
- **GitHub Actions Docs**: https://docs.github.com/en/actions

---

## 🎯 Implementation Priorities

**Week 1 - High Priority**:
1. ✅ Add dependency caching (all workflows)
2. ✅ Add workflow timeouts (all workflows)
3. ✅ Fix artifact naming (all workflows)
4. ✅ Test and verify changes

**Week 2 - Medium Priority**:
1. ✅ Implement parallel validation (RSS pipeline)
2. ✅ Test parallel execution
3. ✅ Monitor for issues

**Week 3 - Enhancement**:
1. ✅ Implement reusable workflow
2. ✅ Refactor workflows one by one
3. ✅ Add path filters

**Week 4 - Optimization**:
1. ✅ Implement parallel reporting
2. ✅ Add artifact cleanup
3. ✅ Performance monitoring

---

## 🔍 Common Issues & Solutions

### Issue: Cache Miss Every Time
**Solution**: Check that cache key matches restore-keys pattern

### Issue: Parallel Jobs Flaky
**Solution**: Add `fail-fast: false` to matrix strategy

### Issue: Timeout Too Short
**Solution**: Increase timeout-minutes based on actual runtime

### Issue: Artifact Upload Fails
**Solution**: Ensure directory exists before upload

---

## 📞 Support

For questions or issues:
1. Check full documentation in `docs/optimization/`
2. Review workflow logs in GitHub Actions
3. Test changes in a feature branch first
4. Monitor Actions usage page for metrics

---

**Last Updated**: 2025-10-29
**Version**: 1.0
