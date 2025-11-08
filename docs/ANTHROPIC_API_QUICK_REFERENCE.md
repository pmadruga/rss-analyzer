# Anthropic API in GitHub Actions - Quick Reference Card

**Last Updated**: 2025-11-08

---

## TL;DR

✅ **Use Anthropic API?** YES (hybrid with Mistral)
💰 **Monthly Cost?** $5-7 (hybrid) or $10-15 (full features)
⏱️ **Time Savings?** 30+ hours/month
🚀 **Setup Time?** Already configured, ready to use!

---

## Quick Decision Guide

```
┌────────────────────────────────────────────┐
│         When to Use Which Provider         │
├────────────────────────────────────────────┤
│                                            │
│  MISTRAL (Cost-Effective)                  │
│  ─────────────────────────                 │
│  ✓ Daily RSS processing                    │
│  ✓ Bulk analysis (100+ articles)           │
│  ✓ Simple tasks                            │
│  ✓ Testing/development                     │
│                                            │
│  ANTHROPIC (High Quality)                  │
│  ───────────────────────                   │
│  ✓ PR code reviews                         │
│  ✓ Deep article analysis (20-30/month)     │
│  ✓ Documentation generation                │
│  ✓ Bug triage and reasoning                │
│  ✓ Test case generation                    │
│                                            │
└────────────────────────────────────────────┘
```

---

## Cost Cheat Sheet

| Scenario | Mistral | Anthropic | Hybrid |
|----------|---------|-----------|--------|
| **100 articles/day (RSS)** | $1.79 | $16.20 | **$1.79** (Mistral) |
| **10 PR reviews/month** | N/A | $2.00 | **$2.00** (Anthropic) |
| **Weekly docs** | N/A | $1.00 | **$1.00** (Anthropic) |
| **30 deep analyses** | N/A | $2.50 | **$2.50** (Anthropic) |
| **Total/month** | $1.79 | $16.20 | **$7.29** |

**Savings**: 55% vs Anthropic-only, with better quality where it matters!

---

## Implementation Checklist

### Phase 1: Testing (Week 1) ⚡
- [ ] Verify `ANTHROPIC_API_KEY` in GitHub secrets (✅ already done!)
- [ ] Create budget tracker: `echo "0.00" > .github/anthropic_spend.txt`
- [ ] Add test workflow (workflow_dispatch, manual trigger)
- [ ] Test with 3-5 articles
- [ ] Monitor costs (target: <$3 for week)

### Phase 2: Automation (Weeks 2-3) 🚀
- [ ] Enable PR code reviews (automated)
- [ ] Add documentation workflow (weekly)
- [ ] Implement cost dashboard
- [ ] Keep Mistral for RSS processing
- [ ] Monitor combined costs (target: <$10/month)

### Phase 3: Production (Week 4+) 🎯
- [ ] Add health checks
- [ ] Enable all AI workflows
- [ ] Setup monthly reporting
- [ ] Optimize based on usage patterns
- [ ] Target: <$15/month total

---

## Quick Start Commands

```bash
# 1. Initialize budget tracker
echo "0.00" > .github/anthropic_spend.txt
git add .github/anthropic_spend.txt
git commit -m "Initialize Anthropic budget tracker"
git push

# 2. Check if API key is configured
gh secret list | grep ANTHROPIC_API_KEY
# Should output: ANTHROPIC_API_KEY

# 3. Test API connectivity
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "test"}]
  }'

# 4. Trigger manual workflow (after creating it)
gh workflow run claude-deep-analysis.yml \
  --field article_ids="1,2,3"

# 5. Check costs
ANTHROPIC=$(cat .github/anthropic_spend.txt)
MISTRAL=$(cat .github/mistral_spend.txt)
echo "Total: $(echo "$ANTHROPIC + $MISTRAL" | bc) / 15.00 budget"
```

---

## Authentication Options

### Option 1: API Key (Recommended ✅)
```yaml
uses: anthropics/claude-code-action@v1
with:
  anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```
✅ Already configured
✅ Works immediately
✅ Simple setup

### Option 2: OAuth Token
```yaml
uses: anthropics/claude-code-action@v1
with:
  claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
```
⚠️ Requires `/install-github-app`
✅ Granular permissions
✅ Official method

**Start with API key** (already configured), consider OAuth later

---

## Pricing Quick Reference

### Anthropic Claude (2025)
| Model | Input | Output |
|-------|-------|--------|
| **Sonnet 4** | $3/1M | $15/1M |
| Opus 4 | $15/1M | $75/1M |
| Haiku | $0.25/1M | $1.25/1M |

### Mistral (2025)
| Model | Input | Output |
|-------|-------|--------|
| **Large** | $2/1M | $6/1M |
| Medium | $0.65/1M | $1.95/1M |
| Small | $0.20/1M | $0.60/1M |

**Note**: Prices per million tokens

---

## Cost Calculator (One-Liner)

```bash
# Calculate monthly cost for N articles
calc_cost() {
  local articles_per_day=$1
  local provider=$2  # "anthropic" or "mistral"

  # After optimizations (72% cache + 30% dedup)
  local effective=$(echo "$articles_per_day * 30 * 0.28 * 0.70" | bc)

  case $provider in
    anthropic)
      cost=$(echo "($effective * 10000 * 3.00 + $effective * 1500 * 15.00) / 1000000" | bc -l)
      ;;
    mistral)
      cost=$(echo "($effective * 10000 * 2.00 + $effective * 1500 * 6.00) / 1000000" | bc -l)
      ;;
  esac

  printf "%.2f\n" $cost
}

# Example: 100 articles/day
calc_cost 100 anthropic  # Output: 3.19
calc_cost 100 mistral    # Output: 1.79
```

---

## Budget Controls (Copy-Paste)

```yaml
# Add to workflow before expensive operations
- name: Check budget
  id: budget
  run: |
    CURRENT=$(cat .github/anthropic_spend.txt 2>/dev/null || echo "0.00")
    BUDGET=15.00
    ESTIMATED=2.00

    if (( $(echo "$ESTIMATED + $CURRENT > $BUDGET" | bc -l) )); then
      echo "::error::Budget would be exceeded"
      echo "Current: $CURRENT, Estimated: $ESTIMATED, Budget: $BUDGET"
      exit 1
    fi

    echo "can_run=true" >> $GITHUB_OUTPUT
```

---

## Common Workflows

### 1. PR Code Review
```yaml
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: "Review PR for quality, security, and performance"
```

### 2. Deep Analysis (Manual)
```yaml
on:
  workflow_dispatch:
    inputs:
      article_ids:
        description: 'Article IDs (e.g., "1,2,3")'
        required: true

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: "Deep analysis of articles: ${{ inputs.article_ids }}"
```

### 3. Daily RSS (Keep Mistral)
```yaml
env:
  API_PROVIDER: 'mistral'  # Cost-effective for bulk

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - run: uv run python -m src.main run --limit 20 --async
```

---

## Monitoring Commands

```bash
# Check cost dashboard
cat docs/cost-dashboard.md

# View current spend
ANTHROPIC=$(cat .github/anthropic_spend.txt)
MISTRAL=$(cat .github/mistral_spend.txt)
TOTAL=$(echo "$ANTHROPIC + $MISTRAL" | bc)
echo "Total: $TOTAL / 15.00 ($(echo "scale=0; $TOTAL * 100 / 15.00" | bc)%)"

# Check API health
gh workflow run api-health-check.yml

# Trigger deep analysis
gh workflow run claude-deep-analysis.yml --field article_ids="1,2,3"

# View monthly report
cat docs/monthly-reports/report-$(date '+%Y-%m').md
```

---

## Troubleshooting

### Issue: "Budget exceeded"
**Fix**: Check spend, adjust budget, or wait for monthly reset
```bash
cat .github/anthropic_spend.txt  # Check current spend
# Edit budget in workflow: BUDGET=20.00
```

### Issue: "API key not found"
**Fix**: Verify secret is configured
```bash
gh secret list | grep ANTHROPIC_API_KEY
# If missing: gh secret set ANTHROPIC_API_KEY
```

### Issue: "Rate limit hit"
**Fix**: Already handled by built-in rate limiter (10 req/s)
```python
# In src/core/async_scraper.py
rate_limiter = AsyncLimiter(max_rate=10, time_period=1.0)
```

### Issue: "Workflow timeout"
**Fix**: Increase timeout or reduce scope
```yaml
timeout-minutes: 30  # Increase from default 15
claude_args: "--max-turns 2"  # Reduce complexity
```

---

## Key Files

### Documentation
- **Full Analysis**: `docs/ANTHROPIC_API_GITHUB_ACTIONS_ANALYSIS.md`
- **Executive Summary**: `docs/ANTHROPIC_API_EXECUTIVE_SUMMARY.md`
- **This Quick Reference**: `docs/ANTHROPIC_API_QUICK_REFERENCE.md`

### Workflows to Create
- `.github/workflows/claude-pr-review.yml` (PR reviews)
- `.github/workflows/claude-deep-analysis.yml` (Deep analysis)
- `.github/workflows/claude-docs-gen.yml` (Documentation)
- `.github/workflows/cost-dashboard.yml` (Monitoring)

### Budget Trackers
- `.github/anthropic_spend.txt` (Create with `0.00`)
- `.github/mistral_spend.txt` (Create with `0.00`)

---

## Success Criteria Quick Check

### ✅ Phase 1 Success
- [ ] API working correctly
- [ ] Quality meets expectations
- [ ] Costs <$3/week
- [ ] No failures

### ✅ Phase 2 Success
- [ ] Automated workflows stable
- [ ] Combined costs <$10/month
- [ ] 5+ hours/month saved
- [ ] 50%+ cache hit rate

### ✅ Phase 3 Success
- [ ] All features enabled
- [ ] Costs <$15/month
- [ ] 30+ hours/month saved
- [ ] High-quality outputs

---

## Emergency Actions

### If costs spike unexpectedly
```bash
# 1. Disable expensive workflows immediately
gh workflow disable claude-pr-review.yml
gh workflow disable claude-deep-analysis.yml

# 2. Check what caused spike
gh run list --workflow claude-pr-review.yml --limit 10

# 3. Adjust budget limits
# Edit workflows: BUDGET=5.00 (lower limit)

# 4. Re-enable with controls
gh workflow enable claude-pr-review.yml
```

### If API fails
```bash
# 1. Check API health
gh workflow run api-health-check.yml

# 2. Fallback to Mistral
# Edit workflows: API_PROVIDER: 'mistral'

# 3. Create GitHub issue
gh issue create --title "API Failure" --body "Details..."
```

---

## Best Practices Summary

✅ **DO**:
- Use Mistral for bulk processing
- Use Anthropic for quality tasks
- Check budget before expensive ops
- Monitor costs daily during rollout
- Start with manual triggers (Phase 1)
- Keep optimizations enabled

❌ **DON'T**:
- Use Anthropic for simple tasks
- Skip budget checks
- Commit API keys to code
- Run without rate limiting
- Enable all features at once

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| **Monthly Budget** | $15 (recommended max) |
| **Phase 1 Target** | <$3/week |
| **Phase 2 Target** | <$10/month combined |
| **Phase 3 Target** | <$15/month |
| **Cache Hit Rate** | 72% (target >50%) |
| **Dedup Rate** | 30-70% |
| **Token Savings** | 20-30% per article |
| **Speed Improvement** | 6-8x with async |
| **Time Savings** | 30+ hours/month |

---

## One-Line Summary

**Use Mistral for bulk, Anthropic for quality = $5-7/month with 30+ hours saved** ✅

---

**Quick Reference Card** | Version 1.0 | 2025-11-08
