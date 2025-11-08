# Claude Code GitHub Actions - Quick Start Guide

## For RSS Article Processing Automation

---

## 5-Minute Setup

### Step 1: Add API Key to GitHub Secrets

1. Get your Anthropic API key from <https://console.anthropic.com>
2. Go to your repository → Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `ANTHROPIC_API_KEY`
5. Value: `sk-ant-...`

### Step 2: Create Basic Workflow

Create `.github/workflows/claude-daily-analysis.yml`:

```yaml
name: Daily Article Analysis

on:
  schedule:
    - cron: '0 9 * * *'  # 9 AM UTC daily
  workflow_dispatch:      # Manual trigger

permissions:
  contents: write

jobs:
  analyze:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python & uv
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync

      - name: Process RSS articles (with optimizations)
        run: |
          mkdir -p data logs output
          if [ -f "data/articles.db" ]; then
            sqlite3 data/articles.db "PRAGMA journal_mode = WAL;"
          fi

          # Async processing for 6-8x speedup
          MAX_CONCURRENT_ARTICLES=8 \
          uv run python -m src.main run --limit 10 --async

      - name: Analyze with Claude
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            Review articles in output/articles_export.json.
            Create a daily summary in docs/daily-summary.md with:
            - Top 3 most significant papers
            - Key research trends
            - Recommended reading order
          claude_args: "--max-turns 2"

      - name: Commit results
        run: |
          git config user.name "Claude Analysis Bot"
          git config user.email "bot@github.com"
          git add docs/
          git commit -m "Daily analysis: $(date +%Y-%m-%d)" || echo "No changes"
          git push
```

### Step 3: Test the Workflow

1. Go to Actions tab in your repository
2. Click "Daily Article Analysis"
3. Click "Run workflow" → "Run workflow"
4. Wait ~10 minutes for completion

### Step 4: Verify Results

Check for new file: `docs/daily-summary.md`

---

## Cost Control (Critical!)

### Set Monthly Budget

Create `.github/workflows/budget-check.yml`:

```yaml
name: Budget Check

on:
  schedule:
    - cron: '0 0 1 * *'  # First day of month

jobs:
  reset-budget:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4

      - name: Reset monthly budget
        run: |
          echo "0.00" > .github/monthly_spend.txt
          git config user.name "Budget Bot"
          git config user.email "budget@github.com"
          git add .github/monthly_spend.txt
          git commit -m "Reset monthly budget"
          git push
```

Update main workflow to check budget:

```yaml
      - name: Check budget (before Claude)
        id: budget
        run: |
          CURRENT=$(cat .github/monthly_spend.txt 2>/dev/null || echo "0.00")
          BUDGET=15.00  # $15/month limit
          REMAINING=$(echo "$BUDGET - $CURRENT" | bc)
          ESTIMATED=0.50  # $0.50 per run

          if (( $(echo "$ESTIMATED > $REMAINING" | bc -l) )); then
            echo "::error::Budget exceeded! Remaining: $REMAINING"
            exit 1
          fi

      - name: Update budget (after Claude)
        if: always()
        run: |
          CURRENT=$(cat .github/monthly_spend.txt 2>/dev/null || echo "0.00")
          NEW=$(echo "$CURRENT + 0.50" | bc)
          echo "$NEW" > .github/monthly_spend.txt
          git add .github/monthly_spend.txt
          git commit -m "Budget: \$$NEW" && git push || true
```

---

## Expected Costs

### With Optimizations (Recommended)

| Scenario | Articles/Day | Monthly Cost | Annual Cost |
|----------|-------------|-------------|-------------|
| Light Usage | 10 | $3.19 | $38.28 |
| Moderate Usage | 50 | $12.50 | $150.00 |
| Heavy Usage | 100 | $15.00 | $180.00 |

**Includes**:

- ✅ Two-tier caching (72% savings)
- ✅ Hash-based dedup (30-70% savings)
- ✅ Token truncation (20-30% savings)
- ✅ Async processing (6-8x faster)

### Without Optimizations (Not Recommended)

| Scenario | Articles/Day | Monthly Cost | Annual Cost |
|----------|-------------|-------------|-------------|
| Light Usage | 10 | $16.20 | $194.40 |
| Moderate Usage | 50 | $81.00 | $972.00 |
| Heavy Usage | 100 | $148.80 | $1,785.60 |

**💡 Tip**: Always use `--async` flag and enable caching!

---

## Optimization Checklist

Before running Claude Code workflows, ensure:

### ✅ RSS Analyzer Optimizations Enabled

```bash
# Check async processing
grep -q "use_async_processing: true" config/config.yaml

# Check caching
grep -q "enable_caching: true" config/config.yaml

# Check token truncation
grep -q "USE_TOKEN_TRUNCATION=true" .env

# Check deduplication
test -f "data/articles.db" && echo "Dedup enabled"
```

### ✅ Workflow Best Practices

- [ ] Set `timeout-minutes: 20` on jobs
- [ ] Use `--max-turns` to limit Claude iterations
- [ ] Enable `--async` flag for processing
- [ ] Implement budget checks
- [ ] Use concurrency groups to prevent conflicts
- [ ] Cache dependencies with `actions/cache@v4`
- [ ] Set up error notifications

---

## Common Issues & Solutions

### Issue: "API rate limit exceeded"

**Solution**: Add rate limiting delay

```yaml
- name: Process with rate limiting
  run: |
    for batch in {1..3}; do
      uv run python -m src.main run --limit 10 --async
      sleep 60  # 1-minute cooldown between batches
    done
```

### Issue: "High unexpected costs"

**Solution**: Check token usage

```bash
# Verify token truncation is active
uv run python -c "
from src.clients.token_utils import count_tokens
from src.core.cache import ContentCache

cache = ContentCache()
stats = cache.get_stats()
print(f'Cache hit rate: {stats[\"hit_rate\"]}%')
print(f'Token limit: 10,000 per article')
"
```

### Issue: "Workflow timeout"

**Solution**: Reduce article limit or increase timeout

```yaml
jobs:
  analyze:
    timeout-minutes: 30  # Increase from default 20
    steps:
      - name: Process fewer articles
        run: uv run python -m src.main run --limit 5 --async
```

### Issue: "Database locked"

**Solution**: Enable WAL mode (already in example)

```yaml
- name: Initialize database
  run: |
    sqlite3 data/articles.db "PRAGMA journal_mode = WAL;"
```

---

## Monitoring Commands

### Check Workflow Status

```bash
# Via GitHub CLI
gh run list --workflow="claude-daily-analysis.yml" --limit 5

# Check latest run
gh run view --log

# Check costs (manual)
cat .github/monthly_spend.txt
```

### Check Cache Performance

```bash
# Run locally
uv run python -c "
from src.core.cache import ContentCache
cache = ContentCache()
stats = cache.get_stats()

print(f'Cache Hit Rate: {stats[\"hit_rate\"]}%')
print(f'L1 (Memory) Size: {stats[\"l1_size_mb\"]} MB')
print(f'L2 (Disk) Entries: {stats[\"l2_entries\"]}')
"
```

### Check Token Usage

```bash
# Estimate monthly usage
uv run python -c "
articles_per_day = 10
tokens_per_article = 10000  # After truncation
cache_hit_rate = 0.72
dedup_rate = 0.30

effective = articles_per_day * 30 * (1 - cache_hit_rate) * (1 - dedup_rate)
monthly_tokens = effective * tokens_per_article
monthly_cost = monthly_tokens * 0.003 / 1000

print(f'Effective articles/month: {effective:.0f}')
print(f'Monthly tokens: {monthly_tokens:,.0f}')
print(f'Estimated cost: \${monthly_cost:.2f}')
"
```

---

## Next Steps

### Phase 1: Test Interactive Review (Week 1)

Create `.github/workflows/claude-interactive.yml`:

```yaml
name: Interactive Article Review

on:
  issue_comment:
    types: [created]

jobs:
  review:
    if: contains(github.event.comment.body, '@claude-review')
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write

    steps:
      - uses: actions/checkout@v4

      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            User requested article review in comment.
            Analyze latest articles and provide insights.
```

**Test**: Comment `@claude-review` on any issue

### Phase 2: Enable Scheduled Analysis (Week 2-3)

Use the workflow from Step 2 above with:

- Budget controls enabled
- Cache monitoring
- Error notifications

**Monitor**: Daily for 1 week to ensure stability

### Phase 3: Add Comprehensive Reports (Week 4+)

Create weekly summary workflow:

```yaml
name: Weekly Comprehensive Report

on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight

jobs:
  weekly-report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync

      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            Generate comprehensive weekly report:
            - Week's highlights
            - Research trends
            - Top papers
            - Reading recommendations

            Save to docs/weekly-reports/YYYY-MM-DD.md
          claude_args: "--max-turns 3"
```

---

## Support Resources

- **Full Research**: See `docs/CLAUDE_CODE_GITHUB_ACTIONS_RESEARCH.md`
- **Claude Code Docs**: <https://code.claude.com/docs/en/github-actions>
- **GitHub Action**: <https://github.com/anthropics/claude-code-action>
- **Anthropic API**: <https://console.anthropic.com>

---

## Summary Checklist

Before going live:

- [ ] `ANTHROPIC_API_KEY` added to GitHub secrets
- [ ] Basic workflow created and tested
- [ ] Budget controls implemented
- [ ] Monthly spend tracker initialized (`echo "0.00" > .github/monthly_spend.txt`)
- [ ] Optimizations verified (async, caching, dedup, truncation)
- [ ] Timeout limits set (20-30 minutes)
- [ ] Error notifications configured
- [ ] Test run completed successfully
- [ ] Cost monitoring in place

**Estimated Setup Time**: 1-2 hours
**Expected Monthly Cost**: $3-15 (depending on volume)
**Expected ROI**: Immediate (time savings > costs)

---

**Ready to start? Run this:**

```bash
# 1. Create initial budget tracker
echo "0.00" > .github/monthly_spend.txt
git add .github/monthly_spend.txt
git commit -m "Initialize budget tracker"
git push

# 2. Test workflow locally
MAX_CONCURRENT_ARTICLES=8 \
uv run python -m src.main run --limit 5 --async

# 3. Create workflow file (copy example from above)
mkdir -p .github/workflows
# ... create claude-daily-analysis.yml

# 4. Commit and push
git add .github/workflows/
git commit -m "Add Claude Code daily analysis workflow"
git push

# 5. Test in GitHub Actions
# Go to Actions tab → Daily Article Analysis → Run workflow
```

Good luck! 🚀
