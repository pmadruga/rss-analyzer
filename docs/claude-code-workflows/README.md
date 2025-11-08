# Claude Code Workflow Templates

Ready-to-use GitHub Actions workflows for automated RSS article processing with Claude Code.

---

## Quick Start

### 1. Setup (One-time)

```bash
# Create workflow directory
mkdir -p .github/workflows

# Initialize budget tracker
echo "0.00" > .github/monthly_spend.txt
git add .github/monthly_spend.txt
git commit -m "Initialize budget tracker"
git push

# Add Anthropic API key to GitHub secrets
# Go to: Repository Settings → Secrets and variables → Actions → New secret
# Name: ANTHROPIC_API_KEY
# Value: sk-ant-...
```

### 2. Choose Your Phase

#### Phase 1: Interactive Review (Low Risk)

```bash
# Copy interactive review workflow
cp docs/claude-code-workflows/phase1-interactive-review.yml \
   .github/workflows/claude-interactive.yml

# Commit and push
git add .github/workflows/claude-interactive.yml
git commit -m "Add Claude interactive review workflow"
git push

# Test: Comment @claude-review on any issue
```

**Use when**: Testing Claude Code for the first time
**Cost**: $1-2/month (on-demand only)
**Risk**: None (manual trigger)

#### Phase 2: Daily Analysis (Controlled Automation)

```bash
# Copy daily analysis workflow
cp docs/claude-code-workflows/phase2-daily-analysis.yml \
   .github/workflows/claude-daily-analysis.yml

# Copy budget reset workflow
cp docs/claude-code-workflows/budget-reset.yml \
   .github/workflows/budget-reset.yml

# Commit and push
git add .github/workflows/
git commit -m "Add Claude daily analysis workflows"
git push

# Run manually first to test
# Go to: Actions → Daily Article Analysis → Run workflow
```

**Use when**: Ready for daily automation
**Cost**: $3-5/month
**Risk**: Low (budget controls enabled)

#### Phase 3: Weekly Reports (Full Automation)

```bash
# Copy weekly reports workflow
cp docs/claude-code-workflows/phase3-weekly-reports.yml \
   .github/workflows/claude-weekly-reports.yml

# Commit and push
git add .github/workflows/claude-weekly-reports.yml
git commit -m "Add Claude weekly reports workflow"
git push

# Runs automatically every Sunday
```

**Use when**: Production-ready automation
**Cost**: $10-15/month (including daily analysis)
**Risk**: Minimal (proven in Phase 1-2)

---

## Workflow Files

### Phase 1: Interactive Review

**File**: `phase1-interactive-review.yml`

**Trigger**: `@claude-review` mention in issue comments
**Runtime**: ~5 minutes
**Cost**: $0.10-0.20 per review

**What it does**:
- Responds to @claude-review mentions
- Analyzes specific articles on request
- Posts findings as issue comments
- No scheduled automation

**Best for**: Testing and manual reviews

### Phase 2: Daily Analysis

**File**: `phase2-daily-analysis.yml`

**Trigger**: Daily at 9 AM UTC (configurable)
**Runtime**: ~15 minutes
**Cost**: ~$0.50 per run ($15/month max)

**What it does**:
1. Checks monthly budget before running
2. Processes RSS articles (with all optimizations)
3. Analyzes with Claude Code
4. Generates daily summary
5. Commits results
6. Updates budget tracker

**Best for**: Daily automated processing

### Phase 3: Weekly Reports

**File**: `phase3-weekly-reports.yml`

**Trigger**: Sunday midnight UTC (configurable)
**Runtime**: ~20-30 minutes
**Cost**: ~$3 per run ($12/month)

**What it does**:
1. Queries database statistics
2. Generates comprehensive weekly report
3. Creates research trend analysis
4. Generates reading lists
5. Exports structured data
6. Deploys to GitHub Pages

**Best for**: Comprehensive weekly summaries

### Budget Management

**File**: `budget-reset.yml`

**Trigger**: First day of each month
**Runtime**: <1 minute
**Cost**: $0

**What it does**:
1. Archives previous month's spend
2. Resets budget tracker to $0.00
3. Generates budget summary

**Best for**: Automatic budget tracking

---

## Configuration

### Environment Variables

Set in workflow files or repository settings:

```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}  # Required
  MAX_CONCURRENT_ARTICLES: 8                           # Optional
```

### Budget Control

Edit budget limit in `phase2-daily-analysis.yml`:

```yaml
MONTHLY_BUDGET=15.00  # Change to your desired limit
```

### Schedule Changes

Edit cron expressions:

```yaml
on:
  schedule:
    - cron: '0 9 * * *'  # Daily at 9 AM UTC
    # Examples:
    # - cron: '0 */6 * * *'  # Every 6 hours
    # - cron: '0 0 * * 1'    # Monday midnight
    # - cron: '30 14 * * 5'  # Friday 2:30 PM
```

Cron reference: https://crontab.guru/

---

## Monitoring

### Check Workflow Status

```bash
# List recent runs
gh run list --workflow="claude-daily-analysis.yml" --limit 5

# View specific run
gh run view --log

# View workflow summary
gh run view --log-failed
```

### Check Budget

```bash
# View current month spend
cat .github/monthly_spend.txt

# View budget history
ls -lh .github/budget-history/
```

### Check Cache Performance

```bash
# Run locally
uv run python -c "
from src.core.cache import ContentCache
cache = ContentCache()
stats = cache.get_stats()
print(f'Cache hit rate: {stats[\"hit_rate\"]}%')
print(f'L1 size: {stats[\"l1_size_mb\"]} MB')
print(f'L2 entries: {stats[\"l2_entries\"]}')
"
```

---

## Troubleshooting

### Issue: "Budget exceeded"

**Solution**: Increase monthly budget or wait for reset

```yaml
# In phase2-daily-analysis.yml
MONTHLY_BUDGET=20.00  # Increase limit
```

### Issue: "Workflow failed"

**Solution**: Check error logs and retry

```bash
# View error logs
gh run view --log-failed

# Rerun failed workflow
gh run rerun
```

### Issue: "No articles processed"

**Solution**: Verify RSS feed and database

```bash
# Test RSS feed
uv run python -c "
from src.core.rss_parser import RSSParser
parser = RSSParser()
entries = parser.fetch_feed('https://bg.raindrop.io/rss/public/57118738')
print(f'Found {len(entries)} entries')
"

# Check database
sqlite3 data/articles.db "SELECT COUNT(*) FROM articles;"
```

### Issue: "Cache not working"

**Solution**: Verify cache configuration

```bash
# Check cache settings
grep "enable_caching" config/config.yaml

# Rebuild cache
uv run python -c "
from src.core.cache import ContentCache
cache = ContentCache('data/cache.db')
cache.clear()  # Reset cache
print('Cache cleared and rebuilt')
"
```

---

## Cost Optimization Tips

### 1. Use Async Processing

Already enabled in workflows via `--async` flag:

```yaml
- name: Process articles
  run: uv run python -m src.main run --limit 10 --async
```

**Benefit**: 6-8x faster = less GitHub Actions time

### 2. Limit Max Turns

Control Claude iterations:

```yaml
claude_args: "--max-turns 2"  # Limit to 2 conversation rounds
```

**Benefit**: Predictable token usage

### 3. Set Timeouts

Prevent runaway jobs:

```yaml
jobs:
  analyze:
    timeout-minutes: 20  # Max 20 minutes
```

**Benefit**: Prevents excessive GitHub Actions usage

### 4. Enable Caching

Leverage two-tier cache (already enabled in RSS Analyzer):

```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/uv
      .venv
    key: uv-${{ hashFiles('**/pyproject.toml') }}
```

**Benefit**: Faster workflow execution

### 5. Batch Processing

Process multiple articles in one workflow run:

```yaml
- name: Process in batches
  run: |
    uv run python -m src.main run --limit 50 --async
```

**Benefit**: Better cost per article

---

## Next Steps

1. **Start with Phase 1** (interactive review)
   - Test Claude Code capabilities
   - Validate quality
   - Understand costs

2. **Deploy Phase 2** (daily analysis) after 1 week
   - Monitor costs daily
   - Adjust prompts as needed
   - Optimize cache hit rate

3. **Enable Phase 3** (weekly reports) after 2-3 weeks
   - Comprehensive automation
   - Full production deployment

---

## Support

- **Full Research**: `docs/CLAUDE_CODE_GITHUB_ACTIONS_RESEARCH.md`
- **Quick Start**: `docs/CLAUDE_CODE_QUICKSTART.md`
- **Summary**: `docs/CLAUDE_CODE_INTEGRATION_SUMMARY.md`
- **Claude Code Docs**: https://code.claude.com/docs/en/github-actions
- **GitHub Action**: https://github.com/anthropics/claude-code-action

---

**Ready to deploy?**

```bash
# Quick deploy all workflows
cp docs/claude-code-workflows/*.yml .github/workflows/
git add .github/workflows/
git commit -m "Add Claude Code workflows (all phases)"
git push

# Or deploy one at a time (recommended)
cp docs/claude-code-workflows/phase1-interactive-review.yml .github/workflows/
# Test, then add more
```

Good luck! 🚀
