# Disabled Workflows - API Key Usage

**Status**: Temporarily disabled to use only Claude Code OAuth workflow
**Date**: 2025-11-08

## Summary

Disabled 4 workflows that require API keys to focus exclusively on the **Claude Code RSS Analyzer** workflow which uses the OAuth token.

## Disabled Workflows

### 1. ✅ rss-complete-pipeline.yml → rss-complete-pipeline.yml.disabled
**Purpose**: Main RSS analysis pipeline using Mistral/Anthropic API
**Trigger**: Daily at 8:00 AM UTC + manual
**Why Disabled**: Uses `MISTRAL_API_KEY` - replaced by Claude Code workflow
**Cost**: $1.79/month (Mistral)

### 2. ✅ force-refresh-now.yml → force-refresh-now.yml.disabled
**Purpose**: Force immediate refresh of RSS feed analysis
**Trigger**: Manual only
**Why Disabled**: Uses API keys for processing
**Cost**: Variable based on usage

### 3. ✅ refresh-titles.yml → refresh-titles.yml.disabled
**Purpose**: Refresh article titles from RSS feed
**Trigger**: Manual only
**Why Disabled**: Uses API keys for title processing
**Cost**: Minimal

### 4. ✅ test-pipeline.yml → test-pipeline.yml.disabled
**Purpose**: Test pipeline components on PRs
**Trigger**: Pull requests to main
**Why Disabled**: Uses API keys for testing
**Cost**: Minimal (only on PRs)

## Active Workflows (No API Keys)

### Claude Code OAuth Workflows
- ✅ **rss-claude-code-analyzer.yml** - Main RSS analysis (OAuth token)
- ✅ **claude-code-review.yml** - PR code reviews (OAuth token)

### GitHub Automation (No API Keys)
- ✅ **pr-automation.yml** - Auto-labeling and PR checks
- ✅ **pr-checks.yml** - PR validation
- ✅ **code-review-swarm.yml** - AI code review agents
- ✅ **automated-code-review.yml** - Automated reviews
- ✅ **deploy-website.yml** - GitHub Pages deployment
- ✅ **deduplication-check.yml** - Duplicate detection
- ✅ **ui-tests.yml** - UI testing
- ✅ **claude.yml** - General Claude workflows

### Utility
- ✅ **_reusable-setup.yml** - Reusable setup steps

## Current Setup

### RSS Analysis
**Before**:
- Main pipeline (Mistral API) at 8 AM UTC → $1.79/month
- Manual refresh tools

**After**:
- Claude Code analyzer (OAuth token) at 9 AM UTC → $3-7/month
- Higher quality, uses OAuth instead of API key

### Benefits of Current Setup

✅ **Single RSS workflow**: Claude Code handles everything
✅ **OAuth authentication**: No API key management
✅ **Better quality**: Claude's deep reasoning
✅ **Tool access**: Bash, Read, Write, Edit, Git
✅ **Adaptive**: Adjusts to content complexity

### Trade-offs

⚠️ **Cost**: $3-7/month vs $1.79/month (but better quality)
⚠️ **Single point**: No fallback if Claude Code fails
ℹ️ **Schedule**: Runs at 9 AM instead of 8 AM

## How to Re-Enable Workflows

If you want to re-enable any workflow:

### Re-enable All (Hybrid Approach)
```bash
# From project root
cd /home/mess/dev/rss-analyzer

# Re-enable all disabled workflows
mv .github/workflows/rss-complete-pipeline.yml.disabled .github/workflows/rss-complete-pipeline.yml
mv .github/workflows/force-refresh-now.yml.disabled .github/workflows/force-refresh-now.yml
mv .github/workflows/refresh-titles.yml.disabled .github/workflows/refresh-titles.yml
mv .github/workflows/test-pipeline.yml.disabled .github/workflows/test-pipeline.yml

git add .github/workflows/
git commit -m "Re-enable API key workflows for hybrid approach"
git push
```

**Result**: Both approaches run (8 AM Mistral + 9 AM Claude Code)
**Cost**: $5-10/month total

### Re-enable Just Main Pipeline (Fallback)
```bash
# Re-enable only the main RSS pipeline
mv .github/workflows/rss-complete-pipeline.yml.disabled .github/workflows/rss-complete-pipeline.yml

git add .github/workflows/
git commit -m "Re-enable main RSS pipeline as fallback"
git push
```

**Result**: Both main (Mistral) and Claude Code run
**Cost**: $5-8/month total
**Benefit**: Redundancy/fallback

### Re-enable Specific Workflow
```bash
# Example: Re-enable just the test pipeline
mv .github/workflows/test-pipeline.yml.disabled .github/workflows/test-pipeline.yml

git add .github/workflows/
git commit -m "Re-enable test pipeline for PR validation"
git push
```

## Quick Re-Enable Script

Created a convenience script:

```bash
# Re-enable all workflows
uv run python tools/reenable_workflows.py --all

# Re-enable specific workflow
uv run python tools/reenable_workflows.py --workflow rss-complete-pipeline

# List disabled workflows
uv run python tools/reenable_workflows.py --list
```

## Monitoring Claude Code Workflow

### Check if it's running
```bash
gh workflow list | grep "Claude Code"
```

Expected output:
```
🤖 RSS Analysis with Claude Code  active  19190291720
```

### View recent runs
```bash
gh run list --workflow="rss-claude-code-analyzer.yml" --limit 5
```

### Manually trigger
```bash
# Quick test
gh workflow run "🤖 RSS Analysis with Claude Code" \
  --field max_articles=3 \
  --field analysis_mode=quick

# Standard run
gh workflow run "🤖 RSS Analysis with Claude Code" \
  --field max_articles=5 \
  --field analysis_mode=standard

# Deep analysis
gh workflow run "🤖 RSS Analysis with Claude Code" \
  --field max_articles=10 \
  --field analysis_mode=deep
```

## Cost Comparison

| Configuration | Monthly Cost | Workflows Active |
|---------------|--------------|------------------|
| **Current (OAuth only)** | **$3-7** | **1 RSS workflow** |
| Previous (Mistral only) | $1.79 | 4 workflows |
| Hybrid (Both) | $5-10 | 5 workflows |
| Claude API only | $16+ | 4 workflows |

## Recommendations

### Keep Current Setup If:
- ✅ You want the best quality analysis
- ✅ You prefer OAuth token authentication
- ✅ $3-7/month is acceptable
- ✅ You want Claude's adaptive reasoning

### Re-Enable Main Pipeline If:
- ✅ You want a fallback option
- ✅ You want to compare Mistral vs Claude quality
- ✅ You don't mind managing two workflows
- ✅ Cost is flexible ($5-10/month is OK)

### Switch Back to Mistral Only If:
- ✅ Budget is tight (need $1.79/month)
- ✅ Basic analysis quality is sufficient
- ✅ Volume is very high (100+ articles/day)

## Next Steps

### Immediate
1. ✅ Disabled API key workflows
2. ✅ Claude Code workflow is active
3. ⏳ Wait for first scheduled run (9 AM UTC)
4. ⏳ Monitor quality and cost

### This Week
- Test Claude Code workflow manually
- Review analysis quality vs expectations
- Check OAuth token usage/costs
- Decide if current setup meets needs

### Optional
- Re-enable main pipeline for redundancy
- Adjust Claude Code schedule/frequency
- Fine-tune analysis modes
- Add custom reporting

## Rollback Plan

If Claude Code workflow has issues:

1. **Quick Fix**: Re-enable main pipeline immediately
   ```bash
   mv .github/workflows/rss-complete-pipeline.yml.disabled \
      .github/workflows/rss-complete-pipeline.yml
   git add . && git commit -m "Restore main pipeline" && git push
   ```

2. **Investigate**: Check Claude Code logs
   ```bash
   gh run list --workflow="rss-claude-code-analyzer.yml" --limit 1
   gh run view <run-id> --log
   ```

3. **Decide**: Keep both, or switch back to Mistral only

## Files Changed

### Renamed (Disabled)
- `.github/workflows/rss-complete-pipeline.yml.disabled`
- `.github/workflows/force-refresh-now.yml.disabled`
- `.github/workflows/refresh-titles.yml.disabled`
- `.github/workflows/test-pipeline.yml.disabled`

### Active
- `.github/workflows/rss-claude-code-analyzer.yml` ← **Main RSS workflow**
- `.github/workflows/claude-code-review.yml` ← PR reviews
- All other non-API workflows

### Documentation
- `docs/DISABLED_WORKFLOWS.md` ← This file
- `docs/CLAUDE_CODE_RSS_WORKFLOW.md` ← Claude Code workflow guide
- `tools/reenable_workflows.py` ← Re-enable script

## Summary

✅ **Status**: 4 API-key workflows disabled
✅ **Active**: Claude Code RSS analyzer (OAuth)
✅ **Cost**: $3-7/month (was $1.79)
✅ **Quality**: Enhanced with Claude reasoning
✅ **Reversible**: Easy to re-enable anytime

All RSS analysis now goes through the Claude Code OAuth workflow! 🚀
