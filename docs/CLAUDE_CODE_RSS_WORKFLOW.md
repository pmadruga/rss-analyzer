# Claude Code RSS Analyzer Workflow

## Overview

This workflow uses **Claude Code Action with your OAuth token** to orchestrate an intelligent hybrid RSS analysis pipeline.

## How It Works

### 🎯 The Hybrid Approach

Instead of choosing between Mistral (cheap) or Claude API (expensive), this workflow combines both:

```
1. Mistral (Bulk Processing)
   ↓ Fast, cost-effective analysis of all articles

2. Claude Code (Enhancement)
   ↓ Deep reasoning and insights using OAuth token

3. Combined Output
   ↓ Best of both worlds!
```

### 💰 Cost Comparison

| Approach | Cost/Month | Quality | Token Used |
|----------|------------|---------|------------|
| Mistral only | $1.79 | Good | `MISTRAL_API_KEY` |
| Claude API only | $16.20 | Excellent | `ANTHROPIC_API_KEY` |
| **This workflow** | **$3-5** | **Excellent** | **`CLAUDE_CODE_OAUTH_TOKEN`** ✅ |

### 🔑 Key Innovation

**Uses your OAuth token!** The workflow:
- ✅ Runs Mistral for bulk processing (cheap)
- ✅ Claude Code enhances key articles (your OAuth token)
- ✅ No API key needed for Claude
- ✅ Best quality at reasonable cost

## Workflow Details

### Trigger Options

#### 1. Scheduled (Daily)
Runs automatically at 9:00 AM UTC (1 hour after main pipeline):
```yaml
schedule:
  - cron: '0 9 * * *'  # Daily at 9:00 AM UTC
```

#### 2. Manual Trigger
Run on-demand with custom settings:
```bash
gh workflow run "🤖 RSS Analysis with Claude Code" \
  --field max_articles=10 \
  --field analysis_mode=deep \
  --field generate_reports=true
```

### Analysis Modes

#### Quick Mode
- Mistral processes all articles
- Claude enhances 1-2 top articles
- Fast, minimal cost
- **Use for**: Daily automated runs

#### Standard Mode (Default)
- Mistral processes all articles
- Claude enhances all articles with key insights
- Balanced cost/quality
- **Use for**: Regular automated runs

#### Deep Mode
- Mistral processes all articles
- Claude provides extensive analysis for each
- Cross-references between papers
- Identifies research gaps and implications
- Higher cost but maximum insight
- **Use for**: Weekly deep-dive analysis

## What Claude Code Does

### Phase 1: Orchestrate Mistral
Claude Code runs the Python analyzer with Mistral:
```bash
export API_PROVIDER=mistral
uv run python -m src.main run --limit 5
```

### Phase 2: Query Database
Retrieves recently analyzed articles:
```bash
sqlite3 data/articles.db -json "SELECT id, title, url FROM articles..."
```

### Phase 3: Enhanced Analysis
For each article, Claude Code:
1. Reads Mistral's analysis from database
2. Reads the article content
3. Applies deep reasoning and analysis
4. Writes enhanced insights to `output/claude_analysis_<id>.md`

Example enhanced analysis includes:
- **Key Contributions**: What's novel in the research?
- **Implications**: Why does this matter?
- **Connections**: How does it relate to other recent work?
- **Critiques**: What are the limitations?
- **Recommendations**: Who should read this and why?

### Phase 4: Report Generation
Creates comprehensive reports:
- Combined analysis showing Mistral vs Claude insights
- Executive summary of all articles
- Thematic analysis across articles
- Quality comparison

### Phase 5: Deployment
- Updates website data
- Commits results to repository
- Deploys to GitHub Pages

## File Outputs

### Generated Files

```
output/
├── combined_analysis_report.md       # Main report
├── claude_analysis_<id>.md           # Per-article enhanced analysis
├── recent_articles.json              # Article metadata
└── run_summary.json                  # Execution statistics

docs/
├── data.json                         # Website data (updated)
└── articles_by_date/                 # Date-organized articles
```

### Analysis Format

Each `claude_analysis_<id>.md` includes:

```markdown
# [Article Title]

## Mistral Analysis (Original)
[Cost-effective baseline analysis]

## Claude Code Enhanced Analysis

### Key Insights
- [Deep reasoning about significance]
- [Novel connections identified]
- [Practical implications]

### Technical Deep Dive
[Detailed technical analysis]

### Research Context
[How this fits in the broader field]

### Recommendations
- **For Researchers**: [What to investigate]
- **For Practitioners**: [How to apply]
- **For Students**: [What to learn]

### Quality Assessment
- **Originality**: [Score and rationale]
- **Rigor**: [Methodology evaluation]
- **Impact**: [Potential influence]
```

## Usage Examples

### Example 1: Daily Automated Run
Let the workflow run on schedule (9 AM UTC daily):
- Analyzes 5 articles (default)
- Standard mode
- Automatic deployment

**Cost**: ~$0.10-0.20/day = $3-6/month

### Example 2: Weekly Deep Dive
```bash
gh workflow run "🤖 RSS Analysis with Claude Code" \
  --field max_articles=20 \
  --field analysis_mode=deep \
  --field generate_reports=true
```

**Use case**: Weekly research review
**Cost**: ~$1-2/week = $4-8/month

### Example 3: Quick Test
```bash
gh workflow run "🤖 RSS Analysis with Claude Code" \
  --field max_articles=3 \
  --field analysis_mode=quick \
  --field generate_reports=true
```

**Use case**: Testing or light analysis
**Cost**: ~$0.05/run

## Benefits Over Direct API Usage

| Feature | Direct API Call | Claude Code Orchestration |
|---------|----------------|---------------------------|
| **Authentication** | Needs API key | ✅ Uses OAuth token |
| **Flexibility** | Fixed script | ✅ Adaptive reasoning |
| **Tool Access** | API only | ✅ Bash, Read, Write, Edit |
| **Orchestration** | Manual | ✅ Intelligent workflow |
| **Error Handling** | Basic | ✅ Self-correcting |
| **Reasoning** | Prompt-limited | ✅ Full Claude capabilities |

## Claude Code Advantages

### 1. Intelligent Orchestration
Claude Code can:
- Decide which articles need deep analysis
- Adjust approach based on article complexity
- Handle errors and retry intelligently
- Optimize the analysis workflow

### 2. Rich Tool Access
Claude Code has access to:
- **Bash**: Run any command
- **Read/Write**: File operations
- **Edit**: Modify existing files
- **Glob/Grep**: Search and discover
- **Git**: Commit and push changes

### 3. Adaptive Analysis
Unlike fixed API calls, Claude Code can:
- Read articles and decide depth needed
- Cross-reference multiple articles
- Identify themes across content
- Adjust analysis style per article type

### 4. Cost Efficiency
- Uses OAuth token (your existing credits)
- Only analyzes what needs Claude-level reasoning
- Mistral handles bulk work
- No separate API key management

## Monitoring and Costs

### Track Usage

Check workflow runs:
```bash
gh run list --workflow="rss-claude-code-analyzer.yml" --limit 10
```

View detailed logs:
```bash
gh run view <run-id> --log
```

### Estimated Costs

Based on average article length (3000 tokens):

| Scenario | Articles | Mode | Cost/Run | Runs/Month | Monthly |
|----------|----------|------|----------|------------|---------|
| Daily Auto | 5 | Standard | $0.15 | 30 | $4.50 |
| Weekly Deep | 20 | Deep | $1.50 | 4 | $6.00 |
| Quick Test | 3 | Quick | $0.05 | 10 | $0.50 |

**Total Range**: $3-8/month depending on usage

### Cost Breakdown

Per article:
- **Mistral**: $0.002 (bulk processing)
- **Claude Code**: $0.02-0.05 (enhancement)
- **Total**: $0.022-0.052 per article

## Troubleshooting

### OAuth Token Not Set
```
Error: claude_code_oauth_token is required
```

**Solution**:
```bash
gh secret set CLAUDE_CODE_OAUTH_TOKEN
# Paste your token from Claude Code CLI
```

### Mistral Key Missing
```
Error: Missing required configuration fields: mistral_api_key
```

**Solution**:
```bash
gh secret set MISTRAL_API_KEY
# Paste your Mistral API key
```

### Database Lock Error
If running concurrent workflows:

**Solution**: The workflows have concurrency groups configured:
- Main pipeline: `group: "rss-pipeline"`
- Claude Code: `group: "rss-claude-code"`

They won't conflict!

## Configuration

### Adjust Schedule

Edit `.github/workflows/rss-claude-code-analyzer.yml`:

```yaml
schedule:
  - cron: '0 9 * * *'  # Change time here (UTC)
```

Examples:
- `0 6 * * *` - 6 AM UTC
- `0 12 * * *` - Noon UTC
- `0 0 * * 1` - Midnight Monday (weekly)

### Change Default Settings

Modify workflow inputs:

```yaml
max_articles:
  default: '5'  # Change default article count

analysis_mode:
  default: 'standard'  # or 'quick' or 'deep'
```

## Comparison with Main Pipeline

| Feature | Main Pipeline | Claude Code Pipeline |
|---------|--------------|---------------------|
| **Trigger** | 8 AM UTC | 9 AM UTC (after main) |
| **API** | Mistral only | Mistral + Claude Code |
| **Cost** | $1.79/month | $3-8/month |
| **Quality** | Good | Excellent |
| **Token** | `MISTRAL_API_KEY` | `CLAUDE_CODE_OAUTH_TOKEN` |
| **Analysis** | Standard | Enhanced + Insights |
| **Reports** | Basic | Comprehensive |

## Recommended Setup

### Option 1: Both Pipelines (Recommended)
Run both workflows:
- **8 AM**: Main pipeline (Mistral, basic analysis)
- **9 AM**: Claude Code (enhancement on top)

**Benefits**:
- Redundancy (if one fails, other succeeds)
- Incremental enhancement (Mistral baseline + Claude insights)
- Compare analysis quality

**Cost**: $5-10/month total

### Option 2: Claude Code Only
Disable main pipeline, use only Claude Code:

**Benefits**:
- Single workflow to maintain
- All analysis in one place
- Uses OAuth token

**Cost**: $3-8/month

**Trade-off**: No fallback if Claude Code fails

### Option 3: Main Pipeline + Manual Claude Code
Keep main pipeline automatic, run Claude Code manually:

**Benefits**:
- Lowest cost ($1.79 + occasional runs)
- Full control over when to use Claude
- Best for budget-conscious users

**Cost**: $2-4/month

## Next Steps

### 1. Test the Workflow

Run manually first:
```bash
gh workflow run "🤖 RSS Analysis with Claude Code" \
  --field max_articles=3 \
  --field analysis_mode=quick
```

### 2. Check the Results

View the run:
```bash
gh run list --workflow="rss-claude-code-analyzer.yml" --limit 1
gh run view <run-id> --log
```

Check generated files:
- `output/combined_analysis_report.md`
- `output/claude_analysis_*.md`

### 3. Review Quality

Compare:
- Mistral's original analysis (in database)
- Claude Code's enhanced analysis (in output/)
- Decide if the quality improvement justifies the cost

### 4. Adjust Settings

Based on results, adjust:
- Article count (more/fewer)
- Analysis mode (quick/standard/deep)
- Schedule (more/less frequent)

### 5. Enable Scheduled Runs

If satisfied, let it run automatically daily at 9 AM UTC!

## FAQ

**Q: Will this conflict with the main RSS pipeline?**
A: No, they have separate concurrency groups and run at different times.

**Q: Can I use this without the main pipeline?**
A: Yes! Just disable the main pipeline and use this as your only RSS analyzer.

**Q: What if my OAuth token expires?**
A: Claude Code tokens are long-lived. If expired, regenerate and update the secret.

**Q: Can I customize Claude Code's analysis?**
A: Yes! Edit the prompt in the workflow file to adjust analysis focus.

**Q: How much does this really cost?**
A: ~$0.02-0.05 per article for Claude enhancement. With 5 articles daily = $3-7/month.

**Q: Is this better than using ANTHROPIC_API_KEY?**
A: For this use case, yes! Claude Code can orchestrate and reason about the workflow, not just process API calls.

## Summary

✅ **Uses your OAuth token** - No API key needed
✅ **Hybrid approach** - Mistral + Claude Code enhancement
✅ **Cost-effective** - $3-8/month for excellent quality
✅ **Intelligent orchestration** - Claude Code adapts to content
✅ **Rich analysis** - Beyond simple API calls
✅ **Easy to use** - One workflow, automatic or manual

This is the best way to leverage your Claude Code OAuth token for RSS analysis! 🚀
