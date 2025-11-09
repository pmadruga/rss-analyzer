# Anthropic Authentication: OAuth Token vs API Key

## TL;DR

**No, the OAuth token cannot replace the API key** - they serve different purposes:

- **CLAUDE_CODE_OAUTH_TOKEN**: Only works with `anthropics/claude-code-action@v1`
- **ANTHROPIC_API_KEY**: Required for direct API calls (RSS analysis, custom scripts)

## The Two Types of Authentication

### 1. CLAUDE_CODE_OAUTH_TOKEN (OAuth Token)

**What it is**: An OAuth token that gives the `claude-code-action` GitHub Action permission to run Claude Code on your behalf.

**Where it works**:
```yaml
- uses: anthropics/claude-code-action@v1
  with:
    claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
    prompt: "Review this code..."
```

**Limitations**:
- ❌ Cannot be used for direct API calls
- ❌ Cannot be used in Python scripts
- ❌ Cannot be used in custom workflows
- ✅ Only works with the official Claude Code GitHub Action

**Current status in your repo**:
- `.github/workflows/claude-code-review.yml` is configured to use it
- Workflow will fail if secret is not set

---

### 2. ANTHROPIC_API_KEY (API Key)

**What it is**: A direct API key for making calls to the Claude API.

**Where it works**:
```python
import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(...)
```

**Use cases**:
- ✅ RSS article analysis (your main use case)
- ✅ Custom Python scripts
- ✅ Direct API integrations
- ✅ Automated processing pipelines

**Current status in your repo**:
- Secret exists but may not be set
- RSS pipeline needs this for Anthropic-based analysis

---

## Why Can't OAuth Token Replace API Key?

The OAuth token is **scoped specifically** for the Claude Code Action:

```
CLAUDE_CODE_OAUTH_TOKEN
    ↓ (only works with)
anthropics/claude-code-action@v1
    ↓ (internally uses)
Claude Code CLI
    ↓ (makes authenticated calls)
Anthropic API
```

Your Python scripts can't use this token because:
1. It's not a standard API key format
2. It requires the Claude Code infrastructure
3. It's designed for interactive coding sessions, not batch API calls

---

## What You Can Do With Each

### With CLAUDE_CODE_OAUTH_TOKEN

✅ **PR Code Reviews** (already configured)
```yaml
# .github/workflows/claude-code-review.yml
- uses: anthropics/claude-code-action@v1
  with:
    claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
    prompt: |
      Review this PR and provide feedback
      Use gh pr comment to post your review
```

✅ **Interactive Code Analysis**
```yaml
- uses: anthropics/claude-code-action@v1
  with:
    claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
    prompt: "Analyze test coverage and suggest improvements"
```

✅ **Documentation Generation**
```yaml
- uses: anthropics/claude-code-action@v1
  with:
    claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
    prompt: "Generate API documentation from the code"
```

### With ANTHROPIC_API_KEY

✅ **RSS Article Analysis** (your main pipeline)
```python
# src/core/ai_clients/anthropic_client.py
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": article_content}]
)
```

✅ **Custom Workflows**
```yaml
- name: Analyze commit messages
  run: |
    python scripts/analyze_commits.py  # Uses ANTHROPIC_API_KEY
```

✅ **Batch Processing**
```yaml
- name: Process 100 articles
  run: |
    python -m src.main run --limit 100  # Uses ANTHROPIC_API_KEY
```

---

## Your Current Setup

Based on the repository analysis:

### Configured ✅
- `MISTRAL_API_KEY` - Working, currently used for RSS pipeline
- `GITHUB_TOKEN` - Automatic, works for all workflows

### Possibly Configured ❓
- `CLAUDE_CODE_OAUTH_TOKEN` - May or may not be set (check with `gh secret list`)
- `ANTHROPIC_API_KEY` - Secret exists but may be empty

### Recommendation

You have **two options**:

#### Option 1: Keep Current Setup (Recommended)
```yaml
# RSS Pipeline: Use Mistral (already working)
API_PROVIDER: 'mistral'
MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}

# PR Reviews: Use Claude Code OAuth
claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
```

**Cost**: $1.79/month (Mistral only)

**Pros**:
- ✅ Everything works now
- ✅ Very cost-effective
- ✅ OAuth token works for PR reviews

**Cons**:
- Lower quality article analysis vs Claude

---

#### Option 2: Hybrid Approach (Best Quality)
```yaml
# RSS Pipeline: Use Claude API
API_PROVIDER: 'anthropic'
ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}  # Need to set this

# PR Reviews: Use Claude Code OAuth
claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
```

**Cost**: $5-7/month

**Pros**:
- ✅ Best quality article analysis
- ✅ PR reviews still work with OAuth
- ✅ Consistent Claude experience

**Cons**:
- Need to set up ANTHROPIC_API_KEY (separate from OAuth)
- Higher cost

---

## How to Get Each Token

### To Get ANTHROPIC_API_KEY

1. Go to https://console.anthropic.com/
2. Navigate to "API Keys"
3. Create a new key
4. Add to GitHub secrets:
   ```bash
   gh secret set ANTHROPIC_API_KEY
   # Paste your key when prompted
   ```

**Cost**: Pay-as-you-go, ~$16.20/month for your workload (before optimizations)

### To Get CLAUDE_CODE_OAUTH_TOKEN

1. Install Claude Code CLI: https://claude.com/code
2. Authenticate: `claude auth login`
3. Generate OAuth token for GitHub Actions
4. Add to GitHub secrets:
   ```bash
   gh secret set CLAUDE_CODE_OAUTH_TOKEN
   # Paste your OAuth token when prompted
   ```

**Cost**: Included with Claude Code subscription or API credits

---

## Check Your Current Secrets

Run this to see what's configured:
```bash
gh secret list
```

Expected output:
```
ANTHROPIC_API_KEY    Updated 2025-11-08
CLAUDE_CODE_OAUTH_TOKEN    Updated 2025-11-08
MISTRAL_API_KEY      Updated 2025-11-08
```

---

## Decision Matrix

| Scenario | Use OAuth Token | Use API Key |
|----------|----------------|-------------|
| PR code reviews with Claude Code | ✅ Yes | ❌ No |
| RSS article analysis | ❌ No | ✅ Yes |
| Custom Python scripts | ❌ No | ✅ Yes |
| Interactive coding tasks | ✅ Yes | ❌ No |
| Batch processing | ❌ No | ✅ Yes |
| Documentation generation (Action) | ✅ Yes | ❌ No |
| Documentation generation (script) | ❌ No | ✅ Yes |

---

## Recommended Action Plan

Since you already have the OAuth token, here's what you can do:

### Immediate (Free/Low Cost)

1. **Set up OAuth token for PR reviews** (if not already done):
   ```bash
   gh secret set CLAUDE_CODE_OAUTH_TOKEN
   ```

2. **Keep Mistral for RSS pipeline** (already working):
   - Cost: $1.79/month
   - No changes needed

**Result**: PR reviews with Claude + cost-effective RSS analysis

### Optional (Better Quality)

1. **Get an ANTHROPIC_API_KEY** from console.anthropic.com

2. **Set it in GitHub**:
   ```bash
   gh secret set ANTHROPIC_API_KEY
   ```

3. **Update RSS pipeline** to use it:
   ```yaml
   API_PROVIDER: 'anthropic'
   ```

**Result**: Best quality everywhere, ~$5-7/month total

---

## Summary

**Can you use OAuth token for RSS analysis?**
❌ **No** - OAuth token only works with `anthropics/claude-code-action@v1`

**What should you do?**
✅ **Use OAuth token for**: PR reviews, interactive coding
✅ **Use API key for**: RSS analysis, batch processing, Python scripts

**Best setup**:
- OAuth token → PR reviews (already configured)
- Mistral API → RSS pipeline (working, cheap)
- Optional: Get API key → Upgrade RSS to Claude when budget allows

---

## Questions?

**Q: Can I convert my OAuth token to an API key?**
A: No, they're completely different authentication mechanisms.

**Q: Can Claude Code Action access my API credits?**
A: Yes, if you authenticated Claude Code with your Anthropic account, it uses your API credits.

**Q: Should I get both?**
A:
- OAuth token: Yes, for PR reviews (free to use, consumes API credits when running)
- API key: Optional, only if you want Claude for RSS analysis instead of Mistral

**Q: Which one am I currently using?**
A: Based on the code:
- RSS pipeline: `MISTRAL_API_KEY` (switched from Anthropic)
- PR reviews: `CLAUDE_CODE_OAUTH_TOKEN` (configured but may not be set)
