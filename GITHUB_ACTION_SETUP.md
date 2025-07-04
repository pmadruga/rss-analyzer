# GitHub Actions Setup for RSS Analyzer

This guide will help you set up the RSS analyzer to run automatically on GitHub Actions, storing all outputs directly in your GitHub repository.

## 🚀 Quick Setup

1. **Run the setup script:**
   ```bash
   ./setup_github_action.sh
   ```

2. **Configure your API key** (choose one):
   - Anthropic Claude (recommended)
   - Mistral AI  
   - OpenAI

3. **Push to GitHub** and the analyzer will start running automatically once daily at 2 AM UTC!

## 📋 Manual Setup

If you prefer manual setup or the script doesn't work:

### 1. Repository Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

#### Required (choose one API provider):

**For Anthropic Claude:**
```
API_PROVIDER = anthropic
ANTHROPIC_API_KEY = sk-your-anthropic-key-here
```

**For Mistral AI:**
```
API_PROVIDER = mistral  
MISTRAL_API_KEY = your-mistral-key-here
```

**For OpenAI:**
```
API_PROVIDER = openai
OPENAI_API_KEY = sk-your-openai-key-here
```

### 2. Repository Variables (Optional)

Go to Settings → Secrets and variables → Actions → Variables tab:

```
RSS_FEED_URL = https://your-custom-feed.com/rss
MAX_LINKED_ARTICLES = 3
SCRAPER_DELAY = 1.0
REQUEST_TIMEOUT = 30
```

### 3. Permissions

Go to Settings → Actions → General:
- ✅ **Workflow permissions:** "Read and write permissions"
- ✅ **Allow GitHub Actions to create and approve pull requests**

## ⚙️ Workflow Features

### 🕐 Automatic Scheduling
- **Runs every hour** at the top of the hour
- **No maintenance required** - fully automated

### 📊 Outputs Stored in Repository
- `output/articles_by_date.md` - Main analysis file organized by date
- `output/run_summary.json` - Run statistics and metadata
- Database backed up as GitHub artifacts

### 🎯 Manual Triggering
Run manually from GitHub web interface or CLI:

```bash
# GitHub CLI
gh workflow run rss-analyzer.yml

# With custom parameters
gh workflow run rss-analyzer.yml \
  --field max_articles=20 \
  --field follow_links=true
```

### 📦 Automatic Releases
Creates releases automatically:
- Every 50 articles processed
- Daily at midnight UTC
- Includes analysis files and statistics

## 🔍 Monitoring

### GitHub Actions Tab
- View run status and logs
- Download artifacts (database backups, logs)
- Trigger manual runs

### Repository Files
- **Latest analysis:** `output/articles_by_date.md`
- **Run summary:** `output/run_summary.json`
- **Commit history** shows automatic updates

### Releases
- **Periodic snapshots** with download links
- **Statistics** in release notes
- **Archive** of historical data

## 🛠️ Customization

### Workflow Schedule
Edit `.github/workflows/rss-analyzer.yml`:
```yaml
on:
  schedule:
    - cron: '0 */2 * * *'  # Every 2 hours
    - cron: '0 9 * * 1-5'  # Weekdays at 9 AM
```

### Processing Limits
Set repository variables:
- `MAX_ARTICLES_PER_RUN` - Articles per execution
- `MAX_LINKED_ARTICLES` - Linked articles to follow
- `SCRAPER_DELAY` - Delay between requests

### Custom RSS Feeds
Set the `RSS_FEED_URL` variable to your feed URL.

## 🔧 Troubleshooting

### Common Issues

**1. Workflow fails with "API key not found"**
- Check that secrets are set correctly
- Verify secret names match exactly
- Ensure API provider matches your key type

**2. No permission to push commits**
- Check workflow permissions are "Read and write"
- Verify repository isn't archived or restricted

**3. Database not persisting between runs**
- Artifacts are automatically managed
- Database state carries over between runs
- Check artifact retention settings

**4. RSS feed not updating**
- Verify RSS_FEED_URL is accessible
- Check feed format is valid
- Review workflow logs for errors

### Getting Help

1. **Check workflow logs** in GitHub Actions tab
2. **Review error messages** in run summaries
3. **Verify configuration** against this guide
4. **Test locally** first if issues persist

## 💡 Tips

- **Start with defaults** before customizing
- **Monitor first few runs** to ensure stability  
- **Use manual triggers** for testing
- **Check rate limits** for your API provider
- **Set reasonable delays** to respect target sites

## 🎉 Benefits

✅ **Zero maintenance** - runs automatically  
✅ **Always available** - no local machine required  
✅ **Version controlled** - all outputs tracked in git  
✅ **Collaborative** - team can access results  
✅ **Reliable** - GitHub's infrastructure  
✅ **Scalable** - easy to modify and extend