# GitHub Pages Setup Guide

This guide will help you enable GitHub Pages for your RSS Analyzer dashboard.

## 🚀 Quick Setup (2 minutes)

### Step 1: Enable GitHub Pages
1. Go to your repository on GitHub: `https://github.com/pmadruga/rss-analyzer`
2. Click the **Settings** tab
3. Scroll down to **Pages** section (left sidebar)
4. Under **Source**, select **"GitHub Actions"**
5. Click **Save**

### Step 2: Trigger First Deployment
1. Go to **Actions** tab in your repository
2. Click **"Deploy to GitHub Pages"** workflow
3. Click **"Run workflow"** button
4. Click **"Run workflow"** (with default settings)

### Step 3: Access Your Dashboard
After the workflow completes (2-3 minutes), your dashboard will be available at:
**https://pmadruga.github.io/rss-analyzer/**

## 🔧 Troubleshooting

### Issue: "404 Page Not Found"
**Cause**: GitHub Pages not enabled or workflow hasn't run yet

**Solution**:
1. Verify GitHub Pages is enabled (Step 1 above)
2. Check that the deploy workflow has completed successfully
3. Wait 5-10 minutes for DNS propagation

### Issue: "Website shows but no articles"
**Cause**: No data.json file or empty database

**Solution**:
1. Run the RSS analyzer first to populate the database
2. The "Update Website Data" workflow will automatically generate data.json
3. Or manually trigger the "Update Website Data" workflow

### Issue: "Workflow failed"
**Cause**: Various deployment issues

**Solution**:
1. Check the Actions tab for error details
2. Ensure all required files are in the `docs/` directory
3. Verify the workflow has proper permissions

## 🔄 Automatic Updates

Once set up, your website will automatically update:
- **Every 3 hours**: Checks for new articles and updates the dashboard
- **After RSS processing**: Automatically updates when new articles are analyzed
- **Manual trigger**: You can manually update via GitHub Actions

## 📊 Features Available

### Dashboard Features
- ✅ **14 articles** ready to display
- ✅ **Real-time search** and filtering
- ✅ **Responsive design** for mobile/desktop
- ✅ **Professional styling** with modern UI
- ✅ **Accessibility features** for all users

### Current Data
- **Articles**: 14 analyzed articles
- **Date Range**: July 2, 2025
- **AI Provider**: Anthropic Claude
- **Average Confidence**: 75%
- **High Confidence Articles**: 10/14

## 🎯 Next Steps

1. **Enable GitHub Pages** (most important!)
2. **Run the deployment workflow**
3. **Bookmark your dashboard**: https://pmadruga.github.io/rss-analyzer/
4. **Set up RSS processing** to get new articles regularly

## 💡 Pro Tips

- **Custom Domain**: You can add a custom domain in GitHub Pages settings
- **HTTPS**: GitHub Pages automatically provides HTTPS
- **Analytics**: The dashboard is privacy-focused with no tracking
- **Performance**: Optimized for fast loading on all devices

---

**Need Help?** Check the Actions tab in your repository for detailed logs and error messages.
