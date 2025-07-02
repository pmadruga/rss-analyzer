#!/bin/bash

# Setup script for GitHub Actions RSS Analyzer

echo "🚀 Setting up GitHub Actions for RSS Analyzer"
echo "=============================================="

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ This directory is not a git repository."
    echo "Please run 'git init' first or run this script from a git repository."
    exit 1
fi

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "⚠️ GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    echo "You can still manually set up the secrets in GitHub web interface."
    MANUAL_SETUP=true
else
    MANUAL_SETUP=false
fi

echo ""
echo "📋 Configuration Steps:"
echo ""

# 1. Repository setup
echo "1. 📁 Repository Setup"
echo "   ✅ GitHub Action workflow created at .github/workflows/rss-analyzer.yml"
echo ""

# 2. Secrets setup
echo "2. 🔐 Required Secrets Setup"
echo ""
echo "   You need to set up the following secrets in your GitHub repository:"
echo "   (Go to: Settings > Secrets and variables > Actions > New repository secret)"
echo ""

echo "   🤖 API_PROVIDER (optional, defaults to 'anthropic'):"
echo "       - 'anthropic' for Claude"
echo "       - 'mistral' for Mistral AI"  
echo "       - 'openai' for OpenAI"
echo ""

echo "   🔑 At least one API key is required:"
echo "       - ANTHROPIC_API_KEY=sk-your-anthropic-key-here"
echo "       - MISTRAL_API_KEY=your-mistral-key-here"
echo "       - OPENAI_API_KEY=sk-your-openai-key-here"
echo ""

# 3. Variables setup
echo "3. ⚙️ Optional Variables Setup"
echo "   (Go to: Settings > Secrets and variables > Actions > Variables tab)"
echo ""
echo "   - RSS_FEED_URL (default: https://bg.raindrop.io/rss/public/57118738)"
echo "   - MAX_LINKED_ARTICLES (default: 3)"
echo "   - SCRAPER_DELAY (default: 1.0)"
echo "   - REQUEST_TIMEOUT (default: 30)"
echo ""

# 4. Permissions setup
echo "4. 🔒 Repository Permissions"
echo "   Go to: Settings > Actions > General"
echo "   ✅ Ensure 'Read and write permissions' is enabled for GITHUB_TOKEN"
echo "   ✅ Allow GitHub Actions to create and approve pull requests"
echo ""

# 5. GitHub CLI setup (if available)
if [ "$MANUAL_SETUP" = false ]; then
    echo "5. 🛠️ Automated Setup with GitHub CLI"
    echo ""
    
    read -p "Would you like to set up secrets automatically with GitHub CLI? (y/n): " setup_auto
    
    if [[ $setup_auto == [Yy]* ]]; then
        echo ""
        echo "Setting up repository secrets..."
        
        # Check if user is logged in
        if ! gh auth status &> /dev/null; then
            echo "Please log in to GitHub CLI first:"
            gh auth login
        fi
        
        # API Provider
        read -p "Enter API provider (anthropic/mistral/openai) [anthropic]: " api_provider
        api_provider=${api_provider:-anthropic}
        gh secret set API_PROVIDER --body "$api_provider"
        echo "✅ Set API_PROVIDER"
        
        # API Key based on provider
        case $api_provider in
            anthropic)
                read -s -p "Enter your Anthropic API key: " api_key
                echo
                gh secret set ANTHROPIC_API_KEY --body "$api_key"
                echo "✅ Set ANTHROPIC_API_KEY"
                ;;
            mistral)
                read -s -p "Enter your Mistral API key: " api_key
                echo
                gh secret set MISTRAL_API_KEY --body "$api_key"
                echo "✅ Set MISTRAL_API_KEY"
                ;;
            openai)
                read -s -p "Enter your OpenAI API key: " api_key
                echo
                gh secret set OPENAI_API_KEY --body "$api_key"
                echo "✅ Set OPENAI_API_KEY"
                ;;
        esac
        
        # Optional: RSS Feed URL
        read -p "Enter RSS feed URL [use default]: " rss_url
        if [[ -n "$rss_url" ]]; then
            gh variable set RSS_FEED_URL --body "$rss_url"
            echo "✅ Set RSS_FEED_URL"
        fi
        
        echo ""
        echo "✅ Secrets configured successfully!"
    fi
else
    echo "5. 📝 Manual Setup Required"
    echo "   Please manually configure secrets in GitHub web interface"
fi

echo ""
echo "6. 🚀 Deployment"
echo ""

# Commit and push the workflow
if [ -n "$(git status --porcelain)" ]; then
    echo "Committing GitHub Action workflow..."
    git add .github/workflows/rss-analyzer.yml
    git add setup_github_action.sh
    
    if git diff --staged --quiet; then
        echo "No changes to commit"
    else
        git commit -m "Add GitHub Actions workflow for RSS analyzer

- Runs automatically every hour
- Stores outputs in repository
- Supports manual triggering
- Creates releases for significant updates

🤖 Generated with Claude Code"
        
        echo "✅ Committed workflow files"
        
        read -p "Push to GitHub now? (y/n): " push_now
        if [[ $push_now == [Yy]* ]]; then
            git push
            echo "✅ Pushed to GitHub"
            
            if command -v gh &> /dev/null; then
                REPO_URL=$(gh repo view --json url -q .url)
                echo ""
                echo "🎉 Setup complete!"
                echo ""
                echo "Your RSS analyzer is now configured to run automatically!"
                echo "📊 View runs at: $REPO_URL/actions"
                echo "📝 View outputs at: $REPO_URL/blob/main/output/articles_by_date.md"
                echo ""
                echo "🔧 To trigger manually: gh workflow run rss-analyzer.yml"
                echo "📱 Or use the GitHub web interface: $REPO_URL/actions/workflows/rss-analyzer.yml"
            fi
        fi
    fi
else
    echo "✅ No changes to commit - workflow already exists"
fi

echo ""
echo "📚 Next Steps:"
echo "1. Verify secrets are set correctly in GitHub"
echo "2. Check that repository has write permissions for Actions"
echo "3. The workflow will run automatically every hour"
echo "4. Manual runs can be triggered from the Actions tab"
echo "5. Outputs will be committed to the repository automatically"
echo ""
echo "🔍 Monitor progress:"
echo "   - GitHub Actions tab for run status"
echo "   - output/articles_by_date.md for results"
echo "   - Releases for periodic snapshots"