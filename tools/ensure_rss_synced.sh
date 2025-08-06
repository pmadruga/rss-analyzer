#!/bin/bash
set -e

# Ensure RSS Feed is Synced
# This script checks if all RSS feed articles are analyzed and processes any missing ones

echo "🔍 Checking RSS feed synchronization..."

# Run the RSS sync checker
if uv run python tools/check_rss_sync.py; then
    echo "✅ RSS feed is already synchronized"
else
    echo "📥 Found unprocessed articles, running analyzer..."

    # Run the RSS analyzer to process missing articles
    # Use a reasonable limit to avoid overwhelming the system
    uv run python -m src.main run --limit 20

    # Verify sync after processing
    echo "🔍 Verifying synchronization after processing..."
    if uv run python tools/check_rss_sync.py; then
        echo "✅ RSS feed is now synchronized"
    else
        echo "⚠️  Some articles may still need processing"
        # Don't fail here, let the workflow continue
    fi
fi

# Regenerate website data with any new articles
echo "📊 Regenerating website data..."
uv run python tools/generate_website_data.py

echo "🎉 RSS synchronization complete!"
