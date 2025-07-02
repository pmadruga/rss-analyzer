#!/bin/bash

# Check status of RSS analyzer hourly service

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "RSS Analyzer Hourly Service Status"
echo "=================================="

# Check if LaunchAgent exists (macOS)
SERVICE_NAME="com.rss-analyzer.hourly"
PLIST_FILE="$HOME/Library/LaunchAgents/$SERVICE_NAME.plist"

if [[ -f "$PLIST_FILE" ]]; then
    echo "LaunchAgent service found: $SERVICE_NAME"
    
    # Check if service is loaded
    if launchctl list | grep -q "$SERVICE_NAME"; then
        echo "✅ Service is loaded and active"
        echo "Service details:"
        launchctl list "$SERVICE_NAME" 2>/dev/null || echo "Could not get service details"
    else
        echo "❌ Service exists but is not loaded"
        echo "To start: launchctl load $PLIST_FILE"
    fi
    
elif crontab -l 2>/dev/null | grep -q "run_hourly.sh"; then
    echo "Cron job found for RSS analyzer"
    echo "✅ Cron service is active"
    echo "Cron entry:"
    crontab -l | grep "run_hourly.sh"
    
else
    echo "❌ No hourly service found"
    echo "Run ./setup_hourly_service.sh to set up the service"
fi

echo ""
echo "Recent Activity"
echo "==============="

# Check for recent log entries
if [[ -f "logs/hourly_runs.log" ]]; then
    echo "Last 10 entries from hourly_runs.log:"
    tail -10 "logs/hourly_runs.log"
else
    echo "No hourly run logs found yet"
fi

echo ""
echo "Database Statistics"
echo "=================="
if [[ -f "data/articles.db" ]]; then
    python -c "
import sqlite3
conn = sqlite3.connect('data/articles.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM articles WHERE status = \"completed\"')
completed = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM articles')
total = cursor.fetchone()[0]
cursor.execute('SELECT MAX(processed_date) FROM articles WHERE status = \"completed\"')
last_processed = cursor.fetchone()[0]
conn.close()
print(f'Total articles: {total}')
print(f'Completed: {completed}')
print(f'Last processed: {last_processed or \"None\"}')
"
else
    echo "No database found"
fi

echo ""
echo "Files"
echo "====="
echo "Articles by date file:"
if [[ -f "output/articles_by_date.md" ]]; then
    echo "✅ output/articles_by_date.md exists"
    echo "   Size: $(wc -c < output/articles_by_date.md) bytes"
    echo "   Modified: $(stat -f '%Sm' output/articles_by_date.md)"
else
    echo "❌ output/articles_by_date.md not found"
fi