#!/bin/bash

# RSS Analyzer Hourly Runner
# This script runs the RSS analyzer and generates only the articles by date file

set -e

# Change to the project directory
cd "$(dirname "$0")"

# Log file for tracking runs
LOG_FILE="logs/hourly_runs.log"

# Ensure log directory exists
mkdir -p logs

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log "Starting hourly RSS analyzer run"

# Run the analyzer
log "Running RSS analyzer..."
if docker compose run --rm rss-analyzer run; then
    log "RSS analyzer completed successfully"
else
    log "ERROR: RSS analyzer failed with exit code $?"
    exit 1
fi

# Generate only the articles by date file
log "Generating articles by date file..."
if python generate_articles_by_date.py; then
    log "Articles by date file generated successfully"
else
    log "ERROR: Failed to generate articles by date file"
    exit 1
fi

# Optional: Clean up old orphaned containers
log "Cleaning up orphaned containers..."
docker container prune -f > /dev/null 2>&1 || true

log "Hourly run completed successfully"
log "----------------------------------------"