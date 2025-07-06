#!/bin/bash

# API Status Checker Script
# This script provides automated logs to monitor API health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 RSS Analyzer API Status Monitor${NC}"
echo "========================================="

# Check if we're in the right directory
if [[ ! -f "config/config.yaml" ]]; then
    echo -e "${RED}❌ Error: Please run this script from the RSS analyzer root directory${NC}"
    exit 1
fi

# Function to log with timestamp
log_with_timestamp() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a logs/api_status.log
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Header for this monitoring session
echo "" >> logs/api_status.log
log_with_timestamp "=== API Status Check Started ==="

echo -e "\n${YELLOW}⚡ Quick API Status Check${NC}"
echo "================================="

# Run quick check with Docker
if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
    echo "Using Docker environment..."
    log_with_timestamp "Running Docker-based API check"

    # Quick test of each API
    echo -e "\n${BLUE}Testing Anthropic API...${NC}"
    if docker compose run --rm rss-analyzer python tools/quick_api_check.py 2>&1 | grep -q "Anthropic.*✅"; then
        echo -e "${GREEN}✅ Anthropic API: Working${NC}"
        log_with_timestamp "Anthropic API: WORKING"
    else
        echo -e "${RED}❌ Anthropic API: Failed${NC}"
        log_with_timestamp "Anthropic API: FAILED"
    fi

    echo -e "\n${BLUE}Testing Mistral API...${NC}"
    if docker compose run --rm rss-analyzer python tools/quick_api_check.py 2>&1 | grep -q "Mistral.*✅"; then
        echo -e "${GREEN}✅ Mistral API: Working${NC}"
        log_with_timestamp "Mistral API: WORKING"
    else
        echo -e "${RED}❌ Mistral API: Failed${NC}"
        log_with_timestamp "Mistral API: FAILED"
    fi

    echo -e "\n${BLUE}Testing OpenAI API...${NC}"
    if docker compose run --rm rss-analyzer python tools/quick_api_check.py 2>&1 | grep -q "OpenAI.*✅"; then
        echo -e "${GREEN}✅ OpenAI API: Working${NC}"
        log_with_timestamp "OpenAI API: WORKING"
    else
        echo -e "${RED}❌ OpenAI API: Failed${NC}"
        log_with_timestamp "OpenAI API: FAILED"
    fi

else
    echo "Docker not available, using local Python..."
    log_with_timestamp "Running local Python API check"

    if python tools/quick_api_check.py; then
        log_with_timestamp "API check completed successfully"
    else
        log_with_timestamp "API check failed"
    fi
fi

echo -e "\n${YELLOW}📊 Detailed API Analysis${NC}"
echo "============================="

# Run comprehensive monitoring if requested
if [[ "$1" == "--detailed" ]]; then
    echo "Running comprehensive API health monitoring..."
    log_with_timestamp "Starting detailed API analysis"

    if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
        docker compose run --rm rss-analyzer python tools/api_health_monitor.py 2>&1 | tee -a logs/api_status.log
    else
        python tools/api_health_monitor.py 2>&1 | tee -a logs/api_status.log
    fi

    log_with_timestamp "Detailed API analysis completed"

    echo -e "\n${GREEN}📄 Reports generated:${NC}"
    echo "- logs/api_status.log (this session)"
    echo "- logs/api_health_report.json (detailed JSON)"
    echo "- logs/api_health.log (monitor logs)"
fi

# Check recent processing failures
echo -e "\n${YELLOW}📋 Recent Processing History${NC}"
echo "================================"

echo "Checking database for recent API failures..."
log_with_timestamp "Checking recent processing failures"

# Query recent processing failures
if [[ -f "data/articles.db" ]]; then
    echo -e "\n${BLUE}Last 5 processing attempts:${NC}"

    FAILURES=$(sqlite3 data/articles.db "
        SELECT timestamp, action, error_message
        FROM processing_log
        WHERE action = 'pipeline_failed'
        ORDER BY timestamp DESC
        LIMIT 5;
    ")

    if [[ -n "$FAILURES" ]]; then
        echo "$FAILURES" | while IFS='|' read -r timestamp action error; do
            echo "  🔴 $timestamp: $error"
            log_with_timestamp "Processing failure: $error"
        done
    else
        echo -e "  ${GREEN}✅ No recent failures found${NC}"
        log_with_timestamp "No recent processing failures"
    fi

    # Check when articles were last successfully processed
    LAST_SUCCESS=$(sqlite3 data/articles.db "
        SELECT MAX(processed_date)
        FROM articles;
    ")

    if [[ -n "$LAST_SUCCESS" ]]; then
        echo -e "\n${BLUE}Last successful processing:${NC} $LAST_SUCCESS"
        log_with_timestamp "Last successful processing: $LAST_SUCCESS"
    fi

else
    echo -e "${RED}❌ Database not found${NC}"
    log_with_timestamp "Database not found"
fi

# Show current RSS feed status
echo -e "\n${YELLOW}📡 RSS Feed Status${NC}"
echo "===================="

echo "Checking RSS feed connectivity..."
log_with_timestamp "Checking RSS feed status"

if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
    RSS_STATUS=$(docker compose run --rm rss-analyzer test-rss 2>&1)
    echo "$RSS_STATUS"

    if echo "$RSS_STATUS" | grep -q "✅.*Entries:"; then
        ENTRY_COUNT=$(echo "$RSS_STATUS" | grep -o "Entries: [0-9]*" | grep -o "[0-9]*")
        log_with_timestamp "RSS feed working: $ENTRY_COUNT entries found"
    else
        log_with_timestamp "RSS feed check failed"
    fi
fi

# Recommendations
echo -e "\n${YELLOW}💡 Recommendations${NC}"
echo "==================="

echo "Based on the API status:"

# Check which APIs are working
WORKING_APIS=0

if docker compose run --rm rss-analyzer python tools/quick_api_check.py 2>&1 | grep -q "Working APIs: [1-9]"; then
    WORKING_COUNT=$(docker compose run --rm rss-analyzer python tools/quick_api_check.py 2>&1 | grep "Working APIs:" | grep -o "[0-9]*/[0-9]*" | cut -d'/' -f1)

    if [[ "$WORKING_COUNT" -gt 0 ]]; then
        echo -e "${GREEN}✅ $WORKING_COUNT API(s) are working - system can process articles${NC}"
        log_with_timestamp "Recommendation: System ready - $WORKING_COUNT APIs working"

        echo -e "\n${BLUE}To process pending articles:${NC}"
        echo "  docker compose run rss-analyzer run --limit 5"
    else
        echo -e "${RED}❌ No APIs are working - system cannot process articles${NC}"
        log_with_timestamp "Recommendation: All APIs failed - manual intervention needed"

        echo -e "\n${BLUE}Troubleshooting steps:${NC}"
        echo "  1. Check API keys and credits"
        echo "  2. Verify network connectivity"
        echo "  3. Wait for rate limits to reset"
        echo "  4. Contact API providers if issues persist"
    fi
else
    echo -e "${YELLOW}⚠️ Unable to determine API status${NC}"
    log_with_timestamp "Warning: Could not determine API status"
fi

# Footer
log_with_timestamp "=== API Status Check Completed ==="
echo ""
echo -e "${GREEN}📝 Full logs saved to: logs/api_status.log${NC}"

if [[ "$1" == "--detailed" ]]; then
    echo -e "${GREEN}📊 JSON report saved to: logs/api_health_report.json${NC}"
fi

echo -e "${BLUE}💡 Run with --detailed for comprehensive analysis${NC}"
