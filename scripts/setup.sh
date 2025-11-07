#!/bin/bash
# RSS Analyzer Setup Script
# Automated setup for development environment

set -e  # Exit on error

echo "🚀 RSS Analyzer Setup Script"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check prerequisites
info "Checking prerequisites..."

# Check Python version
if ! command_exists python3; then
    error "Python 3 is not installed. Please install Python 3.11 or 3.12."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d '.' -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d '.' -f 2)

if [ "$PYTHON_MAJOR" -eq 3 ] && ([ "$PYTHON_MINOR" -eq 11 ] || [ "$PYTHON_MINOR" -eq 12 ]); then
    success "Python $PYTHON_VERSION detected"
else
    error "Python 3.11 or 3.12 required, found $PYTHON_VERSION"
    exit 1
fi

# Step 2: Install UV package manager
info "Checking UV package manager..."

if ! command_exists uv; then
    info "Installing UV package manager..."

    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        warning "Please install UV manually: https://github.com/astral-sh/uv"
        exit 1
    fi

    # Add UV to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    success "UV installed"
else
    success "UV already installed ($(uv --version))"
fi

# Step 3: Install dependencies
info "Installing Python dependencies..."

uv sync

success "Dependencies installed"

# Step 4: Set up environment file
info "Setting up environment configuration..."

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        success "Created .env from .env.example"
        warning "Please edit .env and add your API keys"
    else
        info "Creating .env file..."
        cat > .env << 'EOF'
# API Configuration
API_PROVIDER=anthropic  # or 'mistral' or 'openai'

# Anthropic Claude (default)
ANTHROPIC_API_KEY=sk-ant-your-key-here
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# RSS Feed
RSS_FEED_URL=https://rss.arxiv.org/rss/cs.AI

# Processing
MAX_ARTICLES_PER_RUN=10
SCRAPER_DELAY=1.0
FOLLOW_LINKS=true
MAX_LINKED_ARTICLES=3

# Cache
CACHE_ENABLED=true

# Database
DB_POOL_SIZE=5
EOF
        success "Created .env file"
        warning "Please edit .env and add your API keys"
    fi
else
    warning ".env file already exists (skipping)"
fi

# Step 5: Create required directories
info "Creating required directories..."

mkdir -p data
mkdir -p output
mkdir -p logs

success "Directories created"

# Step 6: Install Lefthook (optional)
info "Installing Lefthook git hooks (optional)..."

if command_exists lefthook; then
    lefthook install
    success "Lefthook hooks installed"
elif command_exists npm; then
    read -p "Lefthook not found. Install via npm? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        npm install -g @arkweid/lefthook
        lefthook install
        success "Lefthook installed and hooks configured"
    else
        warning "Skipping Lefthook installation"
    fi
else
    warning "Lefthook not installed (optional)"
    info "To install: brew install lefthook (macOS) or see https://github.com/evilmartians/lefthook"
fi

# Step 7: Run initial tests
info "Running initial validation..."

# Test imports
if uv run python -c "from src.core import DatabaseManager, ContentCache" 2>/dev/null; then
    success "Core imports working"
else
    error "Core imports failed"
    exit 1
fi

# Initialize database (creates tables)
if uv run python -c "from src.core import DatabaseManager; DatabaseManager('data/articles.db').init_database()" 2>/dev/null; then
    success "Database initialized"
else
    error "Database initialization failed"
    exit 1
fi

# Step 8: Summary
echo ""
echo "================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Edit .env and add your API key:"
echo -e "     ${BLUE}nano .env${NC}"
echo ""
echo "  2. Test API connection:"
echo -e "     ${BLUE}uv run python -m src.main test-api${NC}"
echo ""
echo "  3. Run your first analysis:"
echo -e "     ${BLUE}uv run python -m src.main run --limit 3${NC}"
echo ""
echo "  4. View cache stats:"
echo -e "     ${BLUE}uv run python -m src.main cache-stats${NC}"
echo ""
echo "  5. Run tests:"
echo -e "     ${BLUE}uv run pytest${NC}"
echo ""
echo "Documentation:"
echo "  - Quick Start: docs/QUICK_START.md"
echo "  - Architecture Review: docs/ARCHITECTURAL_REVIEW.md"
echo "  - Implementation Roadmap: docs/IMPLEMENTATION_ROADMAP.md"
echo ""
echo -e "${GREEN}Happy analyzing! 🚀${NC}"
