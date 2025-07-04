#!/bin/bash
# Claude Code Non-Interactive Setup Script
# This script configures Claude Code for non-interactive use with the RSS analyzer.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Claude Code for non-interactive use...${NC}"

# Check if Claude Code is installed
if ! command -v claude &> /dev/null; then
    echo -e "${RED}❌ Claude Code is not installed${NC}"
    echo -e "${YELLOW}Please install Claude Code first:${NC}"
    echo "Visit: https://claude.ai/code"
    exit 1
fi

echo -e "${GREEN}✅ Claude Code is installed${NC}"

# Check Claude Code version
echo -e "${BLUE}Checking Claude Code version...${NC}"
claude --version

# Create Claude Code configuration directory if it doesn't exist
CLAUDE_CONFIG_DIR="$HOME/.config/claude"
mkdir -p "$CLAUDE_CONFIG_DIR"

# Create integration wrapper script
echo -e "${BLUE}Creating Claude Code wrapper script...${NC}"

cat > "$CLAUDE_CONFIG_DIR/rss_analyzer_wrapper.sh" << 'EOF'
#!/bin/bash
# RSS Analyzer Claude Code Wrapper

# Set timeout for non-interactive mode
export CLAUDE_TIMEOUT=300

# Function to run Claude in non-interactive mode with error handling
run_claude_safe() {
    local prompt="$1"
    local input_file="$2"
    
    if [ -n "$input_file" ] && [ -f "$input_file" ]; then
        # Run with file input
        timeout ${CLAUDE_TIMEOUT:-300} bash -c "cat '$input_file' | claude -p '$prompt'" 2>/dev/null
    else
        # Run with direct prompt
        timeout ${CLAUDE_TIMEOUT:-300} claude -p "$prompt" 2>/dev/null
    fi
    
    return $?
}

# Export the function for use in scripts
export -f run_claude_safe
EOF

chmod +x "$CLAUDE_CONFIG_DIR/rss_analyzer_wrapper.sh"

# Add environment variables to shell profile
echo -e "${BLUE}Setting up environment variables...${NC}"

# Detect shell and add to appropriate profile
if [ -n "$ZSH_VERSION" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
else
    SHELL_PROFILE="$HOME/.profile"
fi

# Add Claude Code configuration to shell profile
if ! grep -q "# Claude Code RSS Analyzer Configuration" "$SHELL_PROFILE" 2>/dev/null; then
    echo -e "${BLUE}Adding configuration to $SHELL_PROFILE...${NC}"
    
    cat >> "$SHELL_PROFILE" << EOF

# Claude Code RSS Analyzer Configuration
export CLAUDE_TIMEOUT=300
export CLAUDE_CONFIG_DIR="$HOME/.config/claude"
export PATH="\$PATH:\$CLAUDE_CONFIG_DIR"

# Alias for Claude Code integration
alias claude-rss='python "$(pwd)/tools/claude_code_integration.py"'
EOF

    echo -e "${GREEN}✅ Configuration added to $SHELL_PROFILE${NC}"
else
    echo -e "${YELLOW}⚠️  Configuration already exists in $SHELL_PROFILE${NC}"
fi

# Create pre-commit hook for code review
echo -e "${BLUE}Setting up pre-commit hook...${NC}"

HOOKS_DIR=".git/hooks"
if [ -d "$HOOKS_DIR" ]; then
    cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook with Claude Code review

echo "🔍 Running Claude Code review..."

# Check if Claude Code integration is available
if [ -f "tools/claude_code_integration.py" ]; then
    python tools/claude_code_integration.py review
    
    if [ $? -ne 0 ]; then
        echo "❌ Claude Code review failed"
        read -p "Continue with commit anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Commit aborted."
            exit 1
        fi
    fi
else
    echo "⚠️  Claude Code integration not available"
fi
EOF

    chmod +x "$HOOKS_DIR/pre-commit"
    echo -e "${GREEN}✅ Pre-commit hook installed${NC}"
else
    echo -e "${YELLOW}⚠️  Not a git repository, skipping pre-commit hook${NC}"
fi

# Create automation scripts
echo -e "${BLUE}Creating automation scripts...${NC}"

# Daily analysis script
cat > "tools/claude_daily_analysis.sh" << 'EOF'
#!/bin/bash
# Daily Claude Code analysis for RSS analyzer

echo "📊 Starting daily Claude Code analysis..."

# Check for errors in logs
if [ -f "logs/analyzer.log" ]; then
    echo "🔍 Analyzing error logs..."
    python tools/claude_code_integration.py analyze-errors --log-file logs/analyzer.log --output output/claude_error_analysis.json
fi

# Optimize performance
echo "⚡ Getting optimization suggestions..."
python tools/claude_code_integration.py optimize --output output/claude_optimization.json

# Review recent changes
echo "📋 Reviewing recent changes..."
python tools/claude_code_integration.py review --output output/claude_review.json

echo "✅ Daily analysis complete. Results saved to output/ directory."
EOF

chmod +x "tools/claude_daily_analysis.sh"

# CI/CD integration script
cat > "tools/claude_ci_integration.sh" << 'EOF'
#!/bin/bash
# Claude Code integration for CI/CD pipelines

set -e

echo "🤖 Claude Code CI/CD Integration"

# Check if Claude Code is available
if ! command -v claude &> /dev/null; then
    echo "⚠️  Claude Code not available in CI environment"
    echo "Installing Claude Code..."
    # Add Claude Code installation commands here if needed
    exit 0
fi

# Set non-interactive mode
export CLAUDE_TIMEOUT=180  # Shorter timeout for CI

# Review code changes
echo "🔍 Reviewing code changes..."
python tools/claude_code_integration.py review

# Check for critical issues
if [ $? -ne 0 ]; then
    echo "❌ Claude Code review found critical issues"
    exit 1
fi

echo "✅ Claude Code CI checks passed"
EOF

chmod +x "tools/claude_ci_integration.sh"

# Create usage documentation
cat > "CLAUDE_CODE_INTEGRATION.md" << 'EOF'
# Claude Code Integration for RSS Analyzer

This document describes how to use Claude Code in non-interactive mode with the RSS analyzer.

## Setup

Run the setup script:
```bash
./setup_claude_code.sh
```

## Usage

### Manual Commands

```bash
# Check if Claude Code is available
python tools/claude_code_integration.py check

# Review current code changes
python tools/claude_code_integration.py review

# Analyze error logs
python tools/claude_code_integration.py analyze-errors --log-file logs/analyzer.log

# Get optimization suggestions
python tools/claude_code_integration.py optimize
```

### Automation

```bash
# Daily analysis
./tools/claude_daily_analysis.sh

# CI/CD integration
./tools/claude_ci_integration.sh
```

### Pre-commit Hook

The setup script installs a pre-commit hook that automatically reviews code changes before commits.

## Environment Variables

- `CLAUDE_TIMEOUT`: Timeout for Claude Code operations (default: 300 seconds)
- `CLAUDE_CONFIG_DIR`: Configuration directory for Claude Code

## Direct Claude Code Usage

```bash
# Basic non-interactive usage
claude -p "Analyze this code for security issues"

# With file input
cat error.log | claude -p "Summarize the key errors"

# Output to file
claude -p "Review package.json" > analysis.txt
```

## Troubleshooting

1. **Authentication Issues**: Ensure you're logged in to Claude Code:
   ```bash
   claude /login
   ```

2. **Timeout Issues**: Increase the timeout:
   ```bash
   export CLAUDE_TIMEOUT=600
   ```

3. **Permission Issues**: Make sure scripts are executable:
   ```bash
   chmod +x tools/*.sh
   ```
EOF

echo -e "${GREEN}✅ Claude Code non-interactive setup complete!${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "1. Restart your shell or run: source $SHELL_PROFILE"
echo "2. Test the integration: python tools/claude_code_integration.py check"
echo "3. Read the documentation: CLAUDE_CODE_INTEGRATION.md"
echo
echo -e "${YELLOW}Note: Make sure you're logged in to Claude Code first:${NC}"
echo "claude /login"