# Claude Code Non-Interactive Setup Guide

This guide shows you how to configure Claude Code for non-interactive use with the RSS analyzer, enabling automated code analysis, error debugging, and optimization suggestions.

## 🚀 Quick Start

### 1. Install Claude Code
```bash
# Visit https://claude.ai/code and follow installation instructions
# Or use the installer script:
curl -sSL https://claude.ai/install.sh | bash
```

### 2. Authenticate Claude Code
```bash
claude /login
```

### 3. Run Setup Script
```bash
./setup_claude_code.sh
```

### 4. Test Integration
```bash
# Restart shell or source profile
source ~/.bashrc  # or ~/.zshrc

# Test Claude Code integration
python tools/claude_code_integration.py check
```

## 📋 Features

### 🔍 **Code Review**
Automatically review code changes before commits:
```bash
# Manual review
python tools/claude_code_integration.py review

# Automatic pre-commit hook (installed by setup script)
git commit -m "Your changes"  # Triggers automatic review
```

### 🐛 **Error Analysis**
Analyze error logs and get fix suggestions:
```bash
python tools/claude_code_integration.py analyze-errors --log-file logs/analyzer.log
```

### ⚡ **Performance Optimization**
Get optimization suggestions for your code:
```bash
python tools/claude_code_integration.py optimize
```

### 🤖 **Automated Analysis**
Run daily automated analysis:
```bash
./tools/claude_daily_analysis.sh
```

## 📚 Usage Examples

### Basic Non-Interactive Commands
```bash
# Analyze code for security issues
claude -p "Review this codebase for security vulnerabilities"

# Analyze log files
cat logs/analyzer.log | claude -p "Summarize errors and suggest fixes"

# Code review with output to file
claude -p "Review recent changes for bugs and improvements" > code_review.txt
```

### Integration Commands
```bash
# Check if Claude Code is working
claude-rss check

# Review current changes
claude-rss review

# Analyze API errors
claude-rss analyze-errors --log-file logs/analyzer.log

# Get performance suggestions
claude-rss optimize
```

### Docker Integration
```bash
# Build Claude Code enabled container
docker compose -f docker-compose.claude.yml build

# Run code review in container
docker compose -f docker-compose.claude.yml run rss-analyzer-claude claude-review

# Run optimization analysis
docker compose -f docker-compose.claude.yml run rss-analyzer-claude claude-optimize

# Analyze errors in container
docker compose -f docker-compose.claude.yml run rss-analyzer-claude claude-analyze-errors
```

## 🔧 Configuration

### Environment Variables
```bash
# Claude Code timeout (seconds)
export CLAUDE_TIMEOUT=300

# Enable non-interactive mode
export CLAUDE_NON_INTERACTIVE=true

# Configuration directory
export CLAUDE_CONFIG_DIR="$HOME/.config/claude"
```

### Project-Specific Settings
Add to your `.env` file:
```bash
# Claude Code Configuration
CLAUDE_TIMEOUT=300
CLAUDE_NON_INTERACTIVE=true
ENABLE_CLAUDE_REVIEW=true
```

## 🔄 Automation Workflows

### Pre-Commit Code Review
Automatically installed by setup script:
```bash
# .git/hooks/pre-commit
#!/bin/bash
echo "🔍 Running Claude Code review..."
python tools/claude_code_integration.py review
```

### Daily Analysis Cron Job
```bash
# Add to crontab: crontab -e
0 9 * * * cd /path/to/rss-analyzer && ./tools/claude_daily_analysis.sh
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Claude Code Review
  run: |
    ./tools/claude_ci_integration.sh
```

## 📊 Output Examples

### Code Review Output
```
📋 Code Review Results:
==================================================
✅ Overall code quality is good
⚠️  Found 2 potential improvements:

1. In src/mistral_client.py:
   - Consider adding type hints to improve code clarity
   - Rate limiting implementation looks good

2. In tools/claude_code_integration.py:
   - Good error handling patterns
   - Suggest adding more detailed logging

🔒 Security: No vulnerabilities detected
⚡ Performance: Code is well-optimized
```

### Error Analysis Output
```
🔍 Error Analysis Results:
==================================================
🎯 Root Cause: API rate limiting issues

🔧 Fixes:
1. Implement exponential backoff for failed requests
2. Add request queue with proper spacing
3. Monitor API usage more closely

📝 Prevention:
- Add rate limit monitoring
- Implement circuit breaker pattern
- Cache responses when possible
```

### Optimization Suggestions
```
⚡ Optimization Suggestions:
==================================================
🚀 Performance Improvements:

1. Database Queries:
   - Use batch inserts for better performance
   - Add database indexes for frequent queries

2. API Calls:
   - Implement connection pooling
   - Add response caching for repeated requests

3. Memory Usage:
   - Use generators for large dataset processing
   - Implement lazy loading for article content
```

## 🛠️ Advanced Usage

### Custom Prompts
```bash
# Custom analysis prompts
claude -p "Analyze this code for Python best practices and PEP 8 compliance"
claude -p "Find potential race conditions and concurrency issues"
claude -p "Suggest refactoring opportunities for better maintainability"
```

### Batch File Analysis
```bash
# Analyze multiple files
find src/ -name "*.py" | head -5 | xargs -I {} claude -p "Review this Python file: {}"
```

### Integration with Other Tools
```bash
# Combine with git hooks
git diff HEAD~1 | claude -p "Review these changes for potential issues"

# Combine with linting
pylint src/ 2>&1 | claude -p "Explain these linting errors and suggest fixes"

# Combine with testing
pytest --tb=short 2>&1 | claude -p "Analyze these test failures and suggest solutions"
```

## 🐛 Troubleshooting

### Common Issues

1. **Claude Code not found**
   ```bash
   # Install Claude Code
   curl -sSL https://claude.ai/install.sh | bash
   # Or visit https://claude.ai/code
   ```

2. **Authentication errors**
   ```bash
   # Re-authenticate
   claude /login
   ```

3. **Timeout issues**
   ```bash
   # Increase timeout
   export CLAUDE_TIMEOUT=600
   ```

4. **Permission denied**
   ```bash
   # Make scripts executable
   chmod +x tools/*.sh setup_claude_code.sh
   ```

5. **Non-interactive mode issues**
   ```bash
   # Ensure proper environment variables
   export CLAUDE_NON_INTERACTIVE=true
   ```

### Debug Mode
```bash
# Enable verbose logging
export CLAUDE_DEBUG=true
python tools/claude_code_integration.py review
```

## 📈 Benefits

- **🔍 Automated Code Review**: Catch issues before they reach production
- **🐛 Intelligent Error Analysis**: Get specific fix suggestions for errors
- **⚡ Performance Optimization**: Identify bottlenecks and optimization opportunities
- **📚 Documentation Generation**: Auto-generate comprehensive documentation
- **🤖 CI/CD Integration**: Seamlessly integrate into your development workflow
- **🔒 Security Analysis**: Detect security vulnerabilities early

## 🔗 Resources

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Code GitHub Issues](https://github.com/anthropics/claude-code/issues)
- [Non-Interactive Mode Discussion](https://github.com/anthropics/claude-code/issues/837)

## 📝 Notes

- Claude Code requires an active internet connection
- Non-interactive mode has some limitations with slash commands
- Rate limiting applies to API calls
- Results quality depends on code complexity and context provided

---

**Ready to get started?** Run `./setup_claude_code.sh` and begin using Claude Code for automated code analysis!
