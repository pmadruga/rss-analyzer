# Security Guidelines

## 🔒 Protecting Sensitive Information

### ✅ What's Protected by .gitignore

- **API Keys & Secrets**: `.env`, `*.key`, `*_secret`, etc.
- **Database Files**: `*.db`, `*.sqlite` (contain processed data)
- **Log Files**: `logs/`, `*.log` (may contain sensitive information)
- **Cache Files**: `__pycache__/`, `*.pyc`, `.cache/`
- **IDE Files**: `.vscode/`, `.idea/`, `.DS_Store`
- **Virtual Environments**: `venv/`, `.env/`

### 🚨 NEVER Commit These Files

- `.env` files containing real API keys
- Database files with processed content
- Log files that may contain sensitive data
- Backup files or temporary exports
- Any file containing credentials or tokens

### ✅ Safe to Commit

- `.env.example` - Template with placeholder values
- Source code files (`.py`, `.yaml`, etc.)
- Documentation files (`.md`)
- Configuration templates
- Requirements and dependencies
- Docker configurations
- GitHub workflow files

## 🛠️ Setup Guidelines

### For Local Development

1. **Copy the template:**
   ```bash
   cp .env.example .env
   ```

2. **Add your real API keys to .env** (this file is ignored by git)

3. **Never edit .env.example with real keys** - keep it as a template

### For GitHub Actions

1. **Use GitHub Secrets** for API keys:
   - Go to Settings → Secrets and variables → Actions
   - Add secrets like `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, etc.

2. **Use Repository Variables** for non-sensitive config:
   - RSS feed URLs
   - Processing limits
   - Timeout values

### For Production

- **Use environment variables** or secure secret management
- **Never hardcode credentials** in configuration files
- **Rotate API keys regularly**
- **Monitor usage** for unexpected spikes

## 🔍 Security Checklist

Before pushing to GitHub:

- [ ] Check `.env` is in `.gitignore` and not staged
- [ ] Verify no API keys in source code
- [ ] Confirm database files are ignored
- [ ] Review git status for sensitive files
- [ ] Use `.env.example` as template only

### Quick Security Check

```bash
# Check for potential leaks
git status --porcelain | grep -E '\.(env|key|secret)'
grep -r "sk-" . --exclude-dir=.git | grep -v .env.example

# Verify .env is ignored
git check-ignore .env
```

## 📞 If Credentials Are Compromised

1. **Immediately revoke** the exposed API key
2. **Generate a new key** from the provider's console
3. **Update your local .env** and GitHub secrets
4. **Check git history** for any commits containing the key
5. **Consider the repository compromised** and may need to be deleted/recreated

## 🔗 Provider Security Docs

- **Anthropic**: https://docs.anthropic.com/en/api/getting-started
- **Mistral**: https://docs.mistral.ai/platform/
- **OpenAI**: https://platform.openai.com/docs/guides/safety-best-practices
