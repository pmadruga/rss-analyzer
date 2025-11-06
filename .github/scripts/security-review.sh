#!/bin/bash
set -e

# Security Review Script for Code Review Swarm
# Usage: ./security-review.sh <PR_NUMBER> <CHANGED_FILES>

PR_NUM=$1
CHANGED_FILES=$2

echo "🔒 Running security review for PR #$PR_NUM..."

ISSUES=""
SEVERITY="none"
FINDINGS_COUNT=0

# Split comma-separated files
IFS=',' read -ra FILES <<< "$CHANGED_FILES"

for file in "${FILES[@]}"; do
    # Only process Python files
    if [[ "$file" == *.py ]] && [ -f "$file" ]; then
        echo "  Analyzing: $file"

        # Check 1: Hardcoded secrets
        if grep -nE "(API_KEY|PASSWORD|SECRET|TOKEN|PRIVATE_KEY)\s*=\s*['\"][^'\"]{10,}['\"]" "$file"; then
            ISSUES="$ISSUES\n- 🔴 **CRITICAL**: Potential hardcoded secrets in \`$file\`"
            SEVERITY="critical"
            ((FINDINGS_COUNT++))
        fi

        # Check 2: SQL injection risks
        if grep -nE "execute\s*\([^)]*%[^)]*\)|execute\s*\([^)]*\.format\(" "$file"; then
            ISSUES="$ISSUES\n- 🟡 **HIGH**: SQL injection risk in \`$file\` (string formatting in execute)"
            [ "$SEVERITY" != "critical" ] && SEVERITY="high"
            ((FINDINGS_COUNT++))
        fi

        # Check 3: Dangerous function usage
        if grep -nE "eval\(|exec\(|__import__\(|compile\(" "$file"; then
            ISSUES="$ISSUES\n- 🟡 **HIGH**: Dangerous function usage in \`$file\` (eval/exec/compile)"
            [ "$SEVERITY" != "critical" ] && SEVERITY="high"
            ((FINDINGS_COUNT++))
        fi

        # Check 4: Shell command injection
        if grep -nE "os\.system\(|subprocess\.call\([^)]*shell\s*=\s*True" "$file"; then
            ISSUES="$ISSUES\n- 🟠 **MEDIUM**: Shell injection risk in \`$file\` (shell=True detected)"
            [ "$SEVERITY" = "none" ] && SEVERITY="medium"
            ((FINDINGS_COUNT++))
        fi

        # Check 5: Insecure random usage
        if grep -nE "import random[^_]|from random import" "$file"; then
            if grep -q "password\|token\|key\|secret" "$file"; then
                ISSUES="$ISSUES\n- 🟠 **MEDIUM**: Insecure random usage in \`$file\` (use secrets module for crypto)"
                [ "$SEVERITY" = "none" ] && SEVERITY="medium"
                ((FINDINGS_COUNT++))
            fi
        fi

        # Check 6: Missing error handling in API clients
        if echo "$file" | grep -q "_client.py"; then
            if ! grep -q "try:" "$file" || ! grep -q "except" "$file"; then
                ISSUES="$ISSUES\n- 🟢 **LOW**: Missing error handling in API client: \`$file\`"
                [ "$SEVERITY" = "none" ] && SEVERITY="low"
                ((FINDINGS_COUNT++))
            fi
        fi

        # Check 7: Insecure deserialization
        if grep -nE "pickle\.loads\(|yaml\.load\([^)]*Loader\s*=\s*yaml\.Loader" "$file"; then
            ISSUES="$ISSUES\n- 🟡 **HIGH**: Insecure deserialization in \`$file\`"
            [ "$SEVERITY" != "critical" ] && SEVERITY="high"
            ((FINDINGS_COUNT++))
        fi
    fi
done

# Generate review output
if [ -n "$ISSUES" ]; then
    cat > review-output.md << EOF
## 🔒 Security Review Results

**Status**: ⚠️ **Issues Found**
**Severity**: **$SEVERITY**
**Findings**: $FINDINGS_COUNT issue(s) detected

---

### 🚨 Security Issues

$ISSUES

---

### 📋 Recommendations

1. **Secrets Management**:
   - Use environment variables or secrets management service
   - Never commit API keys, passwords, or tokens
   - Use \`.env\` files with \`.gitignore\`

2. **SQL Security**:
   - Use parameterized queries: \`cursor.execute("SELECT * FROM table WHERE id = ?", (id,))\`
   - Avoid string formatting in SQL statements
   - Use ORM (SQLAlchemy) for complex queries

3. **Input Validation**:
   - Validate and sanitize all user inputs
   - Use allowlists over denylists
   - Implement rate limiting for API endpoints

4. **Secure Coding**:
   - Avoid \`eval()\`, \`exec()\`, \`compile()\` with untrusted input
   - Use \`subprocess\` with \`shell=False\` and list arguments
   - Use \`secrets\` module for cryptographic operations

5. **Deserialization**:
   - Use \`yaml.safe_load()\` instead of \`yaml.load()\`
   - Avoid \`pickle\` for untrusted data
   - Validate deserialized objects

---

### 🔗 References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Bandit Security Linter](https://bandit.readthedocs.io/)

---

🤖 *Automated Security Review by Code Review Swarm*
EOF
else
    cat > review-output.md << EOF
## 🔒 Security Review Results

**Status**: ✅ **All Checks Passed**
**Severity**: none
**Findings**: No security issues detected

---

### ✅ Security Checks Performed

- ✅ No hardcoded secrets detected
- ✅ No SQL injection vulnerabilities found
- ✅ No dangerous function usage (eval/exec)
- ✅ No shell injection risks
- ✅ Secure random usage verified
- ✅ Deserialization patterns are safe

---

### 💡 Security Best Practices

While no issues were found, always remember:

1. Use environment variables for sensitive data
2. Implement parameterized queries for database operations
3. Validate and sanitize all user inputs
4. Keep dependencies updated with security patches
5. Enable security scanning in CI/CD

---

🤖 *Automated Security Review by Code Review Swarm*
EOF
fi

echo "✅ Security review complete"
echo "📊 Findings: $FINDINGS_COUNT"
echo "🎯 Severity: $SEVERITY"

# Exit with appropriate code
if [ "$SEVERITY" = "critical" ]; then
    exit 1  # Block PR for critical issues
else
    exit 0  # Warning or success
fi
