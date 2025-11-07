#!/bin/bash
##
# Quality Gate Checker
#
# Runs specific quality gate checks for the RSS Analyzer project.
# Usage: ./tools/check_quality_gate.sh [gate_name]
#
# Gates: code-quality, security, async-patterns, test-coverage, documentation, performance, all
##

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Gate status tracking
GATES_PASSED=0
GATES_FAILED=0
GATES_TOTAL=0

##
# Print colored message
##
print_status() {
    local status=$1
    local message=$2

    if [ "$status" = "pass" ]; then
        echo -e "${GREEN}✅ PASS${NC}: $message"
    elif [ "$status" = "fail" ]; then
        echo -e "${RED}❌ FAIL${NC}: $message"
    elif [ "$status" = "warn" ]; then
        echo -e "${YELLOW}⚠️  WARN${NC}: $message"
    elif [ "$status" = "info" ]; then
        echo -e "${BLUE}ℹ️  INFO${NC}: $message"
    fi
}

##
# Check if command exists
##
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

##
# Run gate and track status
##
run_gate() {
    local gate_name=$1
    local gate_command=$2

    GATES_TOTAL=$((GATES_TOTAL + 1))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Running Gate: $gate_name${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval "$gate_command"; then
        print_status "pass" "$gate_name"
        GATES_PASSED=$((GATES_PASSED + 1))
        return 0
    else
        print_status "fail" "$gate_name"
        GATES_FAILED=$((GATES_FAILED + 1))
        return 1
    fi
}

##
# Gate 1: Code Quality
##
check_code_quality() {
    echo "Checking code quality (Ruff, Mypy, Pylint)..."

    # Check Ruff
    echo ""
    echo "→ Running Ruff linter..."
    if ! command_exists ruff; then
        print_status "fail" "Ruff not installed. Run: pip install ruff"
        return 1
    fi

    if ! ruff check src/; then
        print_status "fail" "Ruff linting failed"
        return 1
    fi

    # Check Ruff formatting
    echo ""
    echo "→ Checking Ruff formatting..."
    if ! ruff format --check src/; then
        print_status "fail" "Ruff formatting check failed. Run: ruff format ."
        return 1
    fi

    # Check Mypy
    echo ""
    echo "→ Running Mypy type checker..."
    if ! command_exists mypy; then
        print_status "warn" "Mypy not installed. Skipping type checking."
    else
        if [ -f "mypy.ini" ]; then
            if ! mypy src/ --config-file=mypy.ini; then
                print_status "fail" "Mypy type checking failed"
                return 1
            fi
        else
            if ! mypy src/; then
                print_status "fail" "Mypy type checking failed"
                return 1
            fi
        fi
    fi

    # Check Pylint
    echo ""
    echo "→ Running Pylint quality analysis..."
    if ! command_exists pylint; then
        print_status "warn" "Pylint not installed. Skipping quality analysis."
    else
        PYLINT_OUTPUT=$(pylint src/ --score=y --exit-zero 2>&1)
        PYLINT_SCORE=$(echo "$PYLINT_OUTPUT" | grep "Your code has been rated" | grep -oP '\d+\.\d+' || echo "0")

        echo "Pylint score: $PYLINT_SCORE/10.0"

        if (( $(echo "$PYLINT_SCORE < 8.0" | bc -l) )); then
            print_status "fail" "Pylint score $PYLINT_SCORE < 8.0 threshold"
            return 1
        fi
    fi

    print_status "pass" "All code quality checks passed"
    return 0
}

##
# Gate 2: Security
##
check_security() {
    echo "Checking security (Bandit, Safety)..."

    # Check Bandit
    echo ""
    echo "→ Running Bandit security scan..."
    if ! command_exists bandit; then
        print_status "fail" "Bandit not installed. Run: pip install bandit"
        return 1
    fi

    BANDIT_CONFIG=""
    if [ -f ".bandit" ]; then
        BANDIT_CONFIG="-c .bandit"
    fi

    if ! bandit -r src/ $BANDIT_CONFIG -f json -o bandit-report.json; then
        print_status "warn" "Bandit scan completed with findings"
    fi

    # Check for critical issues
    if [ -f "bandit-report.json" ]; then
        CRITICAL_COUNT=$(jq '[.results[] | select(.issue_severity=="HIGH" or .issue_severity=="CRITICAL")] | length' bandit-report.json 2>/dev/null || echo "0")

        if [ "$CRITICAL_COUNT" -gt 0 ]; then
            print_status "fail" "$CRITICAL_COUNT critical/high security issue(s) found"
            echo ""
            echo "Critical issues:"
            jq '.results[] | select(.issue_severity=="HIGH" or .issue_severity=="CRITICAL") | "\(.filename):\(.line_number): \(.issue_text)"' bandit-report.json 2>/dev/null || true
            return 1
        fi
    fi

    # Check Safety
    echo ""
    echo "→ Running Safety dependency scan..."
    if ! command_exists safety; then
        print_status "warn" "Safety not installed. Skipping dependency scan."
    else
        if ! safety check --json > safety-report.json 2>&1; then
            if [ -f "safety-report.json" ]; then
                VULN_COUNT=$(jq '.vulnerabilities | length' safety-report.json 2>/dev/null || echo "0")
                if [ "$VULN_COUNT" -gt 0 ]; then
                    print_status "fail" "$VULN_COUNT vulnerable dependencies found"
                    return 1
                fi
            fi
        fi
    fi

    print_status "pass" "All security checks passed"
    return 0
}

##
# Gate 3: Async Patterns
##
check_async_patterns() {
    echo "Checking async patterns..."

    # Check for blocking I/O
    echo ""
    echo "→ Checking for blocking I/O in async functions..."
    if [ ! -f "tools/check_async_patterns.py" ]; then
        print_status "fail" "Async pattern checker not found"
        return 1
    fi

    if ! python tools/check_async_patterns.py; then
        return 1
    fi

    # Check connection pool usage
    echo ""
    echo "→ Checking connection pool usage..."
    if [ ! -f "tools/check_pool_usage.py" ]; then
        print_status "fail" "Connection pool checker not found"
        return 1
    fi

    if ! python tools/check_pool_usage.py; then
        return 1
    fi

    print_status "pass" "All async pattern checks passed"
    return 0
}

##
# Gate 4: Test Coverage
##
check_test_coverage() {
    echo "Checking test coverage..."

    if ! command_exists pytest; then
        print_status "fail" "Pytest not installed. Run: pip install pytest pytest-cov"
        return 1
    fi

    echo ""
    echo "→ Running tests with coverage..."

    # Run tests with coverage
    if ! pytest --cov=src --cov-report=json --cov-report=term -v; then
        print_status "fail" "Tests failed"
        return 1
    fi

    # Check coverage threshold
    if [ -f "coverage.json" ]; then
        COVERAGE=$(jq '.totals.percent_covered' coverage.json 2>/dev/null || echo "0")
        echo ""
        echo "Coverage: $COVERAGE%"

        if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            print_status "fail" "Coverage $COVERAGE% < 80% threshold"
            return 1
        fi
    else
        print_status "warn" "Coverage report not found"
    fi

    print_status "pass" "All tests passed with coverage ≥ 80%"
    return 0
}

##
# Gate 5: Documentation
##
check_documentation() {
    echo "Checking documentation..."

    echo ""
    echo "→ Checking for missing docstrings..."

    # Count functions without docstrings
    MISSING_DOCSTRINGS=$(grep -r "^def \|^async def " src/ --include="*.py" | wc -l || echo "0")
    HAS_DOCSTRINGS=$(grep -r '"""' src/ --include="*.py" | wc -l || echo "0")

    echo "Functions found: $MISSING_DOCSTRINGS"
    echo "Docstrings found: $HAS_DOCSTRINGS"

    # Simple heuristic - should have at least 50% docstring coverage
    if [ "$MISSING_DOCSTRINGS" -gt 0 ]; then
        COVERAGE_PCT=$((HAS_DOCSTRINGS * 100 / MISSING_DOCSTRINGS))
        echo "Estimated docstring coverage: $COVERAGE_PCT%"

        if [ "$COVERAGE_PCT" -lt 50 ]; then
            print_status "fail" "Insufficient docstring coverage ($COVERAGE_PCT% < 50%)"
            return 1
        fi
    fi

    echo ""
    echo "→ Checking for missing type hints..."

    # Check for functions with type hints
    FUNCTIONS_WITH_HINTS=$(grep -rE "def.*->" src/ --include="*.py" | wc -l || echo "0")

    if [ "$MISSING_DOCSTRINGS" -gt 0 ]; then
        HINT_COVERAGE=$((FUNCTIONS_WITH_HINTS * 100 / MISSING_DOCSTRINGS))
        echo "Type hint coverage: $HINT_COVERAGE%"

        if [ "$HINT_COVERAGE" -lt 70 ]; then
            print_status "warn" "Low type hint coverage ($HINT_COVERAGE%)"
        fi
    fi

    print_status "pass" "Documentation checks passed"
    return 0
}

##
# Gate 6: Performance
##
check_performance() {
    echo "Checking performance..."

    print_status "info" "Performance regression testing not yet implemented"
    print_status "info" "Manual benchmarking recommended for performance-critical changes"

    # Future: Add performance benchmark comparison
    # if [ -f "tools/performance_benchmark.py" ]; then
    #     python tools/performance_benchmark.py --compare-baseline
    # fi

    return 0
}

##
# Print summary
##
print_summary() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}QUALITY GATE SUMMARY${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Total gates checked: $GATES_TOTAL"
    echo -e "${GREEN}Passed: $GATES_PASSED${NC}"
    echo -e "${RED}Failed: $GATES_FAILED${NC}"
    echo ""

    if [ $GATES_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All quality gates passed!${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}❌ Some quality gates failed.${NC}"
        echo ""
        echo "Please fix the issues above before merging."
        echo ""
        return 1
    fi
}

##
# Main
##
main() {
    local gate="${1:-all}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}RSS Analyzer - Quality Gate Checker${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Checking gate(s): $gate"

    case "$gate" in
        code-quality)
            run_gate "Code Quality" "check_code_quality"
            ;;
        security)
            run_gate "Security" "check_security"
            ;;
        async-patterns)
            run_gate "Async Patterns" "check_async_patterns"
            ;;
        test-coverage)
            run_gate "Test Coverage" "check_test_coverage"
            ;;
        documentation)
            run_gate "Documentation" "check_documentation"
            ;;
        performance)
            run_gate "Performance" "check_performance"
            ;;
        all)
            run_gate "Code Quality" "check_code_quality"
            run_gate "Security" "check_security"
            run_gate "Async Patterns" "check_async_patterns"
            run_gate "Test Coverage" "check_test_coverage"
            run_gate "Documentation" "check_documentation"
            run_gate "Performance" "check_performance"
            ;;
        *)
            echo "Unknown gate: $gate"
            echo ""
            echo "Available gates:"
            echo "  code-quality   - Ruff, Mypy, Pylint"
            echo "  security       - Bandit, Safety"
            echo "  async-patterns - Async/await validation"
            echo "  test-coverage  - Pytest with coverage"
            echo "  documentation  - Docstrings, type hints"
            echo "  performance    - Performance regression"
            echo "  all            - Run all gates"
            exit 1
            ;;
    esac

    print_summary
}

# Run main
main "$@"
