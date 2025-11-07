# Integration Tests - Quick Start Guide

## Installation

```bash
# Install dependencies (includes pytest)
uv sync

# Or with pip
pip install -r requirements.txt
```

## Running Tests

### Run All Integration Tests
```bash
# All tests
uv run pytest tests/integration/ -v

# With coverage
uv run pytest tests/integration/ --cov=src --cov-report=term-missing
```

### Run Specific Test Suites

```bash
# Full pipeline tests
uv run pytest tests/integration/test_full_pipeline.py -v

# Performance benchmarks (with detailed output)
uv run pytest tests/integration/test_performance.py -v -s

# Stress tests (exclude slow tests)
uv run pytest tests/integration/test_stress.py -v -m "not slow"

# Security tests
uv run pytest tests/integration/test_security.py -v

# Cost analysis
uv run pytest tests/integration/test_cost_tracking.py -v -s

# Regression tests
uv run pytest tests/integration/test_regression.py -v

# Configuration matrix
uv run pytest tests/integration/test_matrix.py -v
```

### Run Specific Tests

```bash
# Single test class
uv run pytest tests/integration/test_performance.py::TestSyncVsAsyncPerformance -v

# Single test method
uv run pytest tests/integration/test_performance.py::TestSyncVsAsyncPerformance::test_sync_vs_async_comparison -v
```

## Test Markers

```bash
# Run only performance tests
uv run pytest tests/integration/ -v -m performance

# Run only security tests
uv run pytest tests/integration/ -v -m security

# Exclude slow tests
uv run pytest tests/integration/ -v -m "not slow"

# Run stress tests only
uv run pytest tests/integration/ -v -m stress
```

## Generating Reports

### HTML Report
```bash
# Generate HTML test report
uv run pytest tests/integration/ --html=reports/integration_tests.html --self-contained-html
```

### Coverage Report
```bash
# Generate HTML coverage report
uv run pytest tests/integration/ --cov=src --cov-report=html

# Open in browser
open htmlcov/index.html
```

### JSON Report (for CI/CD)
```bash
# Generate machine-readable report
uv run pytest tests/integration/ --json-report --json-report-file=reports/integration_tests.json
```

## Performance Benchmarks

### Quick Performance Check
```bash
# Run key performance tests
uv run pytest tests/integration/test_performance.py::TestSyncVsAsyncPerformance -v -s
uv run pytest tests/integration/test_performance.py::TestDatabasePerformance -v -s
uv run pytest tests/integration/test_performance.py::TestCachePerformance -v -s
```

### Full Performance Suite
```bash
# All performance benchmarks with output
uv run pytest tests/integration/test_performance.py -v -s
```

Expected output:
```
=== Async Performance Benchmark ===
Sync baseline: 1.00s
Async duration: 0.15s
Speedup: 6.7x
Target: 6-8x speedup ✓

=== Database Bulk Insert ===
Inserted 500 articles in 0.25s
Rate: 2000.0 inserts/second ✓

=== Cache Performance ===
Cache hit rate: 80.0%
Target: 85%+ ✓
```

## Stress Testing

### Quick Stress Test
```bash
# Run without slow tests
uv run pytest tests/integration/test_stress.py -v -m "not slow"
```

### Full Stress Test (includes 100+ article tests)
```bash
# Run all stress tests including slow ones
uv run pytest tests/integration/test_stress.py -v
```

## Security Validation

```bash
# Run all security tests
uv run pytest tests/integration/test_security.py -v -s

# Specific security tests
uv run pytest tests/integration/test_security.py::TestSQLInjectionPrevention -v
uv run pytest tests/integration/test_security.py::TestXSSPrevention -v
uv run pytest tests/integration/test_security.py::TestRateLimitingSecurity -v
```

## Cost Analysis

```bash
# Run cost tracking tests with detailed output
uv run pytest tests/integration/test_cost_tracking.py -v -s
```

Expected output:
```
=== Cost Analysis Report ===

API Usage:
  Total Requests: 50
  API Calls: 10
  Cache Hits: 40
  Cache Hit Rate: 80.0%

Costs:
  Total Cost: $0.54
  Cost per Request: $0.0108

Savings:
  Without Cache: $2.70
  With Cache: $0.54
  Savings: $2.16
  Savings %: 80.0% ✓
```

## Configuration Matrix Testing

```bash
# Test all 12 configuration combinations
uv run pytest tests/integration/test_matrix.py -v

# Test specific provider
uv run pytest tests/integration/test_matrix.py::TestProviderMatrix -v --provider=anthropic

# Test with parametrization
uv run pytest tests/integration/test_matrix.py::TestFullMatrix::test_full_configuration_matrix -v
```

## Troubleshooting

### Tests Fail with "Unknown provider: test"
**Fix**: Tests are configured to use mock providers. Update tests to use valid provider or ensure proper mocking.

### Tests Fail with "DatabaseManager.insert_article() got unexpected keyword argument"
**Fix**: API signature changed. Remove `description` parameter from test calls.

### Async Tests Timeout
**Fix**: Increase timeout in test or reduce number of test articles.

### Connection Pool Errors
**Fix**: Ensure database connections are properly cleaned up. Use context managers.

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run integration tests
        run: |
          uv run pytest tests/integration/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Quick Verification Checklist

Use this checklist to verify optimizations:

```bash
# ✓ Performance: Async 6-8x faster
uv run pytest tests/integration/test_performance.py::TestSyncVsAsyncPerformance::test_async_processing_performance -v -s

# ✓ Database: Batch operations 8x faster
uv run pytest tests/integration/test_performance.py::TestDatabasePerformance::test_batch_inserts_performance -v -s

# ✓ Cache: 85%+ hit rate
uv run pytest tests/integration/test_performance.py::TestCachePerformance::test_cache_hit_rate -v -s

# ✓ Security: SQL injection prevented
uv run pytest tests/integration/test_security.py::TestSQLInjectionPrevention -v

# ✓ Cost: 70-90% reduction
uv run pytest tests/integration/test_cost_tracking.py::TestCacheCostSavings::test_90_percent_cost_reduction_claim -v -s

# ✓ Regression: All features work
uv run pytest tests/integration/test_regression.py::TestFeatureRegression -v
```

## Best Practices

1. **Always run tests before committing**
   ```bash
   uv run pytest tests/integration/ -v
   ```

2. **Check coverage regularly**
   ```bash
   uv run pytest tests/integration/ --cov=src --cov-report=term-missing
   ```

3. **Run performance benchmarks on changes**
   ```bash
   uv run pytest tests/integration/test_performance.py -v -s
   ```

4. **Verify security after changes**
   ```bash
   uv run pytest tests/integration/test_security.py -v
   ```

5. **Test all configurations**
   ```bash
   uv run pytest tests/integration/test_matrix.py -v
   ```

## Support

For issues or questions:
- Check [INTEGRATION_TEST_REPORT.md](INTEGRATION_TEST_REPORT.md) for detailed documentation
- Review test output for specific error messages
- Ensure all dependencies are installed: `uv sync`
- Check Python version: 3.11+ required

---

**Last Updated**: 2025-11-07
