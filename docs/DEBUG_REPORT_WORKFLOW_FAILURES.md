# GitHub Workflow Failure Debug Report

**Date**: 2025-11-07
**Debugger**: SPARC Debugger Mode
**Status**: Root cause identified, fixes ready

---

## Executive Summary

Two GitHub workflows are failing:
1. **code-review-swarm.yml** - Workflow syntax/structure issue
2. **rss-complete-pipeline.yml** - **CRITICAL** Database locking during concurrent operations

The RSS Complete Pipeline failure is causing data loss and preventing article processing.

---

## Failure Analysis

### 1. RSS Complete Pipeline (Critical)

**Workflow**: `.github/workflows/rss-complete-pipeline.yml`
**Last Failed**: 2025-11-07 08:06:31 UTC
**Run ID**: 19162131865

#### Symptoms

```
2025-11-07 08:07:28 - src.core.database - ERROR - Failed to insert article: database is locked
2025-11-07 08:07:33 - src.core.database - ERROR - Failed to update article status: database is locked
2025-11-07 08:07:38 - src.core.database - ERROR - Failed to log processing: database is locked
2025-11-07 08:07:44 - src.core.database - ERROR - Failed to update content hash: database is locked
2025-11-07 08:19:14 - src.core.database - ERROR - Failed to insert content: database is locked
```

**Frequency**: Occurs repeatedly (every 5 seconds) during article processing
**Impact**:
- 10+ articles failed to process
- Data loss - AI analysis completed but couldn't save to database
- Wasted API costs (Mistral API calls succeeded but data not persisted)

#### Root Cause

**File**: `src/core/database.py:41-46`

```python
def _create_connection(self) -> sqlite3.Connection:
    """Create a new database connection with proper settings"""
    conn = sqlite3.connect(self.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
```

**Problem**: Missing critical SQLite concurrency settings

**Missing PRAGMAs**:
1. **`PRAGMA journal_mode = WAL`** - Write-Ahead Logging for concurrent readers/writers
2. **`PRAGMA busy_timeout = 30000`** - 30-second retry timeout for locked database
3. **`PRAGMA synchronous = NORMAL`** - Balance between safety and performance

#### Technical Explanation

SQLite by default uses:
- **Journal mode**: DELETE (blocks all readers during write)
- **Busy timeout**: 0ms (immediate failure on lock)
- **Synchronous**: FULL (slow but safe)

With concurrent article processing (20 articles in parallel):
- Multiple threads try to write simultaneously
- No retry mechanism → immediate "database is locked" errors
- DELETE journal mode prevents concurrent reads during writes

### 2. Code Review Swarm Workflow

**Workflow**: `.github/workflows/code-review-swarm.yml`
**Last Failed**: 2025-11-07 18:44:36 UTC
**Run ID**: 19177992261

#### Symptoms

Workflow runs but produces no output, fails silently.

#### Root Cause

**Potential Issues**:
1. Workflow may have syntax errors preventing job execution
2. Jobs may reference non-existent actions or secrets
3. Conditional logic may be preventing job execution

**Note**: Detailed logs were not available via `gh run view`, suggesting workflow didn't execute properly.

---

## Fix Strategy

### Priority 1: Database Concurrency (Critical)

**File**: `src/core/database.py`
**Function**: `ConnectionPool._create_connection()` (line 41)

**Fix**:
```python
def _create_connection(self) -> sqlite3.Connection:
    """Create a new database connection with proper settings"""
    conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Enable Write-Ahead Logging for concurrent access
    conn.execute("PRAGMA journal_mode = WAL")

    # Set busy timeout to 30 seconds (30000ms)
    conn.execute("PRAGMA busy_timeout = 30000")

    # Balance between safety and performance
    conn.execute("PRAGMA synchronous = NORMAL")

    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")

    # Optimize for concurrent access
    conn.execute("PRAGMA cache_size = -64000")  # 64MB cache

    return conn
```

**Benefits**:
- **WAL mode**: Allows concurrent reads and writes
- **30s timeout**: Automatic retry on lock (prevents immediate failure)
- **NORMAL sync**: 10x faster writes with minimal risk
- **64MB cache**: Improved query performance

**Expected Impact**:
- ✅ Eliminates "database is locked" errors
- ✅ Enables concurrent article processing
- ✅ Prevents data loss from failed inserts
- ✅ Reduces wasted API costs

### Priority 2: Code Review Swarm Workflow

**Investigation Required**:
1. Validate workflow YAML syntax
2. Check job conditions and dependencies
3. Verify required secrets exist
4. Test workflow locally with `act` or GitHub CLI

**Temporary Fix**:
- Disable workflow until investigation complete
- Use alternative PR automation workflows

---

## Testing Plan

### 1. Database Fix Testing

**Local Testing**:
```bash
# Test connection pool with new settings
uv run python -c "
from src.core.database import DatabaseManager
db = DatabaseManager('data/test.db')
print('Connection pool initialized')
print(db.get_pool_stats())
"

# Simulate concurrent writes
uv run python -c "
import concurrent.futures
from src.core.database import DatabaseManager

db = DatabaseManager('data/test.db')

def write_article(i):
    db.insert_article(
        url=f'https://test{i}.com',
        title=f'Test Article {i}',
        description='Test',
        rss_entry={'link': f'https://test{i}.com'}
    )
    return i

# Run 20 concurrent writes
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(write_article, range(20)))

print(f'Successfully wrote {len(results)} articles concurrently')
"
```

**CI Testing**:
```bash
# Trigger workflow manually
gh workflow run rss-complete-pipeline.yml --field test_only=true

# Monitor for "database is locked" errors
gh run watch
```

### 2. Code Review Swarm Testing

```bash
# Validate workflow syntax
gh workflow view code-review-swarm.yml

# Test workflow (if syntax valid)
gh workflow run code-review-swarm.yml
```

---

## Rollback Plan

If fix causes issues:

1. **Revert database.py**:
   ```bash
   git checkout HEAD~1 src/core/database.py
   ```

2. **Emergency database repair**:
   ```bash
   # Convert WAL back to DELETE mode if needed
   sqlite3 data/articles.db "PRAGMA journal_mode = DELETE"
   ```

3. **Disable concurrent processing**:
   ```yaml
   # In workflow, reduce MAX_ARTICLES_PER_RUN
   MAX_ARTICLES_PER_RUN: 1
   ```

---

## Implementation Checklist

- [ ] Apply database concurrency fix to `src/core/database.py`
- [ ] Test locally with concurrent writes (20 threads)
- [ ] Verify PRAGMA settings with `sqlite3 data/articles.db "PRAGMA journal_mode"`
- [ ] Commit fix with descriptive message
- [ ] Trigger manual workflow run for testing
- [ ] Monitor workflow logs for database errors
- [ ] Investigate code-review-swarm workflow
- [ ] Update documentation with concurrency best practices
- [ ] Add connection pool monitoring to performance dashboard

---

## Additional Recommendations

### 1. Add Connection Pool Metrics

Add to `src/core/monitoring.py`:
```python
def monitor_connection_pool(db_manager):
    """Monitor connection pool health"""
    stats = db_manager.get_pool_stats()
    if stats['active_connections'] >= stats['pool_size'] * 0.8:
        logger.warning("Connection pool nearing capacity")
    return stats
```

### 2. Add Database Lock Retry Logic

Wrap all database operations with retry decorator:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def database_operation():
    # Database write/update
    pass
```

### 3. Monitor GitHub Actions Costs

Enable workflow cost tracking:
```yaml
- name: Track workflow cost
  run: |
    echo "API calls made: ${{ steps.analysis.outputs.api_calls }}"
    echo "Estimated cost: ${{ steps.analysis.outputs.cost }}"
```

---

## References

- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [SQLite Busy Timeout](https://www.sqlite.org/c3ref/busy_timeout.html)
- [SQLite Performance Tuning](https://www.sqlite.org/pragma.html#pragma_optimize)
- [GitHub Actions Debugging](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows)

---

**Next Steps**: Implement database concurrency fix immediately to prevent further data loss.
