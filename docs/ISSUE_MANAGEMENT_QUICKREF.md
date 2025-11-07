# Issue Management Quick Reference

Quick reference guide for the RSS Analyzer GitHub issue management system.

## Issue Templates

| Template | When to Use | Auto Labels |
|----------|-------------|-------------|
| **Bug Report** | Crashes, errors, unexpected behavior | `type:bug`, `status:needs-triage` |
| **Feature Request** | New features or enhancements | `type:enhancement`, `status:needs-triage` |
| **Documentation** | Missing, unclear, or outdated docs | `type:documentation`, `status:needs-triage` |
| **Performance** | Slow processing, memory/CPU issues | `type:performance`, `priority:medium`, `status:needs-triage` |

## Essential Labels

### Type (pick one)
- `type:bug` - Something is broken
- `type:enhancement` - New feature or improvement
- `type:documentation` - Documentation issue
- `type:performance` - Performance problem

### Priority (pick one)
- `priority:critical` - System down, security issue (SLA: 24h)
- `priority:high` - Major feature broken (SLA: 3 days)
- `priority:medium` - Normal priority (SLA: 1 week)
- `priority:low` - Nice to have (SLA: best effort)

### Status (pick one)
- `status:needs-triage` - Needs team review
- `status:needs-info` - Waiting for reporter
- `status:confirmed` - Verified and ready
- `status:in-progress` - Being worked on
- `status:blocked` - Blocked by dependency
- `status:needs-review` - PR submitted

### Components (pick multiple)
- `component:async-scraper` - Async web scraping
- `component:database` - Database operations
- `component:cache` - Caching system
- `component:ai-clients` - AI integrations
- `component:async-processing` - Async pipeline

### Optimization Features (pick multiple)
- `opt:async` - Async processing issues
- `opt:cache` - Cache-related issues
- `opt:connection-pool` - Database pooling
- `opt:deduplication` - Hash deduplication
- `opt:monitoring` - Performance monitoring

## Common Workflows

### Reporting a Bug

1. Use **Bug Report** template
2. Select component (e.g., `async-scraper`)
3. Specify processing mode (async/sync)
4. Include error logs and stack traces
5. Provide environment details
6. Check optimization features in use

### Requesting a Feature

1. Use **Feature Request** template
2. Describe the problem it solves
3. Provide specific use cases
4. Indicate priority level
5. Note any breaking changes
6. Offer to contribute (optional)

### Improving Documentation

1. Use **Documentation** template
2. Specify location and section
3. Describe what's unclear
4. Suggest improved wording
5. Provide examples if needed

### Reporting Performance Issues

1. Use **Performance** template
2. Provide specific metrics (time, memory, CPU)
3. Include workload details
4. Compare against benchmarks
5. Share cache/pool statistics
6. Include configuration

## Auto-Triage Keywords

Issues are automatically labeled based on keywords:

### Bug Detection
Keywords: `bug`, `error`, `crash`, `exception`, `fail`, `broken`, `traceback`

### Performance Detection
Keywords: `slow`, `memory`, `cpu`, `timeout`, `hanging`, `inefficient`

### Component Detection
- **Async**: `async`, `asyncio`, `concurrent`, `--async flag`
- **Cache**: `cache`, `l1 cache`, `l2 cache`, `cache hit`
- **Database**: `database`, `sqlite`, `connection pool`, `sql query`

### Priority Detection
- **Critical**: `critical`, `urgent`, `security`, `data loss`
- **High**: `blocking`, `cannot use`, `regression`, `main feature broken`

## Issue Queries

### Triage Queue
```
is:issue is:open label:status:needs-triage label:priority:high,priority:critical
```

### Performance Issues
```
is:issue is:open label:type:performance
```

### Async-Related
```
is:issue is:open label:opt:async
```

### Good First Issues
```
is:issue is:open label:good-first-issue
```

### Blocked Issues
```
is:issue is:open label:status:blocked
```

## Quick Actions

### For Reporters

- **Search first**: Check existing issues before creating new ones
- **Use templates**: Always use appropriate issue template
- **Be specific**: Provide clear reproduction steps
- **Respond quickly**: Reply to requests for info within 7 days
- **Test latest**: Verify bug exists in latest version

### For Maintainers

- **Triage daily**: Review `needs-triage` label
- **Request info**: Use `needs-info` with specific questions
- **Set priority**: Based on impact and user base
- **Link issues**: Use "fixes #123", "relates to #456"
- **Update status**: Keep labels current
- **Close stale**: Auto-closes after 60 days inactive

## SLA Targets

| Priority | First Response | Resolution |
|----------|----------------|------------|
| Critical | 4 hours | 24 hours |
| High | 24 hours | 3 days |
| Medium | 3 days | 1 week |
| Low | 1 week | Best effort |

## File Paths for Component Detection

| Pattern | Auto-Labeled As |
|---------|-----------------|
| `src/core/async_scraper.py` | `component:async-scraper`, `opt:async` |
| `src/etl_orchestrator.py` | `component:async-processing`, `opt:async` |
| `src/core/database.py` | `component:database`, `opt:connection-pool` |
| `src/core/cache.py` | `component:cache`, `opt:cache` |
| `src/ai_clients/` | `component:ai-clients` |
| `docker-compose.yml` | `component:docker` |
| `config/config.yaml` | `component:config` |
| `docs/` | `type:documentation` |

## Error Type Mapping

| Exception | Auto-Labeled As |
|-----------|-----------------|
| `MemoryError` | `component:cache`, `priority:critical` |
| `DatabaseError` | `component:database`, `priority:high` |
| `asyncio.TimeoutError` | `component:async-processing`, `priority:high` |
| `ConnectionError` | `component:ai-clients`, `priority:high` |
| `aiohttp.ClientError` | `component:async-scraper`, `priority:medium` |

## Best Practices

### Writing Good Issue Titles

- ✅ `[BUG] Async mode crashes with MemoryError on large articles`
- ✅ `[FEATURE] Add support for concurrent cache warming`
- ✅ `[PERF] Database queries taking 5s with pool size 5`
- ✅ `[DOCS] Missing examples in async migration guide`
- ❌ `It doesn't work`
- ❌ `Help needed`
- ❌ `Question about the app`

### Providing Good Reproduction Steps

```markdown
1. Start application: `docker compose run rss-analyzer run --limit 10 --async`
2. Configure: `API_PROVIDER=anthropic`, `MAX_CONCURRENT_ARTICLES=8`
3. Observe: Application crashes after processing 5 articles
4. Expected: Should process all 10 articles successfully
5. Actual: MemoryError at article 5

Environment:
- Python 3.11.5
- Docker deployment
- 4GB RAM available
- Cache enabled (L1: 256MB)
```

### Including Helpful Diagnostics

```bash
# Cache statistics
docker compose run rss-analyzer cache-stats

# Pool statistics
docker compose run rss-analyzer pool-stats

# Database state
docker compose run rss-analyzer sqlite3 /app/data/articles.db \
  "SELECT status, COUNT(*) FROM articles GROUP BY status;"

# System metrics
docker compose run rss-analyzer stats
```

## Related Documentation

- **[Full Issue Management Guide](./ISSUE_MANAGEMENT.md)** - Complete documentation
- **[Optimization Results](./OPTIMIZATION_RESULTS.md)** - Performance benchmarks
- **[Async Guide](./ASYNC_GUIDE.md)** - Async processing patterns
- **[Cache Usage](./CACHE_USAGE.md)** - Caching system details
- **[Connection Pooling](./CONNECTION_POOLING.md)** - Database optimization

---

**Need Help?** Check [GitHub Discussions](https://github.com/mess/rss-analyzer/discussions) for questions and community support.
