# Issue Management System

Comprehensive issue management configuration for RSS Analyzer with automated triage, labeling taxonomy, and workflow states.

## Table of Contents

- [Labeling Taxonomy](#labeling-taxonomy)
- [Auto-Triage Rules](#auto-triage-rules)
- [Issue Workflow](#issue-workflow)
- [Issue Templates](#issue-templates)
- [Automation Rules](#automation-rules)

---

## Labeling Taxonomy

### Type Labels (Mutually Exclusive)

| Label | Description | Use Cases |
|-------|-------------|-----------|
| `type:bug` | Bug reports or broken functionality | Crashes, errors, unexpected behavior |
| `type:enhancement` | New features or improvements | Feature requests, optimizations |
| `type:documentation` | Documentation updates | Missing docs, typos, clarifications |
| `type:performance` | Performance issues | Slow processing, high memory, inefficiency |
| `type:maintenance` | Code maintenance tasks | Refactoring, dependency updates |
| `type:security` | Security vulnerabilities | CVEs, security fixes |

### Priority Labels (Mutually Exclusive)

| Label | Description | SLA | Criteria |
|-------|-------------|-----|----------|
| `priority:critical` | System broken, security issue | 24 hours | Production down, data loss, security breach |
| `priority:high` | Major functionality impaired | 3 days | Core features broken, significant performance degradation |
| `priority:medium` | Normal priority | 1 week | Minor bugs, feature requests |
| `priority:low` | Nice to have | Best effort | Documentation, minor improvements |

### Component Labels (Multiple Allowed)

| Label | Description | Related Files |
|-------|-------------|---------------|
| `component:rss-parser` | RSS feed parsing | `src/rss_parser.py` |
| `component:scraper` | Web scraping (sync) | `src/scraper.py` |
| `component:async-scraper` | Async web scraping | `src/core/async_scraper.py` |
| `component:database` | Database operations | `src/core/database.py` |
| `component:cache` | Caching system | `src/core/cache.py` |
| `component:ai-clients` | AI API integrations | `src/ai_clients/*.py` |
| `component:async-processing` | Async pipeline | `src/etl_orchestrator.py` |
| `component:reports` | Report generation | `src/core/report_generator.py` |
| `component:config` | Configuration system | `config/config.yaml` |
| `component:docker` | Docker/deployment | `Dockerfile`, `docker-compose.yml` |
| `component:monitoring` | Performance monitoring | `src/core/monitoring.py` |
| `component:deduplication` | Hash-based deduplication | `tools/remove_duplicates.py` |

### Status Labels (Mutually Exclusive)

| Label | Description | Who Sets | Next Action |
|-------|-------------|----------|-------------|
| `status:needs-triage` | New issue, needs review | Auto | Team reviews and categorizes |
| `status:needs-info` | Waiting for more information | Team | Reporter provides details |
| `status:needs-reproduction` | Cannot reproduce | Team | Reporter provides repro steps |
| `status:confirmed` | Issue verified | Team | Ready for implementation |
| `status:in-progress` | Work in progress | Team | Active development |
| `status:blocked` | Blocked by dependency | Team | Wait for blocker resolution |
| `status:needs-review` | PR submitted, needs review | Team | Review and merge |
| `status:wontfix` | Will not fix | Team | Close with explanation |
| `status:duplicate` | Duplicate of another issue | Team | Close and link to original |

### Optimization Feature Labels (Multiple Allowed)

| Label | Description | Related Docs |
|-------|-------------|--------------|
| `opt:connection-pool` | Connection pooling | `docs/CONNECTION_POOLING.md` |
| `opt:cache` | Two-tier caching | `docs/CACHE_USAGE.md` |
| `opt:async` | Async processing | `docs/ASYNC_GUIDE.md` |
| `opt:rate-limiting` | Rate limiting | `docs/RATE_LIMITING_QUICKREF.md` |
| `opt:deduplication` | Hash-based deduplication | `docs/DEDUPLICATION.md` |
| `opt:monitoring` | Performance monitoring | `docs/MONITORING_GUIDE.md` |

### Special Labels

| Label | Description | Use |
|-------|-------------|-----|
| `good-first-issue` | Good for newcomers | Simple, well-defined issues |
| `help-wanted` | Community help requested | Complex or low-priority issues |
| `breaking-change` | Breaking API change | Major version bumps |
| `needs-discussion` | Requires team discussion | Design decisions, architecture |
| `regression` | Previously worked, now broken | After update or change |

---

## Auto-Triage Rules

### Keyword-Based Labeling

#### Type Detection

```yaml
type:bug:
  title_keywords:
    - bug
    - error
    - crash
    - exception
    - fail
    - broken
    - not working
    - traceback
  description_keywords:
    - stack trace
    - error message
    - exception raised
    - unexpected behavior

type:performance:
  title_keywords:
    - slow
    - performance
    - memory
    - cpu
    - timeout
    - hanging
    - inefficient
    - optimization
  description_keywords:
    - takes too long
    - high memory usage
    - high cpu usage
    - processing time
    - performance metrics

type:documentation:
  title_keywords:
    - docs
    - documentation
    - readme
    - typo
    - unclear
    - missing documentation
    - guide
  description_keywords:
    - documentation is
    - docs are
    - not documented
    - unclear documentation

type:enhancement:
  title_keywords:
    - feature
    - enhancement
    - improvement
    - add
    - support
    - enable
  description_keywords:
    - would be nice
    - feature request
    - new functionality
```

#### Component Detection

```yaml
component:async-scraper:
  file_patterns:
    - src/core/async_scraper.py
    - tests/test_async_scraper.py
  keywords:
    - async scraper
    - asyncwebscraper
    - async scraping
    - concurrent scraping

component:database:
  file_patterns:
    - src/core/database.py
    - data/articles.db
  keywords:
    - database
    - sqlite
    - databasemanager
    - connection pool
    - sql query
    - articles table

component:cache:
  file_patterns:
    - src/core/cache.py
  keywords:
    - cache
    - caching
    - contentcache
    - l1 cache
    - l2 cache
    - cache hit
    - cache miss

component:async-processing:
  file_patterns:
    - src/etl_orchestrator.py
    - src/main.py
  keywords:
    - async processing
    - asyncarticleprocessor
    - concurrent processing
    - async mode
    - --async flag

component:ai-clients:
  file_patterns:
    - src/ai_clients/
    - src/ai_clients/anthropic_client.py
    - src/ai_clients/mistral_client.py
    - src/ai_clients/openai_client.py
  keywords:
    - ai client
    - claude
    - mistral
    - openai
    - api provider
    - api key

component:monitoring:
  file_patterns:
    - src/core/monitoring.py
    - tools/api_health_monitor.py
  keywords:
    - monitoring
    - performance monitor
    - metrics
    - health check
    - alert
```

#### Priority Detection

```yaml
priority:critical:
  keywords:
    - critical
    - production down
    - data loss
    - security vulnerability
    - security issue
    - exploit
    - urgent
  error_patterns:
    - SegmentationFault
    - MemoryError
    - SystemExit
    - KeyboardInterrupt

priority:high:
  keywords:
    - blocking
    - cannot use
    - main feature broken
    - regression
    - async mode broken
  error_patterns:
    - RuntimeError
    - ConnectionError
    - TimeoutError
    - DatabaseError

priority:medium:
  default_for:
    - type:bug
    - type:enhancement
    - type:performance

priority:low:
  keywords:
    - nice to have
    - minor
    - cosmetic
    - suggestion
  default_for:
    - type:documentation
```

#### Optimization Feature Detection

```yaml
opt:async:
  keywords:
    - async
    - asyncio
    - concurrent
    - MAX_CONCURRENT_ARTICLES
    - --async flag
  file_patterns:
    - src/etl_orchestrator.py
    - src/core/async_scraper.py

opt:cache:
  keywords:
    - cache
    - l1 cache
    - l2 cache
    - cache hit rate
    - cache statistics
  file_patterns:
    - src/core/cache.py

opt:connection-pool:
  keywords:
    - connection pool
    - pool size
    - database connections
    - pool statistics
  file_patterns:
    - src/core/database.py

opt:deduplication:
  keywords:
    - deduplication
    - duplicate
    - content hash
    - hash-based
  file_patterns:
    - tools/remove_duplicates.py
```

### Stack Trace Analysis

```yaml
error_type_mapping:
  TypeError: [component:ai-clients, priority:medium]
  KeyError: [component:config, priority:medium]
  FileNotFoundError: [component:config, priority:medium]
  ConnectionError: [component:ai-clients, priority:high]
  TimeoutError: [component:ai-clients, priority:medium]
  DatabaseError: [component:database, priority:high]
  MemoryError: [component:cache, priority:critical]
  asyncio.TimeoutError: [component:async-processing, priority:high]
  aiohttp.ClientError: [component:async-scraper, priority:medium]
```

### File Path Detection

```yaml
file_path_rules:
  - pattern: src/core/async_scraper.py
    labels: [component:async-scraper, opt:async]

  - pattern: src/etl_orchestrator.py
    labels: [component:async-processing, opt:async]

  - pattern: src/core/database.py
    labels: [component:database, opt:connection-pool]

  - pattern: src/core/cache.py
    labels: [component:cache, opt:cache]

  - pattern: src/ai_clients/
    labels: [component:ai-clients]

  - pattern: docker-compose.yml
    labels: [component:docker]

  - pattern: config/config.yaml
    labels: [component:config]

  - pattern: docs/
    labels: [type:documentation]

  - pattern: tests/
    labels: [component:tests]
```

---

## Issue Workflow

### State Diagram

```
                    ┌──────────────────┐
                    │  Issue Created   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ needs-triage     │◄────────┐
                    └────────┬─────────┘         │
                             │                   │
                ┌────────────┼────────────┐     │
                │            │            │     │
                ▼            ▼            ▼     │
       ┌──────────────┐ ┌─────────┐ ┌────────────┐
       │ needs-info   │ │confirmed│ │ wontfix    │
       └──────┬───────┘ └────┬────┘ └──────┬─────┘
              │              │              │
              │              ▼              ▼
              │     ┌──────────────┐   ┌────────┐
              │     │ in-progress  │   │ Closed │
              │     └──────┬───────┘   └────────┘
              │            │
              │            ▼
              │     ┌──────────────┐
              │     │ needs-review │
              │     └──────┬───────┘
              │            │
              │            ▼
              │     ┌──────────────┐
              └────►│   Merged     │
                    └──────────────┘
```

### State Transitions

| From State | To State | Trigger | Action Required |
|------------|----------|---------|-----------------|
| `needs-triage` | `needs-info` | Missing details | Reporter adds information |
| `needs-triage` | `confirmed` | Issue verified | Assign priority and component |
| `needs-triage` | `wontfix` | Won't implement | Close with explanation |
| `needs-triage` | `duplicate` | Already exists | Link to original issue |
| `needs-info` | `confirmed` | Info provided | Verify and confirm |
| `needs-info` | `closed` | No response (14 days) | Auto-close |
| `confirmed` | `in-progress` | Work started | Developer assigned |
| `in-progress` | `blocked` | Dependency issue | Document blocker |
| `in-progress` | `needs-review` | PR submitted | Review code |
| `blocked` | `in-progress` | Blocker resolved | Resume work |
| `needs-review` | `in-progress` | Changes requested | Address feedback |
| `needs-review` | `merged` | PR approved | Merge and close |

### SLA by Priority

| Priority | First Response | Resolution Target |
|----------|----------------|-------------------|
| Critical | 4 hours | 24 hours |
| High | 24 hours | 3 days |
| Medium | 3 days | 1 week |
| Low | 1 week | Best effort |

---

## Issue Templates

### Available Templates

1. **Bug Report** (`bug_report.yml`)
   - Comprehensive bug reporting with async context
   - Environment details and reproduction steps
   - Stack traces and error logs
   - Optimization feature checkboxes

2. **Feature Request** (`feature_request.yml`)
   - Problem statement and proposed solution
   - Use cases and expected impact
   - Technical approach suggestions
   - Breaking change assessment

3. **Documentation** (`documentation.yml`)
   - Documentation type and location
   - Current vs suggested content
   - Target audience specification
   - Example needs

4. **Performance Issue** (`performance.yml`)
   - Performance metrics and benchmarks
   - Workload details
   - Configuration and environment
   - Cache and pool statistics

### Template Usage

Issues automatically receive labels based on template selection:

| Template | Auto Labels |
|----------|-------------|
| Bug Report | `type:bug`, `status:needs-triage` |
| Feature Request | `type:enhancement`, `status:needs-triage` |
| Documentation | `type:documentation`, `status:needs-triage` |
| Performance | `type:performance`, `priority:medium`, `status:needs-triage` |

---

## Automation Rules

### GitHub Actions Integration

#### Auto-Label on Issue Creation

```yaml
name: Auto-label Issues
on:
  issues:
    types: [opened]

jobs:
  auto_label:
    runs-on: ubuntu-latest
    steps:
      - name: Apply labels based on keywords
        uses: actions/github-script@v6
        with:
          script: |
            const issue = context.payload.issue;
            const title = issue.title.toLowerCase();
            const body = issue.body?.toLowerCase() || '';
            const labels = [];

            // Type detection
            if (title.includes('async') || body.includes('async mode')) {
              labels.push('opt:async');
            }
            if (title.includes('cache') || body.includes('cache')) {
              labels.push('opt:cache');
            }
            if (title.includes('performance') || title.includes('slow')) {
              labels.push('type:performance');
            }

            // Component detection
            if (body.includes('async_scraper.py')) {
              labels.push('component:async-scraper');
            }
            if (body.includes('database.py') || body.includes('sqlite')) {
              labels.push('component:database');
            }

            // Priority detection
            if (title.includes('critical') || title.includes('urgent')) {
              labels.push('priority:critical');
            }

            // Add labels
            if (labels.length > 0) {
              await github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: issue.number,
                labels: labels
              });
            }
```

#### Stale Issue Management

```yaml
name: Close Stale Issues
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  stale:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/stale@v8
        with:
          stale-issue-message: |
            This issue has been automatically marked as stale because it has not had recent activity.
            It will be closed if no further activity occurs in 7 days.
          close-issue-message: |
            This issue was automatically closed due to inactivity.
            Please reopen if you believe this is still relevant.
          days-before-stale: 60
          days-before-close: 7
          stale-issue-label: 'status:stale'
          exempt-issue-labels: 'priority:critical,priority:high,status:blocked'
```

#### Needs Info Auto-Close

```yaml
name: Close Needs Info Issues
on:
  schedule:
    - cron: '0 0 * * *'

jobs:
  close_needs_info:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v6
        with:
          script: |
            const issues = await github.rest.issues.listForRepo({
              owner: context.repo.owner,
              repo: context.repo.repo,
              state: 'open',
              labels: 'status:needs-info'
            });

            const now = Date.now();
            const fourteenDays = 14 * 24 * 60 * 60 * 1000;

            for (const issue of issues.data) {
              const updated = new Date(issue.updated_at).getTime();

              if (now - updated > fourteenDays) {
                await github.rest.issues.createComment({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  issue_number: issue.number,
                  body: 'Closing due to lack of requested information after 14 days.'
                });

                await github.rest.issues.update({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  issue_number: issue.number,
                  state: 'closed'
                });
              }
            }
```

### Issue Queries

#### High Priority Triage Queue

```
is:issue is:open label:status:needs-triage label:priority:high,priority:critical
```

#### Performance Issues

```
is:issue is:open label:type:performance
```

#### Good First Issues

```
is:issue is:open label:good-first-issue
```

#### Async-Related Issues

```
is:issue is:open label:opt:async
```

#### Blocked Issues

```
is:issue is:open label:status:blocked
```

---

## Best Practices

### For Reporters

1. **Use appropriate template** for your issue type
2. **Search existing issues** before creating new ones
3. **Provide complete information** - environment, logs, reproduction steps
4. **Respond promptly** to requests for additional information
5. **Test with latest version** before reporting bugs

### For Maintainers

1. **Triage daily** - review `needs-triage` issues
2. **Request info promptly** - use `needs-info` label with clear questions
3. **Set appropriate priority** based on impact and urgency
4. **Update status regularly** - keep issue status current
5. **Link related issues** - use keywords like "fixes", "relates to", "duplicate of"
6. **Use milestones** for release planning
7. **Apply optimization labels** to track feature-specific issues

### Label Management

- **Keep labels organized** - remove obsolete labels
- **Use consistent naming** - follow `category:value` pattern
- **Document custom labels** - update this guide when adding labels
- **Review quarterly** - clean up and reorganize as needed

---

## Issue Management Metrics

Track these metrics to measure issue management effectiveness:

### Response Metrics
- Time to first response (by priority)
- Time to triage (apply labels and confirm)
- Time to resolution (by priority)

### Volume Metrics
- Open issues by type
- Open issues by component
- Open issues by priority
- Issues opened vs closed per week

### Quality Metrics
- Issues requiring additional information (%)
- Issues marked as duplicate (%)
- Issues resolved on first attempt (%)
- Average labels per issue

### Component Health
- Open bugs by component
- Performance issues by optimization feature
- Documentation gaps by component

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-11-07 | Initial issue management system | Claude Code |

---

## References

- [GitHub Issue Templates Documentation](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests)
- [GitHub Actions for Issue Management](https://docs.github.com/en/actions/managing-issues-and-pull-requests)
- [Optimization Results](./OPTIMIZATION_RESULTS.md)
- [Async Migration Guide](./ASYNC_MIGRATION.md)
