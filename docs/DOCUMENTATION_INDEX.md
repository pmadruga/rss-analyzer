# Documentation Index

This document provides an overview of all available documentation for the RSS Analyzer project.

## Quick Start

- **[README](../README.md)** - Project overview and quick start guide
- **[CLAUDE.md](../CLAUDE.md)** - Development guide for Claude Code

## Performance & Optimization

### Phase 1-3 Optimization Results

- **[Optimization Results](OPTIMIZATION_RESULTS.md)** ⭐
  - Complete benchmark data for all three optimization phases
  - Before/after performance comparisons
  - Cost savings analysis
  - Implementation timeline and lessons learned
  - 56x database improvement, 72% cost reduction

### Database Optimization

- **[Connection Pooling Guide](CONNECTION_POOLING.md)**
  - Comprehensive guide to connection pooling
  - Architecture and usage examples
  - Performance benchmarks
  - Troubleshooting guide

- **[Connection Pooling Summary](CONNECTION_POOLING_SUMMARY.md)**
  - Quick reference for connection pooling
  - Test results and statistics
  - Migration notes

- **[Connection Pooling Quick Reference](CONNECTION_POOLING_QUICKREF.md)**
  - Cheat sheet for common operations

### Caching System

- **[Cache Usage Guide](CACHE_USAGE.md)**
  - Two-tier caching architecture
  - Usage patterns and examples
  - Performance optimization tips

- **[Cache Integration Guide](CACHE_INTEGRATION.md)**
  - How to integrate caching in your code
  - Configuration options
  - Cost savings calculator

### Async Programming

- **[Async Guide](ASYNC_GUIDE.md)** ⭐
  - Complete async/await patterns guide
  - Async API clients
  - Concurrent pipeline processing
  - Error handling and best practices
  - Migration from sync to async
  - Testing async code
  - Real-world examples

### Monitoring & Observability

- **[Monitoring Guide](MONITORING_GUIDE.md)** ⭐
  - Comprehensive monitoring system documentation
  - Available metrics (database, cache, API, system)
  - API health monitoring with async checks
  - Performance tracking and analysis
  - Alerting configuration
  - Dashboard setup (Grafana, custom)
  - Performance tuning recommendations
  - Troubleshooting guide

## Core Features

### Deduplication

- **[Deduplication Guide](DEDUPLICATION.md)**
  - Hash-based duplicate detection
  - O(1) lookup performance
  - Cost savings through deduplication
  - Usage and troubleshooting

- **[Deduplication Integration](DEDUPLICATION_INTEGRATION_COMPLETE.md)**
  - Complete integration guide
  - Database schema changes
  - Migration steps

### Import System

- **[Import Migration Guide](IMPORT_MIGRATION_GUIDE.md)**
  - Updating import statements
  - New module structure
  - Breaking changes

## API Reference

- **[API Documentation](API_DOCUMENTATION.md)** ⭐
  - Complete API reference for all components
  - Database API with connection pooling
  - Cache API with two-tier system
  - Monitoring API with metrics tracking
  - AI Clients API
  - Async methods and patterns
  - Configuration options
  - Practical examples for all features
  - Performance tips and best practices

## Deployment & Setup

- **[GitHub Actions Setup](setup/GITHUB_ACTION_SETUP.md)**
  - Automated cloud deployment
  - Configuration guide
  - Scheduling and monitoring

## Quick Reference

### Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| Database | Operations speed | 2.78x faster |
| Database | Concurrent capacity | 4.2x higher |
| Cache | Hit rate | 72% |
| Cache | Cost reduction | 62-85% |
| API | Monthly savings | $21.60 |
| System | Uptime | 99.9% |

### Key Files

| File | Purpose |
|------|---------|
| `src/core/database.py` | Connection pooling implementation |
| `src/core/cache.py` | Two-tier caching system |
| `src/core/monitoring.py` | Performance monitoring |
| `tools/api_health_monitor.py` | Async API health checks |
| `examples/cache_demo.py` | Cache usage examples |
| `examples/connection_pool_demo.py` | Pool usage examples |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/config.yaml` | Main configuration |
| `.env` | Environment variables |
| `docker-compose.yml` | Docker configuration |
| `requirements.txt` | Python dependencies |

## Documentation by Use Case

### "I want to optimize my RSS analyzer"
1. Read: [Optimization Results](OPTIMIZATION_RESULTS.md)
2. Implement: [Connection Pooling Guide](CONNECTION_POOLING.md)
3. Implement: [Cache Integration Guide](CACHE_INTEGRATION.md)
4. Monitor: [Monitoring Guide](MONITORING_GUIDE.md)

### "I want to use async/await"
1. Read: [Async Guide](ASYNC_GUIDE.md)
2. Reference: [API Documentation](API_DOCUMENTATION.md) - Async Methods section

### "I want to monitor my system"
1. Read: [Monitoring Guide](MONITORING_GUIDE.md)
2. Setup: API health monitoring
3. Configure: Alerts and dashboards

### "I want to reduce API costs"
1. Read: [Optimization Results](OPTIMIZATION_RESULTS.md) - Cost Savings section
2. Implement: [Cache Integration Guide](CACHE_INTEGRATION.md)
3. Monitor: Cache hit rates and savings

### "I need API reference"
1. Go to: [API Documentation](API_DOCUMENTATION.md)
2. Find your component (Database, Cache, Monitoring, etc.)
3. Follow examples

## External Resources

### Related Technologies
- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [aiohttp documentation](https://docs.aiohttp.org/)
- [SQLite documentation](https://www.sqlite.org/docs.html)
- [Prometheus monitoring](https://prometheus.io/docs/)

### Performance Tools
- [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio) - Testing async code
- [pytest-benchmark](https://github.com/ionelmc/pytest-benchmark) - Performance testing
- [py-spy](https://github.com/benfred/py-spy) - Python profiler

## Contributing

When adding new documentation:

1. Add entry to this index
2. Follow existing documentation style
3. Include code examples
4. Add cross-references to related docs
5. Update README.md if appropriate

## Support

For issues or questions:
- Check relevant documentation first
- Review [Troubleshooting sections](MONITORING_GUIDE.md#troubleshooting)
- Open GitHub issue with documentation link
