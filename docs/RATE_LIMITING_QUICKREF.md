# Rate Limiting Quick Reference

## Quick Configuration

### Environment Variables

```bash
# Set rate limit (requests per second)
export RATE_LIMIT_RPS=10

# Set burst size
export RATE_LIMIT_BURST=20

# Run with custom rate limits
docker compose run rss-analyzer run --limit 10
```

### Python Code

```python
from src.core.async_scraper import AsyncWebScraper

# Conservative rate limiting (academic publishers)
scraper = AsyncWebScraper(
    rate_limit_rps=5.0,
    rate_limit_burst=10
)

# Aggressive rate limiting (authorized scraping)
scraper = AsyncWebScraper(
    rate_limit_rps=20.0,
    rate_limit_burst=50
)
```

### Config File

```yaml
# config/config.yaml
scraper:
  rate_limit_rps: 10  # Requests per second
  rate_limit_burst: 20  # Maximum burst size
```

## Common Scenarios

### Too Many IP Bans

```bash
# Reduce rate limit
export RATE_LIMIT_RPS=2
export SCRAPER_DELAY_BETWEEN_REQUESTS=3.0
```

### Slow Processing

```bash
# Increase rate limit (if permitted)
export RATE_LIMIT_RPS=20
export RATE_LIMIT_BURST=50
```

### Academic Publishers (arXiv, IEEE)

```bash
# Conservative settings
export RATE_LIMIT_RPS=5
export RATE_LIMIT_BURST=10
export SCRAPER_DELAY_BETWEEN_REQUESTS=2.0
```

## Testing

```bash
# Run rate limiting tests
uv run python tests/test_rate_limiting.py

# Or with pytest
uv run pytest tests/test_rate_limiting.py -v
```

## Key Benefits

- ✅ **Prevents DoS attacks**
- ✅ **Avoids IP bans**
- ✅ **Respects server resources**
- ✅ **Sustainable scraping**
- ✅ **Configurable limits**

## Default Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `RATE_LIMIT_RPS` | 10.0 | Requests per second |
| `RATE_LIMIT_BURST` | 20 | Maximum burst size |
| `SCRAPER_DELAY` | 1.0 | Additional delay (seconds) |

## Documentation

Full documentation: [docs/RATE_LIMITING.md](RATE_LIMITING.md)
