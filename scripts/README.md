# Scripts Directory

This directory contains shell scripts for RSS Analyzer operations and deployment.

## Available Scripts

### Development Scripts
- `lint.sh` - Code linting and quality checks
- `check_service_status.sh` - Service status monitoring

### Deployment Scripts
- `setup_daily_service.sh` - Set up daily processing service
- `run_daily.sh` - Daily execution script
- `setup_github_action.sh` - GitHub Actions setup

## Usage

Make scripts executable before running:
```bash
chmod +x scripts/*.sh
```

Run scripts from the project root:
```bash
./scripts/lint.sh
./scripts/setup_daily_service.sh
```
