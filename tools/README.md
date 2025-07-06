# Tools Directory

This directory contains Python utility tools and scripts for the RSS Analyzer project.

## Available Tools

### Data Generation
- `generate_website_data.py` - Generate JSON data for GitHub Pages website
- `generate_articles_by_date.py` - Generate articles organized by date
- `generate_comprehensive_reports.py` - Generate comprehensive analysis reports

### Testing & Development
- `test_link_following.py` - Test link following functionality

## Usage

Run tools from the project root directory:

```bash
# Generate website data
python tools/generate_website_data.py

# Generate reports
python tools/generate_comprehensive_reports.py

# Test link following
python tools/test_link_following.py
```

## Command Line Options

Most tools support command-line arguments. Use `--help` for details:

```bash
python tools/generate_website_data.py --help
```
