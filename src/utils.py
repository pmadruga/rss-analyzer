"""
Utility Functions

Common utilities for the RSS article analyzer including
configuration management, logging setup, and helper functions.
"""

import hashlib
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        except Exception as e:
            root_logger.warning(f"Failed to setup file logging: {e}")

    return root_logger


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from environment variables and config file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()

    # Default configuration
    config = {
        # API Configuration
        'api_provider': os.getenv('API_PROVIDER', 'anthropic'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
        'mistral_api_key': os.getenv('MISTRAL_API_KEY', ''),
        'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
        'claude_model': os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
        'mistral_model': os.getenv('MISTRAL_MODEL', 'mistral-large-latest'),
        'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4'),

        # RSS Configuration
        'rss_feed_url': os.getenv('RSS_FEED_URL', 'https://bg.raindrop.io/rss/public/57118738'),
        'user_agent': os.getenv('USER_AGENT', 'RSS-Article-Analyzer/1.0'),

        # Database Configuration
        'db_path': os.getenv('DB_PATH', 'data/articles.db'),

        # Processing Configuration
        'max_articles_per_run': int(os.getenv('MAX_ARTICLES_PER_RUN', '10')),
        'scraper_delay': float(os.getenv('SCRAPER_DELAY', '1.0')),
        'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '30')),

        # Output Configuration
        'output_dir': os.getenv('OUTPUT_DIR', 'output'),
        'report_filename': os.getenv('REPORT_FILENAME', 'article_analysis_report.md'),

        # Logging Configuration
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', None),

        # Processing Options
        'force_refresh': os.getenv('FORCE_REFRESH', 'false').lower() == 'true',
        'skip_scraping': os.getenv('SKIP_SCRAPING', 'false').lower() == 'true',
        'skip_analysis': os.getenv('SKIP_ANALYSIS', 'false').lower() == 'true',
    }

    # Load from config file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Handle nested API configuration
                    if 'api' in file_config:
                        api_config = file_config['api']
                        if 'provider' in api_config:
                            config['api_provider'] = api_config['provider']
                        if 'anthropic' in api_config:
                            config['claude_model'] = api_config['anthropic'].get('model', config['claude_model'])
                        if 'mistral' in api_config:
                            config['mistral_model'] = api_config['mistral'].get('model', config['mistral_model'])
                        if 'openai' in api_config:
                            config['openai_model'] = api_config['openai'].get('model', config['openai_model'])
                        del file_config['api']

                    config.update(file_config)
        except Exception as e:
            logging.warning(f"Failed to load config file {config_path}: {e}")

    return config


def validate_config(config: dict[str, Any]) -> bool:
    """
    Validate configuration and check for required values
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    # Validate based on API provider
    api_provider = config.get('api_provider', 'anthropic')
    required_fields = ['rss_feed_url']

    if api_provider == 'anthropic':
        required_fields.append('anthropic_api_key')
    elif api_provider == 'mistral':
        required_fields.append('mistral_api_key')
    elif api_provider == 'openai':
        required_fields.append('openai_api_key')
    else:
        logging.error(f"Unsupported API provider: {api_provider}")
        return False

    missing_fields = []

    for field in required_fields:
        if not config.get(field):
            missing_fields.append(field)

    if missing_fields:
        logging.error(f"Missing required configuration fields: {', '.join(missing_fields)}")
        return False

    # Validate API key format
    api_provider = config.get('api_provider', 'anthropic')

    if api_provider == 'anthropic':
        api_key = config['anthropic_api_key']
        if not api_key.startswith('sk-'):
            logging.error("Anthropic API key appears to be invalid (should start with 'sk-')")
            return False
    elif api_provider == 'mistral':
        api_key = config['mistral_api_key']
        # Mistral API keys don't have a specific prefix pattern
        if len(api_key) < 10:
            logging.error("Mistral API key appears to be too short")
            return False
    elif api_provider == 'openai':
        api_key = config['openai_api_key']
        if not api_key.startswith('sk-'):
            logging.error("OpenAI API key appears to be invalid (should start with 'sk-')")
            return False

    # Validate RSS URL
    try:
        parsed = urlparse(config['rss_feed_url'])
        if not parsed.scheme or not parsed.netloc:
            logging.error("RSS feed URL appears to be invalid")
            return False
    except Exception:
        logging.error("RSS feed URL appears to be invalid")
        return False

    return True


def create_content_hash(content: str) -> str:
    """
    Create MD5 hash of content for duplicate detection
    
    Args:
        content: Content to hash
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s\-_\.]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)

    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:96] + ext

    return filename.strip('-_')


def format_timestamp(timestamp: float | None = None) -> str:
    """
    Format timestamp for display
    
    Args:
        timestamp: Unix timestamp, defaults to current time
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()

    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def ensure_directory(path: str) -> bool:
    """
    Ensure directory exists, create if necessary
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        return False


def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def extract_domain(url: str) -> str:
    """
    Extract domain from URL
    
    Args:
        url: URL
        
    Returns:
        Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a simple progress bar string
    
    Args:
        current: Current progress
        total: Total items
        width: Width of progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "] 100%"

    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = int(progress * 100)

    return f"[{bar}] {percentage}% ({current}/{total})"


def get_file_age_days(file_path: str) -> int | None:
    """
    Get file age in days
    
    Args:
        file_path: Path to file
        
    Returns:
        Age in days or None if file doesn't exist
    """
    try:
        if not os.path.exists(file_path):
            return None

        file_time = os.path.getmtime(file_path)
        current_time = datetime.now().timestamp()
        age_seconds = current_time - file_time
        age_days = int(age_seconds / (24 * 60 * 60))

        return age_days
    except Exception:
        return None


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        import json
        return json.loads(json_str)
    except Exception:
        return default


def retry_on_exception(func, max_retries: int = 3, delay: float = 1.0,
                      exceptions: tuple = (Exception,)):
    """
    Decorator for retrying function calls on exceptions
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        import time

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

    return wrapper
