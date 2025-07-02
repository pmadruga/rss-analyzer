"""
RSS Feed Parser Module

Handles fetching and parsing RSS feeds, extracting metadata,
and implementing duplicate detection.
"""

import feedparser
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from dateutil import parser as date_parser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RSSEntry:
    """Represents a single RSS feed entry"""
    
    def __init__(self, title: str, link: str, description: str, 
                 publication_date: Optional[datetime] = None, guid: Optional[str] = None):
        self.title = title
        self.link = link
        self.description = description
        self.publication_date = publication_date
        self.guid = guid or link
        self.content_hash = self._generate_content_hash()
    
    def _generate_content_hash(self) -> str:
        """Generate a hash of the entry content for duplicate detection"""
        content = f"{self.title}{self.link}{self.description}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert entry to dictionary"""
        return {
            'title': self.title,
            'link': self.link,
            'description': self.description,
            'publication_date': self.publication_date.isoformat() if self.publication_date else None,
            'guid': self.guid,
            'content_hash': self.content_hash
        }


class RSSParser:
    """RSS Feed Parser with duplicate detection and error handling"""
    
    def __init__(self, user_agent: str = "RSS-Article-Analyzer/1.0"):
        self.user_agent = user_agent
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({'User-Agent': self.user_agent})
        return session
    
    def fetch_feed(self, feed_url: str, timeout: int = 30) -> List[RSSEntry]:
        """
        Fetch and parse RSS feed from URL
        
        Args:
            feed_url: RSS feed URL
            timeout: Request timeout in seconds
            
        Returns:
            List of RSSEntry objects
            
        Raises:
            Exception: If feed cannot be fetched or parsed
        """
        try:
            logger.info(f"Fetching RSS feed from: {feed_url}")
            
            # Fetch the feed content
            response = self.session.get(feed_url, timeout=timeout)
            response.raise_for_status()
            
            # Parse the feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")
            
            if not hasattr(feed, 'entries') or not feed.entries:
                raise Exception("No entries found in RSS feed")
            
            logger.info(f"Found {len(feed.entries)} entries in RSS feed")
            
            # Convert to RSSEntry objects
            entries = []
            for entry in feed.entries:
                try:
                    rss_entry = self._parse_entry(entry)
                    entries.append(rss_entry)
                except Exception as e:
                    logger.warning(f"Failed to parse entry '{entry.get('title', 'Unknown')}': {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(entries)} entries")
            return entries
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching RSS feed: {e}")
            raise Exception(f"Failed to fetch RSS feed: {e}")
        except Exception as e:
            logger.error(f"Error parsing RSS feed: {e}")
            raise Exception(f"Failed to parse RSS feed: {e}")
    
    def _parse_entry(self, entry) -> RSSEntry:
        """Parse a single feed entry"""
        # Extract title
        title = entry.get('title', 'Untitled').strip()
        
        # Extract link
        link = entry.get('link', '').strip()
        if not link:
            raise Exception("Entry missing required link")
        
        # Extract description/summary
        description = entry.get('summary', entry.get('description', '')).strip()
        
        # Extract publication date
        publication_date = None
        for date_field in ['published', 'updated', 'created']:
            if date_field in entry:
                try:
                    publication_date = date_parser.parse(entry[date_field])
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse date from {date_field}: {e}")
                    continue
        
        # Extract GUID
        guid = entry.get('id', entry.get('guid', link))
        
        return RSSEntry(
            title=title,
            link=link,
            description=description,
            publication_date=publication_date,
            guid=guid
        )
    
    def filter_new_entries(self, entries: List[RSSEntry], 
                          existing_hashes: set) -> List[RSSEntry]:
        """
        Filter out entries that already exist based on content hash
        
        Args:
            entries: List of RSS entries
            existing_hashes: Set of existing content hashes
            
        Returns:
            List of new entries
        """
        new_entries = []
        for entry in entries:
            if entry.content_hash not in existing_hashes:
                new_entries.append(entry)
            else:
                logger.debug(f"Skipping duplicate entry: {entry.title}")
        
        logger.info(f"Found {len(new_entries)} new entries out of {len(entries)} total")
        return new_entries
    
    def get_feed_info(self, feed_url: str) -> Dict:
        """
        Get basic information about the RSS feed
        
        Args:
            feed_url: RSS feed URL
            
        Returns:
            Dictionary with feed information
        """
        try:
            response = self.session.get(feed_url, timeout=30)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            return {
                'title': feed.feed.get('title', 'Unknown'),
                'description': feed.feed.get('description', ''),
                'link': feed.feed.get('link', ''),
                'language': feed.feed.get('language', ''),
                'updated': feed.feed.get('updated', ''),
                'entry_count': len(feed.entries) if hasattr(feed, 'entries') else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get feed info: {e}")
            return {
                'title': 'Unknown',
                'description': '',
                'link': feed_url,
                'language': '',
                'updated': '',
                'entry_count': 0
            }