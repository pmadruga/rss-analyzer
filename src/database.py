"""
Database Module

Handles SQLite database operations for articles, content, and processing logs.
Includes duplicate detection, migrations, and efficient querying.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for RSS article analyzer"""

    def __init__(self, db_path: str = "data/articles.db"):
        self.db_path = db_path
        self.ensure_directory_exists()
        self.init_database()

    def ensure_directory_exists(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        return conn

    def init_database(self):
        """Initialize database with required tables"""
        try:
            with self.get_connection() as conn:
                # Create articles table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        url TEXT NOT NULL UNIQUE,
                        publication_date TIMESTAMP,
                        rss_guid TEXT,
                        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        content_hash TEXT NOT NULL UNIQUE,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create content table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS content (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER NOT NULL,
                        original_content TEXT,
                        methodology_detailed TEXT,
                        technical_approach TEXT,
                        key_findings TEXT,
                        research_design TEXT,
                        metadata JSON,
                        confidence_score INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES articles (id) ON DELETE CASCADE
                    )
                ''')

                # Create processing log table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS processing_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        article_id INTEGER,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        processing_step TEXT,
                        duration_seconds REAL,
                        FOREIGN KEY (article_id) REFERENCES articles (id) ON DELETE CASCADE
                    )
                ''')

                # Create indices for better performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles (content_hash)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_articles_url ON articles (url)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_articles_status ON articles (status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_processing_log_timestamp ON processing_log (timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_content_article_id ON content (article_id)')

                # Run migrations to handle schema changes
                self._run_migrations(conn)

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _run_migrations(self, conn: sqlite3.Connection):
        """Run database migrations to handle schema changes"""
        try:
            # Check if old columns exist and migrate to new schema
            cursor = conn.execute("PRAGMA table_info(content)")
            columns = [row[1] for row in cursor.fetchall()]

            old_columns = ['summary', 'methodology_focus', 'practical_applications', 'novel_contributions', 'significance']
            new_columns = ['methodology_detailed', 'research_design']

            if any(col in columns for col in old_columns):
                logger.info("Migrating content table to new schema...")

                # Add new columns if they don't exist
                if 'methodology_detailed' not in columns:
                    conn.execute('ALTER TABLE content ADD COLUMN methodology_detailed TEXT')
                if 'research_design' not in columns:
                    conn.execute('ALTER TABLE content ADD COLUMN research_design TEXT')

                # Migrate data from old columns to new ones
                conn.execute('''
                    UPDATE content 
                    SET methodology_detailed = COALESCE(methodology_focus, ''),
                        research_design = ''
                    WHERE methodology_detailed IS NULL
                ''')

                logger.info("Content table migration completed")

        except Exception as e:
            logger.warning(f"Migration warning (non-critical): {e}")

    def insert_article(self, title: str, url: str, content_hash: str,
                      rss_guid: str | None = None,
                      publication_date: datetime | None = None) -> int:
        """
        Insert new article into database
        
        Returns:
            Article ID of inserted article
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO articles (title, url, content_hash, rss_guid, publication_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (title, url, content_hash, rss_guid, publication_date))

                article_id = cursor.lastrowid
                logger.debug(f"Inserted article with ID {article_id}: {title}")
                return article_id

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: articles.url" in str(e):
                logger.debug(f"Article with URL already exists: {url}")
                return self.get_article_by_url(url)['id']
            elif "UNIQUE constraint failed: articles.content_hash" in str(e):
                logger.debug(f"Article with content hash already exists: {content_hash}")
                return self.get_article_by_content_hash(content_hash)['id']
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to insert article: {e}")
            raise

    def insert_content(self, article_id: int, original_content: str,
                      analysis: dict) -> int:
        """Insert content and analysis for an article"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO content (
                        article_id, original_content, methodology_detailed,
                        technical_approach, key_findings, research_design, 
                        metadata, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article_id,
                    original_content,
                    analysis.get('methodology_detailed', ''),
                    analysis.get('technical_approach', ''),
                    analysis.get('key_findings', ''),
                    analysis.get('research_design', ''),
                    json.dumps(analysis.get('metadata', {})),
                    analysis.get('confidence_score', 0)
                ))

                content_id = cursor.lastrowid
                logger.debug(f"Inserted content with ID {content_id} for article {article_id}")
                return content_id

        except Exception as e:
            logger.error(f"Failed to insert content for article {article_id}: {e}")
            raise

    def log_processing(self, article_id: int | None, status: str,
                      error_message: str | None = None,
                      processing_step: str | None = None,
                      duration_seconds: float | None = None):
        """Log processing information"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT INTO processing_log (article_id, status, error_message, processing_step, duration_seconds)
                    VALUES (?, ?, ?, ?, ?)
                ''', (article_id, status, error_message, processing_step, duration_seconds))

        except Exception as e:
            logger.error(f"Failed to log processing: {e}")

    def get_existing_content_hashes(self) -> set[str]:
        """Get set of existing content hashes for duplicate detection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('SELECT content_hash FROM articles')
                return {row['content_hash'] for row in cursor.fetchall()}

        except Exception as e:
            logger.error(f"Failed to get existing content hashes: {e}")
            return set()

    def get_analyzed_content_hashes(self) -> set[str]:
        """Get set of content hashes for articles that have been fully analyzed"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT a.content_hash 
                    FROM articles a 
                    JOIN content c ON a.id = c.article_id 
                    WHERE a.status = 'completed'
                ''')
                return {row['content_hash'] for row in cursor.fetchall()}

        except Exception as e:
            logger.error(f"Failed to get analyzed content hashes: {e}")
            return set()

    def get_article_by_url(self, url: str) -> sqlite3.Row | None:
        """Get article by URL"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('SELECT * FROM articles WHERE url = ?', (url,))
                return cursor.fetchone()

        except Exception as e:
            logger.error(f"Failed to get article by URL: {e}")
            return None

    def get_article_by_content_hash(self, content_hash: str) -> sqlite3.Row | None:
        """Get article by content hash"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('SELECT * FROM articles WHERE content_hash = ?', (content_hash,))
                return cursor.fetchone()

        except Exception as e:
            logger.error(f"Failed to get article by content hash: {e}")
            return None

    def update_article_status(self, article_id: int, status: str):
        """Update article processing status"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    UPDATE articles 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (status, article_id))

        except Exception as e:
            logger.error(f"Failed to update article status: {e}")

    def get_articles_for_processing(self, limit: int | None = None,
                                  status: str = 'pending') -> list[sqlite3.Row]:
        """Get articles that need processing"""
        try:
            with self.get_connection() as conn:
                query = 'SELECT * FROM articles WHERE status = ? ORDER BY created_at DESC'
                params = [status]

                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)

                cursor = conn.execute(query, params)
                return cursor.fetchall()

        except Exception as e:
            logger.error(f"Failed to get articles for processing: {e}")
            return []

    def get_completed_articles_with_content(self, limit: int | None = None) -> list[dict]:
        """Get completed articles with their content and analysis"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT 
                        a.id, a.title, a.url, a.publication_date, a.processed_date,
                        c.original_content, c.summary, c.methodology_focus,
                        c.key_findings, c.technical_approach, c.practical_applications,
                        c.novel_contributions, c.significance, c.metadata, c.confidence_score
                    FROM articles a
                    JOIN content c ON a.id = c.article_id
                    WHERE a.status = 'completed'
                    ORDER BY a.processed_date DESC
                '''

                if limit:
                    query += ' LIMIT ?'
                    cursor = conn.execute(query, [limit])
                else:
                    cursor = conn.execute(query)

                articles = []
                for row in cursor.fetchall():
                    article = dict(row)
                    # Parse JSON metadata
                    try:
                        article['metadata'] = json.loads(article['metadata']) if article['metadata'] else {}
                    except json.JSONDecodeError:
                        article['metadata'] = {}

                    articles.append(article)

                return articles

        except Exception as e:
            logger.error(f"Failed to get completed articles: {e}")
            return []

    def get_processing_statistics(self) -> dict:
        """Get processing statistics"""
        try:
            with self.get_connection() as conn:
                stats = {}

                # Total articles
                cursor = conn.execute('SELECT COUNT(*) as count FROM articles')
                stats['total_articles'] = cursor.fetchone()['count']

                # Articles by status
                cursor = conn.execute('''
                    SELECT status, COUNT(*) as count 
                    FROM articles 
                    GROUP BY status
                ''')
                stats['by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}

                # Recent processing activity
                cursor = conn.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM processing_log
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                ''')
                stats['recent_activity'] = [dict(row) for row in cursor.fetchall()]

                # Average confidence score
                cursor = conn.execute('''
                    SELECT AVG(confidence_score) as avg_confidence
                    FROM content
                    WHERE confidence_score > 0
                ''')
                result = cursor.fetchone()
                stats['average_confidence'] = result['avg_confidence'] if result['avg_confidence'] else 0

                return stats

        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {}

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old processing logs"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(f'''
                    DELETE FROM processing_log 
                    WHERE timestamp < datetime('now', '-{days_to_keep} days')
                ''')

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old processing log entries")

        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")

    def backup_database(self, backup_path: str):
        """Create a backup of the database"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise

    def get_database_info(self) -> dict:
        """Get information about the database"""
        try:
            info = {
                'database_path': self.db_path,
                'file_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
                'exists': os.path.exists(self.db_path)
            }

            with self.get_connection() as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                info['tables'] = [row['name'] for row in cursor.fetchall()]

                cursor = conn.execute("PRAGMA user_version")
                info['user_version'] = cursor.fetchone()[0]

            return info

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {'error': str(e)}
