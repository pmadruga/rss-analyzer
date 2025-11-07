"""
Database Module

Handles SQLite database operations for articles, content, and processing logs.
Includes duplicate detection, migrations, efficient querying, and connection pooling.
"""

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from queue import Empty, Queue
from typing import Optional

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Thread-safe SQLite connection pool"""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._active_connections = 0
        self._total_created = 0
        self._closed = False

        # Pre-populate the pool with connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
            self._total_created += 1

        logger.debug(f"Initialized connection pool with {pool_size} connections")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper settings for concurrency"""
        # Set timeout=30.0 for automatic retry on database lock
        conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0
        )
        conn.row_factory = sqlite3.Row  # Enable dict-like access

        # Enable Write-Ahead Logging for concurrent read/write access
        # WAL mode allows readers to operate while writer is active
        conn.execute("PRAGMA journal_mode = WAL")

        # Set busy timeout to 30 seconds (30000ms)
        # Retry automatically when database is locked instead of failing immediately
        conn.execute("PRAGMA busy_timeout = 30000")

        # Balance between safety and performance
        # NORMAL sync provides good durability with better performance than FULL
        conn.execute("PRAGMA synchronous = NORMAL")

        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")

        # Optimize cache size for better performance (64MB)
        # Negative value = size in KB
        conn.execute("PRAGMA cache_size = -64000")

        return conn

    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Validate that a connection is still usable"""
        try:
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool (context manager)

        Usage:
            with pool.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM articles")
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        conn = None
        try:
            # Try to get connection from pool with timeout
            conn = self._pool.get(timeout=30)

            # Validate connection and recreate if necessary
            if not self._validate_connection(conn):
                logger.warning("Invalid connection detected, creating new one")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = self._create_connection()

            with self._lock:
                self._active_connections += 1

            yield conn

        except Empty:
            raise RuntimeError(
                "Timeout waiting for database connection from pool"
            )
        finally:
            if conn:
                try:
                    # Return connection to pool
                    self._pool.put(conn, block=False)
                except Exception as e:
                    logger.error(f"Failed to return connection to pool: {e}")
                finally:
                    with self._lock:
                        self._active_connections -= 1

    def get_pool_stats(self) -> dict:
        """Get statistics about the connection pool"""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "active_connections": self._active_connections,
                "idle_connections": self._pool.qsize(),
                "total_connections_created": self._total_created,
                "closed": self._closed,
            }

    def close_pool(self):
        """Close all connections in the pool"""
        if self._closed:
            return

        with self._lock:
            self._closed = True

        # Close all connections in the pool
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except (Empty, Exception) as e:
                logger.debug(f"Error closing pooled connection: {e}")

        logger.info("Connection pool closed")


class DatabaseManager:
    """SQLite database manager for RSS article analyzer with connection pooling"""

    def __init__(self, db_path: str = "data/articles.db", pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.ensure_directory_exists()

        # Initialize connection pool
        self._pool: Optional[ConnectionPool] = ConnectionPool(db_path, pool_size)

        self.init_database()

    def ensure_directory_exists(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

    @contextmanager
    def get_connection(self):
        """
        Get database connection from pool (backward compatible)

        Usage:
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM articles")
        """
        with self._pool.get_connection() as conn:
            yield conn

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics"""
        return self._pool.get_pool_stats()

    def close_pool(self):
        """Close the connection pool and all connections"""
        if self._pool:
            self._pool.close_pool()
            self._pool = None

    def __del__(self):
        """Cleanup connection pool on object destruction"""
        try:
            self.close_pool()
        except Exception:
            pass  # Ignore errors during cleanup

    def init_database(self):
        """Initialize database with required tables"""
        try:
            with self.get_connection() as conn:
                # Create articles table
                conn.execute("""
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
                """)

                # Create content table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS content (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id INTEGER NOT NULL,
                        original_content TEXT,
                        methodology_detailed TEXT,
                        technical_approach TEXT,
                        key_findings TEXT,
                        research_design TEXT,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (article_id) REFERENCES articles (id) ON DELETE CASCADE
                    )
                """)

                # Create processing log table
                conn.execute("""
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
                """)

                # Create indices for better performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles (content_hash)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_articles_url ON articles (url)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_articles_status ON articles (status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_processing_log_timestamp ON processing_log (timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_content_article_id ON content (article_id)"
                )

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

            old_columns = [
                "summary",
                "methodology_focus",
                "practical_applications",
                "novel_contributions",
                "significance",
            ]
            if any(col in columns for col in old_columns):
                logger.info("Migrating content table to new schema...")

                # Add new columns if they don't exist
                if "methodology_detailed" not in columns:
                    conn.execute(
                        "ALTER TABLE content ADD COLUMN methodology_detailed TEXT"
                    )
                if "research_design" not in columns:
                    conn.execute("ALTER TABLE content ADD COLUMN research_design TEXT")

                # Migrate data from old columns to new ones
                conn.execute("""
                    UPDATE content
                    SET methodology_detailed = COALESCE(methodology_focus, ''),
                        research_design = ''
                    WHERE methodology_detailed IS NULL
                """)

                logger.info("Content table migration completed")

        except Exception as e:
            logger.warning(f"Migration warning (non-critical): {e}")

    def insert_article(
        self,
        title: str,
        url: str,
        content_hash: str,
        rss_guid: str | None = None,
        publication_date: datetime | None = None,
    ) -> int:
        """
        Insert new article into database

        Returns:
            Article ID of inserted article
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO articles (title, url, content_hash, rss_guid, publication_date)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (title, url, content_hash, rss_guid, publication_date),
                )

                article_id = cursor.lastrowid
                logger.debug(f"Inserted article with ID {article_id}: {title}")
                return article_id

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: articles.url" in str(e):
                logger.debug(f"Article with URL already exists: {url}")
                return self.get_article_by_url(url)["id"]
            elif "UNIQUE constraint failed: articles.content_hash" in str(e):
                logger.debug(
                    f"Article with content hash already exists: {content_hash}"
                )
                return self.get_article_by_content_hash(content_hash)["id"]
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to insert article: {e}")
            raise

    def insert_content(
        self, article_id: int, original_content: str, analysis: dict
    ) -> int:
        """Insert content and analysis for an article"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO content (
                        article_id, original_content, methodology_detailed,
                        technical_approach, key_findings, research_design,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        article_id,
                        original_content,
                        analysis.get("methodology_detailed", ""),
                        analysis.get("technical_approach", ""),
                        analysis.get("key_findings", ""),
                        analysis.get("research_design", ""),
                        json.dumps(analysis.get("metadata", {})),
                    ),
                )

                content_id = cursor.lastrowid
                logger.debug(
                    f"Inserted content with ID {content_id} for article {article_id}"
                )
                return content_id

        except Exception as e:
            logger.error(f"Failed to insert content for article {article_id}: {e}")
            raise

    def log_processing(
        self,
        article_id: int | None,
        status: str,
        error_message: str | None = None,
        processing_step: str | None = None,
        duration_seconds: float | None = None,
    ):
        """Log processing information"""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO processing_log (article_id, status, error_message, processing_step, duration_seconds)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        article_id,
                        status,
                        error_message,
                        processing_step,
                        duration_seconds,
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to log processing: {e}")

    def get_existing_content_hashes(self) -> set[str]:
        """Get set of existing content hashes for duplicate detection"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT content_hash FROM articles")
                return {row["content_hash"] for row in cursor.fetchall()}

        except Exception as e:
            logger.error(f"Failed to get existing content hashes: {e}")
            return set()

    def get_analyzed_content_hashes(self) -> set[str]:
        """Get set of content hashes for articles that have been fully analyzed"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT a.content_hash
                    FROM articles a
                    JOIN content c ON a.id = c.article_id
                    WHERE a.status = 'completed'
                """)
                return {row["content_hash"] for row in cursor.fetchall()}

        except Exception as e:
            logger.error(f"Failed to get analyzed content hashes: {e}")
            return set()

    def is_content_already_processed(self, content_hash: str) -> bool:
        """Check if content with this hash has already been processed"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 1 
                    FROM articles a
                    JOIN content c ON a.id = c.article_id
                    WHERE a.content_hash = ? AND a.status = 'completed'
                    LIMIT 1
                """, (content_hash,))
                return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Failed to check if content is processed: {e}")
            return False

    def update_article_content_hash(self, article_id: int, content_hash: str):
        """Update the content hash of an article after scraping"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE articles 
                    SET content_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (content_hash, article_id))

        except Exception as e:
            logger.error(f"Failed to update content hash for article {article_id}: {e}")

    def get_article_by_url(self, url: str) -> sqlite3.Row | None:
        """Get article by URL"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM articles WHERE url = ?", (url,))
                return cursor.fetchone()

        except Exception as e:
            logger.error(f"Failed to get article by URL: {e}")
            return None

    def get_article_by_content_hash(self, content_hash: str) -> sqlite3.Row | None:
        """Get article by content hash"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM articles WHERE content_hash = ?", (content_hash,)
                )
                return cursor.fetchone()

        except Exception as e:
            logger.error(f"Failed to get article by content hash: {e}")
            return None

    def update_article_status(self, article_id: int, status: str):
        """Update article processing status"""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE articles
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (status, article_id),
                )

        except Exception as e:
            logger.error(f"Failed to update article status: {e}")

    def get_articles_for_processing(
        self, limit: int | None = None, status: str = "pending"
    ) -> list[sqlite3.Row]:
        """Get articles that need processing"""
        try:
            with self.get_connection() as conn:
                query = (
                    "SELECT * FROM articles WHERE status = ? ORDER BY created_at DESC"
                )
                params = [status]

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                return cursor.fetchall()

        except Exception as e:
            logger.error(f"Failed to get articles for processing: {e}")
            return []

    def get_completed_articles_with_content(
        self, limit: int | None = None
    ) -> list[dict]:
        """Get completed articles with their content and analysis"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT
                        a.id, a.title, a.url, a.publication_date, a.processed_date,
                        c.original_content, c.summary, c.methodology_focus,
                        c.key_findings, c.technical_approach, c.practical_applications,
                        c.novel_contributions, c.significance, c.metadata
                    FROM articles a
                    JOIN content c ON a.id = c.article_id
                    WHERE a.status = 'completed'
                    ORDER BY a.processed_date DESC
                """

                if limit:
                    query += " LIMIT ?"
                    cursor = conn.execute(query, [limit])
                else:
                    cursor = conn.execute(query)

                articles = []
                for row in cursor.fetchall():
                    article = dict(row)
                    # Parse JSON metadata
                    try:
                        article["metadata"] = (
                            json.loads(article["metadata"])
                            if article["metadata"]
                            else {}
                        )
                    except json.JSONDecodeError:
                        article["metadata"] = {}

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
                cursor = conn.execute("SELECT COUNT(*) as count FROM articles")
                stats["total_articles"] = cursor.fetchone()["count"]

                # Articles by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM articles
                    GROUP BY status
                """)
                stats["by_status"] = {
                    row["status"]: row["count"] for row in cursor.fetchall()
                }

                # Recent processing activity
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM processing_log
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """)
                stats["recent_activity"] = [dict(row) for row in cursor.fetchall()]

                return stats

        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {}

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old processing logs"""
        try:
            with self.get_connection() as conn:
                # Use parameterized query to prevent SQL injection
                cursor = conn.execute("""
                    DELETE FROM processing_log
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """, (days_to_keep,))

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(
                        f"Cleaned up {deleted_count} old processing log entries"
                    )

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
                "database_path": self.db_path,
                "file_size_mb": os.path.getsize(self.db_path) / (1024 * 1024)
                if os.path.exists(self.db_path)
                else 0,
                "exists": os.path.exists(self.db_path),
            }

            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                info["tables"] = [row["name"] for row in cursor.fetchall()]

                cursor = conn.execute("PRAGMA user_version")
                info["user_version"] = cursor.fetchone()[0]

            return info

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}

    def insert_articles_batch(self, articles: list[dict]) -> list[int]:
        """
        Insert multiple articles in a single transaction

        Args:
            articles: List of article dicts with keys: title, url, content_hash,
                     rss_guid (optional), publication_date (optional)

        Returns:
            List of inserted article IDs (may include existing IDs for duplicates)

        Example:
            articles = [
                {
                    "title": "Paper 1",
                    "url": "https://example.com/1",
                    "content_hash": "abc123",
                    "rss_guid": "guid1",
                    "publication_date": datetime.now()
                },
                ...
            ]
            article_ids = db.insert_articles_batch(articles)
        """
        if not articles:
            return []

        article_ids = []
        batch_size = 50  # Process in chunks of 50

        try:
            for i in range(0, len(articles), batch_size):
                batch = articles[i : i + batch_size]

                with self.get_connection() as conn:
                    # Begin explicit transaction for atomicity
                    conn.execute("BEGIN TRANSACTION")

                    try:
                        for article in batch:
                            try:
                                cursor = conn.execute(
                                    """
                                    INSERT INTO articles (title, url, content_hash, rss_guid, publication_date)
                                    VALUES (?, ?, ?, ?, ?)
                                """,
                                    (
                                        article["title"],
                                        article["url"],
                                        article["content_hash"],
                                        article.get("rss_guid"),
                                        article.get("publication_date"),
                                    ),
                                )
                                article_ids.append(cursor.lastrowid)

                            except sqlite3.IntegrityError as e:
                                # Handle duplicates gracefully by fetching existing ID
                                if "UNIQUE constraint failed: articles.url" in str(e):
                                    existing = self.get_article_by_url(article["url"])
                                    if existing:
                                        article_ids.append(existing["id"])
                                        logger.debug(
                                            f"Article already exists (URL): {article['url']}"
                                        )
                                elif "UNIQUE constraint failed: articles.content_hash" in str(
                                    e
                                ):
                                    existing = self.get_article_by_content_hash(
                                        article["content_hash"]
                                    )
                                    if existing:
                                        article_ids.append(existing["id"])
                                        logger.debug(
                                            f"Article already exists (hash): {article['content_hash']}"
                                        )
                                else:
                                    raise

                        conn.commit()
                        logger.debug(
                            f"Batch inserted {len(batch)} articles (batch {i // batch_size + 1})"
                        )

                    except Exception as e:
                        conn.rollback()
                        logger.error(
                            f"Failed to insert article batch {i // batch_size + 1}: {e}"
                        )
                        raise

            logger.info(f"Batch inserted {len(article_ids)} articles total")
            return article_ids

        except Exception as e:
            logger.error(f"Failed to batch insert articles: {e}")
            raise

    def insert_content_batch(self, contents: list[dict]) -> list[int]:
        """
        Insert multiple content records in a single transaction

        Args:
            contents: List of content dicts with keys: article_id, original_content,
                     analysis (dict with methodology_detailed, technical_approach,
                     key_findings, research_design, metadata)

        Returns:
            List of inserted content IDs

        Example:
            contents = [
                {
                    "article_id": 1,
                    "original_content": "Full text...",
                    "analysis": {
                        "methodology_detailed": "...",
                        "technical_approach": "...",
                        "key_findings": "...",
                        "research_design": "...",
                        "metadata": {...}
                    }
                },
                ...
            ]
            content_ids = db.insert_content_batch(contents)
        """
        if not contents:
            return []

        content_ids = []
        batch_size = 50

        try:
            for i in range(0, len(contents), batch_size):
                batch = contents[i : i + batch_size]

                with self.get_connection() as conn:
                    conn.execute("BEGIN TRANSACTION")

                    try:
                        for content in batch:
                            analysis = content.get("analysis", {})

                            cursor = conn.execute(
                                """
                                INSERT INTO content (
                                    article_id, original_content, methodology_detailed,
                                    technical_approach, key_findings, research_design,
                                    metadata
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    content["article_id"],
                                    content.get("original_content", ""),
                                    analysis.get("methodology_detailed", ""),
                                    analysis.get("technical_approach", ""),
                                    analysis.get("key_findings", ""),
                                    analysis.get("research_design", ""),
                                    json.dumps(analysis.get("metadata", {})),
                                ),
                            )
                            content_ids.append(cursor.lastrowid)

                        conn.commit()
                        logger.debug(
                            f"Batch inserted {len(batch)} content records (batch {i // batch_size + 1})"
                        )

                    except Exception as e:
                        conn.rollback()
                        logger.error(
                            f"Failed to insert content batch {i // batch_size + 1}: {e}"
                        )
                        raise

            logger.info(f"Batch inserted {len(content_ids)} content records total")
            return content_ids

        except Exception as e:
            logger.error(f"Failed to batch insert content: {e}")
            raise

    def update_status_batch(self, updates: list[tuple[int, str]]) -> int:
        """
        Update status for multiple articles in a single transaction

        Args:
            updates: List of tuples (article_id, status)

        Returns:
            Number of articles updated

        Example:
            updates = [(1, "completed"), (2, "failed"), (3, "completed")]
            count = db.update_status_batch(updates)
        """
        if not updates:
            return 0

        batch_size = 100
        total_updated = 0

        try:
            for i in range(0, len(updates), batch_size):
                batch = updates[i : i + batch_size]

                with self.get_connection() as conn:
                    conn.execute("BEGIN TRANSACTION")

                    try:
                        for article_id, status in batch:
                            conn.execute(
                                """
                                UPDATE articles
                                SET status = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            """,
                                (status, article_id),
                            )

                        conn.commit()
                        total_updated += len(batch)
                        logger.debug(
                            f"Batch updated {len(batch)} article statuses (batch {i // batch_size + 1})"
                        )

                    except Exception as e:
                        conn.rollback()
                        logger.error(
                            f"Failed to update status batch {i // batch_size + 1}: {e}"
                        )
                        raise

            logger.info(f"Batch updated {total_updated} article statuses total")
            return total_updated

        except Exception as e:
            logger.error(f"Failed to batch update statuses: {e}")
            raise

    def log_processing_batch(self, logs: list[dict]) -> int:
        """
        Insert multiple processing log entries in a single transaction

        Args:
            logs: List of log dicts with keys: article_id (optional), status,
                 error_message (optional), processing_step (optional),
                 duration_seconds (optional)

        Returns:
            Number of log entries inserted

        Example:
            logs = [
                {
                    "article_id": 1,
                    "status": "success",
                    "processing_step": "scraping",
                    "duration_seconds": 2.5
                },
                {
                    "article_id": 2,
                    "status": "error",
                    "error_message": "Timeout",
                    "processing_step": "analysis",
                    "duration_seconds": 5.0
                },
                ...
            ]
            count = db.log_processing_batch(logs)
        """
        if not logs:
            return 0

        batch_size = 100
        total_logged = 0

        try:
            for i in range(0, len(logs), batch_size):
                batch = logs[i : i + batch_size]

                with self.get_connection() as conn:
                    conn.execute("BEGIN TRANSACTION")

                    try:
                        for log in batch:
                            conn.execute(
                                """
                                INSERT INTO processing_log (
                                    article_id, status, error_message,
                                    processing_step, duration_seconds
                                )
                                VALUES (?, ?, ?, ?, ?)
                            """,
                                (
                                    log.get("article_id"),
                                    log["status"],
                                    log.get("error_message"),
                                    log.get("processing_step"),
                                    log.get("duration_seconds"),
                                ),
                            )

                        conn.commit()
                        total_logged += len(batch)
                        logger.debug(
                            f"Batch logged {len(batch)} processing entries (batch {i // batch_size + 1})"
                        )

                    except Exception as e:
                        conn.rollback()
                        logger.error(
                            f"Failed to log processing batch {i // batch_size + 1}: {e}"
                        )
                        # Don't raise for logging errors - log and continue
                        logger.warning(
                            f"Continuing despite logging error in batch {i // batch_size + 1}"
                        )

            logger.info(f"Batch logged {total_logged} processing entries total")
            return total_logged

        except Exception as e:
            logger.error(f"Failed to batch log processing: {e}")
            return total_logged  # Return partial count instead of raising
