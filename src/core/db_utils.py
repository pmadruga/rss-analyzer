"""
Database Utilities for Concurrent Write Operations

Provides retry logic with exponential backoff and advisory locking
to reduce database lock errors during batch operations.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def exponential_backoff_retry(
    max_retries: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    catch_exceptions: tuple = (sqlite3.OperationalError,),
):
    """
    Decorator for retrying database operations with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential calculation
        catch_exceptions: Tuple of exceptions to catch and retry

    Example:
        @exponential_backoff_retry(max_retries=5)
        def insert_article(conn, article):
            conn.execute("INSERT INTO articles ...")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except catch_exceptions as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Only retry on lock errors
                    if "locked" not in error_msg and "busy" not in error_msg:
                        raise

                    if attempt >= max_retries:
                        logger.error(
                            f"Failed after {max_retries} retries: {func.__name__}. "
                            f"Error: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    logger.warning(
                        f"Database lock in {func.__name__}, attempt {attempt + 1}/{max_retries}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


class AdvisoryLock:
    """
    Advisory locking mechanism for coordinating concurrent writes

    Uses SQLite's application_id as a simple advisory lock.
    For production, consider using a separate lock table.

    Example:
        with AdvisoryLock(conn, "batch_insert"):
            # Perform batch operation
            conn.executemany(...)
    """

    def __init__(self, conn: sqlite3.Connection, lock_name: str):
        self.conn = conn
        self.lock_name = lock_name
        self.lock_id = hash(lock_name) % (2**31)  # 32-bit integer
        self.acquired = False

    def __enter__(self):
        """Acquire advisory lock"""
        self._acquire_lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release advisory lock"""
        self._release_lock()
        return False

    @exponential_backoff_retry(max_retries=10, base_delay=0.05, max_delay=5.0)
    def _acquire_lock(self):
        """Attempt to acquire lock with retry"""
        # Check if lock table exists
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_advisory_locks'"
        )

        if not cursor.fetchone():
            # Create lock table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _advisory_locks (
                    lock_id INTEGER PRIMARY KEY,
                    lock_name TEXT NOT NULL,
                    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acquired_by TEXT
                )
                """
            )
            self.conn.commit()

        # Try to acquire lock
        try:
            self.conn.execute(
                "INSERT INTO _advisory_locks (lock_id, lock_name, acquired_by) VALUES (?, ?, ?)",
                (self.lock_id, self.lock_name, f"pid-{id(self)}")
            )
            self.conn.commit()
            self.acquired = True
            logger.debug(f"Acquired advisory lock: {self.lock_name}")

        except sqlite3.IntegrityError:
            # Lock already held
            raise sqlite3.OperationalError(f"Advisory lock {self.lock_name} is held by another operation")

    def _release_lock(self):
        """Release the advisory lock"""
        if self.acquired:
            try:
                self.conn.execute(
                    "DELETE FROM _advisory_locks WHERE lock_id = ?",
                    (self.lock_id,)
                )
                self.conn.commit()
                logger.debug(f"Released advisory lock: {self.lock_name}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to release advisory lock {self.lock_name}: {e}")
            finally:
                self.acquired = False


@contextmanager
def batch_write_context(conn: sqlite3.Connection, lock_name: Optional[str] = None):
    """
    Context manager for batch write operations with optimizations

    - Temporarily disables synchronous mode for speed
    - Uses advisory locking if lock_name provided
    - Automatically commits or rolls back

    Args:
        conn: Database connection
        lock_name: Optional advisory lock name

    Example:
        with batch_write_context(conn, "article_batch") as cursor:
            cursor.executemany("INSERT INTO articles ...", articles)
    """
    # Store original settings
    original_sync = conn.execute("PRAGMA synchronous").fetchone()[0]

    # Optimize for batch writes
    conn.execute("PRAGMA synchronous = OFF")  # Faster, still safe with WAL

    # Acquire advisory lock if requested
    lock = AdvisoryLock(conn, lock_name) if lock_name else None

    try:
        if lock:
            lock.__enter__()

        cursor = conn.cursor()
        yield cursor

        # Commit if no errors
        conn.commit()

    except Exception as e:
        # Rollback on error
        conn.rollback()
        logger.error(f"Batch write failed, rolled back: {e}")
        raise

    finally:
        # Restore original settings
        conn.execute(f"PRAGMA synchronous = {original_sync}")

        if lock:
            lock.__exit__(None, None, None)


def optimize_for_batch_writes(conn: sqlite3.Connection):
    """
    Optimize connection settings for batch write operations

    Call this before performing large batch operations.
    Settings are temporary and connection-specific.

    Args:
        conn: Database connection
    """
    # These optimizations are safe with WAL mode
    conn.execute("PRAGMA synchronous = OFF")  # Faster writes, safe with WAL
    conn.execute("PRAGMA locking_mode = EXCLUSIVE")  # Exclusive lock for this connection
    conn.execute("PRAGMA temp_store = MEMORY")  # Keep temp data in memory

    logger.debug("Optimized connection for batch writes")


def restore_normal_mode(conn: sqlite3.Connection):
    """
    Restore normal connection settings after batch operations

    Args:
        conn: Database connection
    """
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA locking_mode = NORMAL")

    logger.debug("Restored normal connection settings")


@exponential_backoff_retry(max_retries=3)
def safe_executemany(
    conn: sqlite3.Connection,
    query: str,
    data: list,
    use_advisory_lock: bool = True,
    lock_name: Optional[str] = None
) -> sqlite3.Cursor:
    """
    Execute a batch query with retry and optional advisory locking

    Args:
        conn: Database connection
        query: SQL query with placeholders
        data: List of tuples/dicts for executemany
        use_advisory_lock: Whether to use advisory locking
        lock_name: Custom lock name (auto-generated if None)

    Returns:
        Cursor after execution

    Example:
        cursor = safe_executemany(
            conn,
            "INSERT INTO articles (title, url) VALUES (?, ?)",
            [(title1, url1), (title2, url2)]
        )
    """
    if not data:
        return conn.cursor()

    # Generate lock name from query if not provided
    if use_advisory_lock and lock_name is None:
        # Extract table name from query
        query_lower = query.lower()
        if "insert into" in query_lower:
            table = query_lower.split("insert into")[1].split()[0].strip()
            lock_name = f"batch_{table}"
        else:
            lock_name = "batch_operation"

    if use_advisory_lock:
        with batch_write_context(conn, lock_name) as cursor:
            cursor.executemany(query, data)
            return cursor
    else:
        cursor = conn.cursor()
        cursor.executemany(query, data)
        conn.commit()
        return cursor


class BatchWriter:
    """
    Buffered batch writer for efficient bulk inserts

    Accumulates writes and flushes when buffer is full.
    Reduces database lock contention by batching operations.

    Example:
        with BatchWriter(conn, "articles", batch_size=100) as writer:
            for article in articles:
                writer.add(
                    "INSERT INTO articles (title, url) VALUES (?, ?)",
                    (article.title, article.url)
                )
            # Auto-flushes on exit
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        batch_size: int = 100,
        auto_commit: bool = True
    ):
        self.conn = conn
        self.table_name = table_name
        self.batch_size = batch_size
        self.auto_commit = auto_commit
        self.buffer = []
        self.query = None

    def add(self, query: str, params: tuple):
        """Add a record to the batch"""
        if self.query is None:
            self.query = query
        elif self.query != query:
            # Different query, flush current batch first
            self.flush()
            self.query = query

        self.buffer.append(params)

        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """Flush buffered writes to database"""
        if not self.buffer or not self.query:
            return

        try:
            safe_executemany(
                self.conn,
                self.query,
                self.buffer,
                use_advisory_lock=True,
                lock_name=f"batch_{self.table_name}"
            )

            logger.debug(f"Flushed {len(self.buffer)} records to {self.table_name}")
            self.buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush batch to {self.table_name}: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Normal exit, flush remaining
            self.flush()
        else:
            # Error occurred, log buffer size
            logger.warning(f"Batch writer exiting with error, {len(self.buffer)} records in buffer")
        return False
