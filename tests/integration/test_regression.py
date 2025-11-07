"""
Regression Testing Suite

Ensures backward compatibility and that optimizations
don't break existing functionality:
- All existing features still work
- Database schema migrations
- Configuration compatibility
- API client compatibility
- Report generation compatibility
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.clients import AIClientFactory
from src.core import DatabaseManager, RSSParser, WebScraper
from src.core.cache import ContentCache
from src.processors.article_processor import ArticleProcessor, ProcessingConfig


class TestDatabaseBackwardCompatibility:
    """Test database backward compatibility"""

    def test_legacy_database_schema_migration(self):
        """Test that legacy databases can be migrated"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"

            # Create legacy schema (without content_hash)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("""
                    CREATE TABLE articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        url TEXT UNIQUE NOT NULL,
                        description TEXT,
                        content TEXT,
                        status TEXT DEFAULT 'pending',
                        processed_date TEXT
                    )
                """)

                # Insert legacy data
                conn.execute("""
                    INSERT INTO articles (title, url, description)
                    VALUES (?, ?, ?)
                """, ("Legacy Article", "https://example.com/legacy", "Description"))
                conn.commit()

            # Now use DatabaseManager which should handle migration
            db = DatabaseManager(str(db_path))

            # Verify migration worked
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("PRAGMA table_info(articles)")
                columns = {row[1] for row in cursor.fetchall()}

                print(f"\n=== Schema Migration ===")
                print(f"Columns: {columns}")

                # Should have content_hash column after migration
                assert "content_hash" in columns

                # Verify existing data preserved
                cursor = conn.execute("SELECT title, url FROM articles")
                row = cursor.fetchone()
                assert row[0] == "Legacy Article"
                assert row[1] == "https://example.com/legacy"

    def test_database_indices_preserved(self):
        """Test that database indices are preserved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DatabaseManager(str(db_path))

            # Verify indices exist
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='index'
                """)
                indices = [row[0] for row in cursor.fetchall()]

                print(f"\n=== Database Indices ===")
                print(f"Indices: {indices}")

                # Should have indices on critical fields
                # Note: sqlite_autoindex_* are created for UNIQUE constraints
                assert any("url" in idx.lower() or "autoindex" in idx for idx in indices)


class TestConfigurationCompatibility:
    """Test configuration backward compatibility"""

    def test_legacy_config_format(self):
        """Test that legacy config format still works"""
        legacy_config = {
            "db_path": "data/articles.db",
            "rss_feed_url": "https://example.com/feed.xml",
            "api_provider": "anthropic",
            "anthropic_api_key": "test-key-123",
            # Old format without new optimization settings
        }

        # Should work with legacy config
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                legacy_config["db_path"] = str(Path(tmpdir) / "test.db")
                legacy_config["cache_db_path"] = str(Path(tmpdir) / "cache.db")
                legacy_config["output_dir"] = str(Path(tmpdir) / "output")
                Path(legacy_config["output_dir"]).mkdir()

                processor = ArticleProcessor(legacy_config)
                assert processor is not None
                assert processor.db is not None
                assert processor.cache is not None

                print("\n=== Legacy Config Compatibility ===")
                print("Legacy config format: OK")
        except Exception as e:
            pytest.fail(f"Legacy config failed: {e}")

    def test_new_config_format(self):
        """Test that new config format with optimizations works"""
        new_config = {
            "db_path": "data/articles.db",
            "cache_db_path": "data/cache.db",
            "rss_feed_url": "https://example.com/feed.xml",
            "api_provider": "anthropic",
            "anthropic_api_key": "test-key-123",
            # New optimization settings
            "pool_size": 10,
            "max_concurrent": 5,
            "enable_cache": True,
        }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                new_config["db_path"] = str(Path(tmpdir) / "test.db")
                new_config["cache_db_path"] = str(Path(tmpdir) / "cache.db")
                new_config["output_dir"] = str(Path(tmpdir) / "output")
                Path(new_config["output_dir"]).mkdir()

                processor = ArticleProcessor(new_config)
                assert processor is not None

                print("\n=== New Config Format ===")
                print("New config with optimizations: OK")
        except Exception as e:
            pytest.fail(f"New config failed: {e}")


class TestAPIClientBackwardCompatibility:
    """Test API client backward compatibility"""

    def test_synchronous_client_still_works(self):
        """Test that synchronous clients still work"""
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "methodology_detailed": "Analysis",
            "technical_approach": "Technical",
            "key_findings": "Findings",
            "research_design": "Design",
            "extracted_title": "Title",
            "metadata": {"provider": "test"}
        })
        mock_response.content = [mock_content]

        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_instance = MagicMock()
            mock_instance.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_instance

            # Create synchronous client
            client = AIClientFactory.create_client(
                provider="anthropic",
                api_key="test-key-123"
            )

            # Use synchronous API
            result = client.analyze_article(
                title="Test Article",
                content="Test content"
            )

            assert result is not None
            assert "methodology_detailed" in result

            print("\n=== Synchronous Client ===")
            print("Synchronous client API: OK")

    def test_async_client_coexists_with_sync(self):
        """Test that async clients don't break sync clients"""
        # Both should be available
        sync_client = AIClientFactory.create_client(
            provider="anthropic",
            api_key="test-key-123"
        )

        async_client = AIClientFactory.create_async_client(
            provider="anthropic",
            api_key="test-key-123"
        )

        assert sync_client is not None
        assert async_client is not None
        assert sync_client != async_client

        print("\n=== Client Coexistence ===")
        print("Sync and async clients coexist: OK")


class TestFeatureRegression:
    """Test that all existing features still work"""

    def test_rss_parsing_unchanged(self):
        """Test RSS parsing still works as before"""
        mock_feed = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Test Article</title>
              <link>https://example.com/test</link>
              <description>Description</description>
            </item>
          </channel>
        </rss>
        """

        mock_response = MagicMock()
        mock_response.text = mock_feed
        mock_response.status_code = 200

        with patch('requests.get', return_value=mock_response):
            parser = RSSParser("https://example.com/feed.xml")
            entries = parser.fetch_feed()

            assert len(entries) == 1
            assert entries[0]["title"] == "Test Article"
            assert entries[0]["link"] == "https://example.com/test"

            print("\n=== RSS Parsing ===")
            print("RSS parsing unchanged: OK")

    def test_web_scraping_unchanged(self):
        """Test web scraping still works as before"""
        mock_html = """
        <html>
        <body>
            <article>
                <h1>Test Article</h1>
                <p>Test content</p>
            </article>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.status_code = 200
        mock_response.content = mock_html.encode('utf-8')

        with patch('requests.get', return_value=mock_response):
            scraper = WebScraper()
            content = scraper.scrape_article("https://example.com/test")

            assert content is not None
            assert "Test Article" in content
            assert "Test content" in content

            print("\n=== Web Scraping ===")
            print("Web scraping unchanged: OK")

    def test_report_generation_unchanged(self):
        """Test report generation still works as before"""
        from src.core.report_generator import ReportGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            db = DatabaseManager(str(db_path))

            # Add test article
            article_id = db.insert_article(
                title="Test Article",
                url="https://example.com/test",
                description="Description",
                content="Content",
                content_hash="hash_test"
            )

            db.store_content(article_id, {
                "methodology_detailed": "Analysis",
                "technical_approach": "Technical",
                "key_findings": "Findings",
                "research_design": "Design"
            })

            # Generate report
            generator = ReportGenerator(str(output_dir))
            articles = db.get_recent_articles(limit=10)
            reports = generator.generate_all_reports(articles)

            assert reports is not None
            assert "markdown_report" in reports
            assert Path(reports["markdown_report"]).exists()

            print("\n=== Report Generation ===")
            print("Report generation unchanged: OK")


class TestDataIntegrity:
    """Test data integrity after optimizations"""

    def test_no_data_corruption(self):
        """Test that optimizations don't corrupt data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "integrity.db"
            db = DatabaseManager(str(db_path))

            # Insert test data
            test_data = {
                "title": "Test Article with Special Chars: @#$%^&*()",
                "url": "https://example.com/test?param=value&other=123",
                "description": "Description with unicode: \u00e9\u00e8\u00ea",
                "content": "Content with\nNewlines\tTabs and emoji \U0001f60a",
                "content_hash": "hash_special"
            }

            article_id = db.insert_article(**test_data)

            # Retrieve and verify exact match
            article = db.get_article_by_id(article_id)

            assert article["title"] == test_data["title"]
            assert article["url"] == test_data["url"]
            assert article["description"] == test_data["description"]
            assert article["content"] == test_data["content"]

            print("\n=== Data Integrity ===")
            print("No data corruption: OK")

    def test_concurrent_writes_maintain_consistency(self):
        """Test that concurrent writes maintain data consistency"""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "concurrent.db")
            db = DatabaseManager(str(db_path))

            errors = []
            inserted_ids = []

            def worker(worker_id: int):
                try:
                    for i in range(10):
                        article_id = db.insert_article(
                            title=f"Worker {worker_id} Article {i}",
                            url=f"https://example.com/w{worker_id}/a{i}",
                            description="Description",
                            content="Content",
                            content_hash=f"hash_{worker_id}_{i}"
                        )
                        inserted_ids.append(article_id)
                except Exception as e:
                    errors.append((worker_id, str(e)))

            threads = []
            for worker_id in range(5):
                thread = threading.Thread(target=worker, args=(worker_id,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Verify no errors
            assert len(errors) == 0

            # Verify all articles inserted
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM articles")
                count = cursor.fetchone()[0]
                assert count == 50  # 5 workers × 10 articles

            # Verify all IDs are unique
            assert len(inserted_ids) == len(set(inserted_ids))

            print("\n=== Concurrent Consistency ===")
            print("Concurrent writes maintain consistency: OK")


class TestPerformanceRegression:
    """Test that performance hasn't regressed"""

    def test_database_query_performance(self):
        """Test database queries haven't slowed down"""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "perf.db"
            db = DatabaseManager(str(db_path))

            # Insert test data
            for i in range(100):
                db.insert_article(
                    title=f"Article {i}",
                    url=f"https://example.com/{i}",
                    description="Description",
                    content="Content",
                    content_hash=f"hash_{i}"
                )

            # Test query performance
            start = time.time()
            articles = db.get_recent_articles(limit=50)
            duration = time.time() - start

            print(f"\n=== Query Performance ===")
            print(f"Retrieved 50 articles in {duration*1000:.2f}ms")

            assert len(articles) == 50
            assert duration < 0.1  # Should be fast (<100ms)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
