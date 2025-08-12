#!/usr/bin/env python
"""
Cleanup duplicate content records in the database.

This script identifies articles with multiple content records and keeps only
the best/most complete analysis, removing duplicates and failed attempts.
"""

import sqlite3
from pathlib import Path


def cleanup_duplicate_content():
    """Remove duplicate content records, keeping the best analysis for each article."""

    # Database path
    db_path = Path(__file__).parent.parent / "data" / "articles.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Find articles with duplicate content records
        cursor.execute("""
            SELECT article_id, COUNT(*) as count
            FROM content
            GROUP BY article_id
            HAVING COUNT(*) > 1
            ORDER BY article_id
        """)

        duplicates = cursor.fetchall()

        if not duplicates:
            print("No duplicate content records found.")
            return

        print(f"Found {len(duplicates)} articles with duplicate content records:")
        for article_id, count in duplicates:
            print(f"  Article {article_id}: {count} content records")

        # Process each article with duplicates
        total_removed = 0
        for article_id, _count in duplicates:
            # Get all content records for this article
            cursor.execute(
                """
                SELECT
                    rowid,
                    LENGTH(COALESCE(methodology_detailed, '')) as meth_len,
                    LENGTH(COALESCE(technical_approach, '')) as tech_len,
                    LENGTH(COALESCE(key_findings, '')) as key_len,
                    LENGTH(COALESCE(research_design, '')) as res_len,
                    methodology_detailed,
                    technical_approach,
                    key_findings
                FROM content
                WHERE article_id = ?
                ORDER BY
                    (meth_len + tech_len + key_len + res_len) DESC,
                    rowid ASC
            """,
                (article_id,),
            )

            records = cursor.fetchall()

            # Keep the first record (best/most complete)
            keep_rowid = records[0][0]
            total_length = sum(records[0][1:5])

            # Check if the best record has meaningful content
            has_failed_text = (
                "Analysis parsing failed" in (records[0][5] or "")
                or "Analysis parsing failed" in (records[0][6] or "")
                or "Analysis parsing failed" in (records[0][7] or "")
            )

            print(f"\n  Article {article_id}:")
            print(
                f"    Keeping record with rowid {keep_rowid} (total length: {total_length} chars)"
            )
            if has_failed_text:
                print(
                    "    WARNING: Best record contains 'Analysis parsing failed' text"
                )

            # Delete the duplicate records
            for record in records[1:]:
                rowid = record[0]
                record_length = sum(record[1:5])
                print(
                    f"    Removing record with rowid {rowid} (total length: {record_length} chars)"
                )
                cursor.execute("DELETE FROM content WHERE rowid = ?", (rowid,))
                total_removed += 1

        # Commit the changes
        conn.commit()
        print(f"\n✅ Successfully removed {total_removed} duplicate content records")

        # Verify the cleanup
        cursor.execute("""
            SELECT article_id, COUNT(*) as count
            FROM content
            GROUP BY article_id
            HAVING COUNT(*) > 1
        """)

        remaining_duplicates = cursor.fetchall()
        if remaining_duplicates:
            print(
                f"\n⚠️  Warning: {len(remaining_duplicates)} articles still have duplicates"
            )
        else:
            print("\n✅ All duplicate content records have been cleaned up")

        # Show final statistics
        cursor.execute("SELECT COUNT(DISTINCT article_id) FROM content")
        unique_articles = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM content")
        total_content = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]

        print("\nFinal database statistics:")
        print(f"  Total articles: {total_articles}")
        print(f"  Articles with content: {unique_articles}")
        print(f"  Total content records: {total_content}")

        if unique_articles == total_content:
            print("  ✅ Each article has exactly one content record")

    except Exception as e:
        print(f"Error during cleanup: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    print("Starting duplicate content cleanup...")
    print("=" * 50)
    cleanup_duplicate_content()
    print("=" * 50)
    print("Cleanup complete!")
