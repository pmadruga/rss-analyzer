#!/usr/bin/env python
"""
Fix JSON-formatted analyses in the database by extracting the actual text content.

This script identifies articles where the analysis fields contain JSON strings
instead of plain text, and extracts the text content from those JSON objects.
"""

import json
import sqlite3
from pathlib import Path


def extract_text_from_json_field(json_str):
    """Extract text content from a JSON-formatted field."""
    if not json_str or json_str == "Analysis parsing failed":
        return json_str

    try:
        # Check if it's already plain text (not JSON)
        if not json_str.strip().startswith("{") and not json_str.strip().startswith(
            "["
        ):
            return json_str

        # Parse the JSON
        data = json.loads(json_str)

        # If it's a dict with nested structure, extract the text
        if isinstance(data, dict):
            # Common patterns in the nested JSON
            if "explanation" in data:
                parts = []
                if "explanation" in data:
                    parts.append(data["explanation"])
                if "analogy" in data:
                    parts.append(f"\nAnalogy: {data['analogy']}")
                if "why_it_matters" in data:
                    parts.append(f"\nWhy it matters: {data['why_it_matters']}")
                if "innovation" in data:
                    parts.append(f"\nInnovation: {data['innovation']}")
                if "implementation_details" in data:
                    parts.append(f"\nImplementation: {data['implementation_details']}")
                if "steps" in data:
                    parts.append(f"\nSteps: {json.dumps(data['steps'], indent=2)}")
                if "innovations" in data:
                    parts.append(
                        f"\nInnovations: {json.dumps(data['innovations'], indent=2)}"
                    )
                if "why_it_works" in data:
                    parts.append(f"\nWhy it works: {data['why_it_works']}")
                return "\n".join(parts)

            # If it's some other dict structure, try to format it nicely
            return json.dumps(data, indent=2)

        # If it's not a dict, return as is
        return str(data)

    except (json.JSONDecodeError, TypeError):
        # If it's not valid JSON, return as is
        return json_str


def fix_json_analyses():
    """Fix JSON-formatted analyses in the database."""

    # Database path
    db_path = Path(__file__).parent.parent / "data" / "articles.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get all content records
        cursor.execute("""
            SELECT
                article_id,
                methodology_detailed,
                technical_approach,
                key_findings,
                research_design
            FROM content
            ORDER BY article_id
        """)

        records = cursor.fetchall()
        print(f"Checking {len(records)} content records...")

        fixed_count = 0

        for record in records:
            article_id = record[0]
            fields = {
                "methodology_detailed": record[1],
                "technical_approach": record[2],
                "key_findings": record[3],
                "research_design": record[4],
            }

            # Check if any field needs fixing
            needs_fix = False
            fixed_fields = {}

            for field_name, field_value in fields.items():
                if field_value and field_value.strip().startswith("{"):
                    # This field might be JSON
                    fixed_value = extract_text_from_json_field(field_value)
                    if fixed_value != field_value:
                        needs_fix = True
                        fixed_fields[field_name] = fixed_value
                    else:
                        fixed_fields[field_name] = field_value
                else:
                    fixed_fields[field_name] = field_value

            if needs_fix:
                print(f"\nFixing article {article_id}...")

                # Update the record
                cursor.execute(
                    """
                    UPDATE content
                    SET
                        methodology_detailed = ?,
                        technical_approach = ?,
                        key_findings = ?,
                        research_design = ?
                    WHERE article_id = ?
                """,
                    (
                        fixed_fields["methodology_detailed"],
                        fixed_fields["technical_approach"],
                        fixed_fields["key_findings"],
                        fixed_fields["research_design"],
                        article_id,
                    ),
                )

                fixed_count += 1
                print(f"  ✅ Fixed JSON fields for article {article_id}")

        # Commit the changes
        if fixed_count > 0:
            conn.commit()
            print(
                f"\n✅ Successfully fixed {fixed_count} articles with JSON-formatted analyses"
            )
        else:
            print(
                "\n✅ No JSON-formatted analyses found - all content is already in plain text format"
            )

        # Show sample of fixed content
        if fixed_count > 0:
            cursor.execute("""
                SELECT
                    a.id,
                    a.title,
                    LENGTH(c.methodology_detailed) as meth_len
                FROM articles a
                JOIN content c ON a.id = c.article_id
                ORDER BY a.id DESC
                LIMIT 5
            """)

            print("\nSample of content after fixing:")
            for row in cursor.fetchall():
                print(
                    f"  Article {row[0]}: {row[1][:50]}... (methodology: {row[2]} chars)"
                )

    except Exception as e:
        print(f"Error during fix: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    print("Starting JSON analyses fix...")
    print("=" * 50)
    fix_json_analyses()
    print("=" * 50)
    print("Fix complete!")
