#!/usr/bin/env python3
"""Fix GitHub Actions workflows to add repository context to gh commands."""

import re
from pathlib import Path


def fix_workflow_file(filepath: Path) -> bool:
    """Fix gh commands in a workflow file."""
    content = filepath.read_text()
    original_content = content

    # Pattern to find blocks with PR_NUM but without REPO
    # Add REPO variable after PR_NUM assignments
    pattern = r'(PR_NUM=\$\{\{ github\.event\.pull_request\.number[^\}]*\}\})\n(\s+)'
    replacement = r'\1\n\2REPO="${{ github.repository }}"\n\2'
    content = re.sub(pattern, replacement, content)

    # Also handle workflow_dispatch input patterns
    pattern2 = r'(PR_NUM=\$\{\{ github\.event\.pull_request\.number \|\| inputs\.[^\}]+\}\})\n(\s+)'
    replacement2 = r'\1\n\2REPO="${{ github.repository }}"\n\2'
    content = re.sub(pattern2, replacement2, content)

    # Fix gh pr commands without --repo flag
    gh_commands = [
        'gh pr view $PR_NUM',
        'gh pr edit $PR_NUM',
        'gh pr comment $PR_NUM',
        'gh pr review $PR_NUM',
        'gh issue view $ISSUE_NUM',
    ]

    for cmd in gh_commands:
        # Only add --repo if not already present
        if cmd in content and '--repo' not in content[content.index(cmd):content.index(cmd)+200]:
            repo_cmd = cmd.replace('$PR_NUM', '$PR_NUM --repo $REPO').replace('$ISSUE_NUM', '$ISSUE_NUM --repo $REPO')
            content = content.replace(cmd, repo_cmd)

    if content != original_content:
        filepath.write_text(content)
        print(f"✅ Fixed {filepath.name}")
        return True
    else:
        print(f"ℹ️  No changes needed for {filepath.name}")
        return False


def main():
    """Fix all workflow files."""
    workflows_dir = Path('.github/workflows')

    if not workflows_dir.exists():
        print("❌ .github/workflows directory not found")
        return

    files_fixed = 0
    for workflow_file in workflows_dir.glob('*.yml'):
        if workflow_file.name.startswith('_'):
            continue  # Skip reusable workflows

        if fix_workflow_file(workflow_file):
            files_fixed += 1

    print(f"\n🎉 Fixed {files_fixed} workflow files")


if __name__ == '__main__':
    main()
