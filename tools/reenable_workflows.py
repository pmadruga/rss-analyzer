#!/usr/bin/env python3
"""Re-enable disabled workflows."""

import argparse
from pathlib import Path


DISABLED_WORKFLOWS = {
    'rss-complete-pipeline': 'Main RSS analysis pipeline (Mistral API)',
    'force-refresh-now': 'Force immediate RSS refresh',
    'refresh-titles': 'Refresh article titles',
    'test-pipeline': 'Test pipeline components on PRs',
}


def list_disabled():
    """List all disabled workflows."""
    workflows_dir = Path('.github/workflows')
    disabled = list(workflows_dir.glob('*.disabled'))

    if not disabled:
        print("✅ No workflows are currently disabled")
        return

    print("📋 Disabled Workflows:\n")
    for workflow in disabled:
        name = workflow.stem.replace('.yml', '')
        desc = DISABLED_WORKFLOWS.get(name, 'Unknown workflow')
        print(f"  - {name}")
        print(f"    {desc}")
        print(f"    File: {workflow.name}\n")


def reenable_workflow(workflow_name: str):
    """Re-enable a specific workflow."""
    workflows_dir = Path('.github/workflows')
    disabled_file = workflows_dir / f"{workflow_name}.yml.disabled"
    enabled_file = workflows_dir / f"{workflow_name}.yml"

    if not disabled_file.exists():
        print(f"❌ Workflow '{workflow_name}' is not disabled or doesn't exist")
        print(f"   Looking for: {disabled_file}")
        return False

    if enabled_file.exists():
        print(f"⚠️  Warning: {enabled_file.name} already exists!")
        response = input("   Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Cancelled.")
            return False

    disabled_file.rename(enabled_file)
    print(f"✅ Re-enabled workflow: {workflow_name}")
    print(f"   File: {enabled_file.name}")

    desc = DISABLED_WORKFLOWS.get(workflow_name, 'Unknown workflow')
    print(f"   Purpose: {desc}")

    return True


def reenable_all():
    """Re-enable all disabled workflows."""
    workflows_dir = Path('.github/workflows')
    disabled = list(workflows_dir.glob('*.disabled'))

    if not disabled:
        print("✅ No workflows to re-enable")
        return

    print(f"📋 Found {len(disabled)} disabled workflows\n")

    for workflow in disabled:
        name = workflow.stem.replace('.yml', '')
        enabled_file = workflows_dir / f"{name}.yml"

        if enabled_file.exists():
            print(f"⚠️  Skipping {name} (already exists as .yml)")
            continue

        workflow.rename(enabled_file)
        desc = DISABLED_WORKFLOWS.get(name, 'Unknown workflow')
        print(f"✅ Re-enabled: {name}")
        print(f"   {desc}\n")

    print("\n🎉 All workflows re-enabled!")
    print("\n⚠️  Remember to:")
    print("   1. git add .github/workflows/")
    print("   2. git commit -m 'Re-enable workflows'")
    print("   3. git push")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Re-enable disabled GitHub Actions workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List disabled workflows
  python tools/reenable_workflows.py --list

  # Re-enable specific workflow
  python tools/reenable_workflows.py --workflow rss-complete-pipeline

  # Re-enable all workflows
  python tools/reenable_workflows.py --all

Available workflows:
  - rss-complete-pipeline: Main RSS analysis pipeline (Mistral API)
  - force-refresh-now: Force immediate RSS refresh
  - refresh-titles: Refresh article titles
  - test-pipeline: Test pipeline components on PRs
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all disabled workflows'
    )
    group.add_argument(
        '--workflow', '-w',
        type=str,
        help='Re-enable a specific workflow (e.g., rss-complete-pipeline)'
    )
    group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Re-enable all disabled workflows'
    )

    args = parser.parse_args()

    if args.list:
        list_disabled()
    elif args.workflow:
        success = reenable_workflow(args.workflow)
        if success:
            print("\n⚠️  Remember to:")
            print("   1. git add .github/workflows/")
            print(f"   2. git commit -m 'Re-enable {args.workflow} workflow'")
            print("   3. git push")
    elif args.all:
        reenable_all()


if __name__ == '__main__':
    main()
