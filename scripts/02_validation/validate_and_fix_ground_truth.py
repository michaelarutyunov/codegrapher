#!/usr/bin/env python3
"""Validate and fix ground_truth.jsonl commit references.

This script:
1. Checks that commit SHAs are valid (40 characters or resolvable)
2. Resolves short SHAs to full SHAs
3. Identifies branch names that need manual resolution
4. Generates a fixed ground_truth.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def run_git_command(args: List[str], cwd: Path) -> Tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"


def resolve_commit_sha(repo_url: str, ref: str, temp_dir: Path) -> Tuple[str, str]:
    """Resolve a git ref to a full 40-character SHA.

    Returns:
        Tuple of (sha, status) where status is one of:
        - "full": Already a full SHA
        - "resolved": Short SHA resolved to full
        - "branch": Branch name (could change)
        - "error": Could not resolve
    """
    repo_name = repo_url.split("/")[-1]
    repo_path = temp_dir / repo_name

    # Check if already a 40-char SHA
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        return ref, "full"

    # Clone if not exists
    if not repo_path.exists():
        success, output = run_git_command(
            ["clone", "--filter=blob:none", "--quiet", repo_url, str(repo_path)],
            cwd=temp_dir
        )
        if not success:
            return ref, "error"

    # Try to resolve to full SHA
    success, stdout = run_git_command(
        ["rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=repo_path
    )

    if success:
        full_sha = stdout
        if len(full_sha) == 40:
            # Check if it was a branch name
            success2, _ = run_git_command(
                ["show-ref", "--verify", "--quiet", f"refs/heads/{ref}"],
                cwd=repo_path
            )
            if success2:
                return full_sha, "branch"
            return full_sha, "resolved"
        return full_sha, "unknown"
    else:
        return ref, "error"


def load_ground_truth(path: Path) -> List[Dict]:
    """Load ground truth JSONL."""
    tasks = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def save_ground_truth(tasks: List[Dict], path: Path):
    """Save ground truth JSONL."""
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate and fix ground_truth.jsonl")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth.jsonl",
        help="Path to ground truth JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for fixed ground truth (default: overwrite input)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report issues, don't fix them"
    )

    args = parser.parse_args()

    # Load ground truth
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    tasks = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(tasks)} tasks from {args.ground_truth}")

    # Create temp dir for repos
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="validate_ground_truth_"))
    print(f"Using temp directory: {temp_dir}")

    # Analyze each task
    issues_by_task = {}
    fixed_tasks = []

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        repo_url = task["repo"]
        commit_before = task["commit_before"]
        commit_after = task["commit_after"]

        print(f"\n[{i}/{len(tasks)}] {task_id}: {task['description'][:60]}...")

        # Resolve commit_before
        sha_before, status_before = resolve_commit_sha(repo_url, commit_before, temp_dir)

        # Resolve commit_after
        sha_after, status_after = resolve_commit_sha(repo_url, commit_after, temp_dir)

        # Track issues
        issues = []
        if status_before in ["branch", "error"]:
            issues.append(f"commit_before: {commit_before} ({status_before})")
        elif status_before == "resolved":
            issues.append(f"commit_before: {commit_before} -> {sha_before}")

        if status_after in ["branch", "error"]:
            issues.append(f"commit_after: {commit_after} ({status_after})")
        elif status_after == "resolved":
            issues.append(f"commit_after: {commit_after} -> {sha_after}")

        if issues:
            issues_by_task[task_id] = issues
            print(f"  ⚠️  Issues found:")
            for issue in issues:
                print(f"     - {issue}")

            # Fix the task if not dry run
            if not args.dry_run:
                task_fixed = task.copy()
                if status_before in ["resolved", "branch"]:
                    task_fixed["commit_before"] = sha_before
                if status_after in ["resolved", "branch"]:
                    task_fixed["commit_after"] = sha_after
                fixed_tasks.append(task_fixed)
            else:
                fixed_tasks.append(task)
        else:
            print(f"  ✅ OK")
            fixed_tasks.append(task)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if issues_by_task:
        print(f"\n⚠️  Found issues in {len(issues_by_task)} tasks:")
        for task_id, issues in sorted(issues_by_task.items()):
            print(f"\n{task_id}:")
            for issue in issues:
                print(f"  - {issue}")

        # Branch names need manual attention
        branch_issues = [
            (tid, iss) for tid, iss_list in issues_by_task.items()
            for iss in iss_list if "branch" in iss
        ]
        if branch_issues:
            print("\n⚠️  WARNING: Branch names detected!")
            print("   These will check out the current tip of the branch,")
            print("   which may change over time. Consider pinning to specific commits.")
    else:
        print("\n✅ All tasks validated successfully!")

    # Save fixed version
    if not args.dry_run and issues_by_task:
        output_path = args.output or args.ground_truth
        save_ground_truth(fixed_tasks, output_path)
        print(f"\n💾 Fixed ground truth saved to: {output_path}")

        # Save backup of original
        backup_path = args.ground_truth.with_suffix(".jsonl.backup")
        save_ground_truth(tasks, backup_path)
        print(f"💾 Original backed up to: {backup_path}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    sys.exit(1 if issues_by_task else 0)


if __name__ == "__main__":
    main()
