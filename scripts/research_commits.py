#!/usr/bin/env python3
"""Research correct commit SHAs for ground truth tasks.

This script:
1. Clones repositories if not already present
2. Searches git history for commits matching task descriptions
3. Finds both commit_after and commit_before SHAs
4. Outputs updated ground_truth.jsonl entries
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_git_command(args: List[str], cwd: Path) -> Tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"


def clone_repo(repo_url: str, repos_dir: Path) -> Path:
    """Clone repository if not already present."""
    repo_name = repo_url.split("/")[-1]
    repo_path = repos_dir / repo_name

    if not repo_path.exists():
        print(f"Cloning {repo_url}...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--quiet", repo_url, str(repo_path)],
            check=True,
            timeout=600
        )
        print(f"Cloned to {repo_path}")

    # Fetch all refs including remote branches
    print(f"Fetching refs for {repo_name}...")
    subprocess.run(
        ["git", "fetch", "--quiet", "--tags", "--force", "origin"],
        cwd=repo_path,
        check=True,
        timeout=300
    )

    return repo_path


def search_commits_by_keywords(repo_path: Path, keywords: List[str], max_results: int = 20) -> List[Dict]:
    """Search git log for commits matching keywords."""
    # Build search pattern - search in commit messages
    pattern = "|".join(re.escape(kw) for kw in keywords[:3])  # Use first 3 keywords
    pattern = pattern.replace(" ", "\\s+")

    # Search git log
    success, output = run_git_command([
        "log",
        "--all",
        "--grep=" + keywords[0],  # Use first keyword for grep
        "--oneline",
        f"-{max_results}"
    ], repo_path)

    if not success:
        return []

    commits = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            short_sha, subject = parts
            commits.append({
                "short_sha": short_sha,
                "subject": subject
            })

    return commits


def get_full_sha(repo_path: Path, short_sha: str) -> Optional[str]:
    """Get full 40-character SHA from short SHA."""
    success, output = run_git_command(["rev-parse", short_sha], repo_path)
    if success and len(output) == 40:
        return output
    return None


def get_commit_parent(repo_path: Path, sha: str) -> Optional[str]:
    """Get the parent commit SHA."""
    success, output = run_git_command(["rev-parse", f"{sha}^"], repo_path)
    if success:
        return output
    return None


def search_commit_by_file_change(repo_path: Path, file_path: str, keywords: List[str]) -> Optional[str]:
    """Search for commits that modified a specific file."""
    # Search for commits that modified the file
    success, output = run_git_command([
        "log",
        "--all",
        "--oneline",
        f"-{max_results}",
        "--",
        file_path
    ], repo_path)

    if not success:
        return None

    # Check each commit for keyword matches
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            short_sha, subject = parts
            # Check if any keyword is in the subject
            if any(kw.lower() in subject.lower() for kw in keywords):
                return short_sha

    return None


def find_commits_for_task(task: Dict, repos_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Find commit_before and commit_after for a task.

    Returns:
        Tuple of (commit_before, commit_after) or (None, None) if not found
    """
    repo_url = task["repo"]
    description = task["description"]
    cursor_file = task.get("cursor_file", "")
    query_terms = task.get("query_terms", [])
    files_edited = task.get("files_edited", [])

    print(f"\n{'='*80}")
    print(f"Researching: {task['task_id']}")
    print(f"Description: {description}")
    print(f"Cursor file: {cursor_file}")
    print(f"Files edited: {files_edited}")
    print(f"{'='*80}")

    # Clone/fetch repo
    repo_path = clone_repo(repo_url, repos_dir)

    # Strategy 1: Search by description keywords
    keywords = description.split()[:5]  # First 5 words from description
    print(f"\nSearching commits by keywords: {keywords}")

    commits = search_commits_by_keywords(repo_path, keywords, max_results=30)

    if commits:
        print(f"\nFound {len(commits)} potential commits:")
        for i, commit in enumerate(commits[:10], 1):
            full_sha = get_full_sha(repo_path, commit["short_sha"])
            print(f"  {i}. {commit['short_sha']}: {commit['subject'][:80]}")
            if full_sha:
                print(f"     Full SHA: {full_sha}")

        # Try to find the best match by checking file changes
        if files_edited:
            print(f"\nChecking which commits modified expected files...")
            for commit in commits[:5]:
                full_sha = get_full_sha(repo_path, commit["short_sha"])
                if not full_sha:
                    continue

                # Check if this commit modified any of the expected files
                for file_edited in files_edited:
                    success, _ = run_git_command([
                        "diff", "--quiet", f"{full_sha}^", full_sha, "--", file_edited
                    ], repo_path)

                    if success:  # File was modified in this commit
                        print(f"\n✓ Found matching commit!")
                        print(f"  Commit: {full_sha}")
                        print(f"  Subject: {commit['subject']}")
                        print(f"  Modified file: {file_edited}")

                        # Get parent as commit_before
                        parent = get_commit_parent(repo_path, full_sha)
                        if parent:
                            print(f"  Parent (commit_before): {parent}")

                        return parent, full_sha

    # Strategy 2: Search by file path if cursor_file is available
    if cursor_file and files_edited:
        print(f"\nSearching for commits that modified {cursor_file}...")

        # Look for commits in the file's history
        success, output = run_git_command([
            "log",
            "--all",
            "--oneline",
            "-20",
            "--",
            cursor_file
        ], repo_path)

        if success:
            print(f"\nRecent commits modifying {cursor_file}:")
            for line in output.split("\n")[:10]:
                print(f"  {line}")

    print(f"\n⚠ Could not find definitive commit match")
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Research correct commit SHAs")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth.jsonl",
        help="Path to ground truth JSONL file"
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("/tmp/ground_truth_repos"),
        help="Directory to clone repositories to"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for fixed ground truth"
    )

    args = parser.parse_args()

    # Create repos directory
    args.repos_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    with open(args.ground_truth, "r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} tasks from {args.ground_truth}")

    # Research each task
    results = []
    for task in tasks:
        commit_before, commit_after = find_commits_for_task(task, args.repos_dir)

        if commit_before or commit_after:
            # Update task with found commits
            updated_task = task.copy()
            if commit_after:
                updated_task["commit_after"] = commit_after
            if commit_before:
                updated_task["commit_before"] = commit_before
            results.append(updated_task)

            print(f"\n✅ Updated {task['task_id']}")
            print(f"   commit_before: {commit_before or 'NOT FOUND'}")
            print(f"   commit_after: {commit_after or 'NOT FOUND'}")
        else:
            results.append(task)
            print(f"\n❌ Could not find commits for {task['task_id']}")

    # Save results
    output_path = args.output or args.ground_truth.with_suffix(".fixed.jsonl")
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
