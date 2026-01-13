#!/usr/bin/env python3
"""Continue research for remaining tasks."""

import subprocess
import json
from pathlib import Path

REPOS_DIR = Path("/tmp/ground_truth_repos")

def run_git(repo_name, args):
    """Run git command in repo."""
    result = subprocess.run(
        ["git"] + args,
        cwd=REPOS_DIR / repo_name,
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.stdout.strip()

def find_commit_by_subject(repo, subject_keywords):
    """Find commit by searching subject."""
    output = run_git(repo, ["log", "--all", "--pretty=format:%H %s", "-100"])
    
    keywords = [kw.lower() for kw in subject_keywords]
    for line in output.split("\n"):
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        sha, subject = parts
        subject_lower = subject.lower()
        # Check if all keywords appear in subject
        if all(kw in subject_lower for kw in keywords[:3]):
            return sha
    return None

# Task search patterns
searches = {
    "task_004": ("werkzeug", ["multipart", "chunk", "boundary", "r\\n"]),
    "task_006": ("flask", ["follow_redirects", "session", "context"]),
    "task_007": ("jinja", ["unique", "async", "generator"]),
    "task_008": ("werkzeug", ["debugger", "pin", "attempt", "timeout"]),
    "task_009": ("pydantic", ["pickle", "MISSING", "model_construct"]),
    "task_010": ("pydantic", ["TypedDict", "forward", "reference", "parent"]),
    "task_012": ("click", ["fish", "completion", "quoted", "parameter"]),
    "task_013": ("click", ["pager", "subprocess", "argument"]),
    "task_014": ("fastapi", ["refactor", "dataclass", "dependency"]),
    "task_015": ("fastapi", ["refactor", "recursion", "dependency"]),
    "task_016": ("fastapi", ["hashable", "Depends", "Security"]),
    "task_017": ("fastapi", ["refactor", "coroutine", "is_coroutine"]),
    "task_018": ("fastapi", ["python", "3.9", "syntax", "3.8"]),
}

results = {}
for task_id, (repo, keywords) in searches.items():
    print(f"\nSearching {task_id} in {repo} with keywords: {keywords}")
    sha = find_commit_by_subject(repo, keywords)
    if sha:
        # Get parent
        parent = run_git(repo, ["rev-parse", f"{sha}^"])
        results[task_id] = {"commit_after": sha, "commit_before": parent}
        print(f"  Found: {sha[:12]}")
        print(f"  Parent: {parent[:12]}")
    else:
        print(f"  Not found")
        results[task_id] = {"commit_after": None, "commit_before": None}

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
for task_id, data in results.items():
    print(f"{task_id}: {data}")
