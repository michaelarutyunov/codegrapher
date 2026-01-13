#!/usr/bin/env python3
"""Separate test cases by repository size.

This script:
1. Counts Python LOC for each unique repository
2. Moves cases from repos >50k LOC to ground_truth_large.jsonl
3. Keeps cases from repos ≤50k LOC in ground_truth.jsonl

The 50k LOC threshold is based on CodeGrapher's FAISS IndexFlatL2 design constraint.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def extract_repo_info(repo_url: str) -> tuple[str, str, str]:
    """Extract owner/repo from GitHub URL.

    Args:
        repo_url: GitHub repo URL

    Returns:
        Tuple of (owner, repo, full_name)
    """
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")

    owner, repo = match.group(1), match.group(2)
    return owner, repo, f"{owner}/{repo}"


def clone_repo(owner: str, repo: str, temp_dir: Path) -> Path:
    """Clone a repository to temp directory.

    Args:
        owner: Repository owner
        repo: Repository name
        temp_dir: Temporary directory path

    Returns:
        Path to cloned repo
    """
    repo_path = temp_dir / repo

    if repo_path.exists():
        logger.info(f"  Using existing clone: {repo_path}")
        return repo_path

    logger.info(f"  Cloning {owner}/{repo}...")

    try:
        # Shallow clone to save time/space
        cmd = [
            "git", "clone",
            "--depth", "1",
            f"https://github.com/{owner}/{repo}.git",
            str(repo_path)
        ]

        subprocess.run(
            cmd,
            cwd=str(temp_dir),
            check=True,
            capture_output=True,
            text=True
        )

        logger.info(f"  ✅ Cloned to {repo_path}")
        return repo_path

    except subprocess.CalledProcessError as e:
        logger.error(f"  ❌ Clone failed: {e.stderr}")
        raise


def count_python_loc(repo_path: Path) -> int:
    """Count Python LOC (non-comment, non-blank lines).

    Args:
        repo_path: Path to repository

    Returns:
        Total Python LOC count
    """
    total = 0
    files_counted = 0

    # Directories to exclude
    exclude_dirs = {
        ".venv", "venv", ".env", "env",
        "site-packages", ".tox", ".pytest_cache",
        "__pycache__", ".git", ".github",
        "node_modules", "build", "dist", ".egg-info"
    }

    for py_file in repo_path.rglob("*.py"):
        # Skip if in excluded directory
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue

        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Count non-blank, non-comment lines
            loc = 0
            in_multiline_string = False

            for line in lines:
                stripped = line.strip()

                # Skip blank lines
                if not stripped:
                    continue

                # Handle multiline strings (docstrings)
                if '"""' in stripped or "'''" in stripped:
                    if in_multiline_string:
                        in_multiline_string = False
                    else:
                        in_multiline_string = True
                    continue

                if in_multiline_string:
                    continue

                # Skip comment lines
                if stripped.startswith("#"):
                    continue

                loc += 1

            total += loc
            files_counted += 1

        except Exception as e:
            logger.debug(f"  Error reading {py_file}: {e}")
            continue

    logger.info(f"  Counted {files_counted} Python files, {total:,} LOC")
    return total


def separate_by_size(input_path: Path, threshold: int = 50000) -> None:
    """Separate tasks by repository size.

    Args:
        input_path: Input JSONL file
        threshold: LOC threshold (default 50000)
    """
    with open(input_path, "r") as f:
        tasks = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"Loaded {len(tasks)} tasks from {input_path}")
    logger.info(f"Threshold: {threshold:,} LOC")
    logger.info("")

    # Group tasks by repository
    tasks_by_repo: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        repo_url = task["repo"]
        if repo_url not in tasks_by_repo:
            tasks_by_repo[repo_url] = []
        tasks_by_repo[repo_url].append(task)

    logger.info(f"Found {len(tasks_by_repo)} unique repositories")
    logger.info("")

    # Count LOC for each repo
    repo_sizes: Dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="separate_repos_") as temp_dir:
        temp_path = Path(temp_dir)

        for i, repo_url in enumerate(sorted(tasks_by_repo.keys()), 1):
            logger.info(f"[{i}/{len(tasks_by_repo)}] {repo_url}")

            try:
                owner, repo, _ = extract_repo_info(repo_url)
                repo_path = clone_repo(owner, repo, temp_path)
                loc = count_python_loc(repo_path)

                repo_sizes[repo_url] = loc

                status = "✅ KEEP" if loc <= threshold else "⚠️  LARGE"
                logger.info(f"  {status}: {loc:,} LOC")

            except Exception as e:
                logger.error(f"  ❌ Error processing repo: {e}")
                # Assume large if we can't count (conservative)
                repo_sizes[repo_url] = threshold + 1

            logger.info("")

    # Separate tasks
    small_repo_tasks = []
    large_repo_tasks = []

    for task in tasks:
        repo_url = task["repo"]
        loc = repo_sizes.get(repo_url, 0)

        if loc <= threshold:
            small_repo_tasks.append(task)
        else:
            large_repo_tasks.append(task)

    # Write outputs
    output_dir = input_path.parent

    # Main dataset (≤50k LOC)
    main_output = output_dir / "ground_truth.jsonl"
    with open(main_output, "w") as f:
        for task in small_repo_tasks:
            f.write(json.dumps(task) + "\n")

    logger.info(f"✅ Small repos ({len(small_repo_tasks)} tasks) -> {main_output}")

    # Large dataset (>50k LOC)
    if large_repo_tasks:
        large_output = output_dir / "ground_truth_large.jsonl"
        with open(large_output, "w") as f:
            for task in large_repo_tasks:
                f.write(json.dumps(task) + "\n")

        logger.info(f"✅ Large repos ({len(large_repo_tasks)} tasks) -> {large_output}")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Small repos (≤{threshold:,} LOC): {len(small_repo_tasks)} tasks")
    logger.info(f"Large repos (>{threshold:,} LOC): {len(large_repo_tasks)} tasks")
    logger.info("")

    logger.info("Repository sizes:")
    for repo_url in sorted(repo_sizes.keys(), key=lambda r: repo_sizes[r], reverse=True):
        loc = repo_sizes[repo_url]
        count = len(tasks_by_repo[repo_url])
        status = "LARGE" if loc > threshold else "OK"
        repo_name = repo_url.split("/")[-1]
        logger.info(f"  {repo_name:15s}: {loc:6,} LOC ({count} tasks) [{status}]")


def main():
    parser = argparse.ArgumentParser(
        description="Separate test cases by repository size"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50000,
        help="LOC threshold (default: 50000)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    separate_by_size(args.input, args.threshold)
    return 0


if __name__ == "__main__":
    exit(main())
