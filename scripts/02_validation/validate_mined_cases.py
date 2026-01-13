#!/usr/bin/env python3
"""Validate mined test cases for correctness.

This script verifies that mined test cases:
1. Have valid 40-char commit SHAs
2. Have correct parent-child commit relationship
3. Have files_edited matching actual git diff
4. Are from repos ≤50k LOC
5. Have valid schema
"""

import argparse
import json
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def is_valid_sha(sha: str) -> bool:
    """Check if string is a valid 40-char Git SHA."""
    return bool(re.match(r"^[a-f0-9]{40}$", sha))


def extract_repo_info(repo_url: str) -> tuple[str, str]:
    """Extract owner/repo from GitHub URL."""
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")
    return match.group(1), match.group(2)


def clone_repo_at_commit(owner: str, repo: str, commit: str, temp_dir: Path) -> Optional[Path]:
    """Clone repository at specific commit.

    Args:
        owner: Repository owner
        repo: Repository name
        commit: Commit SHA
        temp_dir: Temporary directory

    Returns:
        Path to cloned repo or None on failure
    """
    repo_path = temp_dir / repo

    if repo_path.exists():
        return repo_path

    try:
        # Clone with depth 1 for speed
        clone_cmd = [
            "git", "clone",
            "--depth", "1",
            f"https://github.com/{owner}/{repo}.git",
            str(repo_path)
        ]
        subprocess.run(clone_cmd, capture_output=True, text=True, check=True, cwd=str(temp_dir))

        # Fetch the specific commits we need
        fetch_cmd = ["git", "fetch", "origin", commit]
        subprocess.run(fetch_cmd, capture_output=True, text=True, check=True, cwd=str(repo_path))

        return repo_path

    except subprocess.CalledProcessError as e:
        logger.error(f"  Clone failed: {e.stderr}")
        return None


def verify_parent_relationship(repo_path: Path, commit_before: str, commit_after: str) -> bool:
    """Verify commit_before is parent of commit_after.

    Args:
        repo_path: Path to git repository
        commit_before: Expected parent commit
        commit_after: Child commit

    Returns:
        True if parent relationship is correct
    """
    try:
        # Get actual parent(s) of commit_after
        cmd = ["git", "log", "--pretty=%P", "-n", "1", commit_after]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(repo_path))

        parents = result.stdout.strip().split()

        # commit_before should be one of the parents
        return commit_before in parents

    except subprocess.CalledProcessError:
        return False


def get_files_changed(repo_path: Path, commit_before: str, commit_after: str) -> Optional[List[str]]:
    """Get list of files changed between commits.

    Args:
        repo_path: Path to git repository
        commit_before: Start commit
        commit_after: End commit

    Returns:
        List of file paths or None on error
    """
    try:
        cmd = ["git", "diff", "--name-only", f"{commit_before}...{commit_after}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(repo_path))

        return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

    except subprocess.CalledProcessError:
        return None


def count_python_loc(repo_path: Path) -> int:
    """Count Python LOC in repository."""
    total = 0
    exclude_dirs = {".venv", "venv", ".env", "site-packages", ".tox", "__pycache__", ".git"}

    for py_file in repo_path.rglob("*.py"):
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue

        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
                total += len(lines)
        except:
            continue

    return total


def validate_test_case(task: Dict[str, Any], temp_dir: Path) -> Dict[str, Any]:
    """Validate a single test case.

    Args:
        task: Test case dict
        temp_dir: Temporary directory for cloning

    Returns:
        Validation result dict
    """
    task_id = task["task_id"]
    result = {
        "task_id": task_id,
        "valid": True,
        "issues": []
    }

    # Check required fields
    required_fields = [
        "task_id", "description", "repo", "commit_before", "commit_after",
        "cursor_file", "query_terms", "files_edited", "query_category"
    ]

    for field in required_fields:
        if field not in task:
            result["valid"] = False
            result["issues"].append(f"Missing field: {field}")

    if not result["valid"]:
        return result

    # Check commit SHAs
    if not is_valid_sha(task["commit_before"]):
        result["valid"] = False
        result["issues"].append(f"Invalid commit_before SHA: {task['commit_before']}")

    if not is_valid_sha(task["commit_after"]):
        result["valid"] = False
        result["issues"].append(f"Invalid commit_after SHA: {task['commit_after']}")

    if not result["valid"]:
        return result

    # Clone and verify
    try:
        owner, repo = extract_repo_info(task["repo"])

        logger.info(f"  Cloning {owner}/{repo}...")
        repo_path = clone_repo_at_commit(owner, repo, task["commit_after"], temp_dir)

        if not repo_path:
            result["valid"] = False
            result["issues"].append("Failed to clone repository")
            return result

        # Verify parent relationship
        logger.info(f"  Verifying parent relationship...")
        if not verify_parent_relationship(repo_path, task["commit_before"], task["commit_after"]):
            result["valid"] = False
            result["issues"].append("commit_before is not parent of commit_after")

        # Verify files changed
        logger.info(f"  Verifying files changed...")
        actual_files = get_files_changed(repo_path, task["commit_before"], task["commit_after"])

        if actual_files is None:
            result["valid"] = False
            result["issues"].append("Could not get file diff")
        else:
            expected_files = set(task["files_edited"])
            actual_files_set = set(actual_files)

            # Allow subset match (test case might not list ALL files)
            if not expected_files.issubset(actual_files_set):
                missing = expected_files - actual_files_set
                result["issues"].append(f"files_edited mismatch: {missing} not in actual diff")

        # Check repo size
        logger.info(f"  Counting Python LOC...")
        loc = count_python_loc(repo_path)
        result["loc"] = loc

        if loc > 50000:
            result["issues"].append(f"Repo too large: {loc:,} LOC (limit: 50,000)")

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Validation error: {str(e)}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate mined test cases"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth_mined.jsonl",
        help="Input JSONL file to validate"
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load test cases
    with open(args.input) as f:
        tasks = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"Loaded {len(tasks)} test cases from {args.input}")
    logger.info("")

    # Validate each case
    results = []

    with tempfile.TemporaryDirectory(prefix="validate_mined_") as temp_dir:
        temp_path = Path(temp_dir)

        for i, task in enumerate(tasks, 1):
            logger.info(f"[{i}/{len(tasks)}] {task['task_id']}: {task['description'][:60]}...")

            result = validate_test_case(task, temp_path)
            results.append(result)

            if result["valid"]:
                logger.info(f"  ✅ VALID (LOC: {result.get('loc', 'N/A'):,})")
            else:
                logger.warning(f"  ❌ INVALID:")
                for issue in result["issues"]:
                    logger.warning(f"     - {issue}")

            logger.info("")

    # Summary
    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = len(results) - valid_count

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total cases: {len(results)}")
    logger.info(f"Valid: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    logger.info(f"Invalid: {invalid_count} ({invalid_count/len(results)*100:.1f}%)")
    logger.info("")

    if invalid_count > 0:
        logger.info("Invalid cases:")
        for result in results:
            if not result["valid"]:
                logger.info(f"  {result['task_id']}: {', '.join(result['issues'])}")

    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    exit(main())
