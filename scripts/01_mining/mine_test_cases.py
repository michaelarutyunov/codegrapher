#!/usr/bin/env python3
"""Mine test cases from GitHub repositories using gh CLI.

This script searches for closed issues with merged PRs to create test cases
with realistic query terms extracted from user-facing issue descriptions.

Categories:
- symbol: Query contains class/function names
- description: Natural language describing behavior
- error: Contains error keywords/exceptions
- agent-error: Full traceback format
- dependency: Refactoring/structural queries
"""

import argparse
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# GitHub CLI Wrappers
# ============================================================================


def gh_pr_list(repo: str, state: str = "closed", limit: int = 50, search: str = "") -> List[Dict[str, Any]]:
    """List PRs using gh CLI.

    Args:
        repo: Repository in owner/repo format
        state: PR state (open, closed, merged, all)
        limit: Max PRs to return
        search: Search query

    Returns:
        List of PR dicts
    """
    cmd = [
        "gh", "pr", "list",
        "--repo", repo,
        "--state", state,
        "--limit", str(limit),
        "--json", "number,title,body,mergeCommit,baseRefOid,additions,deletions,files,labels,mergedAt"
    ]

    if search:
        cmd.extend(["--search", search])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error listing PRs: {e.stderr}")
        return []


def gh_issue_view(repo: str, number: int) -> Optional[Dict[str, Any]]:
    """View an issue using gh CLI.

    Args:
        repo: Repository in owner/repo format
        number: Issue number

    Returns:
        Issue dict or None
    """
    cmd = [
        "gh", "issue", "view", str(number),
        "--repo", repo,
        "--json", "number,title,body,labels"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError:
        return None


def extract_issue_number_from_pr(pr: Dict[str, Any]) -> Optional[int]:
    """Extract linked issue number from PR body.

    Args:
        pr: PR dict

    Returns:
        Issue number or None
    """
    body = pr.get("body", "")
    if not body:
        return None

    # Look for "fixes #123", "closes #456", etc.
    patterns = [
        r"(?:fixes|closes|resolves|fix|close|resolve)\s+#(\d+)",
        r"(?:fixes|closes|resolves|fix|close|resolve)\s+https://github\.com/[^/]+/[^/]+/issues/(\d+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


# ============================================================================
# Query Term Extraction
# ============================================================================


def extract_query_terms(text: str, max_terms: int = 7) -> List[str]:
    """Extract technical query terms from text (issue title + body).

    This extracts terms a user/agent would realistically use in a query,
    NOT from the code diff (that's circular logic).

    Args:
        text: Issue title + body
        max_terms: Maximum terms to extract

    Returns:
        List of query terms
    """
    # Remove code blocks (avoid extracting implementation details)
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove URLs
    text = re.sub(r"https?://[^\s]+", "", text)

    # Generic words to exclude
    generic_words = {
        "bug", "fix", "add", "update", "change", "make", "use", "get", "set",
        "support", "allow", "enable", "provide", "ensure", "before", "after",
        "stable", "version", "release", "github", "issue", "pull", "request"
    }

    terms = []

    # Extract error types FIRST (highest priority)
    errors = re.findall(r"\b\w*(?:Error|Exception)\b", text)
    terms.extend([e for e in errors if e.lower() not in generic_words][:2])

    # Extract module.attribute patterns (e.g., pytest.approx, sys.modules)
    dotted = re.findall(r"\b[a-z_][a-z_0-9]*\.[a-z_][a-z_0-9]*\b", text)
    terms.extend([d for d in dotted if d.lower() not in generic_words][:3])

    # Extract snake_case identifiers (e.g., import_module, sys_modules)
    snake_case = re.findall(r"\b[a-z][a-z_0-9]*_[a-z_][a-z_0-9]*\b", text)
    terms.extend([t for t in snake_case if len(t) > 4 and t.lower() not in generic_words][:3])

    # Extract CamelCase identifiers (e.g., UserAuth, StreamContext)
    # More strict: at least 2 capital letters
    camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", text)
    terms.extend([c for c in camel_case if c.lower() not in generic_words][:3])

    # Extract function call patterns (e.g., "validate()", "get_value()")
    function_calls = re.findall(r"\b([a-z_][a-z_0-9]*)\(\)", text)
    terms.extend([f for f in function_calls if len(f) > 4 and f not in generic_words][:2])

    # Extract descriptive technical terms (2+ words)
    # e.g., "async generator", "session state", "form parser"
    descriptive = re.findall(r"\b(async|multipart|session|context|parser|validator|decorator|callback)\s+([a-z]+)\b", text, re.IGNORECASE)
    terms.extend([f"{a} {b}" for a, b in descriptive][:2])

    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        term_lower = term.lower()
        if term_lower not in seen and term_lower not in generic_words and len(term) > 2:
            seen.add(term_lower)
            unique_terms.append(term)

    return unique_terms[:max_terms]


def categorize_query_terms(query_terms: List[str], description: str) -> str:
    """Categorize query based on term types.

    Args:
        query_terms: List of query terms
        description: Task description

    Returns:
        Category: symbol/description/error/agent-error/dependency
    """
    text = " ".join(query_terms + [description]).lower()

    # Check for traceback format
    if "traceback" in text or re.search(r"\.py:\d+", text):
        return "agent-error"

    # Check for dependency keywords
    dependency_kw = ["refactor", "rename", "move", "callers", "depends", "imports"]
    if any(kw in text for kw in dependency_kw):
        return "dependency"

    # Check for error keywords
    error_kw = ["error", "exception", "crash", "fails", "failure"]
    has_error = any(kw in text for kw in error_kw)

    # Count symbol-like terms
    symbol_count = sum(1 for t in query_terms if re.match(r"[A-Z]|_", t))

    # Categorization logic
    if has_error and symbol_count >= 2:
        return "error"
    elif symbol_count >= 3:
        return "symbol"
    elif has_error:
        return "error"
    elif symbol_count >= 1:
        return "symbol"
    else:
        return "description"


# ============================================================================
# PR Filtering
# ============================================================================


def meets_criteria(pr: Dict[str, Any]) -> tuple[bool, str]:
    """Check if PR meets test case criteria.

    Args:
        pr: PR dict from gh CLI

    Returns:
        Tuple of (meets_criteria, reason)
    """
    # Must be merged
    if not pr.get("mergedAt"):
        return False, "not merged"

    # Must have merge commit and base
    if not pr.get("mergeCommit", {}).get("oid"):
        return False, "no merge commit"

    if not pr.get("baseRefOid"):
        return False, "no base ref"

    # Check file changes
    files = pr.get("files", [])
    if not files:
        return False, "no files info"

    # Filter for Python files
    py_files = [f for f in files if f.get("path", "").endswith(".py")]
    if not py_files:
        return False, "no Python files"

    # Prefer 2-5 files changed (sweet spot)
    if len(py_files) < 2:
        return False, f"only {len(py_files)} Python file"

    if len(py_files) > 5:
        return False, f"too many files ({len(py_files)})"

    # Check LOC changed (10-500 is reasonable)
    additions = pr.get("additions", 0)
    deletions = pr.get("deletions", 0)
    total_changes = additions + deletions

    if total_changes < 10:
        return False, f"too small ({total_changes} LOC)"

    if total_changes > 500:
        return False, f"too large ({total_changes} LOC)"

    return True, "OK"


# ============================================================================
# Test Case Generation
# ============================================================================


def generate_test_case(
    repo: str,
    pr: Dict[str, Any],
    issue: Optional[Dict[str, Any]],
    next_task_id: int
) -> Optional[Dict[str, Any]]:
    """Generate a test case from a PR and optional issue.

    Args:
        repo: Repository in owner/repo format
        pr: PR dict
        issue: Issue dict (optional)
        next_task_id: Next available task ID number

    Returns:
        Test case dict or None
    """
    # Get description from issue or PR
    if issue:
        description = issue.get("title", pr.get("title", ""))
        query_source = f"{issue.get('title', '')} {issue.get('body', '')}"
    else:
        description = pr.get("title", "")
        query_source = f"{pr.get('title', '')} {pr.get('body', '')}"

    if not description:
        return None

    # Extract query terms
    query_terms = extract_query_terms(query_source)
    if len(query_terms) < 2:
        return None  # Not enough terms for a meaningful query

    # Categorize
    category = categorize_query_terms(query_terms, description)

    # Get commits
    commit_after = pr.get("mergeCommit", {}).get("oid")
    commit_before = pr.get("baseRefOid")

    if not commit_after or not commit_before:
        return None

    # Get files edited
    files = pr.get("files", [])
    py_files = [f["path"] for f in files if f.get("path", "").endswith(".py")]

    # Filter out test files for cursor_file (prefer source files)
    source_files = [f for f in py_files if "/test" not in f and not f.startswith("test_")]
    cursor_file = source_files[0] if source_files else py_files[0]

    # Build test case
    task_id = f"task_{next_task_id:03d}"

    test_case = {
        "task_id": task_id,
        "description": description[:200],  # Truncate long descriptions
        "repo": f"https://github.com/{repo}",
        "commit_before": commit_before,
        "commit_after": commit_after,
        "cursor_file": cursor_file,
        "query_terms": query_terms,
        "query_category": category,
        "files_edited": py_files,
        "expected_bundle_should_contain": source_files[:3] if source_files else py_files[:3],
        "baseline_tokens_sent": 0,
        "expected_tokens_with_codegraph": 0,
        "notes": f"Mined from PR #{pr['number']}" + (f", issue #{issue['number']}" if issue else "") + f". ~{pr.get('additions', 0) + pr.get('deletions', 0)} LOC changed."
    }

    return test_case


# ============================================================================
# Mining Logic
# ============================================================================


def mine_repo(
    repo: str,
    target_count: int = 5,
    existing_task_ids: Optional[set] = None
) -> List[Dict[str, Any]]:
    """Mine test cases from a repository.

    Args:
        repo: Repository in owner/repo format
        target_count: Target number of cases to mine
        existing_task_ids: Set of existing task IDs to avoid duplicates

    Returns:
        List of test case dicts
    """
    if existing_task_ids is None:
        existing_task_ids = set()

    logger.info(f"Mining {repo} (target: {target_count} cases)...")

    # Get merged PRs (recent first)
    prs = gh_pr_list(repo, state="merged", limit=100)
    logger.info(f"  Found {len(prs)} merged PRs")

    test_cases = []
    checked = 0
    next_id = max([int(tid.split('_')[1]) for tid in existing_task_ids], default=0) + 1

    for pr in prs:
        if len(test_cases) >= target_count:
            break

        checked += 1

        # Check criteria
        meets, reason = meets_criteria(pr)
        if not meets:
            logger.debug(f"  PR #{pr['number']}: Skip ({reason})")
            continue

        # Try to get linked issue
        issue_num = extract_issue_number_from_pr(pr)
        issue = gh_issue_view(repo, issue_num) if issue_num else None

        # Generate test case
        test_case = generate_test_case(repo, pr, issue, next_id)

        if test_case:
            logger.info(
                f"  ✅ PR #{pr['number']}: {test_case['query_category']:12s} | "
                f"{', '.join(test_case['query_terms'][:3])}"
            )
            test_cases.append(test_case)
            next_id += 1
        else:
            logger.debug(f"  PR #{pr['number']}: Could not generate test case")

    logger.info(f"  Checked {checked} PRs, mined {len(test_cases)} cases")
    return test_cases


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Mine test cases from GitHub repositories"
    )
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        help="Repositories to mine (owner/repo format)"
    )
    parser.add_argument(
        "--target-per-repo",
        type=int,
        default=5,
        help="Target cases per repository (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth_mined.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--existing",
        type=Path,
        help="Existing ground truth file (to avoid duplicate task IDs)"
    )

    args = parser.parse_args()

    # Load existing task IDs
    existing_task_ids = set()
    if args.existing and args.existing.exists():
        with open(args.existing) as f:
            for line in f:
                task = json.loads(line.strip())
                existing_task_ids.add(task["task_id"])
        logger.info(f"Loaded {len(existing_task_ids)} existing task IDs")

    # Default repos if none specified
    if not args.repos:
        args.repos = [
            "pallets/click",
            "tiangolo/typer",
            "pallets/jinja",
            "encode/httpx",
            "encode/starlette"
        ]

    logger.info(f"Target repositories: {', '.join(args.repos)}")
    logger.info(f"Target per repo: {args.target_per_repo}")
    logger.info("")

    # Mine each repo
    all_test_cases = []

    for repo in args.repos:
        logger.info("=" * 80)
        test_cases = mine_repo(repo, args.target_per_repo, existing_task_ids)
        all_test_cases.extend(test_cases)

        # Update existing IDs
        for tc in test_cases:
            existing_task_ids.add(tc["task_id"])

        logger.info("")

    # Write output
    with open(args.output, "w") as f:
        for test_case in all_test_cases:
            f.write(json.dumps(test_case) + "\n")

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total mined: {len(all_test_cases)} test cases")
    logger.info(f"Output: {args.output}")
    logger.info("")

    # Category distribution
    categories = {}
    for tc in all_test_cases:
        cat = tc.get("query_category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("Category distribution:")
    for cat in sorted(categories.keys()):
        count = categories[cat]
        pct = (count / len(all_test_cases)) * 100 if all_test_cases else 0
        logger.info(f"  {cat:12s}: {count:2d} ({pct:5.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
