#!/usr/bin/env python3
"""Resolve branch names to commit SHAs using GitHub CLI.

This script:
1. Reads ground_truth_fixed.jsonl
2. For each task with branch names in commits
3. Uses `gh` CLI to find the PR and get merge/parent commits
4. Updates the JSONL with resolved commits
"""

import argparse
import json
import subprocess
import re
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def extract_repo_info(repo_url: str) -> tuple[str, str]:
    """Extract owner/repo from GitHub URL.

    Args:
        repo_url: GitHub repo URL like https://github.com/owner/repo

    Returns:
        Tuple of (owner, repo)
    """
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")
    return match.group(1), match.group(2)


def find_pr_by_branch(owner: str, repo: str, branch: str) -> Optional[Dict[str, Any]]:
    """Find PR by branch name using gh CLI.

    Args:
        owner: Repository owner
        repo: Repository name
        branch: Branch name (e.g., "emmanuelthome/fix-split-rn")

    Returns:
        PR info dict with mergeCommit, baseRefOid, etc., or None if not found
    """
    try:
        # Try searching for PR with this branch name
        cmd = [
            "gh", "pr", "list",
            "--repo", f"{owner}/{repo}",
            "--state", "all",  # Include closed PRs
            "--search", f"head:{branch}",
            "--json", "number,mergeCommit,baseRefOid,state,mergedAt",
            "--limit", "5"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        prs = json.loads(result.stdout)

        if not prs:
            logger.warning(f"  No PR found for branch: {branch}")
            return None

        # Prefer merged PRs
        merged_prs = [pr for pr in prs if pr.get("mergedAt")]
        if merged_prs:
            pr = merged_prs[0]
        else:
            pr = prs[0]

        logger.info(f"  Found PR #{pr['number']} for branch {branch}")
        return pr

    except subprocess.CalledProcessError as e:
        logger.error(f"  Error finding PR for {branch}: {e.stderr}")
        return None


def find_pr_by_search_term(owner: str, repo: str, search_term: str) -> Optional[Dict[str, Any]]:
    """Find PR by searching for term in PR title/body.

    Args:
        owner: Repository owner
        repo: Repository name
        search_term: Search term (e.g., "issue-5774", "merged-1782")

    Returns:
        PR info dict or None
    """
    try:
        # Extract number from search term if present
        number_match = re.search(r"(\d+)", search_term)
        if number_match:
            number = number_match.group(1)

            # Try as issue number first
            try:
                cmd = [
                    "gh", "issue", "view", number,
                    "--repo", f"{owner}/{repo}",
                    "--json", "number"
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)

                # Find PRs linked to this issue
                cmd = [
                    "gh", "pr", "list",
                    "--repo", f"{owner}/{repo}",
                    "--search", f"closes:#{number} OR fixes:#{number}",
                    "--state", "closed",
                    "--json", "number,mergeCommit,baseRefOid,state,mergedAt",
                    "--limit", "5"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                prs = json.loads(result.stdout)

                if prs:
                    merged_prs = [pr for pr in prs if pr.get("mergedAt")]
                    pr = merged_prs[0] if merged_prs else prs[0]
                    logger.info(f"  Found PR #{pr['number']} via issue #{number}")
                    return pr

            except subprocess.CalledProcessError:
                pass  # Not an issue, try as PR

            # Try as PR number
            try:
                cmd = [
                    "gh", "pr", "view", number,
                    "--repo", f"{owner}/{repo}",
                    "--json", "number,mergeCommit,baseRefOid,state,mergedAt"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                pr = json.loads(result.stdout)
                logger.info(f"  Found PR #{number}")
                return pr
            except subprocess.CalledProcessError:
                pass

        logger.warning(f"  Could not resolve: {search_term}")
        return None

    except Exception as e:
        logger.error(f"  Error searching for {search_term}: {e}")
        return None


def get_parent_commit(owner: str, repo: str, commit_sha: str) -> Optional[str]:
    """Get parent commit SHA using git via gh CLI.

    Args:
        owner: Repository owner
        repo: Repository name
        commit_sha: Commit SHA to find parent of

    Returns:
        Parent commit SHA or None
    """
    try:
        # Use gh api to get commit info
        cmd = [
            "gh", "api",
            f"/repos/{owner}/{repo}/commits/{commit_sha}",
            "--jq", ".parents[0].sha"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parent_sha = result.stdout.strip()

        if parent_sha and len(parent_sha) == 40:
            logger.info(f"  Parent of {commit_sha[:7]}: {parent_sha[:7]}")
            return parent_sha
        return None

    except subprocess.CalledProcessError as e:
        logger.error(f"  Error getting parent of {commit_sha}: {e.stderr}")
        return None


def is_valid_sha(value: str) -> bool:
    """Check if value is a valid 40-char SHA."""
    return bool(re.match(r"^[a-f0-9]{40}$", value))


def resolve_commit_ref(owner: str, repo: str, ref: str, is_before: bool = False) -> str:
    """Resolve a commit reference (branch name, short SHA, etc.) to full SHA.

    Args:
        owner: Repository owner
        repo: Repository name
        ref: Reference to resolve
        is_before: If True, this is commit_before (may need parent resolution)

    Returns:
        Resolved 40-char SHA or original ref if resolution fails
    """
    # Already a valid SHA
    if is_valid_sha(ref):
        return ref

    logger.info(f"  Resolving: {ref}")

    # Try as branch name (PR head branch)
    if "/" in ref or "-" in ref:
        pr = find_pr_by_branch(owner, repo, ref)
        if not pr:
            pr = find_pr_by_search_term(owner, repo, ref)

        if pr:
            merge_commit = pr.get("mergeCommit", {}).get("oid")
            base_ref = pr.get("baseRefOid")

            if is_before and base_ref:
                logger.info(f"  Using base ref (commit_before): {base_ref[:7]}")
                return base_ref
            elif merge_commit:
                logger.info(f"  Using merge commit (commit_after): {merge_commit[:7]}")
                return merge_commit

    # Try to expand short SHA using gh api
    if re.match(r"^[a-f0-9]{7,}$", ref):
        try:
            cmd = [
                "gh", "api",
                f"/repos/{owner}/{repo}/commits/{ref}",
                "--jq", ".sha"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            full_sha = result.stdout.strip()
            if is_valid_sha(full_sha):
                logger.info(f"  Expanded {ref} -> {full_sha[:7]}")
                return full_sha
        except subprocess.CalledProcessError:
            pass

    logger.warning(f"  ⚠️  Could not resolve: {ref}")
    return ref


def process_ground_truth(input_path: Path, output_path: Path) -> None:
    """Process ground truth file and resolve branch names.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file
    """
    with open(input_path, "r") as f:
        tasks = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"Loaded {len(tasks)} tasks from {input_path}")
    logger.info("")

    resolved_tasks = []

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        logger.info(f"[{i}/{len(tasks)}] {task_id}: {task['description'][:60]}...")

        try:
            owner, repo = extract_repo_info(task["repo"])

            # Resolve commit_before
            commit_before = task["commit_before"]
            if not is_valid_sha(commit_before):
                resolved_before = resolve_commit_ref(owner, repo, commit_before, is_before=True)
                if resolved_before != commit_before:
                    task["commit_before"] = resolved_before

                # If still not valid and we have commit_after, try to get parent
                if not is_valid_sha(task["commit_before"]) and is_valid_sha(task.get("commit_after", "")):
                    parent = get_parent_commit(owner, repo, task["commit_after"])
                    if parent:
                        task["commit_before"] = parent

            # Resolve commit_after
            commit_after = task.get("commit_after", "")
            if commit_after and not is_valid_sha(commit_after):
                resolved_after = resolve_commit_ref(owner, repo, commit_after, is_before=False)
                if resolved_after != commit_after:
                    task["commit_after"] = resolved_after

            # Verify both are valid now
            if is_valid_sha(task["commit_before"]) and is_valid_sha(task.get("commit_after", "")):
                logger.info(f"  ✅ Both commits resolved")
            else:
                logger.warning(f"  ⚠️  Still has unresolved refs")

        except Exception as e:
            logger.error(f"  ❌ Error processing task: {e}")

        resolved_tasks.append(task)
        logger.info("")

    # Write output
    with open(output_path, "w") as f:
        for task in resolved_tasks:
            f.write(json.dumps(task) + "\n")

    logger.info(f"✅ Resolved tasks saved to: {output_path}")

    # Summary
    valid_before = sum(1 for t in resolved_tasks if is_valid_sha(t["commit_before"]))
    valid_after = sum(1 for t in resolved_tasks if is_valid_sha(t.get("commit_after", "")))

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {len(resolved_tasks)}")
    logger.info(f"Valid commit_before: {valid_before}/{len(resolved_tasks)}")
    logger.info(f"Valid commit_after: {valid_after}/{len(resolved_tasks)}")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Resolve branch names to commit SHAs")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth_fixed.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth_resolved.jsonl",
        help="Output JSONL file"
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    process_ground_truth(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
