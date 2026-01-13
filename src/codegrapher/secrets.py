r"""Secret detection pipeline for preventing sensitive data from being indexed.

This module implements PRD Section 8: Secret & Safety Pipeline. It uses
detect-secrets as a CLI wrapper to scan files for:
- AWS keys: AKIA[0-9A-Z]{16}
- Private keys: -----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----
- Generic secrets: (secret|password|token)[\s]*=[\s]*['"][^'"]{8,}['"]
- API keys: api[_-]?key[\s]*=[\s]*['"][^'"]{16,}['"]
- Base64 high-entropy strings

Files with detected secrets are excluded from indexing to prevent
sensitive data from being exposed in search results.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Set


# Module logger
logger = logging.getLogger(__name__)


# Excluded files tracking
EXCLUDED_FILES_NAME = ".codegraph/excluded_files.txt"
BASELINE_NAME = ".secrets.baseline"


class SecretFoundError(Exception):
    """Raised when a secret is detected in a file.

    This exception is used to signal that indexing should skip
    the file while allowing the pipeline to continue processing
    other files.
    """

    def __init__(self, file_path: Path, secret_count: int):
        """Initialize the exception.

        Args:
            file_path: Path to the file containing secrets
            secret_count: Number of secrets detected
        """
        self.file_path = file_path
        self.secret_count = secret_count
        super().__init__(
            f"Detected {secret_count} secret(s) in {file_path}"
        )


def scan_file(
    file_path: Path,
    repo_root: Path,
    baseline_path: Optional[Path] = None
) -> bool:
    """Scan a file for secrets using detect-secrets CLI.

    Args:
        file_path: Path to the file to scan
        repo_root: Repository root directory
        baseline_path: Optional path to .secrets.baseline file.
                      If None, looks for .secrets.baseline at repo_root.

    Returns:
        True if secrets were detected (file should be skipped),
        False if file is clean

    Raises:
        FileNotFoundError: If file_path doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine baseline path
    if baseline_path is None:
        baseline_path = repo_root / BASELINE_NAME

    # Build detect-secrets command
    cmd = ["detect-secrets", "scan", str(file_path)]

    # Add baseline if it exists
    if baseline_path.exists():
        cmd.extend(["--baseline", str(baseline_path)])

    try:
        # Run detect-secrets
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,  # 10 second timeout per file
        )

        # Parse JSON output
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)

                # Check if any secrets were found
                results = output.get("results", {})

                # results is a dict keyed by filename
                # Filter to only secrets from our file
                file_results = results.get(str(file_path), [])

                if file_results:
                    # Secrets detected - log and add to excluded files
                    secret_count = len(file_results)
                    logger.warning(
                        f"Skipped indexing {file_path.relative_to(repo_root)}: "
                        f"{secret_count} secret(s) detected"
                    )
                    _add_to_excluded_files(file_path, repo_root)
                    return True

            except json.JSONDecodeError:
                # Failed to parse output - treat as clean
                logger.warning(
                    f"Failed to parse detect-secrets output for {file_path}"
                )

        return False

    except subprocess.TimeoutExpired:
        logger.warning(f"Secret scan timed out for {file_path}")
        return False
    except FileNotFoundError:
        # detect-secrets not installed - log warning but don't fail
        logger.warning("detect-secrets not found - skipping secret detection")
        return False
    except Exception as e:
        # Other errors - log warning and treat as clean
        logger.warning(f"Failed to scan {file_path} for secrets: {e}")
        return False


def _add_to_excluded_files(file_path: Path, repo_root: Path) -> None:
    """Add a file to the excluded files list.

    Args:
        file_path: Path to the file with secrets
        repo_root: Repository root directory
    """
    excluded_path = repo_root / EXCLUDED_FILES_NAME

    # Ensure directory exists
    excluded_path.parent.mkdir(parents=True, exist_ok=True)

    # Get relative path
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        # file_path is not under repo_root
        rel_path = str(file_path)

    # Read existing excluded files
    excluded: Set[str] = set()
    if excluded_path.exists():
        try:
            with open(excluded_path, 'r') as f:
                excluded = set(line.strip() for line in f if line.strip())
        except IOError as e:
            logger.warning(f"Failed to read excluded files: {e}")

    # Add new file if not already present
    if rel_path not in excluded:
        excluded.add(rel_path)
        try:
            with open(excluded_path, 'w') as f:
                for path in sorted(excluded):
                    f.write(f"{path}\n")
        except IOError as e:
            logger.warning(f"Failed to write excluded files: {e}")


def is_excluded(file_path: Path, repo_root: Path) -> bool:
    """Check if a file is in the excluded files list.

    Args:
        file_path: Path to check
        repo_root: Repository root directory

    Returns:
        True if file is excluded, False otherwise
    """
    excluded_path = repo_root / EXCLUDED_FILES_NAME

    if not excluded_path.exists():
        return False

    # Get relative path
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        # file_path is not under repo_root
        rel_path = str(file_path)

    # Check if in excluded list
    try:
        with open(excluded_path, 'r') as f:
            excluded_files = set(line.strip() for line in f if line.strip())
            return rel_path in excluded_files
    except IOError as e:
        logger.warning(f"Failed to read excluded files: {e}")
        return False


def get_excluded_files(repo_root: Path) -> List[str]:
    """Get the list of excluded files.

    Args:
        repo_root: Repository root directory

    Returns:
        List of excluded file paths (relative to repo_root)
    """
    excluded_path = repo_root / EXCLUDED_FILES_NAME

    if not excluded_path.exists():
        return []

    try:
        with open(excluded_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except IOError as e:
        logger.warning(f"Failed to read excluded files: {e}")
        return []


def clear_excluded_file(repo_root: Path, file_path: Path) -> bool:
    """Remove a file from the excluded files list.

    Useful when a secret has been removed from a file and it should
    be re-indexed.

    Args:
        repo_root: Repository root directory
        file_path: Path to remove from exclusion list

    Returns:
        True if file was removed, False if it wasn't in the list
    """
    excluded_path = repo_root / EXCLUDED_FILES_NAME

    if not excluded_path.exists():
        return False

    # Get relative path
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        # file_path is not under repo_root
        rel_path = str(file_path)

    # Read and filter
    try:
        with open(excluded_path, 'r') as f:
            excluded_files = [line.strip() for line in f if line.strip()]

        # Remove the file if present
        if rel_path in excluded_files:
            excluded_files.remove(rel_path)

            # Write back
            with open(excluded_path, 'w') as f:
                for path in sorted(excluded_files):
                    f.write(f"{path}\n")

            return True

        return False

    except IOError as e:
        logger.warning(f"Failed to update excluded files: {e}")
        return False
