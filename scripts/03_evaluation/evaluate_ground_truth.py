#!/usr/bin/env python3
"""Phase 12 Evaluation Script: Token Savings Measurement

This script evaluates CodeGrapher's token-saving capabilities on real-world tasks.
It measures three key metrics:
- Token Savings: Reduction in tokens sent to LLM (target: ≥30%)
- Recall: Proportion of relevant files retrieved (target: ≥85%)
- Precision: Proportion of retrieved files that are relevant (target: ≤40%)

Usage:
    python scripts/eval_token_save.py --mixed --api-tasks 5  # Default: 5 real API, 15 simulated
    python scripts/eval_token_save.py --simulate              # All simulated (no API cost)
    python scripts/eval_token_save.py --real-api              # All real API calls
    python scripts/eval_token_save.py --resume                # Resume from checkpoint

WSL Memory Optimization Guide:
===============================
If you experience WSL crashes or disconnections during evaluation, it's likely due to
memory pressure. Here are several ways to address this:

1. INCREASE WSL2 MEMORY LIMIT (Recommended):
   Create/Edit %USERPROFILE%\\.wslconfig on Windows:

   [wsl2]
   memory=16GB           # Increase from default 8GB
   swap=4GB              # Add swap space
   swapFile=C:\\\\temp\\\\wsl-swap.vhdx
   pageReporting=false   # May help with stability

   Then restart WSL: wsl --shutdown in PowerShell

2. USE CHECKPOINT/RESUME:
   The script automatically saves progress after each task. If it crashes:
   - Re-run with --resume flag to continue from where it stopped
   - Example: python scripts/eval_token_save.py --resume

3. REDUCE CONCURRENT MEMORY USAGE:
   - Close other applications while running evaluation
   - Avoid running multiple evals in parallel
   - Monitor memory with: htop or free -h

4. RUN IN BATCHES:
   Split the 20 tasks into smaller runs:
   - Edit fixtures/ground_truth.jsonl to subset tasks
   - Or process fewer repos at a time

5. USE SIMULATE MODE FOR TESTING:
   - Start with --simulate to test without API costs
   - Verify stability before using --real-api

MEMORY USAGE PER TASK:
- Clone: ~500MB-2GB depending on repo size
- Index Build: ~1-3GB peak (FAISS index + embedding model)
- Query: ~100-500MB
- Cleanup: Most memory freed after each task

DIAGNOSING WSL CRASHES:
- Check dmesg | grep -i kill for OOM kills
- Check journalctl -xe for WSL errors
- Monitor with: watch -n 1 free -h

Note: Install psutil for memory monitoring (optional):
    pip install psutil
"""

import argparse
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codegrapher.server import codegraph_query

# Access the underlying function from the FastMCP tool wrapper
_codegraph_query_func = codegraph_query.fn

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Token counting will be approximate.", file=sys.stderr)

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic SDK not installed. Real API mode unavailable.", file=sys.stderr)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================================
# Token Counting Utilities
# ============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # psutil not available, return 0
        return 0.0


def log_memory_usage(context: str = ""):
    """Log current memory usage if psutil is available."""
    mem_mb = get_memory_usage_mb()
    if mem_mb > 0:
        logger.debug(f"Memory usage{context}: {mem_mb:.1f} MB")


def robust_rmtree(path: Path, max_retries: int = 3) -> bool:
    """Robustly remove a directory tree, handling WSL .git directory issues.

    WSL's filesystem layer can cause shutil.rmtree() to fail on .git directories
    with "Directory not empty" errors even when the directory appears empty.

    This function implements multiple fallback strategies:
    1. Standard shutil.rmtree with onexc handler
    2. Manual permission fix + retry
    3. subprocess rm -rf (most reliable on WSL)

    Args:
        path: Path to directory to remove
        max_retries: Maximum number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    if not path.exists():
        return True

    def handle_remove_readonly(func, p, exc_info):
        """Handle read-only files on Windows/WSL."""
        import stat
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for attempt in range(max_retries):
        try:
            # Strategy 1: Standard rmtree with onexc handler
            shutil.rmtree(path, onexc=handle_remove_readonly)
            return True
        except (OSError, PermissionError) as e:
            logger.debug(f"Cleanup attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Brief delay before retry
                continue

            # Strategy 2: Try subprocess rm -rf (works better on WSL)
            try:
                subprocess.run(
                    ["rm", "-rf", str(path)],
                    check=False,
                    capture_output=True,
                    timeout=30
                )
                if not path.exists():
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Strategy 3: Manual recursive deletion for stubborn cases
            try:
                for root, dirs, files in os.walk(str(path), topdown=False):
                    for name in files:
                        filepath = os.path.join(root, name)
                        try:
                            os.chmod(filepath, 0o777)
                            os.unlink(filepath)
                        except (OSError, PermissionError):
                            pass
                    for name in dirs:
                        dirpath = os.path.join(root, name)
                        try:
                            os.rmdir(dirpath)
                        except (OSError, PermissionError):
                            pass
                os.rmdir(str(path))
                if not path.exists():
                    return True
            except (OSError, PermissionError):
                pass

    logger.warning(f"Failed to remove directory after {max_retries} attempts: {path}")
    return False


def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
) -> Any:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch

    Returns:
        Return value of func

    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = base_delay

    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                             f"Retrying in {delay:.1f}s...")
                time.sleep(min(delay, max_delay))
                delay *= backoff_factor
            else:
                logger.error(f"All {max_attempts} attempts failed: {e}")

    raise last_exception


def count_tokens_tiktoken(text: str) -> int:
    """Count tokens using tiktoken (Claude uses similar tokenizer)."""
    if not TIKTOKEN_AVAILABLE:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

    try:
        # Claude uses a tokenizer similar to GPT-4
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"tiktoken failed: {e}, using fallback")
        return len(text) // 4


def count_tokens_api(text: str, client: Optional[Any] = None) -> int:
    """Count tokens using Anthropic API (most accurate)."""
    if not ANTHROPIC_AVAILABLE or client is None:
        return count_tokens_tiktoken(text)

    try:
        # Use count_tokens API
        response = client.messages.count_tokens(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens
    except Exception as e:
        logger.warning(f"API token counting failed: {e}, using tiktoken")
        return count_tokens_tiktoken(text)


# ============================================================================
# Ground Truth Loading
# ============================================================================

def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    """Load and validate ground truth dataset from JSONL."""
    tasks = []
    required_fields = [
        "task_id", "description", "repo", "commit_before", "commit_after",
        "cursor_file", "query_terms", "files_edited"
    ]

    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                task = json.loads(line.strip())

                # Validate required fields
                missing = [f for f in required_fields if f not in task]
                if missing:
                    logger.error(f"Line {line_num}: Missing fields: {missing}")
                    continue

                tasks.append(task)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON: {e}")
                continue

    logger.info(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


# ============================================================================
# Repository Setup
# ============================================================================

def setup_test_repo(task: Dict[str, Any], temp_dir: Path) -> Optional[Path]:
    """Clone repository and checkout commit_before with retry logic.

    Handles:
    - Full SHAs (40 chars)
    - Short SHAs (7+ chars)
    - Branch names (local and remote)
    - Pull request refs (e.g., username/branch:ref)
    - Network failures with retry

    Returns:
        Path to cloned repo, or None if failed
    """
    repo_url = task["repo"]
    commit = task["commit_before"]
    repo_name = repo_url.split("/")[-1]
    repo_path = temp_dir / repo_name

    logger.info(f"Cloning {repo_url}...")

    # Remove existing directory if present (for retry scenarios)
    if repo_path.exists():
        shutil.rmtree(repo_path)

    try:
        # Clone with --filter=blob:none for faster partial clone
        # Use retry logic for network operations
        def clone_repo():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--quiet", repo_url, str(repo_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for clone
            )

        retry_with_backoff(
            clone_repo,
            max_attempts=3,
            base_delay=2.0,
            exceptions=(subprocess.CalledProcessError, subprocess.TimeoutExpired)
        )

        # Fetch all refs from origin to find remote branches
        # This is needed for refs like "emmanuelthome/fix-split-rn"
        logger.debug("Fetching all remote refs...")
        fetch_result = subprocess.run(
            ["git", "fetch", "--quiet", "--tags", "--force", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )

        # Try different strategies to checkout the ref
        success = False
        strategies = [
            # Try as a direct SHA (full or short)
            ["git", "checkout", commit],
            # Try as a remote branch (origin/branch-name)
            ["git", "checkout", f"origin/{commit}"],
            # Try as a remote ref with username prefix (for PR branches)
            ["git", "checkout", f"remotes/origin/{commit}"],
        ]

        for i, cmd in enumerate(strategies, 1):
            logger.debug(f"Strategy {i}: trying {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                success = True
                logger.debug(f"Checked out using strategy {i}")
                break

        if not success:
            # As a last resort, try remote fetch with the ref
            logger.debug(f"Last resort: fetching origin/{commit}")
            fetch_commit_result = subprocess.run(
                ["git", "fetch", "--quiet", "origin", commit],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            if fetch_commit_result.returncode == 0:
                checkout_result = subprocess.run(
                    ["git", "checkout", "FETCH_HEAD"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True
                )
                if checkout_result.returncode == 0:
                    success = True
                    logger.debug(f"Checked out via FETCH_HEAD")

        if not success:
            raise subprocess.CalledProcessError(1, "git checkout", stderr="All checkout strategies failed")

        # Get the actual commit SHA we ended up on
        rev_parse_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        actual_commit = rev_parse_result.stdout.strip()

        logger.info(f"Checked out {actual_commit[:12]} (requested: {commit[:12]}) in {repo_path}")
        return repo_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return None


# ============================================================================
# CodeGrapher Index Building
# ============================================================================

def build_codegraph_index(repo_path: Path) -> bool:
    """Initialize and build CodeGrapher index for a repository.

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Building CodeGrapher index for {repo_path}...")

    try:
        # Change to repo directory
        original_cwd = Path.cwd()
        os.chdir(repo_path)

        # Find codegraph command (check .venv/bin first, then system PATH)
        codegraph_cmd = None
        venv_bin = Path(__file__).parent.parent / ".venv" / "bin"
        if (venv_bin / "codegraph").exists():
            codegraph_cmd = str(venv_bin / "codegraph")
        else:
            # Try to find in PATH
            import shutil
            codegraph_cmd = shutil.which("codegraph")

        if codegraph_cmd is None:
            logger.error("codegraph command not found in .venv/bin or PATH")
            return False

        logger.debug(f"Using codegraph at: {codegraph_cmd}")

        # Run codegraph init (with --no-hook to skip git hook installation)
        init_result = subprocess.run(
            [codegraph_cmd, "init", "--no-hook"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        logger.debug(f"Init output: {init_result.stdout}")

        # Run codegraph build --full
        build_result = subprocess.run(
            [codegraph_cmd, "build", "--full"],
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        logger.debug(f"Build output: {build_result.stdout}")

        logger.info(f"Index built successfully for {repo_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"CodeGrapher build failed: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"CodeGrapher build timed out for {repo_path}")
        return False
    finally:
        os.chdir(original_cwd)


# ============================================================================
# Token Measurement
# ============================================================================

def measure_baseline_tokens(
    repo_path: Path,
    use_api: bool = False,
    client: Optional[Any] = None
) -> int:
    """Count tokens for all Python files in repository (baseline).

    This simulates sending the entire relevant codebase as context.
    """
    python_files = list(repo_path.rglob("*.py"))

    # Exclude common non-source directories
    exclude_dirs = {".venv", "venv", "env", ".tox", ".pytest_cache", "__pycache__", "build", "dist", ".git"}
    python_files = [
        f for f in python_files
        if not any(exclude in f.parts for exclude in exclude_dirs)
    ]

    logger.info(f"Counting tokens for {len(python_files)} Python files (baseline)...")

    total_text = ""
    for py_file in python_files:
        try:
            total_text += py_file.read_text(errors="ignore")
            total_text += "\n\n"
        except Exception as e:
            logger.warning(f"Failed to read {py_file}: {e}")

    if use_api and client:
        return count_tokens_api(total_text, client)
    else:
        return count_tokens_tiktoken(total_text)


def measure_codegraph_tokens(
    repo_path: Path,
    cursor_file: str,
    query_terms: List[str],
    use_api: bool = False,
    client: Optional[Any] = None,
    skip_import_closure: bool = False
) -> Tuple[int, List[str]]:
    """Query CodeGrapher and count tokens in returned bundle.

    Returns:
        Tuple of (token_count, list_of_returned_files)
    """
    # Change to repo directory for query
    original_cwd = Path.cwd()
    os.chdir(repo_path)

    try:
        # Build query from terms
        query = " ".join(query_terms)

        logger.info(f"Querying CodeGrapher: '{query}' (cursor: {cursor_file})...")

        # Query CodeGrapher using the underlying function
        # Skip import closure if debugging (pass None for cursor_file)
        effective_cursor = None if skip_import_closure else cursor_file
        if skip_import_closure:
            logger.info(f"DEBUG: Skipping import closure pruning (cursor_file set to None)")

        result = _codegraph_query_func(
            query=query,
            cursor_file=effective_cursor,
            token_budget=10000  # High budget to avoid truncation
        )

        if result["status"] != "success":
            logger.error(f"Query failed: {result.get('message', 'Unknown error')}")
            return 0, []

        # Debug: log the result structure
        logger.debug(f"CodeGrapher result keys: {result.keys()}")
        logger.debug(f"Number of items in result: {len(result.get('files', []))}")

        # Extract file contents from result
        # The result format is: {"files": [{"path": "...", "symbol": "...", "excerpt": "...", "line_range": [...]}]}
        items = result.get("files", [])
        returned_file_paths = []
        total_text = ""

        # Collect unique file paths
        for item in items:
            file_path = item.get("path", "")
            if file_path and file_path not in returned_file_paths:
                returned_file_paths.append(file_path)

        logger.info(f"CodeGrapher returned {len(returned_file_paths)} unique files")

        # Diagnostic: Log what was returned vs expected (for recall investigation)
        logger.debug(f"Returned files: {returned_file_paths[:10]}...")  # First 10
        logger.debug(f"Total candidates before truncation: {result.get('total_candidates', 0)}")
        logger.debug(f"Truncated: {result.get('truncated', False)}")

        # Read actual file content for token counting
        # This gives us the real token count that would be sent to an LLM
        for file_path in returned_file_paths:
            full_path = repo_path / file_path
            try:
                content = full_path.read_text(errors="ignore")
                total_text += f"# {file_path}\n{content}\n\n"
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        # Count tokens
        if use_api and client:
            tokens = count_tokens_api(total_text, client)
        else:
            tokens = count_tokens_tiktoken(total_text)

        return tokens, returned_file_paths

    finally:
        os.chdir(original_cwd)


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_metrics(
    baseline_tokens: int,
    cg_tokens: int,
    returned_files: List[str],
    files_edited: List[str]
) -> Dict[str, Any]:
    """Calculate recall, precision, and token savings.

    Returns:
        Dictionary with metrics
    """
    # Normalize file paths (remove leading ./ and trailing slashes)
    def normalize_path(p: str) -> str:
        return p.lstrip("./").rstrip("/")

    returned_set = {normalize_path(f) for f in returned_files}
    edited_set = {normalize_path(f) for f in files_edited}

    # Calculate recall: what proportion of edited files were retrieved?
    if len(edited_set) > 0:
        recall = len(returned_set & edited_set) / len(edited_set)
    else:
        recall = 0.0

    # Calculate precision: what proportion of retrieved files were edited?
    # Note: Lower precision is acceptable (target ≤40%) because we want
    # to include related context files, not just the exact files edited
    if len(returned_set) > 0:
        precision = len(returned_set & edited_set) / len(returned_set)
    else:
        precision = 0.0

    # Calculate token savings
    if baseline_tokens > 0:
        token_savings = (baseline_tokens - cg_tokens) / baseline_tokens
    else:
        token_savings = 0.0

    return {
        "baseline_tokens": baseline_tokens,
        "cg_tokens": cg_tokens,
        "token_savings_pct": token_savings * 100,
        "recall": recall,
        "precision": precision,
        "files_returned": len(returned_set),
        "files_edited": len(edited_set),
        "files_overlap": len(returned_set & edited_set),
    }


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def evaluate_task(
    task: Dict[str, Any],
    temp_dir: Path,
    use_api: bool = False,
    client: Optional[Any] = None,
    skip_import_closure: bool = False
) -> Optional[Dict[str, Any]]:
    """Evaluate a single task.

    Returns:
        Results dictionary, or None if task failed
    """
    task_id = task["task_id"]
    logger.info(f"\n{'='*80}\nEvaluating {task_id}: {task['description']}\n{'='*80}")
    log_memory_usage(" (start)")

    # Setup repository
    repo_path = setup_test_repo(task, temp_dir)
    if repo_path is None:
        logger.error(f"Failed to setup repo for {task_id}")
        return None
    log_memory_usage(" (after repo setup)")

    # Build CodeGrapher index
    if not build_codegraph_index(repo_path):
        logger.error(f"Failed to build index for {task_id}")
        return None
    log_memory_usage(" (after index build)")

    # Measure baseline tokens
    baseline_tokens = measure_baseline_tokens(repo_path, use_api, client)
    logger.info(f"Baseline tokens: {baseline_tokens:,}")

    # Measure CodeGrapher tokens
    cg_tokens, returned_files = measure_codegraph_tokens(
        repo_path,
        task["cursor_file"],
        task["query_terms"],
        use_api,
        client,
        skip_import_closure
    )
    logger.info(f"CodeGrapher tokens: {cg_tokens:,}")

    # Calculate metrics
    metrics = calculate_metrics(
        baseline_tokens,
        cg_tokens,
        returned_files,
        task["files_edited"]
    )

    # Build result
    result = {
        "task_id": task_id,
        "description": task["description"],
        "repo": task["repo"],
        **metrics,
        "returned_files": returned_files,
        "expected_files": task["files_edited"],
    }

    logger.info(f"Results: {metrics['token_savings_pct']:.1f}% savings, "
                f"{metrics['recall']:.1%} recall, {metrics['precision']:.1%} precision")

    log_memory_usage(" (end of task)")

    return result


def run_evaluation(
    tasks: List[Dict[str, Any]],
    mode: str = "simulate",
    api_tasks: int = 5,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
    skip_import_closure: bool = False
) -> List[Dict[str, Any]]:
    """Run evaluation on all tasks with checkpoint/resume support.

    Args:
        tasks: List of tasks to evaluate
        mode: "simulate", "mixed", or "real-api"
        api_tasks: Number of tasks to run with real API (if mode="mixed")
        checkpoint_path: Path to save checkpoint data
        resume: If True, resume from existing checkpoint
        skip_import_closure: If True, disable import closure pruning for debugging

    Returns:
        List of result dictionaries
    """
    # Initialize API client if needed
    client = None
    if mode in ["real-api", "mixed"]:
        if not ANTHROPIC_AVAILABLE:
            logger.error("Anthropic SDK not available. Install with: pip install anthropic")
            logger.info("Falling back to simulation mode.")
            mode = "simulate"
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found in environment.")
                logger.info("Falling back to simulation mode.")
                mode = "simulate"
            else:
                client = Anthropic(api_key=api_key)
                logger.info(f"Anthropic client initialized (mode: {mode})")

    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(tempfile.gettempdir()) / "codegraph_eval_checkpoint.json"

    # Load existing checkpoint if resuming
    results = []
    completed_task_ids = set()
    start_index = 0

    if resume and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
            results = checkpoint_data.get("results", [])
            completed_task_ids = {r["task_id"] for r in results}
            start_index = len(results)
            logger.info(f"Resumed from checkpoint: {len(results)} completed tasks")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
            start_index = 0

    # Create temp directory for repos
    temp_dir = Path(tempfile.mkdtemp(prefix="codegraph_eval_"))
    logger.info(f"Using temp directory: {temp_dir}")

    # Register cleanup handler
    def cleanup_temp_dirs():
        """Clean up temp directory but preserve checkpoint."""
        if temp_dir.exists():
            robust_rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")

    import atexit
    import signal

    atexit.register(cleanup_temp_dirs)

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"\nReceived signal {signum}, saving checkpoint...")
        save_checkpoint()
        cleanup_temp_dirs()
        sys.exit(130)  # 128 + 2 (SIGINT)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def save_checkpoint():
        """Save current progress to checkpoint file."""
        try:
            checkpoint_data = {
                "results": results,
                "completed_task_ids": list(completed_task_ids),
                "mode": mode,
                "api_tasks": api_tasks,
                "total_tasks": len(tasks),
                "timestamp": Path.cwd().stat().st_mtime if Path.cwd().exists() else 0
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint saved: {len(results)}/{len(tasks)} tasks completed")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    try:
        for i in range(start_index, len(tasks)):
            task = tasks[i]
            task_id = task["task_id"]

            # Skip already completed tasks
            if task_id in completed_task_ids:
                logger.info(f"\nSkipping {task_id} (already completed)")
                continue

            # Determine if this task should use API
            use_api = False
            if mode == "real-api":
                use_api = True
            elif mode == "mixed" and i < api_tasks:
                use_api = True

            logger.info(f"\nTask {i+1}/{len(tasks)} ({'API' if use_api else 'Simulated'})")

            result = evaluate_task(task, temp_dir, use_api, client, skip_import_closure)
            if result:
                results.append(result)
                completed_task_ids.add(task_id)

                # Save checkpoint after each successful task
                save_checkpoint()

            # Clean up this repo to save disk space
            repo_name = task["repo"].split("/")[-1]
            repo_path = temp_dir / repo_name
            if repo_path.exists():
                robust_rmtree(repo_path)

    finally:
        # Save final checkpoint
        save_checkpoint()
        # Clean up temp directory
        cleanup_temp_dirs()

    logger.info(f"Checkpoint available at: {checkpoint_path}")
    logger.info(f"To resume later, use: --resume --checkpoint {checkpoint_path}")

    return results


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Generate markdown report and summary statistics.

    Returns:
        Tuple of (markdown_report, summary_dict)
    """
    if not results:
        return "No results to report.", {}

    # Calculate summary statistics
    token_savings = [r["token_savings_pct"] for r in results]
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]

    summary = {
        "total_tasks": len(results),
        "token_savings_median": statistics.median(token_savings),
        "token_savings_mean": statistics.mean(token_savings),
        "token_savings_min": min(token_savings),
        "token_savings_max": max(token_savings),
        "recall_median": statistics.median(recalls),
        "recall_mean": statistics.mean(recalls),
        "recall_min": min(recalls),
        "recall_max": max(recalls),
        "precision_median": statistics.median(precisions),
        "precision_mean": statistics.mean(precisions),
        "precision_min": min(precisions),
        "precision_max": max(precisions),
    }

    # Check acceptance criteria
    criteria = {
        "token_savings_pass": summary["token_savings_median"] >= 30.0,
        "recall_pass": summary["recall_median"] >= 0.85,
        "precision_pass": summary["precision_median"] <= 0.40,
    }
    all_pass = all(criteria.values())

    # Generate markdown report
    report = f"""# CodeGrapher Phase 12 Evaluation Results

## Summary Statistics

- **Tasks Evaluated:** {summary['total_tasks']}
- **Token Savings (Median):** {summary['token_savings_median']:.1f}% {'✅' if criteria['token_savings_pass'] else '❌'} (target: ≥30%)
- **Recall (Median):** {summary['recall_median']:.1%} {'✅' if criteria['recall_pass'] else '❌'} (target: ≥85%)
- **Precision (Median):** {summary['precision_median']:.1%} {'✅' if criteria['precision_pass'] else '❌'} (target: ≤40%)

## Acceptance Criteria Status

| Criterion | Target | Actual (Median) | Status |
|-----------|--------|-----------------|--------|
| Token Savings | ≥30% | {summary['token_savings_median']:.1f}% | {'✅ PASS' if criteria['token_savings_pass'] else '❌ FAIL'} |
| Recall | ≥85% | {summary['recall_median']:.1%} | {'✅ PASS' if criteria['recall_pass'] else '❌ FAIL'} |
| Precision | ≤40% | {summary['precision_median']:.1%} | {'✅ PASS' if criteria['precision_pass'] else '❌ FAIL'} |

**Overall Status:** {'✅ ALL CRITERIA MET' if all_pass else '❌ SOME CRITERIA NOT MET'}

## Detailed Statistics

### Token Savings
- Mean: {summary['token_savings_mean']:.1f}%
- Min: {summary['token_savings_min']:.1f}%
- Max: {summary['token_savings_max']:.1f}%

### Recall
- Mean: {summary['recall_mean']:.1%}
- Min: {summary['recall_min']:.1%}
- Max: {summary['recall_max']:.1%}

### Precision
- Mean: {summary['precision_mean']:.1%}
- Min: {summary['precision_min']:.1%}
- Max: {summary['precision_max']:.1%}

## Per-Task Results

| Task ID | Description | Token Savings | Recall | Precision | Files Returned |
|---------|-------------|---------------|--------|-----------|----------------|
"""

    for r in results:
        desc = r["description"][:50] + "..." if len(r["description"]) > 50 else r["description"]
        report += f"| {r['task_id']} | {desc} | {r['token_savings_pct']:.1f}% | {r['recall']:.1%} | {r['precision']:.1%} | {r['files_returned']} |\n"

    report += f"""
## Insights

### Token Savings Distribution
The token savings ranged from {summary['token_savings_min']:.1f}% to {summary['token_savings_max']:.1f}%,
with a median of {summary['token_savings_median']:.1f}%. This demonstrates CodeGrapher's ability to
significantly reduce context size while maintaining relevant information.

### Retrieval Quality
With a median recall of {summary['recall_median']:.1%}, CodeGrapher successfully retrieves most of the
files actually edited in real-world development tasks. The precision of {summary['precision_median']:.1%}
indicates that the tool includes additional context files beyond just the edited files, which is expected
and desirable for providing comprehensive context to the LLM.

---

*Report generated by Phase 12 evaluation script*
"""

    return report, summary


def generate_appendable_report(result: Dict[str, Any]) -> str:
    """Generate a markdown report for a single task that can be appended to.

    This format allows running evaluations incrementally and appending results
    to the same file. The user will calculate median statistics at the end.

    Args:
        result: Single task result dictionary

    Returns:
        Markdown report section for this task
    """
    from datetime import datetime

    task_id = result["task_id"]
    desc = result["description"]
    repo = result["repo"]
    repo_name = repo.split("/")[-1]

    # Format returned and expected files
    returned_str = ", ".join(f"`{f}`" for f in result.get("returned_files", []))
    expected_str = ", ".join(f"`{f}`" for f in result.get("expected_files", []))

    # Check if expected files were found
    returned_set = set(result.get("returned_files", []))
    expected_set = set(result.get("expected_files", []))
    found = returned_set & expected_set
    missed = expected_set - returned_set

    status = "✅ PASS" if len(missed) == 0 else "⚠️  PARTIAL"
    if len(found) == 0:
        status = "❌ FAIL"

    report = f"""## {task_id}: {desc}

**Repo:** [{repo_name}]({repo})
**Status:** {status}
**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | {result['token_savings_pct']:.1f}% | ≥30% | {'✅' if result['token_savings_pct'] >= 30 else '❌'} |
| Recall | {result['recall']:.1%} | ≥85% | {'✅' if result['recall'] >= 0.85 else '❌'} |
| Precision | {result['precision']:.1%} | ≤40% | {'✅' if result['precision'] <= 0.40 else '❌'} |

**Baseline Tokens:** {result['baseline_tokens']:,}
**CodeGrapher Tokens:** {result['cg_tokens']:,}
**Files Returned:** {result['files_returned']}

### Files

**Expected Files:** {expected_str}
- ✅ Found: {", ".join(f"`{f}`" for f in found) if found else "None"}
- ❌ Missed: {", ".join(f"`{f}`" for f in missed) if missed else "None"}

**All Returned Files:** {returned_str}

---

"""

    return report


def append_report_to_file(report: str, output_path: Path) -> None:
    """Append a single task report to the evaluation results file.

    Creates the file with a header if it doesn't exist.

    Args:
        report: Single task markdown report
        output_path: Path to the output markdown file
    """
    if output_path.exists():
        with open(output_path, "a") as f:
            f.write(report)
    else:
        # Create new file with header
        header = """# CodeGrapher Phase 12 Evaluation Results (Appendable)

This file accumulates results from individual evaluation runs. Each task is added independently as it's evaluated.

**Targets:** Token Savings ≥30%, Recall ≥85%, Precision ≤40%

---

"""
        with open(output_path, "w") as f:
            f.write(header)
            f.write(report)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CodeGrapher token savings")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated token counting (no API calls, no cost)"
    )
    mode_group.add_argument(
        "--real-api",
        action="store_true",
        help="Use real Anthropic API for all tasks (most accurate, higher cost)"
    )
    mode_group.add_argument(
        "--mixed",
        action="store_true",
        help="Mixed strategy: some real API, some simulated (default)"
    )

    parser.add_argument(
        "--api-tasks",
        type=int,
        default=5,
        help="Number of tasks to run with real API in mixed mode (default: 5)"
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth.jsonl",
        help="Path to ground truth JSONL file"
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "eval_results.json",
        help="Path to save detailed results JSON"
    )

    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "eval_report.md",
        help="Path to save summary report markdown"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from existing checkpoint"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file for saving/resuming progress"
    )

    parser.add_argument(
        "--skip-import-closure",
        action="store_true",
        help="Skip import closure pruning for debugging (useful to isolate 0% recall issue)"
    )

    parser.add_argument(
        "--appendable",
        action="store_true",
        help="Generate appendable report format (one task per section, no summary stats)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine mode
    if args.simulate:
        mode = "simulate"
    elif args.real_api:
        mode = "real-api"
    else:
        mode = "mixed"  # Default

    logger.info(f"Starting evaluation in {mode.upper()} mode")
    if mode == "mixed":
        logger.info(f"Will run {args.api_tasks} tasks with real API, rest simulated")

    # Load ground truth
    if not args.ground_truth.exists():
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    tasks = load_ground_truth(args.ground_truth)
    if not tasks:
        logger.error("No valid tasks loaded")
        sys.exit(1)

    # Run evaluation
    results = run_evaluation(
        tasks,
        mode,
        args.api_tasks,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        skip_import_closure=args.skip_import_closure
    )

    if not results:
        logger.error("No results generated")
        sys.exit(1)

    # Save detailed results JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "results": results,
            "mode": mode,
            "api_tasks": args.api_tasks if mode == "mixed" else None,
        }, f, indent=2)
    logger.info(f"Detailed results saved to {args.output_json}")

    # Handle appendable format vs standard format
    if args.appendable:
        # Appendable format: one task per section
        for result in results:
            report = generate_appendable_report(result)
            append_report_to_file(report, args.output_report)
        logger.info(f"Results appended to {args.output_report}")

        # Print brief console summary
        for result in results:
            print(f"\n{result['task_id']}: {result['token_savings_pct']:.1f}% savings, "
                  f"{result['recall']:.1%} recall, {result['precision']:.1%} precision")

        # Exit with success (appendable format doesn't calculate overall stats)
        sys.exit(0)
    else:
        # Standard format with summary statistics
        report, summary = generate_report(results)

        # Save markdown report
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, "w") as f:
            f.write(report)
        logger.info(f"Summary report saved to {args.output_report}")

        # Print summary to console
        print("\n" + report)

        # Exit with appropriate code
        sys.exit(0 if all([
            summary.get("token_savings_median", 0) >= 30.0,
            summary.get("recall_median", 0) >= 0.85,
            summary.get("precision_median", 1.0) <= 0.40,
        ]) else 1)


if __name__ == "__main__":
    main()
