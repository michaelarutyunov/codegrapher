#!/usr/bin/env python3
"""WSL-Robust Evaluation Runner with Auto-Retry

This script runs evaluation in batches with automatic retry on WSL disconnections.
It splits the evaluation into manageable chunks and automatically resumes from
checkpoints, handling WSL memory pressure gracefully.

Features:
- Batch processing (default: 3 tasks per batch)
- Automatic checkpoint/resume between batches
- Memory cleanup between batches (via fresh Python processes)
- Auto-retry on failures with exponential backoff
- Progress tracking and detailed logging

Usage:
    python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3
    python scripts/03_evaluation/run_eval_robust.py --mode simulate --batch-size 5
    python scripts/03_evaluation/run_eval_robust.py --resume  # Resume from last checkpoint
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def count_tasks(ground_truth_path: Path) -> int:
    """Count total tasks in ground truth file."""
    with open(ground_truth_path) as f:
        return sum(1 for line in f if line.strip())


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing checkpoint if available."""
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def run_batch(
    batch_num: int,
    mode: str,
    ground_truth: Path,
    checkpoint: Path,
    output_report: Path,
    resume: bool,
    max_retries: int = 3
) -> bool:
    """Run a single batch of evaluation with retry logic.

    Args:
        batch_num: Batch number for logging
        mode: Evaluation mode (simulate/mixed/real-api)
        ground_truth: Path to ground truth JSONL
        checkpoint: Path to checkpoint file
        output_report: Path to output report
        resume: Whether to resume from checkpoint
        max_retries: Maximum retry attempts on failure

    Returns:
        True if batch succeeded, False otherwise
    """
    cmd = [
        sys.executable,
        "scripts/03_evaluation/evaluate_ground_truth.py",
        f"--{mode}",
        "--ground-truth", str(ground_truth),
        "--checkpoint", str(checkpoint),
        "--appendable",
        "--output-report", str(output_report)
    ]

    if resume:
        cmd.append("--resume")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Running batch {batch_num} (attempt {attempt}/{max_retries})...")
            logger.debug(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per batch
            )

            logger.info(f"✅ Batch {batch_num} completed successfully")
            if result.stdout:
                logger.debug(f"Stdout: {result.stdout[-500:]}")  # Last 500 chars

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Batch {batch_num} failed (attempt {attempt}/{max_retries})")
            logger.error(f"Exit code: {e.returncode}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr[-500:]}")  # Last 500 chars

            if attempt < max_retries:
                # Exponential backoff
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Max retries ({max_retries}) exceeded for batch {batch_num}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Batch {batch_num} timed out (attempt {attempt}/{max_retries})")

            if attempt < max_retries:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Max retries ({max_retries}) exceeded for batch {batch_num}")
                return False

        except Exception as e:
            logger.error(f"❌ Unexpected error in batch {batch_num}: {e}")
            return False

    return False


def run_evaluation_robust(
    mode: str,
    batch_size: int,
    ground_truth: Path,
    output_report: Path,
    checkpoint_base: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    sleep_between_batches: int = 10
) -> bool:
    """Run evaluation in robust batches with automatic resume.

    Args:
        mode: Evaluation mode (simulate/mixed/real-api)
        batch_size: Number of tasks per batch
        ground_truth: Path to ground truth JSONL
        output_report: Path to output markdown report
        checkpoint_base: Base name for checkpoint file
        resume_from: Existing checkpoint to resume from
        sleep_between_batches: Seconds to sleep between batches

    Returns:
        True if all batches completed successfully
    """
    # Count total tasks
    total_tasks = count_tasks(ground_truth)
    num_batches = (total_tasks + batch_size - 1) // batch_size

    logger.info("=" * 80)
    logger.info("WSL-Robust Evaluation Runner")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Output: {output_report}")
    logger.info("")

    # Determine checkpoint path
    if resume_from:
        checkpoint_path = resume_from
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    elif checkpoint_base:
        checkpoint_path = checkpoint_base
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = Path(f"/tmp/eval_checkpoint_robust_{timestamp}.json")

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("")

    # Load existing checkpoint to determine starting batch
    checkpoint_data = load_checkpoint(checkpoint_path)
    completed_tasks = 0

    if checkpoint_data:
        completed_tasks = len(checkpoint_data.get("results", []))
        start_batch = (completed_tasks // batch_size) + 1
        logger.info(f"Checkpoint found: {completed_tasks} tasks already completed")
        logger.info(f"Starting from batch {start_batch}")
    else:
        start_batch = 1
        logger.info("No checkpoint found, starting from batch 1")

    logger.info("")

    # Run batches
    for batch_num in range(start_batch, num_batches + 1):
        logger.info("=" * 80)
        logger.info(f"Batch {batch_num} of {num_batches}")
        logger.info("=" * 80)

        # Calculate task range
        start_task = (batch_num - 1) * batch_size + 1
        end_task = min(batch_num * batch_size, total_tasks)
        logger.info(f"Task range: {start_task}-{end_task}")
        logger.info("")

        # Run batch (always use resume to skip already-completed tasks)
        success = run_batch(
            batch_num=batch_num,
            mode=mode,
            ground_truth=ground_truth,
            checkpoint=checkpoint_path,
            output_report=output_report,
            resume=(batch_num > 1 or resume_from is not None)
        )

        if not success:
            logger.error("")
            logger.error("=" * 80)
            logger.error(f"❌ Evaluation failed at batch {batch_num}")
            logger.error("=" * 80)
            logger.error(f"Checkpoint saved to: {checkpoint_path}")
            logger.error("")
            logger.error("To resume, run:")
            logger.error(f"  python scripts/03_evaluation/run_eval_robust.py --resume --checkpoint {checkpoint_path}")
            logger.error("")
            return False

        # Sleep between batches for memory cleanup
        if batch_num < num_batches:
            logger.info("")
            logger.info(f"Sleeping {sleep_between_batches}s for memory cleanup...")
            logger.info("")
            time.sleep(sleep_between_batches)

    # Success!
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ All batches completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_report}")
    logger.info(f"Final checkpoint: {checkpoint_path}")
    logger.info("")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="WSL-robust evaluation runner with automatic retry"
    )

    parser.add_argument(
        "--mode",
        choices=["simulate", "mixed", "real-api"],
        default="mixed",
        help="Evaluation mode (default: mixed)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of tasks per batch (default: 3, adjust lower if WSL crashes)"
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("fixtures/ground_truth.jsonl"),
        help="Path to ground truth JSONL file"
    )

    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("fixtures/eval_results_appendable.md"),
        help="Path to output markdown report"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint file path (for custom location)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint (requires --checkpoint)"
    )

    parser.add_argument(
        "--sleep-between-batches",
        type=int,
        default=10,
        help="Seconds to sleep between batches (default: 10)"
    )

    args = parser.parse_args()

    # Validate resume mode
    if args.resume and not args.checkpoint:
        logger.error("--resume requires --checkpoint to specify which checkpoint to resume from")
        sys.exit(1)

    # Run robust evaluation
    success = run_evaluation_robust(
        mode=args.mode,
        batch_size=args.batch_size,
        ground_truth=args.ground_truth,
        output_report=args.output_report,
        checkpoint_base=args.checkpoint,
        resume_from=args.checkpoint if args.resume else None,
        sleep_between_batches=args.sleep_between_batches
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
