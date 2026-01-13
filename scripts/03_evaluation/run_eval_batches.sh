#!/bin/bash
# WSL-Robust Batch Evaluation Runner
#
# This script runs evaluation in small batches to avoid WSL memory issues.
# It automatically resumes from checkpoint between batches, allowing for
# memory cleanup and WSL stability.
#
# Usage:
#   ./scripts/03_evaluation/run_eval_batches.sh [--batch-size N] [--mode simulate|mixed|real-api]

set -e

# Configuration
BATCH_SIZE=${BATCH_SIZE:-3}  # Process 3 tasks at a time (adjustable)
MODE="mixed"
GROUND_TRUTH="fixtures/ground_truth.jsonl"
CHECKPOINT="/tmp/eval_checkpoint_batch.json"
OUTPUT_APPENDABLE="fixtures/eval_results_appendable.md"
SLEEP_BETWEEN_BATCHES=10  # seconds

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--batch-size N] [--mode simulate|mixed|real-api]"
            exit 1
            ;;
    esac
done

# Get total number of tasks
TOTAL_TASKS=$(wc -l < "$GROUND_TRUTH")

echo "=========================================="
echo "WSL-Robust Batch Evaluation Runner"
echo "=========================================="
echo "Total tasks: $TOTAL_TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_APPENDABLE"
echo ""

# Calculate number of batches
NUM_BATCHES=$(( (TOTAL_TASKS + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Will run $NUM_BATCHES batches"
echo ""

# Create a task-specific checkpoint for this run
BATCH_CHECKPOINT="${CHECKPOINT%.json}_$(date +%Y%m%d_%H%M%S).json"

# Run batches
for batch in $(seq 1 $NUM_BATCHES); do
    echo "=========================================="
    echo "Batch $batch of $NUM_BATCHES"
    echo "=========================================="

    # Calculate task range for this batch
    START_TASK=$(( (batch - 1) * BATCH_SIZE + 1 ))
    END_TASK=$(( batch * BATCH_SIZE ))
    if [ $END_TASK -gt $TOTAL_TASKS ]; then
        END_TASK=$TOTAL_TASKS
    fi

    echo "Processing tasks $START_TASK-$END_TASK..."
    echo ""

    # Run evaluation with resume (will skip already completed tasks)
    if [ $batch -eq 1 ]; then
        # First batch: don't use --resume
        .venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py \
            --$MODE \
            --ground-truth "$GROUND_TRUTH" \
            --checkpoint "$BATCH_CHECKPOINT" \
            --appendable \
            --output-report "$OUTPUT_APPENDABLE"
    else
        # Subsequent batches: use --resume
        .venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py \
            --$MODE \
            --ground-truth "$GROUND_TRUTH" \
            --checkpoint "$BATCH_CHECKPOINT" \
            --resume \
            --appendable \
            --output-report "$OUTPUT_APPENDABLE"
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠️  Batch $batch failed or was interrupted (exit code: $EXIT_CODE)"
        echo "Checkpoint saved to: $BATCH_CHECKPOINT"
        echo ""
        echo "To resume, run:"
        echo "  $0 --batch-size $BATCH_SIZE --mode $MODE"
        echo ""
        exit $EXIT_CODE
    fi

    # Memory cleanup between batches
    if [ $batch -lt $NUM_BATCHES ]; then
        echo ""
        echo "✅ Batch $batch complete. Sleeping ${SLEEP_BETWEEN_BATCHES}s for memory cleanup..."
        echo ""
        sleep $SLEEP_BETWEEN_BATCHES

        # Optional: Force Python garbage collection by starting fresh process
        # (the script already does cleanup, but this ensures clean slate)
    fi
done

echo ""
echo "=========================================="
echo "✅ All batches complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_APPENDABLE"
echo "Checkpoint: $BATCH_CHECKPOINT"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_APPENDABLE"
echo ""
