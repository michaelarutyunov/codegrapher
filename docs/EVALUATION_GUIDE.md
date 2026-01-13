# CodeGrapher Evaluation Guide

**Quick Reference:** How to run evaluations on the ground truth dataset, especially on WSL with memory constraints.

---

## TL;DR - Just Run This

```bash
# Clear previous results (backup first!)
mv fixtures/eval_results_appendable.md fixtures/eval_results_appendable_backup_$(date +%Y%m%d).md

# Run robust evaluation (handles WSL disconnections automatically)
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3
```

---

## Problem: WSL Disconnections During Evaluation

**Symptoms:**
- WSL disconnects/crashes during evaluation
- Out of memory errors
- Have to manually run tests one-by-one

**Root Cause:** Memory pressure from:
- Repository cloning (~500MB-2GB per repo)
- FAISS index building (~1-3GB peak)
- Multiple repos in memory simultaneously

---

## Solution: Robust Batch Runner

### Option 1: Python Robust Runner (RECOMMENDED)

**File:** `scripts/03_evaluation/run_eval_robust.py`

**Features:**
- ✅ Runs tasks in small batches (default: 3 at a time)
- ✅ Fresh Python process per batch (full memory cleanup)
- ✅ Automatic checkpoint/resume on crashes
- ✅ Auto-retry with exponential backoff (3 attempts per batch)
- ✅ Detailed progress logging
- ✅ 1-hour timeout per batch

**Basic Usage:**

```bash
# Default: mixed mode, 3 tasks per batch
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3

# Simulate mode (no API costs) - use for testing
python scripts/03_evaluation/run_eval_robust.py --mode simulate --batch-size 5

# Real API mode (all real API calls) - slower batches recommended
python scripts/03_evaluation/run_eval_robust.py --mode real-api --batch-size 2
```

**If It Crashes:**

Just re-run the same command! It will automatically resume:

```bash
# Re-run after crash (uses saved checkpoint)
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3
```

**Advanced Resume:**

```bash
# Find checkpoint file
ls /tmp/eval_checkpoint_robust_*.json

# Resume from specific checkpoint
python scripts/03_evaluation/run_eval_robust.py --resume --checkpoint /tmp/eval_checkpoint_robust_20260113_184530.json
```

---

### Option 2: Bash Batch Runner (Alternative)

**File:** `scripts/03_evaluation/run_eval_batches.sh`

**Usage:**

```bash
./scripts/03_evaluation/run_eval_batches.sh --batch-size 3 --mode mixed
```

Simpler but less robust than the Python version (no auto-retry).

---

### Option 3: Direct Evaluation (Single Run)

**File:** `scripts/03_evaluation/evaluate_ground_truth.py`

**When to Use:**
- You have plenty of memory
- Running on a stable machine (not WSL)
- Want to run all tasks in one go

**Usage:**

```bash
# Mixed mode (5 real API calls, rest simulated)
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --mixed --ground-truth fixtures/ground_truth.jsonl

# Simulate mode (no API costs)
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --simulate --ground-truth fixtures/ground_truth.jsonl

# Real API mode (all real API calls)
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --real-api --ground-truth fixtures/ground_truth.jsonl

# Resume from crash
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --resume --checkpoint /tmp/eval_checkpoint.json
```

---

## Evaluation Modes

| Mode | API Calls | Cost | Accuracy | Use Case |
|------|-----------|------|----------|----------|
| **simulate** | 0 | Free | Approximate | Testing, infrastructure validation |
| **mixed** | 5 real, rest simulated | Low | Good | Default, balanced cost/accuracy |
| **real-api** | All real | Higher | Best | Final validation, production metrics |

---

## Batch Size Tuning

| Batch Size | Batches (23 tasks) | Memory Risk | Speed | Recommendation |
|------------|-------------------|-------------|-------|----------------|
| 1 | 23 | Very Low | Slowest | Ultra-safe for tiny WSL memory |
| 3 | 8 | Low | Good | **Recommended for WSL** |
| 5 | 5 | Medium | Fast | Stable WSL with 8GB+ memory |
| 10 | 3 | High | Fastest | Non-WSL or 16GB+ memory |

**Rule of thumb:**
- If WSL crashes → Reduce batch size
- If too slow → Increase batch size

---

## Before Running Evaluation

### 1. Clear Previous Results

```bash
# Backup existing results
mv fixtures/eval_results_appendable.md fixtures/eval_results_appendable_backup_$(date +%Y%m%d).md

# Or just delete
rm fixtures/eval_results_appendable.md
```

**Why?** The `--appendable` flag appends results. Running twice would duplicate entries.

---

### 2. Increase WSL Memory (Optional but Recommended)

**Edit `%USERPROFILE%\.wslconfig` on Windows:**

```ini
[wsl2]
memory=16GB           # Increase from default 8GB
swap=4GB              # Add swap space
swapFile=C:\\temp\\wsl-swap.vhdx
```

**Restart WSL:**

```powershell
# In PowerShell
wsl --shutdown
```

---

## During Evaluation

### Monitor Progress

The robust runner logs detailed progress:

```
==================================================
Batch 3 of 8
==================================================
Task range: 7-9
Running batch 3 (attempt 1/3)...
✅ Batch 3 completed successfully

Sleeping 10s for memory cleanup...
```

### Monitor Memory (Optional)

In another terminal:

```bash
# Watch memory usage
watch -n 1 free -h

# Check for OOM kills
dmesg | grep -i kill
```

---

## After Evaluation

### View Results

```bash
# View full report
cat fixtures/eval_results_appendable.md

# Count completed tasks
grep "^## task_" fixtures/eval_results_appendable.md | wc -l

# View summary statistics (if using standard format)
cat fixtures/eval_report.md
```

### Calculate Summary Statistics

The appendable format doesn't calculate medians automatically. Use Python:

```python
import re
from pathlib import Path
import statistics

# Read results
text = Path("fixtures/eval_results_appendable.md").read_text()

# Extract metrics
savings = [float(m.group(1)) for m in re.finditer(r"Token Savings.*?(\d+\.\d+)%", text)]
recalls = [float(m.group(1)) for m in re.finditer(r"Recall.*?(\d+\.\d+)%", text)]

print(f"Token Savings Median: {statistics.median(savings):.1f}%")
print(f"Recall Median: {statistics.median(recalls):.1f}%")
```

---

## Troubleshooting

### WSL Still Crashing?

**Try:**
1. Reduce batch size to 2 or 1
2. Close other applications
3. Check memory: `free -h`
4. Check for OOM kills: `dmesg | grep -i kill`
5. Increase WSL memory limit (see above)

### Checkpoint Not Resuming?

**Check:**
```bash
# List checkpoint files
ls /tmp/eval_checkpoint_*.json

# View checkpoint content
cat /tmp/eval_checkpoint_robust_*.json | jq .
```

### Tasks Being Skipped?

The checkpoint/resume system skips already-completed tasks. This is intentional.

**To start fresh:**
```bash
# Delete checkpoint
rm /tmp/eval_checkpoint_*.json

# Delete previous results
rm fixtures/eval_results_appendable.md

# Run again
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3
```

---

## Example: Full Workflow

```bash
# 1. Navigate to project
cd /home/mikhailarutyunov/projects/codegrapher

# 2. Backup previous results
mv fixtures/eval_results_appendable.md fixtures/eval_results_backup_$(date +%Y%m%d).md

# 3. Run robust evaluation
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3

# 4. If it crashes (network/WSL issue), just re-run:
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3
# ↑ Automatically resumes from checkpoint!

# 5. View results
cat fixtures/eval_results_appendable.md
```

---

## Summary

| Task | Command |
|------|---------|
| **Run evaluation (recommended)** | `python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3` |
| **Resume after crash** | Same command (auto-resumes) |
| **Backup results** | `mv fixtures/eval_results_appendable.md fixtures/eval_results_backup_$(date +%Y%m%d).md` |
| **Clear results** | `rm fixtures/eval_results_appendable.md` |
| **View results** | `cat fixtures/eval_results_appendable.md` |
| **Monitor memory** | `watch -n 1 free -h` |

---

**See Also:**
- [Ground Truth Dataset Documentation](GROUND_TRUTH_DATASET.md)
- [fixtures/README.md](../fixtures/README.md)
