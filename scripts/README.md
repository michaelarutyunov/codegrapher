# CodeGrapher Scripts Guide

This directory contains all scripts for test case generation, validation, evaluation, and benchmarking.

**Organization:** Scripts are grouped into 5 numbered workflow phases for easy discovery and understanding.

---

## Directory Structure

```
scripts/
├── 01_mining/           # Mine test cases from GitHub
├── 02_validation/       # Validate and process test cases
├── 03_evaluation/       # Run evaluations and measure performance
├── 04_benchmarking/     # Performance benchmarks
└── 05_testing/          # Test infrastructure and safety checks
```

---

## Workflow Phases

### Phase 1: Mining & Research (`01_mining/`)

Mine test cases from GitHub repositories using real-world commits.

**Scripts:**
- **`mine_test_cases.py`** - Mine test cases from GitHub repos using `gh` CLI
  - Extracts realistic query terms from closed issues with merged PRs
  - Categorizes queries (symbol, description, error, agent-error, dependency)
  - Usage: `python scripts/01_mining/mine_test_cases.py --repos pallets/click --target-per-repo 5`

- **`research_commits.py`** - Research and resolve correct commit SHAs for ground truth tasks
  - Clones repos and searches git history for matching commits
  - Resolves branch names and short SHAs to full 40-char SHAs

---

### Phase 2: Validation & Processing (`02_validation/`)

Validate mined test cases and prepare them for evaluation.

**Scripts:**
- **`validate_mined_cases.py`** - Validate mined test cases for correctness
  - Checks: valid 40-char commit SHAs, correct parent-child relationships, accurate files_edited
  - Verifies repo size ≤50k LOC, valid schema compliance
  - Usage: `python scripts/02_validation/validate_mined_cases.py --input fixtures/ground_truth_mined.jsonl`

- **`resolve_branch_names.py`** - Resolve branch names to commit SHAs using GitHub CLI
  - Updates JSONL with resolved commits

- **`add_query_categories.py`** - Categorize test cases based on query_terms
  - Categories: symbol, description, error, agent-error, dependency

- **`separate_large_repos.py`** - Separate test cases by repository size
  - Moves >50k LOC cases to `ground_truth_large.jsonl`
  - Keeps ≤50k LOC in `ground_truth.jsonl`

- **`validate_and_fix_ground_truth.py`** - Final validation and fixing of ground truth data
  - Checks commit SHA validity, resolves short SHAs to full ones
  - Identifies branch names needing manual resolution

---

### Phase 3: Evaluation (`03_evaluation/`)

Evaluate CodeGrapher's token-saving capabilities on real-world tasks.

**Scripts:**
- **`run_eval_robust.py`** - ⭐ **RECOMMENDED** WSL-robust batch evaluation runner
  - Runs evaluation in batches with auto-retry on failures
  - Checkpoint/resume support, memory cleanup between batches
  - Handles WSL disconnections gracefully
  - **Usage:** `python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3`

- **`evaluate_ground_truth.py`** - Direct evaluation script (single run)
  - Measures token savings (target ≥30%), recall (≥85%), and precision (≤40%)
  - Supports mixed/simulated/real API modes with checkpoints
  - Usage: `python scripts/03_evaluation/evaluate_ground_truth.py --mixed --ground-truth fixtures/ground_truth.jsonl`

- **`run_eval_batches.sh`** - Shell script wrapper for batch evaluation
  - Alternative batch runner (simpler than Python version)
  - Automates checkpoint resumption to avoid WSL memory issues

- **`research_remaining.py`** - Continuation script for researching remaining tasks
  - Works with cloned repos in /tmp

---

### Phase 4: Benchmarking (`04_benchmarking/`)

Benchmark CodeGrapher performance against PRD targets.

**Scripts:**
- **`benchmark.py`** - Performance benchmarking for v1.0
  - Verifies all PRD targets: cold start ≤2s, query latency ≤500ms, incremental index ≤1s
  - Tests: full index 30k LOC ≤30s, RAM idle ≤500MB, disk overhead ≤1.5× repo size
  - Requires: hyperfine and psrecord

- **`benchmark_incremental.py`** - Performance benchmark for incremental indexing
  - Verifies Phase 8 acceptance criteria (incremental update <200ms for small changes)

---

### Phase 5: Testing & Utilities (`05_testing/`)

Unit tests and infrastructure safety checks.

**Scripts:**
- **`test_benchmark.py`** - Smoke test for benchmark.py infrastructure
  - Tests without requiring hyperfine

- **`kill_test.py`** - Kill test for atomic transaction safety
  - Verifies atomic_update() correctly rolls back SQLite and FAISS on process crash

---

## Quick Start

### Mine New Test Cases

```bash
# Mine 5 test cases per repository
python scripts/01_mining/mine_test_cases.py \
  --repos pallets/click encode/httpx \
  --target-per-repo 5 \
  --output fixtures/ground_truth_mined.jsonl

# Validate mined cases
python scripts/02_validation/validate_mined_cases.py \
  --input fixtures/ground_truth_mined.jsonl
```

### Run Evaluation

```bash
# Recommended: Robust batch runner (handles WSL disconnections)
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3

# Alternative: Direct evaluation (single run)
python scripts/03_evaluation/evaluate_ground_truth.py --mixed --ground-truth fixtures/ground_truth.jsonl

# If WSL disconnects, just re-run the same command - it automatically resumes!
```

### Run Benchmarks

```bash
# Performance benchmarks
python scripts/04_benchmarking/benchmark.py

# Incremental index benchmarks
python scripts/04_benchmarking/benchmark_incremental.py
```

---

## Common Workflows

### Complete Test Case Generation Pipeline

```bash
# 1. Mine cases from GitHub
python scripts/01_mining/mine_test_cases.py --repos pallets/click --target-per-repo 5

# 2. Validate mined cases
python scripts/02_validation/validate_mined_cases.py --input fixtures/ground_truth_mined.jsonl

# 3. Categorize queries
python scripts/02_validation/add_query_categories.py --input fixtures/ground_truth_mined.jsonl

# 4. Separate large repos
python scripts/02_validation/separate_large_repos.py --input fixtures/ground_truth.jsonl

# 5. Run evaluation
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3
```

### Fix and Validate Existing Ground Truth

```bash
# 1. Resolve commit SHAs
python scripts/02_validation/resolve_branch_names.py --input fixtures/ground_truth.jsonl

# 2. Validate and fix
python scripts/02_validation/validate_and_fix_ground_truth.py --output fixtures/ground_truth_fixed.jsonl

# 3. Separate by size
python scripts/02_validation/separate_large_repos.py --input fixtures/ground_truth_fixed.jsonl
```

---

## File Organization Benefits

**Numbered Prefixes (01-05):**
- Show the natural workflow progression
- Make it easy to understand the order of operations
- Self-documenting architecture

**Category Grouping:**
- Related scripts are together
- Easy to find scripts by purpose
- Scalable for future additions

---

## Additional Documentation

- **Evaluation Guide:** [docs/EVALUATION_GUIDE.md](../docs/EVALUATION_GUIDE.md)
- **Ground Truth Dataset:** [docs/GROUND_TRUTH_DATASET.md](../docs/GROUND_TRUTH_DATASET.md)
- **Quick Reference:** [fixtures/README.md](../fixtures/README.md)

---

**Updated:** 2026-01-13
