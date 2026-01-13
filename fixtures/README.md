# Ground Truth Test Dataset

**Quick Reference** - See [docs/GROUND_TRUTH_DATASET.md](../docs/GROUND_TRUTH_DATASET.md) for full documentation

---

## Files

| File | Cases | Description |
|------|-------|-------------|
| **ground_truth.jsonl** | 23 | **Main dataset** - Use this for evaluation |
| ground_truth_large.jsonl | 6 | Large repos (>50k LOC) - Future testing |
| ground_truth_unresolved.jsonl | 4 | Unresolved commits - Reference only |
| ground_truth.jsonl.backup | 20 | Original before processing |

---

## Run Evaluation

```bash
# Quick start (recommended) - Robust batch runner
cd /home/mikhailarutyunov/projects/codegrapher
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3

# Alternative: Direct evaluation (single run)
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --mixed --ground-truth fixtures/ground_truth.jsonl
```

**Modes:**
- `--simulate` - No API costs (tests infrastructure only)
- `--mixed` - 5 real API calls, rest simulated (default)
- `--real-api` - All real API calls (most accurate, higher costs)

**Output:** `fixtures/eval_results_appendable.md` + console

---

## Dataset Stats

| Metric | Value |
|--------|-------|
| Total cases | 23 |
| Repositories | 8 (click, httpx, jinja, pytest, starlette, flask, werkzeug, typer) |
| Max repo size | 41k LOC (all ≤50k) |
| Categories | error (35%), symbol (30%), dependency (17%), description (17%) |

---

## Query Categories

- **error** (8) - Error keywords, exceptions
- **symbol** (7) - Function/class names
- **dependency** (4) - Refactoring/structural
- **description** (4) - Natural language

Query terms extracted from user-facing issue descriptions (NOT code diffs) to ensure realistic agent queries.

---

**Full Documentation:** [docs/GROUND_TRUTH_DATASET.md](../docs/GROUND_TRUTH_DATASET.md)
