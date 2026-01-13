# CodeGrapher Phase 12 Evaluation Results (Appendable)

This file accumulates results from individual evaluation runs. Each task is added independently as it's evaluated.

**Targets:** Token Savings ≥30%, Recall ≥85%, Precision ≤40%

---

## task_001: Fix KeyError with --import-mode=importlib in nested directories with same-named subdirectories

**Repo:** [pytest](https://github.com/pytest-dev/pytest)
**Status:** ✅ PASS
**Run Date:** 2026-01-13 18:20:48

### Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Token Savings | 87.1% | ≥30% | ✅ |
| Recall | 100.0% | ≥85% | ✅ |
| Precision | 25.0% | ≤40% | ✅ |

**Baseline Tokens:** 704,104
**CodeGrapher Tokens:** 91,095
**Files Returned:** 8

### Files

**Expected Files:** `src/_pytest/pathlib.py`, `testing/test_pathlib.py`
- ✅ Found: `src/_pytest/pathlib.py`, `testing/test_pathlib.py`
- ❌ Missed: None

**All Returned Files:** `src/_pytest/pathlib.py`, `testing/test_pathlib.py`, `testing/test_pluginmanager.py`, `src/_pytest/config/__init__.py`, `testing/acceptance_test.py`, `testing/_py/test_local.py`, `src/_pytest/_py/path.py`, `testing/test_collection.py`

---

