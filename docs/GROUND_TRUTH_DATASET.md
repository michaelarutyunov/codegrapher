# CodeGrapher Test Dataset

**Status:** ✅ Complete (23 verified test cases)
**Last Updated:** 2026-01-13

---

## Quick Start

### Run Evaluation

```bash
cd /home/mikhailarutyunov/projects/codegrapher

# RECOMMENDED: Robust batch runner (handles WSL disconnections)
python scripts/03_evaluation/run_eval_robust.py --mode mixed --batch-size 3

# Alternative modes with robust runner
python scripts/03_evaluation/run_eval_robust.py --mode simulate --batch-size 5  # No API costs
python scripts/03_evaluation/run_eval_robust.py --mode real-api --batch-size 2  # All real API (slower batches)

# Direct evaluation (single run, no batching)
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --mixed --ground-truth fixtures/ground_truth.jsonl
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --simulate --ground-truth fixtures/ground_truth.jsonl
.venv/bin/python scripts/03_evaluation/evaluate_ground_truth.py --real-api --ground-truth fixtures/ground_truth.jsonl
```

**Output:**
- `fixtures/eval_results_appendable.md` - Markdown report
- Console output with metrics

**Features:**
- Automatic checkpoint/resume (saved to `/tmp/eval_checkpoint_*.json`)
- Safe to Ctrl+C and restart
- WSL-compatible with robust cleanup

---

## Dataset Overview

### Main Dataset: 23 Cases

**File:** `fixtures/ground_truth.jsonl`

| Metric | Value |
|--------|-------|
| **Total cases** | 23 |
| **Repositories** | 8 (click, httpx, jinja, pytest, starlette, flask, werkzeug, typer) |
| **Max repo size** | 40,966 LOC (pytest) |
| **All ≤50k LOC** | ✅ Yes (FAISS IndexFlatL2 constraint) |
| **Valid commit SHAs** | ✅ 100% (40-char format) |

**Query Category Distribution:**

| Category | Count | % | Description |
|----------|-------|---|-------------|
| error | 8 | 35% | Error keywords, exceptions |
| symbol | 7 | 30% | Function/class names |
| dependency | 4 | 17% | Refactoring/structural |
| description | 4 | 17% | Natural language |

**Repository Distribution:**

| Repo | Cases | LOC |
|------|-------|-----|
| jinja, click, httpx | 4 each | 10k-15k |
| pytest, starlette | 3 each | 14k-41k |
| flask, werkzeug | 2 each | 9k-18k |
| typer | 1 | ~8k |

### Additional Datasets

**Large Repos:** `fixtures/ground_truth_large.jsonl` - 6 cases
- Repos >50k LOC (pydantic 90k, fastapi 83k)
- For future testing with improved indexing

**Unresolved:** `fixtures/ground_truth_unresolved.jsonl` - 4 cases
- Could not auto-resolve commit SHAs
- Available for manual resolution if needed

---

## How This Dataset Was Created

### Two-Track Approach

**Track A: Validate Existing Cases (10 cases)**
1. **Resolve commit SHAs** - Used `gh` CLI to find PR merge commits and parent refs
2. **Categorize queries** - Analyzed term patterns (symbols, errors, descriptions)
3. **Filter by size** - Separated repos >50k LOC
4. **Result:** 80% auto-resolution rate (16/20 original cases)

**Track B: Mine New Cases (13 cases)**
1. **GitHub mining** - Queried 461 merged PRs via `gh pr list`
2. **Filter candidates** - 2-5 files, 10-500 LOC changed, Python only
3. **Extract query terms** - From issue descriptions, NOT diffs (critical!)
4. **Validate** - Clone, verify commits, check file diffs
5. **Result:** 65% validation pass rate (13/20 mined)

### Critical Design Principle

**Query terms extracted from user-facing issue descriptions, NOT code diffs.**

**Why:** The agent doesn't have the diff yet - finding relevant files IS the purpose of the query. Extracting from diffs would be circular logic.

**Example:**
```
Issue: "Fix crash when using pytest.approx with Decimal"
Query terms: ["pytest.approx", "Decimal", "FloatOperation", "crash"] ✅
NOT from diff: ["ApproxDecimal", "test_approx.py", "__repr__"] ❌
```

---

## Scripts & Tools

All scripts are reusable for future test case generation:

| Script | Purpose | LOC |
|--------|---------|-----|
| `resolve_branch_names.py` | Resolve PR branches to commit SHAs | ~350 |
| `add_query_categories.py` | Categorize by query type | ~250 |
| `separate_large_repos.py` | Filter by LOC threshold | ~300 |
| `mine_test_cases.py` | Mine from GitHub repos | ~500 |
| `validate_mined_cases.py` | Validate mined cases | ~350 |

**Total:** ~1,750 LOC of test case generation infrastructure

### Mining New Cases

To mine additional cases from other repos:

```bash
# Mine 5 cases per repo
python3 scripts/mine_test_cases.py \
  --repos pallets/click encode/httpx \
  --target-per-repo 5 \
  --output fixtures/ground_truth_new.jsonl \
  --existing fixtures/ground_truth.jsonl

# Validate mined cases
python3 scripts/validate_mined_cases.py \
  --input fixtures/ground_truth_new.jsonl
```

---

## Key Achievements

1. **Automated at scale** - 23 cases vs 4 manual cases (5.75x improvement)
2. **Query realism** - Terms from user descriptions, not implementation
3. **High quality** - 100% valid commits, all repos within size limits
4. **Reusable tools** - Scripts work for future test generation
5. **Balanced categories** - Covers error, symbol, description, dependency queries

---

## Limitations & Future Work

### Current Limitations

1. **Missing agent-error category** (0%) - Traceback format rare in real issues
2. **Below target** - 23 cases vs target of 45
3. **Some low-quality terms** - ~3-5 cases need manual cleanup (e.g., task_026)

### Future Enhancements

**Expand Dataset:**
- Mine from additional repos
- Target underrepresented categories (dependency, agent-error)
- Create synthetic agent-error cases from tracebacks

**Improve Quality:**
- NLP-based term extraction
- Domain-specific vocabulary filtering
- Automated quality scoring

**Better Categorization:**
- Search by PR labels ("refactor" for dependency cases)
- Feature requests for description cases
- Bug reports already favor error cases

---

## Files Reference

**Active Files:**
- `fixtures/ground_truth.jsonl` - **Main dataset (23 cases)** ← Use this
- `fixtures/ground_truth_large.jsonl` - Large repos (6 cases)
- `fixtures/ground_truth_unresolved.jsonl` - Unresolved (4 cases)
- `fixtures/ground_truth.jsonl.backup` - Original backup

**Scripts:**
- `scripts/01_mining/mine_test_cases.py` - GitHub mining
- `scripts/01_mining/research_commits.py` - Research commit SHAs
- `scripts/02_validation/validate_mined_cases.py` - Validation pipeline
- `scripts/02_validation/resolve_branch_names.py` - Commit SHA resolution
- `scripts/02_validation/add_query_categories.py` - Query categorization
- `scripts/02_validation/separate_large_repos.py` - Size filtering
- `scripts/02_validation/validate_and_fix_ground_truth.py` - Fix ground truth
- `scripts/03_evaluation/evaluate_ground_truth.py` - Evaluation harness (single run)
- `scripts/03_evaluation/run_eval_robust.py` - **Robust batch runner (recommended for WSL)**
- `scripts/03_evaluation/run_eval_batches.sh` - Bash batch runner (alternative)
- `scripts/03_evaluation/research_remaining.py` - Research remaining tasks
- `scripts/04_benchmarking/benchmark.py` - Performance benchmarking
- `scripts/04_benchmarking/benchmark_incremental.py` - Incremental benchmarks
- `scripts/05_testing/test_benchmark.py` - Benchmark tests
- `scripts/05_testing/kill_test.py` - Safety tests

**Documentation:**
- `docs/GROUND_TRUTH_DATASET.md` - This file (comprehensive reference)

---

## Methodology Details

### Track A: Commit SHA Resolution

**Challenge:** Original dataset had PR branch names, short SHAs, not permanent refs.

**Solution:**
- `gh pr list --search "head:{branch}"` to find PRs by branch
- `gh api /repos/{owner}/{repo}/commits/{sha}` to get parent commits
- Expand short SHAs to full 40-char format
- **Result:** 16/20 auto-resolved (80%)

**Unresolved cases (4):** Likely squash merges where branch-to-commit mapping failed.

### Track B: Query Term Extraction

**Process:**
1. Remove code blocks and URLs from issue text
2. Extract in priority order:
   - Error types (`KeyError`, `Exception`)
   - Module.method patterns (`pytest.approx`, `sys.modules`)
   - snake_case identifiers (len >4)
   - CamelCase identifiers
   - Function calls (`validate()`)
   - Descriptive phrases (`async generator`)
3. Filter generic words (`bug`, `fix`, `github`)
4. Limit to 7 most relevant terms

**Quality filtering:**
- 65% validation pass rate for mined cases
- Manual review identified ~3 cases needing cleanup
- Most query terms are realistic and specific

### Track B: Validation Checks

For each mined case:
1. ✅ 40-char commit SHAs
2. ✅ Parent-child relationship: `git log --pretty=%P -n 1 {commit_after}`
3. ✅ files_edited matches: `git diff --name-only {before}...{after}`
4. ✅ Repo size ≤50k Python LOC
5. ✅ Schema compliance (all required fields)

**Why 35% failed:** GitHub merge strategies (squash/rebase) create commits without preserving parent relationships. This is expected and filtered out.

---

## Lessons Learned

### What Worked

✅ **`gh` CLI integration** - Handles auth, pagination, rate limits automatically
✅ **Issue-based extraction** - Captures realistic user queries
✅ **Automated validation** - Caught invalid parent relationships early
✅ **LOC filtering** - Prevented large repo issues

### What Didn't Work

❌ **baseRefOid not reliable** - Squash/rebase breaks parent assumptions
❌ **Broad regex patterns** - Initially extracted non-technical terms
❌ **Category balance hard** - Bug reports dominate GitHub (error category over-represented)

### Recommendations

1. **Target specific PR labels** for category balance
2. **Build vocabulary whitelist** for term extraction
3. **Create synthetic cases** for rare categories (agent-error, dependency)
4. **Accept lower yield** - 65% validation rate is typical for automated mining

---

## Dataset Schema

Each test case has:

```json
{
  "task_id": "task_001",
  "description": "Fix KeyError with --import-mode=importlib...",
  "repo": "https://github.com/pytest-dev/pytest",
  "commit_before": "326faa25f4e776f082eea5603d84b0812b57773c",
  "commit_after": "6486c3f3a858a0c8043f5c3f7c24297b82a0abe4",
  "cursor_file": "src/_pytest/pathlib.py",
  "query_terms": ["import_module_using_spec", "sys.modules", "importlib", ...],
  "query_category": "error",
  "files_edited": ["src/_pytest/pathlib.py", "testing/test_pathlib.py"],
  "expected_bundle_should_contain": ["src/_pytest/pathlib.py", ...],
  "baseline_tokens_sent": 0,
  "expected_tokens_with_codegraph": 0,
  "notes": "Bug fix: Ensures parent modules are registered..."
}
```

**Required fields:** task_id, description, repo, commit_before, commit_after, cursor_file, query_terms, files_edited

**Optional fields:** query_category, expected_bundle_should_contain, baseline_tokens_sent, expected_tokens_with_codegraph, notes
