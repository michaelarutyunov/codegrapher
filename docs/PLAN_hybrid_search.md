# Hybrid Retrieval Implementation Plan

**Status:** ✅ Phase 1 Complete | ⚠️ Phase 2 Partial | 📋 Phase 3 Pending
**Created:** 2026-01-14
**Last Updated:** 2026-01-14
**Author:** CodeGrapher Team
**Related:** [PLAN_v1.md](PLAN_v1.md) (completed), [PROGRESS.md](PROGRESS.md) (tracking)

---

## Implementation Status Summary

### Completed (2026-01-14)

**Phase 1: Minimal Viable Hybrid** ✅ COMPLETE
- ✅ 1.0 Query preprocessing (noise filtering)
- ✅ 1.1 rank-bm25 dependency added
- ✅ 1.2 SparseIndex class implemented
- ✅ 1.3 BM25Searcher class implemented
- ✅ 1.4 Tokenization + filename matching
- ✅ 1.5 RRF-based merge (instead of simple score-based)
- ✅ 1.6 Updated codegraph_query with hybrid pipeline
- ✅ 1.7 A/B Test on 4-task pretest (ground_truth_pretest.jsonl)

**Phase 2: Production Hardening** ⚠️ MOSTLY COMPLETE
- ✅ 2.2 Sparse index builds during indexing
- ✅ 2.3 Sparse index loads at startup
- ✅ 2.4 RRF merge implemented (k=60)
- ✅ 2.5 Test-source pairing logic
- ✅ 2.6 Advanced tokenization (CamelCase, dotted.module, ALL_CAPS, etc.)
- ⚠️ 2.1 sparse_terms table (deferred - using in-memory tokenization)
- ⏳ 2.7 Full A/B Test (pending - needs full ground_truth.jsonl run)

**Phase 3: Optimization** 📋 NOT STARTED
- ⏳ 3.1 Query latency baseline measurement
- ⏳ 3.2 Parallel search (conditional on latency >80ms)
- ⏳ 3.3 Weight tuning (if needed)
- ⏳ 3.4 Final A/B Test

### Critical Bug Fixes Applied

1. **Import Closure Pruning (server.py:660)**
   - Problem: Filename-matched symbols pruned when not in import closure
   - Fix: Preserve filename-matched symbols regardless of import closure
   - Impact: task_032 improved 0% → 75% recall

2. **Case Sensitivity (sparse_index.py:311)**
   - Problem: "TestClient" didn't match "testclient.py"
   - Fix: Case-insensitive filename matching
   - Impact: task_040 improved 0% → 50% recall

3. **Cursor File Priority (server.py:644-657)**
   - Problem: Semantic search fails to find cursor file itself
   - Fix: Defensive inclusion of cursor file when not already present
   - Impact: task_039 improved 0% → 50% recall

### Pretest Results (4 tasks)

| Task | Description | Before | After | Fix |
|------|-------------|--------|-------|-----|
| task_032 | Move utils to client | 0% | 75% | Import closure pruning |
| task_039 | async http_exception | 0% | 50% | Cursor file priority |
| task_040 | TestClient timeout | 0% | 50% | Case sensitivity |
| task_028 | compile_templates | 0% | 0% | *Still failing* |

### Remaining Work

1. **Full A/B Test** (Step 2.7): Run on complete ground_truth.jsonl dataset
2. **task_028 Root Cause**: Semantic gap requires stemming or fuzzy matching
3. **Phase 3**: Performance measurement and optimization

---

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Approach](#technical-approach)
- [Implementation Phases](#implementation-phases)
- [Acceptance Criteria](#acceptance-criteria)
- [Testing Strategy](#testing-strategy)
- [Rollback Plan](#rollback-plan)
- [References](#references)

---

## Overview

This plan implements **hybrid retrieval** (dense + sparse search) for CodeGrapher to improve recall when semantic search fails. The addition of keyword-based BM25 search complements the existing vector embedding search, with results merged using **Reciprocal Rank Fusion (RRF)**.

**Scope:** Post-v1.0 enhancement to improve retrieval quality without breaking changes.

**Effort Estimate:** 4-7 days development + 1-2 days testing

---

## Motivation

### Current Problem

CodeGrapher v1.0 uses dense (semantic) retrieval only:
- Query → Embedding → FAISS search → Results
- Works well when semantic match is strong
- **Fails when:** Query has poor embedding, rare terms, or exact symbol matching needed

### Ground Truth Evidence

From Phase 12 evaluation (`fixtures/ground_truth.jsonl`):

| Task | Query Type | Dense Recall | Issue | BM25 Helps? |
|------|-----------|-------------|-------|-------------|
| task_001 | Error keywords | 100% | ✅ Working | - |
| task_002 | Symbol names | TBD | Test-file cursor | Partially (test-source pairing) |
| task_003 | AST concepts | TBD | Semantic weak | Partially |
| task_028 | Generic terms | 0% | "e.g" dilutes signal | Partially (preprocessing) |
| task_032 | Filename-only | 0% | Symbols don't contain filenames | **No** (needs file matching) |
| task_039 | Path ambiguity | TBD | Multiple same-named files | Partially |
| task_040 | Generic terms | TBD | Returns wrong files | Partially |
| task_006 | Test-file cursor | Partial | Import closure prunes source | Partially (test-source pairing) |

**Root cause:** If vector search returns irrelevant candidates at Step 2, downstream scoring (Step 4) cannot recover. Additionally, three specific failure modes identified:

1. **Filename-only queries:** BM25 searches symbol text, not file paths (task_032)
2. **Query noise:** Filler terms ("e.g", "i.e") dilute signal (task_028)
3. **Test-source pairing:** Import closure is unidirectional (test → imports), missing source → test relationship (task_002, task_006)

### Why Hybrid Helps

```
┌─────────────────────────────────────────────────────────────┐
│ Query: "import_module_using_spec KeyError sys.modules"       │
├─────────────────────────────────────────────────────────────┤
│ Dense (embedding):  Finds semantically related code         │
│ Sparse (BM25):       Finds exact symbol names, keywords     │
│                                                              │
│ Together: Coverage when either method fails                 │
└─────────────────────────────────────────────────────────────┘
```

**Industry validation:** Hybrid retrieval is standard practice:
- Elasticsearch (dense + sparse)
- Pinecone (sparse-dense fusion)
- Verba (RRF-based hybrid)

---

## Technical Approach

### Retrieval Methods

| Method | Implementation | Strengths | Weaknesses |
|--------|----------------|-----------|------------|
| **Dense (existing)** | FAISS IndexFlatL2 + jina-embeddings-v2-base-code | Semantic understanding, synonyms | Poor on rare/exact terms |
| **Sparse (new)** | BM25 over tokenized symbol signatures | Exact match, rare terms, code symbols | No semantic understanding |

### Merge Strategy: Reciprocal Rank Fusion (RRF)

**Formula:**
```
RRF_score(item) = Σ (k / (k + rank_position))

Where k = 60 (constant), rank = 1, 2, 3, ...
```

**Why RRF:**
- Score-scale invariant (works with different scoring systems)
- No normalization needed
- Proven robust in research
- Handles items in only one ranking

**Alternative considered:** Weighted score averaging
- Rejected: Requires score normalization, sensitive to weight tuning

### Architecture

```
User Query
    │
    ├─► Dense Search ──────┐
    │   (FAISS)            │
    │                      ├─► RRF Merge ─► Ranked Results
    │                      │
    └─► Sparse Search ─────┘
        (BM25)
```

---

## Implementation Phases

### Phase 1: Minimal Viable Hybrid (1-2 days)

**Goal:** Get sparse search working and validate recall improvement

| Step | Task | File Changes | Acceptance |
|------|------|--------------|------------|
| 1.0 | Add query preprocessing | `src/codegrapher/server.py` (new function) | Noise filtered, technical terms preserved |
| 1.1 | Add `rank_bm25` dependency | `pyproject.toml` | Install succeeds |
| 1.2 | Create `SparseIndex` class | `src/codegrapher/sparse_index.py` (new) | In-memory index builds |
| 1.3 | Create `BM25Searcher` class | `src/codegrapher/sparse_index.py` | Search returns results |
| 1.4 | Add tokenization + filename matching | `src/codegrapher/sparse_index.py` | Handles tokens, underscore modules, AND file paths |
| 1.5 | Simple score-based merge | `src/codegrapher/server.py` | Results merge |
| 1.6 | Update `codegraph_query` | `src/codegrapher/server.py` | Calls both searches |
| 1.7 | **A/B Test** | Run on 2-3 ground truth tasks | Recall > baseline |

**Deliverables:**
- Working sparse search with filename matching
- Query preprocessing for noise reduction
- First A/B results showing recall impact

---

### Phase 2: Production Hardening (2-3 days)

**Goal:** Make sparse index persistent and improve merge robustness

| Step | Task | File Changes | Acceptance |
|------|------|--------------|------------|
| 2.1 | Add `sparse_terms` table schema | `src/codegrapher/models.py` | Migration works |
| 2.2 | Build sparse index during indexing | `src/codegrapher/indexer.py` | Persisted to DB |
| 2.3 | Load sparse index at startup | `src/codegrapher/sparse_index.py` | Loads from DB |
| 2.4 | Implement RRF merge | `src/codegrapher/server.py` | Rank-based scoring |
| 2.5 | Add test-source pairing logic | `src/codegrapher/server.py` (new function) | Test queries include source files |
| 2.6 | Add tokenization improvements | `src/codegrapher/sparse_index.py` | CamelCase, dotted.module |
| 2.7 | **Full A/B Test** | Run on all ground truth tasks | Target: 60-80% recall |

**Deliverables:**
- Persistent sparse index
- RRF-based merging
- Test-source pairing for bidirectional import awareness
- Improved tokenization

---

### Phase 3: Optimization (1-2 days)

**Goal:** Performance and monitoring

| Step | Task | File Changes | Acceptance |
|------|------|--------------|------------|
| 3.1 | Measure query latency baseline | `src/codegrapher/server.py` | Log dense/sparse/merge times |
| 3.2 | Parallel search (IF median >80ms) | `src/codegrapher/server.py` | Both searches run concurrently |
| 3.3 | Weight tuning (if needed) | `src/codegrapher/server.py` | Optimize RRF constant |
| 3.4 | **Final A/B Test** | Run full evaluation | All criteria met |

**Deliverables:**
- Performance metrics and baseline
- Parallel search ONLY if latency warrants it
- Final validation

**Note on parallelization:** Threading has ~1-2ms overhead. Only implement if:
- Sequential median latency >80ms, OR
- Dense search >40ms AND sparse search >40ms
Otherwise, sequential execution is simpler and equally fast.

---

## Acceptance Criteria

### Functional Requirements

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Recall improvement** | ≥ baseline | Ground truth A/B test |
| **No regression** | Existing tests pass | `pytest tests/` |
| **Sparse index builds** | Without errors | Manual verification |
| **Persistent index** | Survives restart | Query after restart works |

### Performance Requirements

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Query latency** | <120ms (sequential OK) | Time from query to results |
| **Parallel search** | Only if median >80ms | Measured baseline first |
| **Index build time** | +10% over baseline | Build time comparison |
| **Memory overhead** | <50MB additional | Memory profiling |
| **Index size** | <5MB on disk | Database file size |

**Note:** Parallelization (Phase 3.2) is conditional. Implement only if measured sequential latency exceeds 80ms median. Otherwise, sequential execution is simpler and equally fast.

### Quality Requirements

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Code coverage** | >80% for new code | pytest coverage |
| **Type checking** | No new errors | mypy |
| **Documentation** | All public functions documented | Docstrings present |

---

## Testing Strategy

### Smoke Tests (Development)

```python
# tests/test_sparse_index.py
def test_sparse_index_build():
    """Verify sparse index builds from symbols."""
    index = SparseIndex()
    index.add_symbols(test_symbols)
    assert len(index) > 0

def test_bm25_search():
    """Verify BM25 search returns results."""
    searcher = BM25Searcher(index)
    results = searcher.search(["import", "module"])
    assert len(results) > 0

def test_rrf_merge():
    """Verify RRF merge combines rankings."""
    dense = [("sym_a", 0.9), ("sym_b", 0.7)]
    sparse = [("sym_b", 0.8), ("sym_c", 0.6)]
    merged = merge_rrf(dense, sparse)
    assert merged[0][0] == "sym_b"  # Should rank first
```

### Ground Truth A/B Tests (Validation)

```bash
# Before hybrid (baseline)
python scripts/03_evaluation/run_eval_robust.py --mode simulate --output before.json

# After Phase 1
python scripts/03_evaluation/run_eval_robust.py --mode simulate --output phase1.json

# After Phase 2
python scripts/03_evaluation/run_eval_robust.py --mode simulate --output phase2.json

# After Phase 3
python scripts/03_evaluation/run_eval_robust.py --mode simulate --output final.json

# Compare results
python scripts/compare_eval_results.py before.json final.json
```

### Regression Guard

After each phase, run full test suite:
```bash
pytest tests/ -v
```

Expected: All existing tests still pass (no breaking changes).

### Edge Case Validation (New Tests)

Specific test cases derived from evaluation failures:

| Test Case | Query Input | Expected Behavior | Verify Against |
|-----------|-------------|-------------------|----------------|
| **Filename-only** | `["_utils.py", "_client.py"]` | Returns symbols from those files | `task_032` |
| **Underscore module** | `["_utils", "KeyError"]` | Matches `_utils` symbols | Internal modules pattern |
| **Noise filtering** | `["e.g", "compile_templates"]` | Searches only for "compile_templates" | `task_028` |
| **Test-source pair** | Query from `test_pathlib.py` | Includes `src/pathlib.py` symbols | `task_002`, `task_006` |
| **Dotted module** | `["sys.modules"]` | Matches module references | Existing tests |
| **Rare symbol** | `["import_module_using_spec"]` | Exact match works | `task_001` |
| **CamelCase type** | `["FloatOperation"]` | Matches type names | `task_002` |

These tests validate the key additions:
1. Query preprocessing removes noise without filtering technical terms
2. Filename matching finds symbols when query contains `.py` files
3. Underscore tokenization handles internal/private modules
4. Test-source pairing augments results when cursor is in a test file

---

## Rollback Plan

### If Recall Doesn't Improve

**Diagnosis:**
1. Check sparse index is built (`SELECT COUNT(*) FROM sparse_terms`)
2. Check BM25 returns results (manual query)
3. Check merge includes sparse results (debug logging)

**Rollback options:**
- **Option A:** Disable sparse in config (feature flag)
- **Option B:** Revert `codegraph_query` to dense-only
- **Option C:** Use git revert on specific commits

### If Performance Degrades

**Thresholds:**
- Query latency >200ms → Investigate bottleneck
- Index build time +50% → Optimize sparse indexing
- Memory +100MB → Review index caching strategy

**Mitigation:**
- Parallel search (Phase 3.1)
- Sparse index lazy-loading
- Reduce BM25 candidate count (k=20 → k=10)

### If Data Corruption

**Recovery:**
1. Delete `.codegraph/symbols.db` (sparse terms in same DB)
2. Run `codegraph build --full`
3. Sparse index rebuilds from scratch

---

## References

### External Resources

- **RRF Paper:** [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- **BM25:** [BM25 documentation](https://en.wikipedia.org/wiki/Okapi_BM25)
- **rank_bm25:** [GitHub repository](https://github.com/dorianbrown/rank_bm25)

### Internal Documents

- [PRD Section 6](PRD_project_codegrapher_v1.0.md#section-6-mcp-server-interface) - MCP server interface
- [PROGRESS.md Phase 12](PROGRESS.md#phase-12-testing--evaluation) - Ground truth evaluation
- [Ground Truth Dataset](GROUND_TRUTH_DATASET.md) - Test cases and metrics

### Code Files to Modify

| File | Changes | Complexity |
|------|---------|------------|
| `pyproject.toml` | Add `rank-bm25>=0.2.2` dependency | Trivial |
| `src/codegrapher/models.py` | Add `sparse_terms` table schema | Low |
| `src/codegrapher/sparse_index.py` | NEW: SparseIndex, BM25Searcher classes | Medium |
| `src/codegrapher/indexer.py` | Build sparse index during indexing | Low |
| `src/codegrapher/server.py` | Query preprocessing, RRF merge, test-source pairing, parallel search (conditional) | Medium |
| `tests/test_sparse_index.py` | NEW: Sparse index tests | Low |

### New Functions to Implement

| Function | Location | Purpose |
|----------|----------|---------|
| `preprocess_query()` | `server.py` | Remove noise terms (conservative filtering) |
| `SparseIndex.__init__()` | `sparse_index.py` | Initialize in-memory inverted index |
| `SparseIndex.add_symbols()` | `sparse_index.py` | Build inverted index from symbols |
| `BM25Searcher.__init__()` | `sparse_index.py` | Initialize BM25 with index |
| `BM25Searcher.search()` | `sparse_index.py` | Return ranked symbol IDs by query tokens |
| `augment_with_filename_matches()` | `sparse_index.py` or `server.py` | Add symbols from .py files in query |
| `merge_rrf()` | `server.py` | Reciprocal Rank Fusion of dense + sparse |
| `is_test_source_pair()` | `server.py` | Check if files are test-source pairs |
| `augment_with_test_source_pairs()` | `server.py` | Add source files when cursor in test |
| `parallel_search()` | `server.py` (CONDITIONAL) | Parallel dense+sparse IF latency >80ms |

---

## Appendix: Technical Deep Dives

### A. RRF Formula Explained

```
For each item appearing in rankings:
  RRF_score = Σ (k / (k + rank_position))

Example with k=60:
  Item A: rank 1 in dense, rank 2 in sparse
    score = 60/(60+1) + 60/(60+2) = 0.984 + 0.968 = 1.95

  Item B: rank 1 in sparse only
    score = 0 + 60/(60+1) = 0.984

Result: A ranks higher (confirmed by both systems)
```

### B. BM25 Parameters

```python
BM25(k1=1.5, b=0.75, epsilon=0.25)

k1=1.5  # Term frequency saturation (higher = less saturation)
b=0.75  # Length normalization (0=ignore, 1=full normalization)
epsilon=0.25  # Floor for idf values (prevents division by zero)
```

Defaults work well for code; tuning usually unnecessary.

### C. Tokenization Strategy

```python
def tokenize_symbol(symbol: Symbol) -> List[str]:
    """Extract searchable tokens from symbol."""
    tokens = []

    # 1. Signature dotted.module.pattern
    tokens.extend(re.findall(r"\b[a-z_][a-z_0-9]*\.[a-z_][a-z_0-9]*\b", symbol.signature))

    # 2. CamelCase symbols
    tokens.extend(re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", symbol.signature))

    # 3. snake_case symbols
    tokens.extend(re.findall(r"\b[a-z][a-z_0-9]*_[a-z_][a-z_0-9]*\b", symbol.signature))

    # 4. Underscore-prefixed modules (internal/private)
    # Matches: _utils, _client, _internal
    # Skips: __init__, __name__ (double underscore pattern)
    tokens.extend(re.findall(r"\b_[a-z][a-z_0-9]*\b", symbol.file))

    # 5. Alphanumeric words
    tokens.extend(re.findall(r"\b[a-zA-Z]{3,}\b", symbol.signature))

    # 6. Docstring tokens (if present)
    if symbol.doc:
        tokens.extend(re.findall(r"\b[a-zA-Z]{3,}\b", symbol.doc[:200]))

    return [t.lower() for t in tokens if len(t) > 2]
```

**Why underscore tokenization matters:** Many Python projects use underscore-prefixed modules (`_utils.py`, `_client.py`) for internal code. Users explicitly search for these by name, so they should be searchable.

**Pattern detail:** `r"\b_[a-z][a-z_0-9]*\b"` matches single-underscore identifiers but not double-underscore dunders (`__init__`, `__name__`), which are Python magic attributes and typically not searched by name.

### D. Query Preprocessing (Noise Filtering)

**Problem:** Queries contain filler terms that dilute signal.

```python
def preprocess_query(query: str) -> str:
    """Remove noise terms while preserving technical keywords."""
    # Conservative noise list - DO NOT filter valid code terms
    FILLER_TERMS = {
        'e.g', 'i.e', 'etc', 'eg', 'ie', 'aka',
        'please', 'thanks', 'help', 'github', 'issue'
        # NOTE: Deliberately NOT including: async, function, class, def, etc.
    }

    words = query.split()
    cleaned = [w for w in words if w.lower() not in FILLER_TERMS]
    return ' '.join(cleaned)

# Example:
# Input:  "e.g. fix async function KeyError in importlib"
# Output: "fix async function KeyError importlib"
#          ^^^^ removed (noise)    ^^^^^^^^ technical terms preserved
```

**Why conservative filtering:** Terms like "async", "function", "class" are valid technical search terms. Over-filtering loses information.

### E. Filename Matching Logic

**Problem:** Filename-only queries (e.g., `["_utils.py"]`) won't match symbols because filenames aren't tokenized.

```python
def augment_with_filename_matches(
    query_tokens: List[str],
    all_symbols: List[Symbol],
    initial_results: Set[str]
) -> Set[str]:
    """Add symbols from files mentioned in query."""
    result_ids = set(initial_results)

    # Check if any token is a filename
    for token in query_tokens:
        if token.endswith('.py'):
            # Find all symbols in this file
            for symbol in all_symbols:
                if token in symbol.file:
                    result_ids.add(symbol.id)

    return result_ids

# Example:
# Query: ["_utils.py", "_client.py"]
# Sparse BM25 finds: 0 results (no "_utils.py" in symbol text)
# Filename matcher adds: all symbols from _utils.py and _client.py
```

**Implementation:** Post-process sparse results to add filename matches before merge.

### F. Test-Source Pairing Logic

**Problem:** Import closure is unidirectional. If cursor is in test file, only test imports are included, missing the source file being tested.

```python
def is_test_source_pair(test_file: str, source_file: str) -> bool:
    """Check if two files are test-source pairs."""
    # Pattern 1: test_ prefix in same directory
    if test_file == f"test_{source_file}":
        return True

    # Pattern 2: tests/ mirrors src/ structure
    if test_file.startswith('tests/') and source_file.startswith('src/'):
        test_suffix = test_file.replace('tests/', '').replace('test_', '')
        source_suffix = source_file.replace('src/', '')
        return test_suffix == source_suffix or test_suffix == f"test_{source_suffix}"

    # Pattern 3: _test.py suffix
    if test_file == f"{source_file}_test":
        return True

    return False

def augment_with_test_source_pairs(
    cursor_file: str,
    candidates: List[Symbol],
    all_symbols: List[Symbol]
) -> List[Symbol]:
    """If cursor is in test file, include corresponding source."""
    if 'test' not in cursor_file.lower():
        return candidates

    candidate_files = {s.file for s in candidates}
    augmented = list(candidates)

    for symbol in all_symbols:
        if symbol.file in candidate_files:
            continue
        if is_test_source_pair(cursor_file, symbol.file):
            augmented.append(symbol)

    return augmented

# Example:
# Cursor: test_pathlib.py
# Import closure finds: test_pathlib.py only
# Test-source pairing adds: src/pathlib.py (the file being tested)
```

---

*Last Updated: 2026-01-14*

**Revision History:**
- 2026-01-14: Initial plan created
- 2026-01-14: Updated to address three evaluation failure modes:
  - Added query preprocessing (Step 1.0)
  - Added filename matching (enhanced Step 1.4)
  - Added test-source pairing (new Step 2.5)
- 2026-01-14: Technical refinements based on peer review:
  - Added underscore-prefixed module tokenization (pattern #4 in tokenization)
  - Updated ground truth table with real evaluation data (tasks 028, 032, 039, 040, 006)
  - Changed parallel search to conditional (measure first, implement if >80ms median)
- 2026-01-14: **PHASE 1 COMPLETE** - Implementation finished with three critical bug fixes:
  - Import closure pruning: Preserve filename-matched symbols (task_032: 0%→75%)
  - Case sensitivity: Case-insensitive filename matching (task_040: 0%→50%)
  - Cursor file priority: Defensive cursor file inclusion (task_039: 0%→50%)
  - Deferred sparse_terms table (using in-memory tokenization instead)
