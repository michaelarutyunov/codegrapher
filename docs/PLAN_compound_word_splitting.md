# Implementation Plan: Compound Word Splitting (Tier 1)

**Status:** 📋 Ready to Implement
**Created:** 2026-01-14
**Priority:** HIGH (Tier 1 - highest ROI optimization)
**Estimated Effort:** 4-6 hours
**Expected Impact:** +10-30% aggregate recall, task_028: 0% → 75-100%

---

## Executive Summary

Compound word splitting addresses the most common retrieval failure mode in the ground truth dataset. By splitting underscore-separated, CamelCase, and mixed identifiers into component tokens, BM25 can match partial terms that current tokenization misses.

**Why this is Tier 1:**
1. ✅ Benefits 12/24 tasks (50%) in ground truth
2. ✅ Solves critical blocker (task_028 at 0% recall)
3. ✅ No external dependencies, +1ms latency
4. ✅ Language-agnostic, future-proof
5. ✅ Enables other optimizations (stemming, fuzzy matching)

---

## Technical Specification

### Tokenization Rules

```python
# Input: Identifier string (function name, class name, variable, etc.)
# Output: List of component tokens

# Rule 1: Underscore splitting
"compile_templates"        → ["compile", "templates"]
"stream_with_context"      → ["stream", "with", "context"]
"import_module_using_spec" → ["import", "module", "using", "spec"]

# Rule 2: Hyphen splitting (rare in Python, but in queries)
"chunk-boundary"           → ["chunk", "boundary"]
"carriage-return"          → ["carriage", "return"]

# Rule 3: CamelCase splitting
"FloatOperation"           → ["Float", "Operation"]
"TestClient"               → ["Test", "Client"]
"UnicodeDecodeError"       → ["Unicode", "Decode", "Error"]

# Rule 4: Mixed (apply rules sequentially)
"root_render_func"         → ["root", "render", "func"]
"generate_async"           → ["generate", "async"]
"http_exception"           → ["http", "exception"]

# Rule 5: Preserve original
"compile_templates"        → ["compile_templates", "compile", "templates"]
# (original + split for both exact and partial matching)
```

### Integration Points

1. **sparse_index.py - `tokenize_symbol()` function**
   - Add `tokenize_compound_words()` helper function
   - Integrate into existing tokenization pipeline
   - Preserve original tokens for exact matching

2. **Indexer (indexer.py)**
   - No changes needed (tokenization happens at query/index time)

3. **Query processing (server.py)**
   - No changes needed (BM25 handles split tokens automatically)

---

## Implementation Tasks

### Task 1: Implement Core Tokenization Function

**File:** `src/codegrapher/sparse_index.py`
**Lines:** Add after line 107 (after `tokenize_symbol()` function)
**Effort:** 30 minutes

```python
def tokenize_compound_word(identifier: str) -> List[str]:
    """Split compound identifier into component tokens.

    Handles underscore-separated, hyphen-separated, and CamelCase
    identifiers. Preserves original identifier for exact matching.

    Args:
        identifier: Single token (word, identifier, filename component)

    Returns:
        List of tokens including original and split components

    Examples:
        >>> tokenize_compound_word("compile_templates")
        ["compile_templates", "compile", "templates"]
        >>> tokenize_compound_word("TestClient")
        ["TestClient", "Test", "Client"]
        >>> tokenize_compound_word("simple")
        ["simple"]
    """
    if len(identifier) <= 2:
        return [identifier]  # Skip single-char tokens

    tokens = [identifier]  # Always preserve original

    # Rule 1: Underscore and hyphen splitting
    underscore_parts = re.split(r'[-_\s]', identifier)
    if len(underscore_parts) > 1:
        tokens.extend(underscore_parts)

    # Rule 2: CamelCase splitting (apply to each part from Rule 1)
    for part in underscore_parts:
        # Match: Capital letter followed by lowercase, OR all caps
        # Examples:
        #   "FloatOperation" → ["Float", "Operation"]
        #   "HTTPServer" → ["HTTP", "Server"]
        #   "parseXML" → ["parse", "XML"]  # NOTE: Won't split this perfectly
        camel_parts = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)', part)
        if len(camel_parts) > 1:
            tokens.extend(camel_parts)

    # Rule 3: Handle edge case - single underscore
    if identifier == '_':
        return ['_']

    # Deduplicate while preserving order
    seen = set()
    unique_tokens = []
    for t in tokens:
        t_lower = t.lower()
        if len(t) > 2 and t_lower not in seen:
            seen.add(t_lower)
            unique_tokens.append(t)

    return unique_tokens
```

**Acceptance Criteria:**
- ✅ Unit test `test_tokenize_compound_word_underscore()` passes
- ✅ Unit test `test_tokenize_compound_word_camelcase()` passes
- ✅ Unit test `test_tokenize_compound_word_mixed()` passes
- ✅ Unit test `test_tokenize_compound_word_simple()` passes (no splitting)

---

### Task 2: Integrate into `tokenize_symbol()`

**File:** `src/codegrapher/sparse_index.py`
**Lines:** Modify `tokenize_symbol()` function (around line 23-107)
**Effort:** 15 minutes

**Change:** Add compound word splitting for all signature tokens:

```python
def tokenize_symbol(symbol: Symbol) -> List[str]:
    """Extract searchable tokens from a symbol.

    [Existing docstring...]

    Returns:
        List of lowercase tokens, filtered to 3+ characters
    """
    tokens = []

    # [Existing tokenization patterns 1-8...]

    # 8. Alphanumeric words (3+ chars)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", symbol.signature)
    tokens.extend(words)

    # NEW: 8.5. Compound word splitting
    # Apply to all alphanumeric tokens extracted above
    for word in words:
        compound_tokens = tokenize_compound_word(word)
        tokens.extend(compound_tokens)

    # 9. Docstring tokens (if present)
    if symbol.doc:
        doc_words = re.findall(r"\b[a-zA-Z]{3,}\b", symbol.doc[:200])
        tokens.extend(doc_words)
        # Also split compound words in docstrings
        for word in doc_words:
            compound_tokens = tokenize_compound_word(word)
            tokens.extend(compound_tokens)

    # [Existing deduplication logic...]
    # [Return statement...]
```

**Acceptance Criteria:**
- ✅ `test_tokenize_symbol_with_compounds()` passes
- ✅ Existing tests still pass (no regression)

---

### Task 3: Add Unit Tests

**File:** `tests/test_sparse_index.py` (or create if not exists)
**Lines:** Add ~80 lines of new tests
**Effort:** 45 minutes

```python
import pytest
from codegrapher.sparse_index import tokenize_compound_word


class TestCompoundWordSplitting:
    """Test compound word tokenization."""

    def test_underscore_separated(self):
        """Underscore-separated identifiers split correctly."""
        result = tokenize_compound_word("compile_templates")
        assert "compile_templates" in result  # Original preserved
        assert "compile" in result
        assert "templates" in result

        result = tokenize_compound_word("stream_with_context")
        assert "stream_with_context" in result
        assert "stream" in result
        assert "with" in result
        assert "context" in result

        result = tokenize_compound_word("import_module_using_spec")
        assert "import_module_using_spec" in result
        assert "import" in result
        assert "module" in result
        assert "using" in result
        assert "spec" in result

    def test_camelcase_splitting(self):
        """CamelCase identifiers split correctly."""
        result = tokenize_compound_word("FloatOperation")
        assert "FloatOperation" in result
        assert "Float" in result
        assert "Operation" in result

        result = tokenize_compound_word("TestClient")
        assert "TestClient" in result
        assert "Test" in result
        assert "Client" in result

        result = tokenize_compound_word("UnicodeDecodeError")
        assert "UnicodeDecodeError" in result
        assert "Unicode" in result
        assert "Decode" in result
        assert "Error" in result

    def test_mixed_identifiers(self):
        """Mixed underscore + CamelCase splits correctly."""
        result = tokenize_compound_word("root_render_func")
        assert "root_render_func" in result
        assert "root" in result
        assert "render" in result
        assert "func" in result

        result = tokenize_compound_word("generate_async")
        assert "generate_async" in result
        assert "generate" in result
        assert "async" in result

        result = tokenize_compound_word("http_exception")
        assert "http_exception" in result
        assert "http" in result
        assert "exception" in result

    def test_simple_identifiers_unchanged(self):
        """Simple identifiers without compound patterns are unchanged."""
        result = tokenize_compound_word("simple")
        assert result == ["simple"]

        result = tokenize_compound_word("compile")
        assert "compile" in result
        assert len(result) == 1  # No splitting

        result = tokenize_compound_word("templates")
        assert "templates" in result
        assert len(result) == 1

    def test_edge_cases(self):
        """Edge cases handled correctly."""
        # Single character
        result = tokenize_compound_word("a")
        assert result == ["a"]

        # Two characters (below threshold)
        result = tokenize_compound_word("ab")
        assert result == ["ab"]

        # Underscore only
        result = tokenize_compound_word("_")
        assert result == ["_"]

        # All caps abbreviation
        result = tokenize_compound_word("HTTP")
        assert "HTTP" in result
        # May or may not split depending on regex

        # Numbers in identifiers
        result = tokenize_compound_word("test_123")
        assert "test_123" in result
        assert "test" in result
        # Numbers filtered by 3-char threshold

    def test_deduplication(self):
        """Duplicate tokens are removed."""
        result = tokenize_compound_word("test_test")
        assert result.count("test") == 1  # Only one "test"
        assert "test_test" in result  # Original preserved

    def test_case_preservation_then_lower(self):
        """Original case preserved, but deduplication is case-insensitive."""
        result = tokenize_compound_word("Test")
        assert "Test" in result
        # Case-insensitive deduplication means "test" won't be added if "Test" exists

    def test_hyphen_separated(self):
        """Hyphen-separated identifiers split correctly."""
        result = tokenize_compound_word("chunk-boundary")
        assert "chunk-boundary" in result
        assert "chunk" in result
        assert "boundary" in result
```

**Acceptance Criteria:**
- ✅ All 8 test cases pass
- ✅ Coverage report shows >90% for new code
- ✅ No test failures in existing test suite

---

### Task 4: Rebuild Index

**Command:** `codegraph build --full`
**Effort:** 5-30 minutes (depends on repo size)
**Purpose:** Re-tokenize all symbols with compound word splitting

**Steps:**
```bash
# Navigate to test repository (e.g., jinja)
cd /tmp/jinja_test

# Rebuild index with new tokenization
codegraph build --full

# Verify index built successfully
ls -lh .codegraph/symbols.db
```

**Acceptance Criteria:**
- ✅ Index builds without errors
- ✅ Index size increased by 10-20% (more tokens)
- ✅ Build time increased by <20% (tokenization overhead)

---

### Task 5: Run Pretest Evaluation

**Script:** `python scripts/03_evaluation/run_eval_robust.py --ground-truth fixtures/ground_truth_pretest.jsonl --mode simulate`
**Effort:** 10 minutes
**Purpose:** Verify task_028 improvement

**Expected Results:**
| Task | Before | After | Status |
|------|--------|-------|--------|
| task_028 | 0% | **≥75%** | ✅ Must pass |
| task_032 | 75% | 75-80% | ✅ Should maintain |
| task_039 | 50% | **≥60%** | ✅ Should improve |
| task_040 | 50% | **≥60%** | ✅ Should improve |

**Acceptance Criteria:**
- ✅ task_028 recall ≥75% (critical)
- ✅ No regression in other tasks
- ✅ Aggregate recall improved by ≥10%

---

### Task 6: Run Full Evaluation

**Script:** `python scripts/03_evaluation/run_eval_robust.py --mode simulate --output results_compound_splitting.json`
**Effort:** 60-120 minutes
**Purpose:** Measure aggregate improvement across all tasks

**Expected Results:**
- **Aggregate recall:** +10-30% improvement
- **Tasks improved:** 8-12 tasks show measurable gains
- **Tasks regressed:** 0 tasks (compound splitting is additive)

**Acceptance Criteria:**
- ✅ Aggregate recall improvement ≥10%
- ✅ No tasks with >5% regression
- ✅ At least 8 tasks show ≥5% improvement

---

### Task 7: Documentation Updates

**Files:**
1. `docs/PLAN_hybrid_search.md` - Mark Tier 1 complete
2. `docs/PROGRESS.md` - Add compound splitting to Phase 12
3. `src/codegrapher/sparse_index.py` - Update docstrings

**Effort:** 30 minutes

**Changes:**

**PLAN_hybrid_search.md:**
```markdown
## Implementation Status Summary

### Completed (2026-01-14)

**Phase 1: Minimal Viable Hybrid** ✅ COMPLETE
[...]

**Phase 2: Production Hardening** ⚠️ MOSTLY COMPLETE
[...]

**NEW - Tier 1 Optimization: Compound Word Splitting** ✅ COMPLETE
- ✅ Implemented tokenize_compound_word() function
- ✅ Integrated into tokenize_symbol() pipeline
- ✅ Unit tests added (8 test cases)
- ✅ Index rebuilt with compound splitting
- ✅ Pretest validation: task_028 0% → 75-100%
- ✅ Full evaluation: +10-30% aggregate recall
- ✅ Documentation updated

**Impact:**
- 12/24 tasks (50%) benefit from compound splitting
- task_028 critical blocker resolved
- No performance degradation (+1ms latency)
```

**sparse_index.py docstring:**
```python
"""Sparse index for BM25 keyword search.

This module implements sparse retrieval using BM25 ranking over tokenized
symbol signatures and filenames. Complements dense vector search for
exact symbol matching and rare term queries.

**Tokenization Strategy:**
- Underscore splitting: compile_templates → compile, templates
- Hyphen splitting: chunk-boundary → chunk, boundary
- CamelCase splitting: FloatOperation → Float, Operation
- Compound word splitting enables partial matching and morphology handling

**Performance:**
- Index overhead: +10-20% size (more tokens per symbol)
- Query latency: +1ms (regex splitting, negligible)
- Recall improvement: +10-30% aggregate (measured)
"""
```

---

## Testing Strategy

### Unit Tests (Task 3)
- **Scope:** `tokenize_compound_word()` function
- **Coverage:** >90% for new code
- **Execution:** <1 second

### Integration Tests (Task 5)
- **Scope:** End-to-end search with pretest dataset
- **Dataset:** `fixtures/ground_truth_pretest.jsonl` (4 tasks)
- **Execution:** ~10 minutes
- **Success:** task_028 ≥75% recall

### Regression Tests (Task 6)
- **Scope:** Full ground truth dataset
- **Dataset:** `fixtures/ground_truth.jsonl` (24 tasks)
- **Execution:** ~60-120 minutes
- **Success:** Aggregate recall ≥+10%, no >5% regressions

### Edge Case Validation
- **Empty string:** Returns empty list
- **Single char:** Returns [char]
- **All underscores:** Returns original
- **Unicode:** Handles non-ASCII (é, ñ, etc.)
- **Numbers:** Preserves but filters by length threshold

---

## Rollback Plan

### If Implementation Fails

**Symptoms:**
- Index build fails
- Tests fail
- Performance degrades >20ms
- Recall regresses >5%

**Rollback Steps:**

1. **Revert code changes:**
   ```bash
   git revert <commit-hash>
   ```

2. **Rebuild index:**
   ```bash
   codegraph build --full
   ```

3. **Verify restoration:**
   ```bash
   # Run pretest
   python scripts/03_evaluation/run_eval_robust.py --ground-truth fixtures/ground_truth_pretest.jsonl --mode simulate
   # Should return to previous results
   ```

### If Partial Rollback Needed

**Scenario:** Compound splitting works but breaks specific tasks

**Mitigation:**
- Add feature flag: `ENABLE_COMPOUND_SPLITTING = True`
- Disable for specific patterns: Add blacklist in code
- File issue with details for investigation

---

## Success Criteria

### Must Have (Blockers)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **task_028 recall** | ≥75% | Pretest evaluation |
| **Build time** | +<20% | Index build duration |
| **Query latency** | +<5ms | Query timing logs |
| **Test coverage** | >90% | pytest coverage |
| **No regressions** | 0 tasks >5% worse | Full evaluation |

### Should Have (Important)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Aggregate recall** | ≥+10% | Full evaluation |
| **Tasks improved** | ≥8 tasks | Count of improved tasks |
| **Index size** | +<30% | Database file size |
| **Memory overhead** | +<10MB | Process memory |

### Nice to Have (Bonus)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Aggregate recall** | ≥+20% | Full evaluation |
| **Tasks improved** | ≥10 tasks | Count of improved tasks |
| **task_028 recall** | ≥90% | Pretest evaluation |
| **Code elegance** | Clean, documented | Code review |

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Index corruption** | Low | High | Backup before rebuild, test on small repo first |
| **Performance regression** | Low | Medium | Measure query latency, add timing logs |
| **Over-tokenization** | Medium | Low | Deduplication logic, 3-char threshold |
| **False positives** | Low | Low | Compound splitting is additive, not replacing |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Build time too long** | Low | Low | Test on small repo, monitor progress |
| **WSL memory issues** | Medium | Low | Run on non-WSL if needed |
| **Test flakiness** | Low | Low | Run tests multiple times, use checkpointing |

---

## Dependencies

### Required (Blockers)
- ✅ Python 3.10+ (already required)
- ✅ Existing sparse_index.py infrastructure
- ✅ BM25 searcher (already implemented)

### Optional (Nice to have)
- ⏳ Debug logging (already added in commit 23ebd2d)
- ⏳ Performance monitoring (Phase 3)

---

## Timeline

| Task | Duration | Dependencies |
|------|----------|--------------|
| Task 1: Core function | 30 min | None |
| Task 2: Integration | 15 min | Task 1 |
| Task 3: Unit tests | 45 min | Task 1 |
| Task 4: Rebuild index | 5-30 min | Task 2 |
| Task 5: Pretest eval | 10 min | Task 4 |
| Task 6: Full eval | 60-120 min | Task 5 |
| Task 7: Documentation | 30 min | Task 6 |
| **Total** | **4-6 hours** | **Sequential** |

**Parallelization opportunities:**
- Tasks 1-3 can be done in parallel (implementation + tests)
- Task 4 can run while writing documentation (Task 7)

---

## Next Steps After Implementation

### If Successful (Expected)
1. ✅ Commit changes with comprehensive message
2. ✅ Update PLAN_hybrid_search.md (Tier 1 complete)
3. ✅ Update PROGRESS.md (Phase 12 progress)
4. ⏳ Proceed to Tier 2 (stemming) if recall still <85%
5. ⏳ Proceed to Phase 3 (performance optimization)

### If Partial Success
1. ⏳ Analyze which tasks didn't improve
2. ⏳ Identify patterns (e.g., need stemming + compound split)
3. ⏳ Consider Tier 2 (stemming) for remaining gaps

### If Unsuccessful
1. ⏳ Rollback changes
2. ⏳ Debug failure mode (logs, unit tests)
3. ⏳ Consider alternative approaches (fuzzy matching)

---

## References

### Analysis Documents
- `ANALYSIS_stemming_vs_fuzzy_matching.md` - Tier 1 recommendation
- `ANALYSIS_compound_word_benefits.md` - 12/24 tasks benefit
- `PLAN_hybrid_search.md` - Overall implementation plan

### Code Files
- `src/codegrapher/sparse_index.py` - Implementation location
- `tests/test_sparse_index.py` - Test location

### Evaluation Scripts
- `scripts/03_evaluation/run_eval_robust.py` - Evaluation harness
- `fixtures/ground_truth_pretest.jsonl` - 4-task pretest
- `fixtures/ground_truth.jsonl` - 24-task full dataset

---

*Plan created: 2026-01-14*
*Author: CodeGrapher Team*
*Status: Ready for implementation*
*Related: PLAN_hybrid_search.md Phase 3.3 (Weight tuning)*
