# CodeGrapher Implementation Plan (Finalized v2.0)
**Target:** Claude Code (GLM)
**Status:** Ready for Execution
**Source:** PRD v1.0 & Critique v1.1

---

## Overview
This plan represents the finalized implementation roadmap for CodeGrapher. It has been refined to address critical gaps in error recovery, performance validation, and end-to-end integration.

**Key Changes from Draft:**
1.  **Added Phase 11.5:** Dedicated performance benchmarking to satisfy AC #5, #6, #7.
2.  **Enhanced Phase 10:** Explicit MCP configuration generation for user onboarding.
3.  **Refined Dependencies:** Strict versioning (`~=`) for stability.
4.  **Refined Prompts:** Added specific acceptance criteria for git hooks, truncation logic, and model verification.

---

## Phase 1: Environment Setup & Scaffolding
**Goal:** Establish the project structure with strict dependency management.

**Prompt:**
> Create a Python project structure for 'codegrapher'.
> 1. Create `pyproject.toml` using strict version constraints (`~=`) to prevent breaking changes:
>    - `fastmcp~=0.2.0`
>    - `faiss-cpu~=1.7.4`
>    - `numpy~=1.24.0`
>    - `transformers~=4.35.0`
>    - `torch~=2.1.0`
>    - `detect-secrets~=1.5.0`
>    - `huggingface-hub~=0.19.0`
>    - `watchdog~=3.0.0`
>    - `networkx~=3.0`
> 2. Create standard directories: `src/codegrapher/`, `tests/`, `scripts/`, `fixtures/`, `config/`.
> 3. Initialize `__init__.py` in source directory.
> 4. Ensure `python_requires = ">=3.10"`.

**Acceptance:**
- [ ] `uv pip install -e .` succeeds.
- [ ] Dependencies are locked to minor versions.

---

## Phase 2: Core Data Models & Database Schema
**Goal:** Define the data structures and persistence layer (PRD Section 7).

**Prompt:**
> Define the core data models in `src/codegrapher/models.py`.
> 1. Create a Pydantic `Symbol` class matching the SQL schema (id, file, start_line, end_line, signature, doc, mutates, embedding).
> 2. Create a `Database` class handling SQLite connections.
> 3. Implement table creation for `symbols`, `edges`, and `index_meta`.
> 4. Ensure embeddings serialize to BLOB but convert to/from numpy arrays in Python.

**Acceptance:**
- [ ] Database tables match PRD Section 7 exactly.
- [ ] `Symbol` validation works correctly.

---

## Phase 3: AST Parser & Symbol Extraction
**Goal:** Extract symbols and imports from Python source code.

**Prompt:**
> Implement the AST parser in `src/codegrapher/parser.py`.
> 1. Write `extract_symbols(file_path: Path) -> List[Symbol]` using `ast`.
> 2. Extract top-level functions, classes, and assignments.
> 3. Capture signatures, first docstring lines, and line numbers.
> 4. Write `extract_imports(file_path: Path)` to return raw import strings.

**Acceptance:**
- [ ] All relevant symbol types are extracted.
- [ ] Imports are captured as strings.

---

## Phase 4: Call Graph & PageRank
**Goal:** Build the graph structure for ranking (PRD Section 5, Recipe 3).

**Prompt:**
> Build the call graph infrastructure in `src/codegrapher/graph.py`.
> 1. Create `EdgeExtractor` to identify function calls, inheritance, and imports from AST.
> 2. Store edges in SQLite `edges` table.
> 3. Implement `compute_pagerank(db: Database) -> Dict[str, float]`:
>    - Use `networkx` to build the graph.
>    - Run PageRank with `alpha=0.85, max_iter=100`.
>    - Return normalized scores.
> 4. Ensure this runs during full index builds.

**Acceptance:**
- [ ] Call graph is persisted correctly.
- [ ] PageRank scores are calculated and cached.

---

## Phase 5: Embeddings & FAISS Indexing
**Goal:** Convert symbols to vectors and enable fast search (PRD Section 5).

**Prompt:**
> Implement the embedding and vector store logic in `src/codegrapher/vector_store.py`.
> 1. Initialize `jina-embeddings-v2-base-code`. Auto-download from Hugging Face if missing.
>    - **Error Handling:** On download failure, print PRD Section 11 options and `exit(5)`.
> 2. Create `FAISSIndexManager` based on PRD Recipe 4.
> 3. Implement `add`, `remove`, and `search` methods.
> 4. Ensure index saves to `.codegraph/index.faiss`.
> 5. **Verification:** Add a test to ensure embedding "def test()" returns a 768-dim vector.

**Acceptance:**
- [ ] Model loads/downloads correctly.
- [ ] Search returns top-k neighbors.
- [ ] Embedding dimensions are verified (768).

---

## Phase 6: Import Resolution Logic
**Goal:** Resolve raw import strings to file paths (PRD Recipe 1).

**Prompt:**
> Implement `src/codegrapher/resolver.py`.
> 1. Implement `resolve_import_to_path(module_name, current_file, repo_root) -> Optional[Path]`.
> 2. Handle relative imports (`..`, `.`) and absolute imports within `repo_root`.
> 3. Return `None` for external libraries (site-packages).
> 4. Add unit tests for edge cases (circular imports, missing files).

**Acceptance:**
- [ ] Relative imports resolve correctly.
- [ ] External imports return `None`.
- [ ] Edge cases are covered by tests.

---

## Phase 7: Secret Detection (Safety Layer)
**Goal:** Prevent secrets from being indexed (PRD Section 8).

**Prompt:**
> Implement the secret detection pipeline in `src/codegrapher/secrets.py`.
> 1. Use `detect-secrets` to scan files.
> 2. Check for `.secrets.baseline` at repo root. Use it if present to manage false positives.
> 3. If a secret is detected, log warning, add to `.codegraph/excluded_files.txt`, and return `True`.
> 4. Ensure indexing loop skips files where secrets are found.

**Acceptance:**
- [ ] Files with secrets are skipped.
- [ ] Baseline file is respected if it exists.

---

## Phase 8: Incremental Indexing Logic
**Goal:** Update index in <1s for changed files (PRD Recipe 2, 5).

**Prompt:**
> Implement the incremental indexer in `src/codegrapher/indexer.py`.
> 1. Create `IncrementalIndexer` with LRU cache for ASTs.
> 2. Implement `update_file` to calculate `SymbolDiff`.
> 3. **Atomicity:** Implement `atomic_update()` context manager from PRD Recipe 5:
>    - Use `BEGIN IMMEDIATE` for SQLite.
>    - Snapshot FAISS state before changes.
>    - Rollback both on exception.
> 4. Wrap diff applications in `atomic_update`.

**Acceptance:**
- [ ] Performance is <200ms for small changes.
- [ ] Transactions are atomic (verified via kill test).

---

## Phase 9: File Watching & Auto-Update
**Goal:** Ensure index freshness automatically (AC #4, #5).

**Prompt:**
> Implement automatic updates in `src/codegrapher/watcher.py`.
> 1. Use `watchdog` to monitor `*.py` changes.
> 2. On change: scan secrets -> update index -> atomic commit.
> 3. Queue bulk changes (>20 files) for background rebuild.
> 4. **Git Hook Support:** Create a `post-commit` hook template installed during `init` that calls `codegraph update --git-changed` to ensure consistency even if the watcher is dead.

**Acceptance:**
- [ ] File edits trigger near-instant updates.
- [ ] Git hook is generated and installed.
- [ ] Bulk changes don't block the main thread.

---

## Phase 10: MCP Server Interface
**Goal:** Expose functionality to Claude Code (PRD Section 6).

**Prompt:**
> Create the MCP server in `src/codegrapher/server.py`.
> 1. Define tool `codegraph_query` with inputs: `query`, `cursor_file`, `max_depth`, `token_budget`.
> 2. Implement search logic: Vector search -> Import Closure Prune -> Weighted Score Rank.
> 3. **Truncation:** Use `truncate_at_token_budget()` from PRD Recipe 3 (break at file boundaries).
> 4. **Error Handling:** Catch `IndexCorruptedError` -> fallback to SQLite text search. Detect stale index -> set flag.
> 5. **MCP Config:** Generate `config/mcp_server.json` template:
>    ```json
>    {
>      "mcpServers": {
>        "codegrapher": {
>          "command": "python",
>          "args": ["-m", "codegrapher.server"]
>        }
>      }
>    }
>    ```

**Acceptance:**
- [ ] Server handles queries per PRD schema.
- [ ] Errors fall back gracefully.
- [ ] `config/mcp_server.json` is generated.

---

## Phase 11: CLI & Build Tools
**Goal:** User-facing commands.

**Prompt:**
> Create CLI in `src/codegrapher/cli.py`.
> 1. `codegraph init`: Setup repo, download model, install git hook.
> 2. `codegraph build --full`: Full index build.
> 3. `codegraph query`: CLI testing.
> 4. `codegraph update --git-changed`: Helper for git hooks.
> 5. **Integration Test:** Run on small repo, modify file, verify update <1s. Kill mid-update, verify no corruption.

**Acceptance:**
- [ ] All CLI commands function.
- [ ] Integration test (kill test) passes.

---

## Phase 11.5: Performance Verification
**Goal:** Validate performance budgets (PRD Section 10, AC #5, #6, #7).

**Prompt:**
> Create performance benchmarking in `scripts/benchmark.py`.
> 1. Use `hyperfine` to measure:
>    - Query latency (<500ms)
>    - Incremental update (<1s)
>    - Full index build (<30s for 30k LOC)
> 2. Use `psrecord` to verify RAM usage (<500MB idle).
> 3. Use `du` to check disk overhead (<1.5× repo size).
> 4. Output results in markdown table.

**Acceptance:**
- [ ] All metrics meet PRD Section 10 targets.
- [ ] Benchmark script runs in CI.

---

## Phase 12: Testing & Evaluation
**Goal:** Verify acceptance criteria (AC #1, #2, #3).

**Prompt:**
> Create testing harness (PRD Section 13).
> 1. **Ground Truth:** Clone 3-4 small MIT repos. Manually create 20 tasks (8 bug, 6 refactor, 6 feature) in `fixtures/ground_truth.jsonl`.
> 2. **Evaluation:** Write `scripts/eval_token_save.py`.
> 3. Run baseline vs. CodeGrapher.
> 4. **Metrics:** Verify token savings ≥30%, Recall ≥85%.

**Acceptance:**
- [ ] Ground truth data is realistic.
- [ ] Evaluation script passes AC #1, #2, #3.

---

## Final Checklist

- [ ] **Security:** No secrets in code; `detect-secrets` active.
- [ ] **Performance:** All benchmarks green.
- [ ] **Recovery:** Survives process kills/crashes.
- [ ] **Usability:** MCP config generated; `init` works smoothly.
- [ ] **Documentation:** `README.md` updated.

---

**End of Finalized Implementation Plan**