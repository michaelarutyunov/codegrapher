# Master Prompt for CodeGrapher Implementation

**Version:** 1.0  
**Target Agent:** Claude Code or similar coding agents  
**Project:** CodeGrapher v1.0 - Token-efficient MCP code search server

---

## Overview

You are implementing CodeGrapher v1.0, an MCP server that provides token-efficient code search for Python repositories. You have access to three critical documents:

1. **PRD_project_codegrapher_v1_0.md** - Complete specification (882 lines)
2. **ENGINEERING_GUIDELINES.md** - Mandatory coding standards (1338 lines)
3. **PLAN_v1.md** - Phase-by-phase implementation roadmap (273 lines)

---

## Critical Instructions - Read First

**BEFORE starting Phase 1:**
1. ✅ Read the complete Engineering Guidelines document (all 1338 lines)
2. ✅ Familiarize yourself with PRD Sections 1-19
3. ✅ Review PLAN_v1.md to understand the full implementation arc

**BEFORE starting any individual phase:**
1. Read the specific phase prompt in PLAN_v1.md
2. Review PRD sections referenced in the phase prompt
3. Read relevant PRD Recipes if mentioned (Section 17)
4. Verify library APIs using Context7 before writing code that uses external libraries

---

## Core Constraints (Non-Negotiable)

### Design Decisions (PRD Section 5)
- **Hybrid retrieval:** Symbol table (precision) + vector similarity (recall) + 1-hop graph expansion (completeness)
- **Embeddings:** `jina-embeddings-v2-base-code` → 768-dim vectors, CPU-only
- **Incremental updates:** AST-level diff, <200ms for ≤5 symbols
- **Ranking formula:** `0.60·cosine + 0.25·PageRank + 0.10·recency + 0.05·test_file`
- **PageRank parameters:** α=0.85, max_iter=100, tolerance=1e-6

### Implementation Style (Engineering Guidelines)
- ✅ **Simple over clever** (Rule 1.1): Direct implementations, no design patterns unless PRD requires
- ✅ **No premature abstraction** (Rule 1.2): No base classes unless ≥3 implementations exist
- ✅ **Limit indirection** (Rule 1.3): Max 2 levels of function delegation
- ✅ **Proper library use** (Rule 2.2): Use libraries as intended, don't reimplement
- ✅ **No scope creep** (Rule 3.1): Only implement PRD Sections 1-19
- ✅ **Standard Python style** (Section 4): PEP 8, type hints, docstrings, 100 char lines

### Scope Boundaries (PRD Section 4)
**In scope:**
- Python 3.10-3.12 mono-repos, ≤50k LOC
- CPU-only embeddings (no GPU)
- MCP stdio transport
- Incremental graph & vector index
- Secret scrubbing
- MIT license, full OSS

**Out of scope (DO NOT implement):**
- Multi-repo, other languages, GPU acceleration
- Web transport, LSP, VS Code extension
- Cross-repo references, binary dependencies
- UI dashboard (CLI only)
- Any features in PRD Section 18 "Open Questions"

---

## Implementation Workflow

### For Each Phase

#### 1. Preparation
State clearly:
```
Starting Phase [N]: [Phase Name]
Referenced documents:
- PLAN_v1.md lines [X-Y]
- PRD Section [Z]
- Engineering Guidelines Rule [A.B]

Library verification needed: [Yes/No]
If yes: Will call Context7 for [library] API verification
```

#### 2. Library API Verification
**CRITICAL:** Before using any external library method, verify it exists:

**When to call Context7:**
- Phase 4: Before using `nx.pagerank()`, `nx.DiGraph()`, `nx.add_edge()`
- Phase 5: Before using `AutoModel.from_pretrained()`, `model.encode()`
- Phase 5: Before using `faiss.IndexFlatL2()`, `faiss.add()`, `faiss.search()`
- Phase 9: Before using `watchdog.Observer()`, `FileSystemEventHandler`

**Verification process:**
```python
# Example: Verify NetworkX PageRank
import networkx as nx

# 1. Create minimal test case
G = nx.DiGraph()
G.add_edge('a', 'b')

# 2. Verify method exists and signature
assert hasattr(nx, 'pagerank'), "PageRank not available"
result = nx.pagerank(G, alpha=0.85, max_iter=100)

# 3. Verify output format
assert isinstance(result, dict)
assert 'a' in result and 'b' in result
```

#### 3. Implementation
- Follow the phase prompt in PLAN_v1.md exactly
- Create files in the directory structure per Engineering Guidelines Rule 5.3
- Keep modules under 600 LOC (Rule 5.2)
- Use code patterns from PRD Section 17 Recipes when applicable

**Directory structure (Rule 5.3):**
```
codegrapher/
├── src/
│   └── codegrapher/
│       ├── __init__.py
│       ├── models.py          # Phase 2
│       ├── parser.py          # Phase 3
│       ├── resolver.py        # Phase 6
│       ├── graph.py           # Phase 4
│       ├── vector_store.py    # Phase 5
│       ├── secrets.py         # Phase 7
│       ├── indexer.py         # Phase 8
│       ├── watcher.py         # Phase 9
│       ├── query.py           # Phase 10
│       ├── server.py          # Phase 10
│       └── cli.py             # Phase 11
├── tests/
│   ├── test_parser.py
│   ├── test_indexer.py
│   ├── test_graph.py
│   └── test_integration.py
├── scripts/
│   ├── eval_token_save.py     # Phase 12
│   └── benchmark.py           # Phase 11.5
├── fixtures/
│   └── ground_truth.jsonl     # Phase 12
├── config/
│   └── mcp_server.json        # Phase 10
├── pyproject.toml
└── README.md
```

#### 4. Verification
Check all acceptance criteria for the phase:
```
Phase [N] Acceptance Criteria:
- [ ] [Criterion 1 from PLAN_v1.md]
- [ ] [Criterion 2 from PLAN_v1.md]
- [ ] Code follows Engineering Guidelines
- [ ] No over-engineering (Rule 1.1-1.3)
- [ ] Libraries used correctly (Rule 2.2)
- [ ] No scope creep (Rule 3.1-3.2)
```

#### 5. Quality Checklist
Before marking phase complete:
- [ ] All acceptance criteria met
- [ ] Module(s) are <600 LOC each
- [ ] Type hints on all public functions
- [ ] Docstrings in Google style for public APIs
- [ ] Error messages are actionable (Rule 4.3: What + Why + How to fix)
- [ ] No magic numbers (all constants named with semantic meaning)
- [ ] Library methods verified via Context7 where applicable
- [ ] No features outside PRD Sections 1-19
- [ ] Code formatted with Black (`--line-length 100`)

#### 6. Commit
```bash
git add .
git commit -m "feat(phase-N): [description]"
```

Format per Engineering Guidelines Rule 10.1:
- `feat`: New feature or capability
- `fix`: Bug fix
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `docs`: Documentation only

---

## Key Technical Specifications

### Database Schema (PRD Section 7)
```sql
-- Symbols table
CREATE TABLE symbols (
  id TEXT PRIMARY KEY,           -- fully qualified name (e.g., "mymodule.MyClass.method")
  file TEXT NOT NULL,            -- relative path from repo root
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  signature TEXT NOT NULL,       -- function/class signature
  doc TEXT,                      -- first sentence of docstring
  mutates TEXT,                  -- comma-separated list of mutated variables
  embedding BLOB NOT NULL        -- 768 float32 values (3072 bytes)
);

-- Call graph edges
CREATE TABLE edges (
  caller_id TEXT NOT NULL,
  callee_id TEXT NOT NULL,
  type TEXT NOT NULL,            -- 'call', 'import', 'inherit'
  FOREIGN KEY (caller_id) REFERENCES symbols(id),
  FOREIGN KEY (callee_id) REFERENCES symbols(id)
);

-- Index metadata
CREATE TABLE index_meta (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### MCP Tool Interface (PRD Section 6)
**Tool name:** `codegraph_query`

**Input schema:**
```json
{
  "query": "string",           // free text or symbol name
  "cursor_file": "string",     // optional: file path for import-closure pruning
  "max_depth": 1,              // graph hop depth 0-3
  "token_budget": 3500,        // hard limit for response
  "include_snippets": false,   // if true, include code; if false, metadata only
  "format": "markdown"         // only option v1
}
```

**Success response:**
```json
{
  "status": "success",
  "files": [
    {
      "path": "src/auth/validator.py",
      "line_range": [23, 45],
      "symbol": "TokenValidator.validate",
      "excerpt": "def validate(self, token: str) -> bool:"
    }
  ],
  "tokens_used": 2847,
  "total_symbols": 12,
  "truncated": false,
  "total_candidates": 47
}
```

**Error response:**
```json
{
  "status": "error",
  "error_type": "index_stale|index_corrupt|parse_error|empty_results",
  "message": "Index last updated 3 hours ago. Run codegraph-rebuild.",
  "fallback_suggestion": "Use grep or full file search",
  "partial_results": []
}
```

### Performance Targets (PRD Section 10)
| Scenario | Target | Verification Command |
|----------|--------|---------------------|
| Cold start | ≤2s | `time codegraph init` |
| Query latency | ≤500ms | `hyperfine 'codegraph query "auth"'` |
| Incremental update (1 file) | ≤1s | `hyperfine 'codegraph update file.py'` |
| Full index 30k LOC | ≤30s | `hyperfine 'codegraph build --full'` |
| RAM idle | ≤500MB | `psrecord` |
| Disk overhead | ≤1.5× repo size | `du -sh .codegraph/` |

---

## Critical Algorithms (PRD Section 17)

### Recipe 1: Import Closure Pruning
**Purpose:** Keep only symbols reachable from user's cursor location.

**Algorithm:**
```python
def prune_by_import_closure(
    candidate_symbols: list[Symbol],
    cursor_file: Path,
    repo_root: Path
) -> list[Symbol]:
    """Keep only symbols reachable via import graph from cursor_file."""
    # 1. Build import graph: file -> list[imported_files]
    import_graph = build_import_graph(repo_root)
    
    # 2. BFS from cursor_file to find reachable files
    reachable = set()
    queue = deque([cursor_file])
    visited = {cursor_file}
    
    while queue:
        current = queue.popleft()
        reachable.add(current)
        for imported in import_graph[current]:
            if imported not in visited:
                visited.add(imported)
                queue.append(imported)
    
    # 3. Filter symbols to only those in reachable files
    return [s for s in candidate_symbols if Path(s.file) in reachable]
```

### Recipe 2: Incremental AST Diff
**Purpose:** Update index in <1s for file changes without full re-parse.

**Key points:**
- LRU cache of 50 pickled ASTs
- Compare symbols by `(node.name, node.__class__.__name__)`
- Compute multiset difference: deleted, added, modified
- Re-embed only if signature/doc/decorators changed
- Total: <50ms parsing + <150ms DB transaction = <200ms

### Recipe 3: Weighted Score Ranking
**Purpose:** Combine vector similarity, PageRank, recency into single score.

**Formula:**
```python
score(s) = (
    0.60 * norm_cosine(query_vec, s.embedding) +
    0.25 * norm_pagerank(s) +
    0.10 * recency_score(s) +
    0.05 * (1.0 if is_test_file(s) else 0.0)
)
```

**Normalization:**
- `norm_cosine`: Map cosine similarity [-1,1] → [0,1] via `(x+1)/2`
- `norm_pagerank`: Divide by max PageRank score
- `recency_score`: Piecewise constant (1.0 if ≤7 days, 0.5 if ≤30 days, 0.1 otherwise)

**Truncation:** Break at file boundaries only (never mid-file)

### Recipe 4: FAISS Operations
**Purpose:** Efficient vector similarity search and incremental updates.

**Key operations:**
```python
class FAISSIndexManager:
    def __init__(self, dim=768):
        self.index = faiss.IndexFlatL2(dim)  # L2 distance for CPU
        self.symbol_ids = []  # parallel list
    
    def add_symbols(self, symbols: list[Symbol]):
        embeddings = np.array([s.embedding for s in symbols], dtype='float32')
        self.index.add(embeddings)
        self.symbol_ids.extend([s.id for s in symbols])
    
    def remove_symbols(self, symbol_ids: list[str]):
        # FAISS doesn't support in-place removal → rebuild
        keep_indices = [i for i, sid in enumerate(self.symbol_ids) 
                       if sid not in set(symbol_ids)]
        old_vectors = np.array([self.index.reconstruct(i) for i in keep_indices])
        
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(old_vectors)
        self.symbol_ids = [self.symbol_ids[i] for i in keep_indices]
    
    def search(self, query_embedding: np.ndarray, k=20) -> list[tuple[str, float]]:
        try:
            distances, indices = self.index.search(
                np.array([query_embedding], dtype='float32'), k
            )
        except RuntimeError:
            raise IndexCorruptedError("FAISS search failed")
        
        return [(self.symbol_ids[idx], float(dist)) 
                for dist, idx in zip(distances[0], indices[0])
                if idx < len(self.symbol_ids)]
```

### Recipe 5: Transaction Safety
**Purpose:** Ensure SQLite + FAISS stay consistent even on crash.

**Implementation:**
```python
@contextmanager
def atomic_update(db_path: Path, faiss_manager: FAISSIndexManager):
    """Context manager for atomic updates across SQLite + FAISS."""
    conn = sqlite3.connect(db_path)
    conn.execute("BEGIN IMMEDIATE")  # lock database
    
    # Snapshot FAISS state
    faiss_backup = {
        'index': faiss_manager.index,
        'symbol_ids': faiss_manager.symbol_ids.copy()
    }
    
    try:
        yield conn  # caller does SQL updates + FAISS modifications
        
        conn.commit()
        faiss_manager.save()  # write to disk
        
    except Exception as e:
        conn.rollback()
        
        # Rollback FAISS to snapshot
        faiss_manager.index = faiss_backup['index']
        faiss_manager.symbol_ids = faiss_backup['symbol_ids']
        
        logger.error(f"Transaction failed: {e}")
        raise
    
    finally:
        conn.close()
```

---

## Common Mistakes to Avoid

### Anti-Pattern: Over-Engineering (Rule 1.1)
❌ **Wrong:**
```python
class SymbolExtractorFactory:
    @staticmethod
    def create_extractor(node_type: str) -> BaseExtractor:
        return _extractors[node_type]()

class FunctionExtractor(BaseExtractor):
    def extract(self, node: ast.FunctionDef) -> Symbol:
        ...
```

✅ **Correct:**
```python
def extract_symbols(tree: ast.AST, file_path: Path) -> List[Symbol]:
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            symbols.append(_extract_function(node, file_path))
        elif isinstance(node, ast.ClassDef):
            symbols.append(_extract_class(node, file_path))
    return symbols
```

### Anti-Pattern: Library Misuse (Rule 2.2)
❌ **Wrong:** Reimplementing PageRank
```python
def my_pagerank(graph):
    """Custom PageRank implementation."""
    # 50 lines of algorithm...
```

✅ **Correct:** Use NetworkX
```python
import networkx as nx

def compute_pagerank(edges: List[Edge]) -> Dict[str, float]:
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge.caller_id, edge.callee_id)
    return nx.pagerank(G, alpha=0.85, max_iter=100)
```

### Anti-Pattern: Scope Creep (Rule 3.2)
❌ **Forbidden features:**
- Web UI or dashboard
- Export to JSON/CSV/YAML
- Configuration file hot-reloading
- Multi-language support
- Plugin system or extension API
- Visualization of call graphs

✅ **Stick to PRD:** Only implement features in Sections 1-19

### Anti-Pattern: Silent Failures (Rule 8.3)
❌ **Wrong:**
```python
try:
    index.add(symbol)
except Exception:
    pass  # Swallows all errors
```

✅ **Correct:**
```python
try:
    index.add(symbol)
except FAISSError as e:
    logger.error(f"Failed to index {symbol.id}: {e}")
    # Continue with other symbols
except MemoryError:
    logger.critical("Out of memory - cannot continue indexing")
    raise  # Re-raise critical errors
```

### Anti-Pattern: Magic Numbers (Rule 4.4)
❌ **Wrong:**
```python
if tokens > 3500:
    truncate()
if days_ago > 30:
    recency = 0.1
```

✅ **Correct:**
```python
DEFAULT_TOKEN_BUDGET = 3500  # From PRD Section 6
RECENCY_MEDIUM_DAYS = 30     # From PRD Recipe 3

if tokens > token_budget:
    truncate()
if days_ago > RECENCY_MEDIUM_DAYS:
    recency = 0.1
```

---

## Decision Flowcharts

### When to Create a Class vs Function (Engineering Guidelines 11.1)
```
Does the implementation need mutable state?
  ├─ YES → Use a class
  │    └─ Examples: IncrementalIndexer (has AST cache),
  │                  FAISSIndexManager (has index state)
  │
  └─ NO → Are there >3 related functions that share data?
       ├─ YES → Use a class as namespace
       │    └─ Examples: Database (shares connection)
       │
       └─ NO → Use module-level functions
            └─ Examples: extract_symbols(), compute_pagerank()
```

### When to Use a Library vs Implement (Engineering Guidelines 11.2)
```
Does an established library solve this problem?
  ├─ YES → Is it already in pyproject.toml?
  │    ├─ YES → Use it (verify API first via Context7)
  │    │
  │    └─ NO → Estimate implementation complexity
  │         ├─ <100 LOC → Implement yourself (avoid dependency)
  │         │
  │         └─ >100 LOC → Add library to pyproject.toml
  │              └─ Verify: Is library maintained? (updated in last 6 months)
  │
  └─ NO → Estimate implementation complexity
       ├─ <200 LOC → Implement
       │
       └─ >200 LOC → Reconsider approach or find alternative library
```

### How to Handle Errors (Engineering Guidelines 11.3)
```
Can the operation succeed with degraded functionality?
  ├─ YES → Use fail-safe pattern
  │    └─ Log warning, return partial results
  │         Example: FAISS corrupt → use SQLite text search
  │
  └─ NO → Is this a user error (bad input)?
       ├─ YES → Raise specific exception with actionable message
       │    └─ Example: ValueError("Query too short: use 3+ chars")
       │
       └─ NO → Is this a system error (OOM, disk full)?
            ├─ YES → Log critical error, raise exception
            │    └─ Example: MemoryError("Out of RAM during indexing")
            │
            └─ Other → Log error, raise CodeGraphError subclass
```

---

## Phase Completion Template

Use this template after completing each phase:

```markdown
## Phase [N] Complete: [Phase Name]

### Files Created/Modified
- `src/codegrapher/[module].py` (XXX LOC)
- `tests/test_[module].py` (YYY LOC)
- [Other files]

### Dependencies Added
- [None] OR [library~=version]

### Acceptance Criteria
- [✓] [Criterion 1]
- [✓] [Criterion 2]
- [✓] Code follows Engineering Guidelines
- [✓] Library APIs verified via Context7

### Quality Checklist
- [✓] Module(s) <600 LOC
- [✓] Type hints on all public functions
- [✓] Docstrings in Google style
- [✓] Error messages are actionable
- [✓] No magic numbers
- [✓] No scope creep

### Warnings/Deviations
[None] OR [List any deviations with justification]

### Git Commit
```bash
git commit -m "feat(phase-N): [description]"
```

### Next Phase
Phase [N+1]: [Phase Name]
```

---

## Audit Checkpoint System

After completing Phases 3, 6, 9, and 12, run this audit:

```markdown
## Audit Checkpoint after Phase [N]

### Module Size Check
```bash
wc -l src/codegrapher/*.py
```
- [ ] All modules <600 LOC

### Code Quality Check
- [ ] No forbidden libraries imported (check pyproject.toml)
- [ ] All public functions have type hints
- [ ] All public functions have docstrings
- [ ] No magic numbers (search for raw integers >10)

### Scope Creep Check
- [ ] No files created outside approved structure
- [ ] No features from PRD Section 18 "Open Questions"
- [ ] No web UI, dashboards, or JSON export

### Performance Sanity Check (after Phase 9+)
- [ ] Index build on small repo (<1000 LOC) completes
- [ ] Query returns results in reasonable time
- [ ] No obvious memory leaks (check with manual test)

### Issues Found
[List any issues and remediation plan]
```

---

## Start Command

When you're ready to begin, state:

```
✅ Pre-Flight Checklist Complete
- Read Engineering Guidelines (1338 lines)
- Read PRD Sections 1-19
- Read PLAN_v1.md
- Understand project scope and constraints

Starting Phase 1: Environment Setup & Scaffolding

Referenced documents:
- PLAN_v1.md lines 19-41
- Engineering Guidelines Rule 5.3 (Directory Structure)

No library verification needed for this phase.

I will now create the project structure...
```

---

## Final Notes

**Remember:**
1. **Verify library APIs** before using (Context7 is your friend)
2. **Keep it simple** - no over-engineering
3. **Stick to scope** - only PRD Sections 1-19
4. **Quality over speed** - meet all acceptance criteria
5. **When in doubt** - consult Engineering Guidelines decision flowcharts

**If you get stuck:**
1. Re-read the relevant PRD section
2. Check Engineering Guidelines for the specific rule
3. Review PRD Recipe implementations
4. Verify library API with Context7

**Success criteria:**
- All 12 phases complete
- All acceptance criteria met
- Performance targets achieved (Phase 11.5)
- Evaluation harness passes (Phase 12)

Good luck! 🚀
