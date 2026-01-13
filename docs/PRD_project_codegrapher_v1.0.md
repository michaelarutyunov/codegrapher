# PRD — Project CodeGrapher  
*Version 1.0 — 2026-01-12*  
*Status: Ready for implementation*

---

## 1. Purpose & Elevator Pitch
CodeGrapher is a **local MCP server** that gives Claude Code (and any other MCP client) a *selective, token-efficient* view of a Python mono-repo.  
It returns **small, high-recall code bundles** (< 4 k tokens) instead of letting the agent burn dollars on full-file greps or giant context dumps.  
Target user: **solo developer** who repeatedly hits the Claude Code plan limit and wants ≥ 30 % token reduction per task without thinking about it.

---

## 2. User Story (primary)
"As a solo dev using Claude Code on a ≤ 50 k LOC Python project, I want the agent to **automatically receive only the files likely to be edited** so that my **monthly token spend drops by at least one third** while I still finish tasks in the same time."

---

## 3. Acceptance Criteria (measurable)
| # | Criterion | Metric | How verified |
|---|-----------|--------|--------------|
| 1 | Token saving | ≥ 30 % median vs baseline | `scripts/eval_token_save.py` run on 20 real tasks |
| 2 | Recall | ≥ 85 % of files that *were* edited must be in top-1 bundle | ground-truth set `fixtures/ground_truth.jsonl` |
| 3 | Precision | ≤ 40 % of files in bundle are *never* touched | same test as #2 |
| 4 | Index freshness | ≤ 10 s lag after `git push` or file-save | CI hook timestamp |
| 5 | Incremental update | ≤ 1 s for ≤ 5 changed files | `hyperfine` per-file hook |
| 6 | Full rebuild | ≤ 30 s for 30 k LOC on 4-core laptop | `codegraph-build --full` |
| 7 | Resource ceiling | ≤ 500 MB RAM idle, ≤ 150 % repo disk | `psrecord`, `du` |
| 8 | Zero secrets | 0 hard-coded credentials embedded | `detect-secrets` gate |
| 9 | Crash safety | agent falls back to full search + warning | manual kill test |
| 10 | Agent task completion | Same success rate as baseline (±5%) | 20 tasks with/without CodeGrapher |

---

## 4. Scope
**In scope**
- Python 3.10+
- Single mono-repo, ≤ 50 k LOC
- Embeddings on CPU (no GPU required)
- MCP stdio transport; Claude Code first client
- Incremental graph & vector index
- Secret scrubbing before embed
- MIT licence, full OSS  

**Out of scope (v1)**  
- Multi-repo, other languages, GPU acceleration  
- Web transport, LSP, VS-Code extension  
- Cross-repo references, binary dependencies  
- UI dashboard (CLI only)  

---

## 5. Core Design Choices (already agreed)
1. **Hybrid retrieval**:  
   exact symbol table **(precision)** + vector similarity **(recall)** + 1-hop graph expansion **(completeness)**.

2. **Single embedding per symbol**:  
   `signature + first-sentence doc + mutates tag` (≤ 40 tokens) → **768-dim vector** via `jina-embeddings-v2-base-code` (307 MB model, CPU-only, 161M parameters).
   Model auto-downloads from Hugging Face on first run to `~/.cache/huggingface/`.

3. **Incremental updates**:  
   Symbol-level AST diff (not text diff) on file-save:
   - Parse new AST, compare with cached previous AST by `(node.name, node.__class__.__name__)`
   - Deleted symbols → remove from SQLite + FAISS
   - Added/moved symbols → insert
   - Changed symbols (signature/doc/decorators) → re-embed and update
   - Minimal re-parse: only changed top-level symbols (functions, classes, module assignments)
   - Update SQLite + FAISS index in single transaction < 200 ms for ≤ 5 symbols
   - Queue bulk changes (> 20 files) for background full-index
   - Keep last 50 ASTs in LRU cache; fallback to background rebuild on failure

4. **Bundle builder algorithm**:  
   - Vector top-k (k = 20)
   - **Prune by build target**: Keep only symbols reachable from the user's working context via import-closure (not by directory/extension). Walk import graph from cursor location; drop symbols with infinite import distance.
   - Expand callers/callees 1 hop (max 5 each direction)
   - **Rank by weighted score**:
     ```
     score(s) = 0.60 · norm_cosine(query_vec, s_vec)
                + 0.25 · norm_page_rank(s)
                + 0.10 · norm_recency(s)
                + 0.05 · is_test_file(s)
     ```
     Where `norm_page_rank` is computed via: PageRank with damping factor α=0.85, max_iterations=100, convergence tolerance=1e-6, on directed call-graph edges only (imports/inheritance excluded). Initial vector: uniform distribution.
   - Truncate at token budget (3,500 tokens) at file boundaries
   - Return markdown table

5. **Local-only & private**:  
   Everything lives in `.codegraph/` folder; no code or embeddings leave the machine.

6. **Metadata-only output (Option A)**:  
   Tool returns file paths + line ranges + 1-line excerpt. Agent must call `read_file()` separately to get actual code. This keeps responses under token budget and allows agents to selectively fetch only what they need.

---

## 6. API Surface (MCP)

### Core Tool

**Tool name**: `codegraph_query`  

**Input**:
```json
{
  "query": "string",           // free text or symbol name
  "cursor_file": "string",     // optional: file path for import-closure pruning (e.g., "src/auth.py")
  "max_depth": 1,              // graph hop depth 0-3
  "token_budget": 3500,        // hard limit for response
  "include_snippets": false,   // if true, include code; if false (default), metadata only
  "format": "markdown"         // only option v1
}
```

**Parameter details:**
- `cursor_file` (optional): If provided, prune results by import-closure from this file (see Recipe 1). If omitted, skip import-closure pruning and return all symbols matching query. Use this when agent knows which file the user is working in.

**Output (include_snippets=false, default)**: 
Markdown table with:
- File path
- Line range (start-end)
- Symbol name
- 1-line excerpt/signature
- Total tokens ≤ budget

**Output (include_snippets=true)**:
Same as above, but includes full code snippets for each entry. Server hard-truncates at file boundaries when budget is exceeded.

**Success response structure**:
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
  "truncated": false,           // true if token_budget was exceeded
  "total_candidates": 47        // how many symbols matched before truncation
}
```

**Truncation behavior:**
When `token_budget` is exceeded, server truncates at file boundaries (never mid-file), sets `truncated: true`, and includes `total_candidates` count. Agent should warn user: "Results truncated at 3500 tokens; try broader query or increase budget."

**Error response structure**:
```json
{
  "status": "error",
  "error_type": "index_stale|index_corrupt|parse_error|empty_results",
  "message": "Index last updated 3 hours ago. Run codegraph-rebuild.",
  "fallback_suggestion": "Use grep or full file search",
  "partial_results": []  // may contain partial data if available
}
```

**Resource URI** (optional v2):  
`graph://callers/<symbol>` can be `@`-mentioned.

---

### Agent Integration Guidelines

**When agents should use this tool:**
- Multi-file editing tasks requiring cross-file coordination
- Queries mentioning specific symbols, classes, or function names (e.g., "fix the UserAuth class")
- Dependency analysis ("update all callers of authenticate()")
- Refactoring tasks spanning multiple modules
- Bug fixes where the affected code location is unknown

**When agents should NOT use this tool:**
- Reading a single specific file by known path → use `read_file()` directly
- Searching for literal strings or regex patterns → use `grep`
- Query is < 5 tokens and very simple → use `find` or `ls`
- Adding new files from scratch → no existing code to retrieve

**Tool selection decision tree for agents:**
```
if (task involves editing AND mentions symbols/classes/functions):
    → use codegraph_query
elif (searching for literal strings/patterns):
    → use grep
elif (known exact file path):
    → use read_file() directly
elif (exploring directory structure):
    → use find/ls
else:
    → try codegraph_query, fall back to grep if empty
```

**Output consumption pattern:**
1. Agent calls `codegraph_query(query="...", token_budget=3500)`
2. Parses markdown response to extract file paths + line ranges
3. Calls `read_file(path, start_line, end_line)` for each file to get actual code
4. Analyzes code and proceeds with editing/analysis task
5. If results seem incomplete, agent may:
   - Broaden query terms
   - Increase `max_depth` parameter (0 → 1 → 2)
   - Fall back to traditional search tools

**Error handling for agents:**
- On `MCP_ERROR` (server crash): Inform user, fall back to full-repo search
- On `status: "index_stale"`: Suggest user runs `codegraph-rebuild`, proceed with grep
- On `status: "empty_results"`: Try broader query terms, or use `find`/`grep` as fallback
- On `status: "index_corrupt"`: Alert user to run `codegraph-rebuild --full`, use grep meanwhile

**Example agent workflow:**
```
User: "Fix the bug in authentication where expired tokens aren't rejected"

Claude Code reasoning:
1. Identifies keywords: "authentication", "expired tokens", "rejected"
2. Calls: codegraph_query(
     query="authentication token validation expiry",
     token_budget=3500,
     max_depth=1
   )
3. Receives metadata:
   - src/auth/validator.py:23-45 (TokenValidator.validate)
   - src/auth/token.py:67-89 (Token.is_expired)
   - tests/test_auth.py:120-135 (test_expired_token_rejection)
4. Calls read_file() for each path to get actual code
5. Analyzes code, identifies bug in validator.py line 38
6. Implements fix and runs tests
```

---

## 7. Index Schema (simplified)
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
  embedding BLOB NOT NULL        -- 384 float32 values from jina-code-small
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

FAISS flat index (CPU) on `embedding` field with L2 distance.

---

## 8. Secret & Safety Pipeline

**Implementation:**
```bash
uv pip install detect-secrets==1.5.2
```

**Integration in incremental pipeline:**
```
file change → detect-secrets scan → if secret detected → skip file, log warning, continue
                                  → if clean → parse AST → embed
```

**Baseline file path:** `<repo-root>/.secrets.baseline`  
If missing, `detect-secrets` runs with default rules (no historical false-positives honored).

**Behavior:**
- On secret detection:
  - Skip file from indexing
  - Log warning once: `"Skipped indexing src/config.py: secret detected"`
  - Add to `.codegraph/excluded_files.txt`
  - Return structured error to agent (see Section 6)
- **Do not modify user's source code** - user must remove secret manually
- File remains excluded from index until secret is removed and file is re-saved

**Detected patterns** (via detect-secrets defaults):
- AWS keys: `AKIA[0-9A-Z]{16}`
- Private keys: `-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----`
- Generic secrets: `(secret|password|token)[\s]*=[\s]*['"][^'"]{8,}['"]`
- API keys: `api[_-]?key[\s]*=[\s]*['"][^'"]{16,}['"]`
- Base64 high-entropy strings


---

## 9. Fallback & Reliability

### Error Recovery Matrix

| Scenario | Error String | Back-off / Limit | Fallback Data | User Message |
|----------|-------------|------------------|---------------|--------------|
| **SQLite locked** | `OperationalError: database is locked` | Exponential back-off: 0.05s → 5s max | Read-only snapshot from memory cache | "Index busy; showing cached results. Retry shortly." |
| **FAISS corrupt** | `RuntimeError` from `faiss.Index.search()` or `faiss.read_index()` | None (immediate) | Exact-text search in SQLite + queue rebuild | "Search degraded (exact-text) while index rebuilt (~2 min)." |
| **Syntax error** | `SyntaxError` | None | Skip file, continue with others | "utils/aws.py line 42: syntax error – file skipped." |
| **OOM embedding** | `_ArrayMemoryError` | None | Insert row without vector | "File too large; keyword search only for this file." |
| **Import error** | `ModuleNotFoundError` | None | Skip symbol, log warning | "Cannot resolve import 'missing_lib' – skipped." |
| **Index stale** | N/A (time-based) | None | Serve stale results with flag | "Index 3h old. Results may be outdated. Run `codegraph-rebuild`." |

### Agent-Facing Behavior

**On index unavailable:**
- MCP server crash → Claude Code receives structured error
- Agent receives:
  ```json
  {
    "status": "error",
    "error_type": "index_unavailable",
    "message": "CodeGrapher index unavailable. Run codegraph-rebuild to restore.",
    "fallback_suggestion": "Use grep -r <pattern> or find . -name '*.py'"
  }
  ```
- Agent continues with traditional search tools
- User sees one-time notification: "CodeGraph index unavailable; using standard search."

**On partial failures:**
- Some files failed to parse → return partial results with warning
- Index is stale (> 1 hour old) → return results with `index_stale` flag
- Empty results → suggest broader query or alternative tools

**Telemetry (opt-in):**
- Uploads event counts only (no code) to help improve recall
- Events logged: query_success, query_empty, query_error, fallback_used
- Privacy: No code, symbols, or file paths are transmitted

---

## 10. Performance Budget Table
| Scenario | Target | CI Command |
|----------|--------|------------|
| Cold start | ≤ 2 s | `time claude mcp start codegraph` |
| Query latency | ≤ 500 ms | `hyperfine -i 10 'mcp-cli call codegraph_query {...}'` |
| Incremental index (1 file) | ≤ 1 s | `hyperfine 'codegraph-index file.py'` |
| Full index 30 k LOC | ≤ 30 s | `hyperfine 'codegraph-build --full'` |
| RAM idle | ≤ 500 MB | `psrecord` |
| Disk overhead | ≤ 1.5 × repo size | `du` |

---

## 11. Installation & Onboarding Flow

### Prerequisites
- Python 3.10+
- Git repository (for auto-detection)
- 500 MB free disk space
- 2 GB RAM
- **Network access** for first-run model download from Hugging Face

### Installation Steps
```bash
# Install via uv pip
uv pip install codegraph

# Initialize in project directory
cd /path/to/your/project
codegraph init

# Auto-detected: repo root, Python files, .gitignore patterns
# Creates: .codegraph/ directory
# Downloads: jina-embeddings-v2-base-code (~307 MB) from Hugging Face
# Builds initial index (may take 10-60s depending on repo size)

# Add to Claude Code MCP config
codegraph mcp-config >> ~/.config/claude/mcp_servers.json
```

**If Hugging Face download fails** (network error, rate limit, offline):
- Server exits with code 5
- Error message: `Failed to download 'jinaai/jina-embeddings-v2-base-code'. Options: (1) Fix network and retry, (2) Pre-download model to ~/.cache/huggingface/ with 'huggingface-cli download jinaai/jina-embeddings-v2-base-code', or (3) Wait for network access.`
- **No keyword-only fallback in v1.0** (embeddings are required; see Section 18 for v2 plans)

### Auto-Detection Logic
1. Walk up directory tree to find `.git` folder → repo root
2. Scan for `*.py` files (respects `.gitignore`)
3. Estimate index time based on LOC count
4. Build initial index with progress bar

### First Successful Query
```bash
# Test from command line
codegraph query "authentication logic" --token-budget 3500

# Or from Claude Code
> Help me understand the authentication flow in this codebase
[Claude Code automatically calls codegraph_query behind the scenes]
```

---

## 12. Schema Evolution & Migration

### Version Management
- Schema version stored in `index_meta` table: `key='schema_version', value='1.0'`
- On startup, check schema version vs. code version

### Migration Strategy (v1.0 → v1.1)
```sql
-- Example: adding a new column in v1.1
ALTER TABLE symbols ADD COLUMN complexity_score INTEGER DEFAULT 0;

-- Update schema version
UPDATE index_meta SET value='1.1' WHERE key='schema_version';
```

### Breaking Changes

**On incompatible schema change:**
1. Server detects version mismatch on startup (reads `index_meta.schema_version`)
2. **Interactive mode** (TTY detected): Prompts user:
   ```
   Index schema v1.0 → v1.1 requires rebuild (~30s). Proceed? [Y/n]
   ```
3. **Non-interactive mode** (CI/scripts): Auto-rebuild with warning to stderr:
   ```
   WARNING: Schema upgraded v1.0 → v1.1, rebuilding index...
   ```
4. **Backup created automatically:** `.codegraph/symbols.db.backup.v1.0` before modification
5. If rebuild fails, restore from backup and exit with error

**User must approve?** Only in interactive mode. Non-interactive auto-rebuilds to prevent agent workflow breakage.

---

## 13. Testing Strategy

### Test Harness Components

**1. Ground Truth Dataset** (`fixtures/ground_truth.jsonl`)
- 20 real-world tasks from open-source Python projects
- **Licence:** All ground-truth repos are MIT or Apache-2.0 licensed. Full attribution in `fixtures/NOTICE.txt`. Dataset is redistributable under same terms as CodeGrapher (MIT).
- Each task includes:
  ```json
  {
    "task_id": "task_001",
    "description": "Fix JWT token expiration bug",
    "repo": "https://github.com/example/auth-service",
    "commit_before": "abc123",
    "files_edited": ["src/auth/token.py", "tests/test_token.py"],
    "query_terms": ["token", "expiration", "jwt"]
  }
  ```

**2. Evaluation Script** (`scripts/eval_token_save.py`)
- Runs each task twice: with and without CodeGrapher
- Measures:
  - Tokens sent to Claude API
  - Task completion rate
  - Time to completion
  - Number of tool calls
- Outputs comparison report

**3. Recall/Precision Tests**
- For each task, call `codegraph_query` with `query_terms`
- Check if `files_edited` are in top-K results
- Compute recall = (files_found / files_edited)
- Compute precision = (files_edited / files_returned)

**4. Representative Task Selection**
- Mix of: bug fixes (8), refactoring (6), feature additions (6)
- Repos sized: 5k LOC (5 tasks), 20k LOC (10 tasks), 40k LOC (5 tasks)
- Diverse domains: web APIs, data processing, CLI tools, ML pipelines

**5. Example Ground Truth Entry**

```json
{
  "task_id": "task_001",
  "description": "Fix JWT token expiration bug where tokens past 24h are still accepted",
  "repo": "https://github.com/pallets/flask-jwt-extended",
  "commit_before": "a1b2c3d4e5f",
  "commit_after": "e4f5g6h7i8j",
  
  "cursor_file": "flask_jwt_extended/jwt_manager.py",
  "query_terms": ["jwt", "token", "expiration", "verify", "decode"],
  
  "files_edited": [
    "flask_jwt_extended/jwt_manager.py",
    "tests/test_decode_tokens.py"
  ],
  
  "expected_bundle_should_contain": [
    "flask_jwt_extended/jwt_manager.py:45-67",
    "flask_jwt_extended/config.py:23-34",
    "flask_jwt_extended/utils.py:89-102",
    "flask_jwt_extended/tokens.py:120-145"
  ],
  
  "baseline_tokens_sent": 8247,
  "expected_tokens_with_codegraph": 2850,
  "expected_token_reduction_pct": 65.4,
  
  "agent_task_should_succeed": true,
  "notes": "Bug is in jwt_manager.py line 52 - missing expiration check"
}
```

This format allows automated evaluation:
1. Run agent with baseline (no CodeGrapher) → measure tokens + success
2. Run agent with CodeGrapher → measure tokens + success
3. Compare actual vs expected token reduction
4. Verify files_edited were in the returned bundle (recall metric)
5. Count false positives (files in bundle but not edited)

---

## 14. Documentation Requirements

### User-Facing Docs

**README.md**
- Installation (uv pip, mcp config)
- Quick start example
- Troubleshooting common issues
- Link to full docs

**docs/user-guide.md**
- How CodeGrapher works (high-level)
- When to use vs. grep/find
- Performance tips (index frequency, token budgets)
- Privacy guarantees

**docs/troubleshooting.md**
- "Index is stale" → run rebuild
- "Empty results" → try broader query
- "Secret detected" → check config
- Performance issues → check RAM/disk

### Developer Docs

**docs/architecture.md**
- Component diagram (parser, indexer, query engine)
- Data flow: file → AST → embeddings → index → query
- Extension points for v2 features

**docs/api-reference.md**
- Full MCP tool specification
- All parameters with types and defaults
- Error response codes and meanings

**CONTRIBUTING.md**
- How to add test cases
- Code style guide (Black, mypy)
- PR review process

---

## 17. Implementation Recipes (for Agent Coding)

### Recipe 1: Build Target Pruning via Import Closure

**Purpose:** Keep only symbols reachable from user's cursor location.

**Algorithm:**
```python
def prune_by_import_closure(
    candidate_symbols: list[Symbol],
    cursor_file: Path,
    repo_root: Path
) -> list[Symbol]:
    """
    Keep only symbols reachable via import graph from cursor_file.
    Drops symbols with infinite import distance.
    """
    # Build import graph: file -> list[imported_files]
    import_graph = defaultdict(list)
    for py_file in repo_root.rglob("*.py"):
        imports = extract_imports(py_file)  # uses ast.Import, ast.ImportFrom
        import_graph[py_file] = imports
    
    # BFS from cursor_file to find reachable files
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
    
    # Filter symbols to only those in reachable files
    return [s for s in candidate_symbols if Path(s.file) in reachable]

def extract_imports(filepath: Path) -> list[Path]:
    """Extract all imported module paths from a Python file."""
    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=str(filepath))
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(resolve_import_to_path(alias.name, filepath))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(resolve_import_to_path(node.module, filepath))
    
    return [p for p in imports if p]  # filter None
```

**Edge cases:**
- Circular imports: BFS naturally handles via `visited` set
- Missing imports: `resolve_import_to_path` returns `None`, gets filtered
- Dynamic imports (`importlib`): Not tracked (acceptable for v1)

---

### Recipe 2: Incremental AST Diff

**Purpose:** Update index in <1s for file changes without full re-parse.

**Algorithm:**
```python
from dataclasses import dataclass
from typing import Optional
import pickle

@dataclass
class SymbolDiff:
    deleted: list[str]      # symbol IDs to remove
    added: list[Symbol]     # new symbols to insert
    modified: list[Symbol]  # changed symbols to re-embed

class IncrementalIndexer:
    def __init__(self, cache_size=50):
        self.ast_cache = {}  # LRU cache of {file_path: pickled_ast}
        self.cache_size = cache_size
    
    def update_file(self, file_path: Path, new_source: str) -> SymbolDiff:
        """
        Compare new AST with cached old AST, return minimal diff.
        Raises SyntaxError if new_source is invalid.
        """
        # Step 1: Parse new AST
        try:
            new_ast = ast.parse(new_source, filename=str(file_path), mode='exec')
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}:{e.lineno} - skipping")
            raise
        
        # Step 2: Load old AST from cache
        old_ast = self.ast_cache.get(file_path)
        if not old_ast:
            # First time seeing this file - all symbols are "added"
            new_symbols = extract_symbols(new_ast, file_path)
            self.ast_cache[file_path] = pickle.dumps(new_ast)
            return SymbolDiff(deleted=[], added=new_symbols, modified=[])
        
        old_ast = pickle.loads(old_ast)
        
        # Step 3: Build symbol multisets keyed by (name, type)
        old_symbols = {
            (node.name, node.__class__.__name__): node 
            for node in old_ast.body 
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
        }
        
        new_symbols = {
            (node.name, node.__class__.__name__): node 
            for node in new_ast.body 
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
        }
        
        # Step 4: Compute diff
        deleted_keys = set(old_symbols.keys()) - set(new_symbols.keys())
        added_keys = set(new_symbols.keys()) - set(old_symbols.keys())
        common_keys = set(old_symbols.keys()) & set(new_symbols.keys())
        
        diff = SymbolDiff(deleted=[], added=[], modified=[])
        
        # Deleted symbols
        diff.deleted = [f"{file_path.stem}.{key[0]}" for key in deleted_keys]
        
        # Added symbols
        for key in added_keys:
            node = new_symbols[key]
            diff.added.append(node_to_symbol(node, file_path))
        
        # Modified symbols (same name/type but different content)
        for key in common_keys:
            old_node, new_node = old_symbols[key], new_symbols[key]
            
            if needs_reembed(old_node, new_node):
                diff.modified.append(node_to_symbol(new_node, file_path))
        
        # Step 5: Update cache
        self.ast_cache[file_path] = pickle.dumps(new_ast)
        if len(self.ast_cache) > self.cache_size:
            # Evict oldest (simple FIFO for v1, can upgrade to LRU)
            oldest = next(iter(self.ast_cache))
            del self.ast_cache[oldest]
        
        return diff

def needs_reembed(old_node: ast.AST, new_node: ast.AST) -> bool:
    """Check if symbol changed in a way that requires re-embedding."""
    # Compare line numbers (moved?)
    if old_node.lineno != new_node.lineno:
        return True
    
    # Compare docstrings
    old_doc = ast.get_docstring(old_node)
    new_doc = ast.get_docstring(new_node)
    if old_doc != new_doc:
        return True
    
    # For functions: compare signature
    if isinstance(old_node, ast.FunctionDef):
        old_sig = format_signature(old_node)
        new_sig = format_signature(new_node)
        if old_sig != new_sig:
            return True
        
        # Compare decorators
        old_decs = [d.id for d in old_node.decorator_list if isinstance(d, ast.Name)]
        new_decs = [d.id for d in new_node.decorator_list if isinstance(d, ast.Name)]
        if old_decs != new_decs:
            return True
    
    return False
```

**Performance guarantee:**
- `ast.parse()`: ~5ms for 500-line file
- Multiset diff: O(n) where n = number of top-level symbols (~10-50)
- Total: <50ms parsing + <150ms DB transaction = <200ms

---

### Recipe 3: Weighted Score Ranking

**Purpose:** Combine vector similarity, PageRank, recency into single score.

**Algorithm:**
```python
import numpy as np
from datetime import datetime, timedelta

def compute_bundle_scores(
    candidates: list[Symbol],
    query_embedding: np.ndarray,
    pagerank_scores: dict[str, float],  # symbol_id -> score
    git_log: dict[str, datetime],       # file_path -> last_modified
) -> list[tuple[Symbol, float]]:
    """
    Compute weighted score for each candidate symbol.
    Returns sorted list of (symbol, score) tuples.
    """
    scored = []
    max_pr = max(pagerank_scores.values()) if pagerank_scores else 1.0
    
    for symbol in candidates:
        # Component 1: Vector similarity (0-1)
        cosine_sim = np.dot(query_embedding, symbol.embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(symbol.embedding)
        )
        norm_cosine = (cosine_sim + 1) / 2  # map [-1,1] to [0,1]
        
        # Component 2: PageRank centrality (0-1)
        raw_pr = pagerank_scores.get(symbol.id, 0.0)
        norm_pr = raw_pr / max_pr if max_pr > 0 else 0.0
        
        # Component 3: Recency (piecewise constant)
        last_mod = git_log.get(symbol.file, datetime.min)
        days_ago = (datetime.now() - last_mod).days
        if days_ago <= 7:
            recency = 1.0
        elif days_ago <= 30:
            recency = 0.5
        else:
            recency = 0.1
        
        # Component 4: Test file bonus
        is_test = 1.0 if Path(symbol.file).stem.startswith("test_") else 0.0
        
        # Weighted combination
        score = (
            0.60 * norm_cosine +
            0.25 * norm_pr +
            0.10 * recency +
            0.05 * is_test
        )
        
        scored.append((symbol, score))
    
    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def truncate_at_token_budget(
    scored_symbols: list[tuple[Symbol, float]],
    token_budget: int
) -> list[Symbol]:
    """
    Truncate symbol list to fit within token budget.
    Always breaks at file boundaries.
    """
    selected = []
    tokens_used = 0
    current_file = None
    file_buffer = []
    
    for symbol, score in scored_symbols:
        symbol_tokens = estimate_tokens(symbol)
        
        # If new file, check if we can fit entire file
        if symbol.file != current_file:
            if current_file is not None:
                # Commit previous file if it fits
                file_tokens = sum(estimate_tokens(s) for s in file_buffer)
                if tokens_used + file_tokens <= token_budget:
                    selected.extend(file_buffer)
                    tokens_used += file_tokens
                else:
                    break  # Can't fit more files
            
            # Start new file buffer
            current_file = symbol.file
            file_buffer = [symbol]
        else:
            file_buffer.append(symbol)
    
    # Commit final file
    if file_buffer:
        file_tokens = sum(estimate_tokens(s) for s in file_buffer)
        if tokens_used + file_tokens <= token_budget:
            selected.extend(file_buffer)
    
    return selected

def estimate_tokens(symbol: Symbol) -> int:
    """
    Estimate tokens for markdown output of one symbol.
    Format: | filepath | line_range | symbol_name | excerpt |
    """
    # Rough heuristic: 1 token ≈ 4 chars
    line_range_str = f"{symbol.start_line}-{symbol.end_line}"
    excerpt = symbol.signature[:50] + "..."
    
    row_chars = (
        len(symbol.file) + 
        len(line_range_str) + 
        len(symbol.id) + 
        len(excerpt) + 
        10  # markdown syntax
    )
    
    return row_chars // 4
```

---

### Recipe 4: FAISS Index Operations

**Purpose:** Efficient vector similarity search and incremental updates.

**Algorithm:**
```python
import faiss
import numpy as np

class FAISSIndexManager:
    def __init__(self, dim=768, index_path=".codegraph/index.faiss"):
        self.dim = dim
        self.index_path = Path(index_path)
        self.index = faiss.IndexFlatL2(dim)  # L2 distance for CPU
        self.symbol_ids = []  # parallel list to index
        
        if self.index_path.exists():
            self.load()
    
    def add_symbols(self, symbols: list[Symbol]):
        """Add new symbols to index."""
        if not symbols:
            return
        
        embeddings = np.array([s.embedding for s in symbols], dtype='float32')
        self.index.add(embeddings)
        self.symbol_ids.extend([s.id for s in symbols])
        self.save()
    
    def remove_symbols(self, symbol_ids: list[str]):
        """
        Remove symbols from index.
        Note: FAISS IndexFlatL2 doesn't support removal, so rebuild.
        """
        if not symbol_ids:
            return
        
        # Filter out removed symbols
        removed_set = set(symbol_ids)
        keep_indices = [i for i, sid in enumerate(self.symbol_ids) 
                       if sid not in removed_set]
        
        if not keep_indices:
            # Empty index
            self.index = faiss.IndexFlatL2(self.dim)
            self.symbol_ids = []
            self.save()
            return
        
        # Rebuild index with kept vectors
        old_vectors = np.array([
            self.index.reconstruct(i) for i in keep_indices
        ], dtype='float32')
        
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(old_vectors)
        self.symbol_ids = [self.symbol_ids[i] for i in keep_indices]
        self.save()
    
    def search(self, query_embedding: np.ndarray, k=20) -> list[tuple[str, float]]:
        """
        Search for k nearest neighbors.
        Returns list of (symbol_id, distance) tuples.
        Raises IndexCorruptedError if FAISS index is corrupted.
        """
        try:
            query = np.array([query_embedding], dtype='float32')
            distances, indices = self.index.search(query, k)
        except RuntimeError as e:
            logger.error(f"FAISS index corrupted: {e}")
            raise IndexCorruptedError("FAISS search failed - index rebuild required")
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.symbol_ids):  # valid index
                results.append((self.symbol_ids[idx], float(dist)))
        
        return results
    
    def save(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        
        # Save symbol_ids separately
        ids_path = self.index_path.with_suffix('.ids')
        with open(ids_path, 'w') as f:
            f.write('\n'.join(self.symbol_ids))
    
    def load(self):
        """Load index from disk. Raises IndexCorruptedError if corrupted."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            
            ids_path = self.index_path.with_suffix('.ids')
            with open(ids_path) as f:
                self.symbol_ids = [line.strip() for line in f]
        except RuntimeError as e:
            logger.error(f"Failed to load FAISS index (corrupted): {e}")
            raise IndexCorruptedError("Cannot load FAISS index - rebuild required")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Fallback: empty index
            self.index = faiss.IndexFlatL2(self.dim)
            self.symbol_ids = []
```

**Performance notes:**
- `IndexFlatL2` is brute-force but fast on CPU for <100k vectors (~10ms for k=20)
- L2 distance equivalent to cosine similarity when vectors are normalized
- For >100k vectors, upgrade to `IndexIVFFlat` (v2)

---

### Recipe 5: Transaction Safety

**Purpose:** Ensure SQLite + FAISS stay consistent even on crash.

**Algorithm:**
```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def atomic_update(db_path: Path, faiss_manager: FAISSIndexManager):
    """
    Context manager for atomic updates across SQLite + FAISS.
    Rolls back both on exception.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("BEGIN IMMEDIATE")  # lock database
    
    # Snapshot FAISS state
    faiss_backup = {
        'index': faiss_manager.index,
        'symbol_ids': faiss_manager.symbol_ids.copy()
    }
    
    try:
        yield conn  # caller does SQL updates
        
        # Caller has updated FAISS in memory
        # Now commit both atomically
        conn.commit()
        faiss_manager.save()  # write to disk
        
    except Exception as e:
        # Rollback SQLite
        conn.rollback()
        
        # Rollback FAISS to snapshot
        faiss_manager.index = faiss_backup['index']
        faiss_manager.symbol_ids = faiss_backup['symbol_ids']
        
        logger.error(f"Transaction failed: {e}")
        raise
    
    finally:
        conn.close()

# Usage example
with atomic_update(db_path, faiss_mgr) as conn:
    # Delete old symbols from SQLite
    conn.executemany(
        "DELETE FROM symbols WHERE id = ?",
        [(sid,) for sid in diff.deleted]
    )
    
    # Delete from FAISS
    faiss_mgr.remove_symbols(diff.deleted)
    
    # Insert new symbols to SQLite
    conn.executemany(
        "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(s.id, s.file, s.start_line, s.end_line, 
          s.signature, s.doc, s.mutates, s.embedding.tobytes()) 
         for s in diff.added]
    )
    
    # Add to FAISS
    faiss_mgr.add_symbols(diff.added)
    
    # If we get here, both commit atomically
```

---

## 18. Open Questions & Nice-to-have (post v1)
- Support for `@dataclass` field-level mutations.  
- Automatic re-rank by test-coverage heat-map.  
- FAISS IVF index for > 100 k symbols (memory scaling).  
- Web transport & auth for team server.
- Multi-language support starting with TypeScript/JavaScript
- Integration with IDE LSP servers
- Real-time collaboration features
- **Keyword-only fallback mode** (`--keyword-only` flag): Exact-text search without embeddings for offline/air-gapped environments. Uses SQLite FTS5 for full-text search on signature + docstring fields.

---

## 19. Definition of Done
- All acceptance criteria in section 3 are **green on CI**
- Test harness with 20 real tasks running nightly
- Documentation complete (README, user guide, API reference, troubleshooting)
- Integration tested with Claude Code via MCP config
- Agent successfully completes tasks at baseline rate (±5%)
- Repo tagged `v1.0.0`
- Published to PyPI as `codegraph`
- Announced in Claude Code community forum
- MIT license file included
- CHANGELOG.md with v1.0 release notes

---

*End of PRD — ready for implementation*
