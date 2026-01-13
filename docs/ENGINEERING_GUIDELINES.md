# CodeGrapher Engineering Guidelines
**For: Coding Agents (Claude Code, Cursor, etc.)**  
**Version: 2.0**  
**Applies to: PRD v1.0 + Implementation Plan v2.0**

---

## Purpose

This document provides explicit constraints and patterns for AI coding agents implementing CodeGrapher. It unpacks the assumptions:
- **Competent implementation** (not over-engineered)
- **Proper use of libraries** (NetworkX, FAISS, detect-secrets)
- **No scope creep** beyond PRD v1.0
- **Standard Python style** (not golfed or verbose)

**CRITICAL**: Read this document in full before starting Phase 1. All phases must follow these guidelines.

---

## 1. Competent Implementation (Not Over-Engineered)

### Rule 1.1: Prefer Simple Over Clever
- **DO**: Use direct implementations with clear logic flow
- **DON'T**: Create design patterns unless the PRD explicitly requires flexibility

**Examples:**

❌ **Over-engineered:**
```python
class SymbolExtractorFactory:
    """Factory pattern for symbol extraction strategies."""
    @staticmethod
    def create_extractor(node_type: str) -> BaseExtractor:
        return _extractors[node_type]()

class FunctionExtractor(BaseExtractor):
    def extract(self, node: ast.FunctionDef) -> Symbol:
        ...
```

✅ **Competent:**
```python
def extract_symbols(tree: ast.AST, file_path: Path) -> List[Symbol]:
    """Extract all symbols from an AST."""
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            symbols.append(_extract_function(node, file_path))
        elif isinstance(node, ast.ClassDef):
            symbols.append(_extract_class(node, file_path))
    return symbols
```

**Why**: The simple version is easier to debug, test, and modify. The factory adds 3x the code with zero flexibility benefit in v1.0.

---

### Rule 1.2: No Premature Abstraction
- **DO**: Write concrete implementations first
- **DON'T**: Create base classes, protocols, or interfaces unless reused ≥3 times

**Test**: If you write `class BaseX(ABC)`, ask: "Do I have 3+ concrete implementations?"
- If no → delete the base class, use concrete types

**Exception**: Pydantic models and dataclasses are not abstractions; they're data structures.

---

### Rule 1.3: Limit Indirection
- **Max 1 level** of function delegation for simple operations
- **Max 2 levels** for complex operations

❌ **Too indirect:**
```python
def query() -> Results:
    return self._execute_query()

def _execute_query() -> Results:
    return self._perform_search()

def _perform_search() -> Results:
    return self._faiss_search()
```

✅ **Direct:**
```python
def query() -> Results:
    """Execute a code search query."""
    candidates = self._faiss_search()
    ranked = self._rank_by_score(candidates)
    return self._format_results(ranked)
```

---

## 2. Proper Use of Libraries

### Rule 2.1: Verify Before Using
**CRITICAL**: Before calling ANY library method, verify it exists in current version.

**Process**:
1. Check if method exists in library's official documentation
2. Use Context7 to verify current API if unsure
3. If still unsure, write a minimal smoke test first:

```python
# Verify before building main code
import networkx as nx
G = nx.DiGraph()
assert hasattr(nx, 'pagerank'), "PageRank not available"
result = nx.pagerank(G)  # Test it works
```

**Why**: Agents frequently hallucinate library APIs. This wastes hours on debugging non-existent methods.

---

### Rule 2.2: Use Libraries As Intended
Each library has a specific purpose. Don't misuse or reimplement.

| Library | Purpose | DO | DON'T |
|---------|---------|----|----|
| **NetworkX** | Graph algorithms | Use `nx.pagerank()` | Reimplement PageRank |
| **FAISS** | Vector search | Use `IndexFlatL2` | Use for general storage |
| **detect-secrets** | Secret scanning | Use as CLI wrapper | Parse its output format |
| **watchdog** | File watching | Use `Observer` pattern | Use for task scheduling |
| **transformers** | Model loading | Use `from_pretrained()` | Implement custom loaders |

❌ **Misuse example:**
```python
def my_pagerank(graph):
    """Custom PageRank implementation."""
    # 50 lines of custom algorithm...
    # This violates Rule 2.2 - NetworkX does this
```

✅ **Correct use:**
```python
import networkx as nx

def compute_pagerank(edges: List[Edge]) -> Dict[str, float]:
    """Compute PageRank scores for call graph."""
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge.caller_id, edge.callee_id)
    return nx.pagerank(G, alpha=0.85, max_iter=100)
```

---

### Rule 2.3: Minimal Library Surface Area
Only use the specific methods required by the PRD. Avoid "exploring" library features.

**Allowed methods by library:**

**NetworkX:**
- `DiGraph()` - create directed graph
- `add_edge()` - add edges
- `pagerank()` - compute PageRank
- **Forbidden**: Graph generators, custom algorithms, visualization

**FAISS:**
- `IndexFlatL2(dim)` - create flat L2 index
- `add(vectors)` - add vectors
- `search(query, k)` - find neighbors
- `write_index()` / `read_index()` - persistence
- **Forbidden**: GPU indices, training, quantization, clustering

**watchdog:**
- `Observer()` - file system watcher
- `FileSystemEventHandler` - event handler base
- `on_modified()` - file change callback
- **Forbidden**: Polling observers, complex pattern matching

**transformers:**
- `AutoModel.from_pretrained(model_id)` - load model
- `AutoTokenizer.from_pretrained(model_id)` - load tokenizer
- `model.encode(text)` - generate embeddings
- **Forbidden**: Training, fine-tuning, pipelines, custom architectures

---

## 3. No Scope Creep Beyond PRD v1.0

### Rule 3.1: PRD as Contract
The PRD defines the complete scope. Do not add features.

**Decision framework:**
1. Is this feature mentioned in PRD Sections 1-19? → **YES**: implement
2. Is this in "Out of scope (v1)" or "Open Questions (post v1)"? → **NO**: skip
3. Is this a "nice to have" or "might be useful"? → **NO**: skip

---

### Rule 3.2: Out-of-Scope Feature List
**DO NOT implement** any of the following, even if they seem helpful:

❌ **Forbidden features:**
- Web UI or dashboard
- Export to JSON/CSV/YAML (only MCP output format)
- Configuration file hot-reloading
- Telemetry beyond PRD Section 9 specifications
- Multi-language support (only Python)
- Plugin system or extension API
- Advanced FAISS indices (IVF, GPU, PQ)
- Visualization of call graphs
- Code complexity metrics
- Dependency graph visualization
- Email notifications
- Slack/Discord integrations

---

### Rule 3.3: No Gold-Plating
Don't improve beyond PRD requirements.

❌ **Gold-plating:**
```python
# PRD says: "log warning once"
logger.warning(f"Skipped {file}: secret detected")

# Agent adds unnecessary features:
class ExclusionReporter:
    def generate_html_report(self): ...
    def send_email_alert(self): ...
    def export_to_csv(self): ...
```

✅ **Spec-compliant:**
```python
logger.warning(f"Skipped indexing {file}: secret detected")
# Done. Nothing more needed.
```

---

### Rule 3.4: Feature Flag Detection
If you catch yourself writing:
- `if advanced_mode:`
- `if enable_experimental:`
- `if use_new_algorithm:`
- `if config.get('feature_x_enabled'):`

**STOP**. These indicate scope creep. Delete them immediately.

**Exception**: The PRD explicitly mentions optional parameters (e.g., `include_snippets` in MCP tool). These are allowed.

---

### Rule 3.5: MCP Configuration Generation
Phase 10 requires generating `config/mcp_server.json`. This is the ONLY configuration file needed.

✅ **Required:**
```json
{
  "mcpServers": {
    "codegrapher": {
      "command": "python",
      "args": ["-m", "codegrapher.server"]
    }
  }
}
```

❌ **Forbidden additions:**
```json
{
  "mcpServers": {
    "codegrapher": {
      "command": "python",
      "args": ["-m", "codegrapher.server"],
      "env": {
        "CODEGRAPHER_LOG_LEVEL": "DEBUG",  // ❌ Not in PRD
        "CODEGRAPHER_CACHE_SIZE": "1000"   // ❌ Not in PRD
      },
      "timeout": 30000,  // ❌ Not in PRD
      "restartOnCrash": true  // ❌ Not in PRD
    }
  }
}
```

Keep it minimal. Users can modify it themselves if needed.

---

## 4. Standard Python Style

### Rule 4.1: PEP 8 with Type Hints
All code must follow PEP 8 with mandatory type hints.

**Template:**
```python
def extract_symbols(
    file_path: Path,
    include_private: bool = False
) -> List[Symbol]:
    """Extract symbols from a Python file.
    
    Args:
        file_path: Path to the Python file
        include_private: Whether to include private symbols (_name)
    
    Returns:
        List of Symbol objects found in the file
        
    Raises:
        SyntaxError: If the file contains invalid Python syntax
    """
    ...
```

**Requirements:**
- Type hints on ALL function signatures (no exceptions)
- Docstrings for public functions (Google style)
- 4-space indentation (no tabs)
- Max line length: **100 characters** (not 80, not 120)
- Use `black` formatter with `--line-length 100`

---

### Rule 4.2: Explicit Over Implicit
Code should be self-documenting through clear names and types.

❌ **Implicit:**
```python
def process(data):
    return [x for x in data if x]
```

✅ **Explicit:**
```python
def filter_empty_symbols(symbols: List[Symbol]) -> List[Symbol]:
    """Remove symbols with empty signatures."""
    return [s for s in symbols if s.signature]
```

---

### Rule 4.3: Error Messages Must Be Actionable
Every error message must tell the user WHAT to do, not just WHAT went wrong.

❌ **Bad:**
```python
raise ValueError("Invalid query")
```

❌ **Still bad:**
```python
raise ValueError(f"Query '{query}' is invalid")
```

✅ **Good:**
```python
raise ValueError(
    f"Query '{query}' must be 3-100 characters long. "
    f"Got {len(query)} characters. "
    f"Try a more specific search term like 'authentication' or 'parse_config'."
)
```

**Pattern**: `[What's wrong] + [Why it's wrong] + [How to fix it]`

---

### Rule 4.4: No Magic Numbers
All constants must be named with clear semantic meaning.

❌ **Bad:**
```python
if tokens > 3500:
    truncate()
    
if days_ago > 30:
    recency = 0.1
```

✅ **Good:**
```python
# From PRD Section 6
DEFAULT_TOKEN_BUDGET = 3500

# From PRD Recipe 3 - recency scoring
RECENCY_RECENT_DAYS = 7
RECENCY_MEDIUM_DAYS = 30

if tokens > token_budget:
    truncate()
    
if days_ago > RECENCY_MEDIUM_DAYS:
    recency = 0.1
```

**Exception**: Mathematical constants (0, 1, -1, 0.5) in obvious contexts don't need names.

---

### Rule 4.5: Import Order (PEP 8)
Organize imports in three groups with blank lines between:

```python
# 1. Standard library
import ast
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party packages
import faiss
import numpy as np
from pydantic import BaseModel
import networkx as nx

# 3. Local application
from codegrapher.models import Symbol, Edge
from codegrapher.parser import extract_symbols
```

Within each group, sort alphabetically.

---

## 5. Code Organization Rules

### Rule 5.1: One Responsibility Per Module
Each `.py` file should have a single clear purpose.

**Module map:**
```
src/codegrapher/
├── models.py          # Data models only (Symbol, Database)
├── parser.py          # AST parsing only
├── resolver.py        # Import resolution only
├── graph.py           # Call graph + PageRank only
├── vector_store.py    # Embeddings + FAISS only
├── secrets.py         # Secret detection only
├── indexer.py         # Incremental indexing only
├── watcher.py         # File watching only
├── query.py           # Query execution + ranking only
├── server.py          # MCP server only
└── cli.py             # CLI commands only
```

**Forbidden module names:**
- ❌ `utils.py` (too vague - what utilities?)
- ❌ `helpers.py` (what does it help?)
- ❌ `common.py` (common to what?)
- ❌ `misc.py` (catch-all dumps)

**Exception**: If you need shared constants, create `constants.py` with only constant definitions.

---

### Rule 5.2: Module Size Limits
- **Target**: 200-400 LOC per module
- **Warning**: 400-600 LOC (consider splitting)
- **Hard limit**: 600 LOC

**If exceeded**, split by sub-responsibility:

**Example split:**
```
# indexer.py is 750 LOC → split into:
indexer.py        # 350 LOC - IncrementalIndexer class
diff.py           # 250 LOC - SymbolDiff computation
transactions.py   # 150 LOC - atomic_update() context manager
```

**How to count**: `wc -l src/codegrapher/indexer.py` (blank lines included)

---

### Rule 5.3: Directory Structure
The project MUST follow this exact structure:

```
codegrapher/
├── src/
│   └── codegrapher/
│       ├── __init__.py
│       ├── models.py
│       ├── parser.py
│       ├── resolver.py
│       ├── graph.py
│       ├── vector_store.py
│       ├── secrets.py
│       ├── indexer.py
│       ├── watcher.py
│       ├── query.py
│       ├── server.py
│       └── cli.py
├── tests/
│   ├── test_parser.py
│   ├── test_indexer.py
│   ├── test_graph.py
│   ├── test_resolver.py
│   └── test_integration.py
├── scripts/
│   ├── eval_token_save.py
│   └── benchmark.py
├── fixtures/
│   ├── ground_truth.jsonl
│   └── test_repos/
├── config/
│   └── mcp_server.json
├── pyproject.toml
├── README.md
└── ENGINEERING_GUIDELINES.md  # This file
```

**Do not create:**
- `src/codegrapher/core/` (unnecessary nesting)
- `src/codegrapher/utils/` (violates Rule 5.1)
- `lib/` or `pkg/` directories
- Multiple `__init__.py` files with logic

---

### Rule 5.4: Git Hook Files
Phase 9 requires creating a git hook template. Store it as:

```
config/
└── post-commit.template
```

**Implementation:**
```bash
#!/bin/bash
# CodeGrapher post-commit hook
# Auto-generated by `codegraph init`

# Run incremental index update on committed files
codegraph update --git-changed

exit 0
```

**During `codegraph init`:**
1. Copy `config/post-commit.template` → `.git/hooks/post-commit`
2. Make it executable: `chmod +x .git/hooks/post-commit`
3. **Do not** overwrite existing hooks without user confirmation

---

## 6. Testing Guidelines

### Rule 6.1: Test Pyramid
Maintain the correct ratio of test types:

```
       /\
      /  \  ← 5-10 integration tests (15%)
     /────\
    / unit \ ← 30-50 unit tests (85%)
   /────────\
```

**Distribution:**
- **85% unit tests**: Fast (<10ms), isolated, no I/O
- **15% integration tests**: Realistic workflows, may use disk/network
- **0% E2E tests in v1**: Defer to Phase 12 evaluation harness

---

### Rule 6.2: Test File Naming
Test files MUST match production file names exactly:

```
src/codegrapher/parser.py    → tests/test_parser.py
src/codegrapher/indexer.py   → tests/test_indexer.py
src/codegrapher/graph.py     → tests/test_graph.py
```

**Special files:**
```
tests/test_integration.py    # Cross-module workflows
tests/conftest.py            # Pytest fixtures only
```

---

### Rule 6.3: Fixtures Over Mocks
**Prefer**: Real small test data  
**Avoid**: Complex mocking frameworks

❌ **Over-mocked:**
```python
@patch('codegrapher.parser.ast.parse')
@patch('codegrapher.parser.Path.read_text')
@patch('codegrapher.parser.extract_imports')
def test_extract(mock_imports, mock_read, mock_parse):
    mock_parse.return_value = MagicMock(spec=ast.Module)
    mock_parse.return_value.body = [MagicMock()]
    # 20 more lines of mock setup...
```

✅ **Fixture-based:**
```python
def test_extract_function():
    """Test that we correctly extract a simple function."""
    test_file = FIXTURES_DIR / "simple_function.py"
    symbols = extract_symbols(test_file)
    
    assert len(symbols) == 1
    assert symbols[0].signature == "def hello() -> str:"
    assert symbols[0].start_line == 1
```

**Create test fixtures:**
```
fixtures/
├── simple_function.py      # def hello(): pass
├── class_with_methods.py   # class Foo: def bar(): pass
├── relative_imports.py     # from ..utils import x
└── has_secret.py           # AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
```

---

### Rule 6.4: Integration Test Requirements
Phase 11 requires a specific integration test. Implement it as:

```python
def test_incremental_update_survives_crash():
    """Verify index integrity after process kill during update."""
    # 1. Create test repo with 500 LOC
    repo = create_test_repo(loc=500)
    
    # 2. Full index build
    run_command(f"codegraph build --full", cwd=repo)
    
    # 3. Modify one file
    modify_file(repo / "src/main.py", add_function="def new_func(): pass")
    
    # 4. Start incremental update in background
    proc = subprocess.Popen(["codegraph", "update"], cwd=repo)
    time.sleep(0.5)  # Let it start
    
    # 5. Kill process mid-update
    proc.kill()
    proc.wait()
    
    # 6. Verify database is not corrupted
    db = Database(repo / ".codegraph/symbols.db")
    assert db.is_valid(), "Database corrupted after crash"
    
    # 7. Verify we can still query
    result = run_command("codegraph query 'main'", cwd=repo)
    assert result.returncode == 0
```

This verifies AC #9: "agent falls back to full search + warning" on crash.

---

## 7. Performance Guidelines

### Rule 7.1: No Premature Optimization
**Process:**
1. Implement clearly (prioritize correctness)
2. Run benchmarks (Phase 11.5)
3. If metric fails → profile → optimize specific bottleneck
4. If metric passes → done, move on

**Don't optimize** without benchmark evidence.

---

### Rule 7.2: Algorithmic Complexity Targets
From PRD Section 10, these are the complexity targets:

| Operation | Target Time | Max Complexity | Algorithm |
|-----------|-------------|----------------|-----------|
| Query | <500ms | O(k log n) | FAISS search + ranking |
| Incremental update | <1s | O(m) | AST diff where m = changed symbols |
| Full build | <30s for 30k LOC | O(n log n) | Parse + embed + PageRank |

**Forbidden patterns:**
- ❌ O(n²) nested loops in query path
- ❌ O(n) linear search through all symbols (defeats FAISS)
- ❌ Loading entire repo into memory at once

---

### Rule 7.3: Memory Constraints
From PRD Section 10:
- **Idle**: <500 MB RAM
- **Active indexing**: <1 GB RAM
- **Disk overhead**: <1.5× repo size

**Techniques to stay within limits:**

✅ **Stream large operations:**
```python
# ❌ Loads everything into memory
all_files = [parse(f) for f in repo.glob("**/*.py")]
for symbols in all_files:
    index.add(symbols)

# ✅ Streams one file at a time
for file in repo.glob("**/*.py"):
    symbols = parse(file)
    index.add(symbols)
    # symbols garbage collected here
```

✅ **Use generators:**
```python
def iter_symbols(repo: Path) -> Iterator[Symbol]:
    """Yield symbols one at a time."""
    for file in repo.glob("**/*.py"):
        yield from extract_symbols(file)
```

✅ **Clear caches aggressively:**
```python
class IncrementalIndexer:
    def __init__(self, cache_size=50):  # PRD specifies 50
        self.ast_cache = {}  # LRU cache
        
    def update_file(self, file_path, source):
        # ... do work ...
        
        # Evict old entries if cache grows
        if len(self.ast_cache) > self.cache_size:
            oldest = next(iter(self.ast_cache))
            del self.ast_cache[oldest]
```

---

### Rule 7.4: Benchmark Script Requirements
Phase 11.5 requires `scripts/benchmark.py`. It must:

1. **Use hyperfine for timing:**
```python
import subprocess

def benchmark_query_latency():
    """Measure query latency using hyperfine."""
    result = subprocess.run([
        "hyperfine",
        "--warmup", "3",
        "--min-runs", "10",
        "codegraph query 'authentication'"
    ], capture_output=True, text=True)
    
    # Parse result, verify <500ms
    assert "Time (mean ± σ):" in result.stdout
```

2. **Use psrecord for memory:**
```python
def benchmark_memory_usage():
    """Measure idle memory usage."""
    proc = subprocess.Popen(["python", "-m", "codegrapher.server"])
    
    # Let it stabilize
    time.sleep(5)
    
    # Record memory
    psrecord_proc = subprocess.Popen([
        "psrecord", str(proc.pid),
        "--duration", "30",
        "--plot", "memory.png"
    ])
    psrecord_proc.wait()
    
    # Parse and verify <500MB
```

3. **Output markdown table:**
```python
def generate_report():
    """Generate PRD Section 10 style report."""
    return f"""
| Scenario | Target | Actual | Status |
|----------|--------|--------|--------|
| Cold start | ≤2s | {cold_start:.2f}s | {'✅' if cold_start <= 2 else '❌'} |
| Query latency | ≤500ms | {query:.0f}ms | {'✅' if query <= 500 else '❌'} |
| Incremental (1 file) | ≤1s | {incr:.2f}s | {'✅' if incr <= 1 else '❌'} |
| Full index 30k LOC | ≤30s | {full:.0f}s | {'✅' if full <= 30 else '❌'} |
| RAM idle | ≤500MB | {ram:.0f}MB | {'✅' if ram <= 500 else '❌'} |
    """
```

---

## 8. Error Handling Guidelines

### Rule 8.1: Structured Errors
Use the error response format from PRD Section 6.

**Implementation:**
```python
from enum import Enum

class ErrorType(str, Enum):
    """Error types from PRD Section 9."""
    INDEX_UNAVAILABLE = "index_unavailable"
    INDEX_STALE = "index_stale"
    INDEX_CORRUPT = "index_corrupt"
    PARSE_ERROR = "parse_error"
    EMPTY_RESULTS = "empty_results"

class CodeGraphError(Exception):
    """Base for all CodeGrapher errors."""
    
    def __init__(self, message: str, error_type: ErrorType, fallback: str):
        self.message = message
        self.error_type = error_type
        self.fallback_suggestion = fallback
        super().__init__(message)
    
    def to_response(self) -> dict:
        """Convert to MCP error response."""
        return {
            "status": "error",
            "error_type": self.error_type.value,
            "message": self.message,
            "fallback_suggestion": self.fallback_suggestion,
            "partial_results": []
        }

class IndexCorruptedError(CodeGraphError):
    """FAISS index is corrupted."""
    def __init__(self, details: str):
        super().__init__(
            message=f"FAISS index corrupted: {details}",
            error_type=ErrorType.INDEX_CORRUPT,
            fallback="Run codegraph-rebuild --full"
        )
```

---

### Rule 8.2: Fail-Safe Defaults
When in doubt, prefer serving degraded results over hard failures.

✅ **Fail-safe:**
```python
def query(text: str) -> dict:
    """Query the code index."""
    try:
        results = self._faiss_search(text)
    except IndexCorruptedError:
        logger.error("FAISS corrupted, falling back to text search")
        results = self._sqlite_text_search(text)
        return {
            "status": "success",
            "degraded_mode": True,
            "message": "Using text search (vector index corrupted)",
            "results": results
        }
```

❌ **Hard failure:**
```python
def query(text: str) -> dict:
    results = self._faiss_search(text)  # Crashes on corruption
    return {"status": "success", "results": results}
```

**Principle**: It's better to return approximate results with a warning than to fail completely.

---

### Rule 8.3: Never Fail Silently
Every exception must be logged or propagated. No silent `except: pass`.

❌ **Silent failure:**
```python
try:
    index.add(symbol)
except Exception:
    pass  # ❌ Swallows all errors
```

❌ **Still bad:**
```python
try:
    index.add(symbol)
except Exception as e:
    pass  # ❌ Logs nothing
```

✅ **Proper handling:**
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

---

### Rule 8.4: Error Recovery Matrix Implementation
PRD Section 9 defines specific error recovery behaviors. Implement exactly as specified:

```python
def _handle_db_locked(self, retry_count: int = 0) -> None:
    """Handle SQLite 'database is locked' error with exponential backoff."""
    if retry_count >= 5:
        raise DatabaseLockedError("Database locked after 5 retries")
    
    # Exponential backoff: 0.05s → 0.1s → 0.2s → 0.4s → 0.8s
    delay = 0.05 * (2 ** retry_count)
    logger.warning(f"Database locked, retrying in {delay}s (attempt {retry_count + 1}/5)")
    time.sleep(delay)
```

---

## 9. Documentation Requirements

### Rule 9.1: Docstring Coverage
- **100%** for public functions/classes
- **50%** for private helpers (only if complex logic)
- **0%** for trivial getters/setters

**Minimum acceptable docstring:**
```python
def resolve_import_to_path(
    module_name: str,
    current_file: Path,
    repo_root: Path
) -> Optional[Path]:
    """Resolve import string to absolute file path within repo.
    
    Handles both relative imports (e.g., '..utils', '.config') and
    absolute imports (e.g., 'mypackage.foo'). Returns None for
    external libraries.
    
    Args:
        module_name: Import string from ast.Import or ast.ImportFrom
        current_file: File containing the import statement
        repo_root: Root directory of the repository
    
    Returns:
        Absolute Path if import resolves to a repo file, None if
        import is external (stdlib or site-packages)
        
    Examples:
        >>> resolve_import_to_path(
        ...     '..utils',
        ...     Path('src/sub/mod.py'),
        ...     Path('.')
        ... )
        Path('src/utils.py')
        
        >>> resolve_import_to_path('os.path', Path('main.py'), Path('.'))
        None  # stdlib import
    """
```

**Required sections:**
1. One-line summary (what it does)
2. Additional context (how it works, caveats)
3. Args (all parameters)
4. Returns (what it returns)
5. Raises (if applicable)
6. Examples (for complex functions)

---

### Rule 9.2: Comment Complexity, Not Syntax
Comments should explain WHY, not WHAT.

❌ **Useless comments:**
```python
# Increment counter
counter += 1

# Check if symbol is a function
if isinstance(node, ast.FunctionDef):
    ...
```

✅ **Helpful comments:**
```python
# PageRank requires 100+ iterations to converge for repos with >1000 files.
# This was determined empirically from the test harness (see fixtures/).
scores = nx.pagerank(graph, max_iter=100)

# FAISS L2 distance is equivalent to cosine similarity when vectors are
# L2-normalized. We normalize during embedding, so this works correctly.
index = faiss.IndexFlatL2(dim)
```

**Comment when:**
- Algorithm choice is non-obvious
- Magic numbers come from empirical tuning
- Workarounds for library bugs/limitations
- Performance-critical sections

**Don't comment:**
- Standard library usage
- Self-explanatory conditionals
- Obvious loops

---

### Rule 9.3: README.md Content
The README must follow this structure (no more, no less):

```markdown
# CodeGrapher

Local MCP server providing token-efficient code search for Claude Code.

## Installation

```bash
uv pip install codegraph
cd /path/to/your/project
codegraph init
```

## Quick Start

```bash
# Build initial index
codegraph build --full

# Test a query
codegraph query "authentication logic"

# Add to Claude Code
codegraph mcp-config >> ~/.config/claude/mcp_servers.json
```

## Features

- **Hybrid retrieval**: Symbol table + vector similarity + call graph
- **Incremental updates**: <1s index refresh on file changes
- **Token-efficient**: ≥30% token reduction vs full-file search
- **Local & private**: All data stays in `.codegraph/` folder

## Requirements

- Python 3.8+
- 2GB RAM
- 500MB disk space

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md)

## License

MIT
```

**Do not add:**
- Badges (stars, CI status, coverage)
- Architecture diagrams
- Detailed usage examples (those go in docs/)
- Contributor guidelines (separate CONTRIBUTING.md)

---

## 10. Git Commit Guidelines

### Rule 10.1: Commit Per Phase
At the end of each implementation phase, commit with this format:

```bash
git add .
git commit -m "feat(phase-N): <description>"
```

**Type prefixes:**
- `feat`: New feature or capability
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or updating tests
- `docs`: Documentation only

**Examples:**
```bash
git commit -m "feat(phase-3): implement AST parser and symbol extraction"
git commit -m "feat(phase-8): add incremental indexing with transaction safety"
git commit -m "test(phase-12): add ground truth evaluation harness"
git commit -m "fix(phase-10): handle empty query results gracefully"
```

---

### Rule 10.2: Commits Should Build
Every commit must pass basic checks:

```bash
# Before committing:
python -m pytest tests/          # All tests pass
python -m mypy src/              # Type checks pass
python -m black src/ tests/      # Code formatted
```

**Never commit:**
- Broken code (syntax errors)
- Failing tests
- Type errors
- Unformatted code

**Exception**: Work-in-progress commits on feature branches (not main).

---

### Rule 10.3: Commit Message Body (Optional)
For complex changes, add a body explaining WHY:

```bash
git commit -m "feat(phase-6): implement import resolution with namespace support

Handles PEP 420 namespace packages by checking for __init__.py
absence. This was needed because modern repos (like portions of
the standard library) use implicit namespace packages.

Fixes edge case where 'import foo.bar' would fail if foo/ had
no __init__.py despite bar.py existing."
```

---

## 11. Decision Flowcharts

### Flowchart 11.1: When to Create a Class vs Function

```
Does the implementation need mutable state?
  ├─ YES → Use a class
  │    └─ Examples: IncrementalIndexer (has AST cache),
  │                  FAISSIndexManager (has index state)
  │
  └─ NO → Are there >3 related functions that share data?
       ├─ YES → Use a class as namespace
       │    └─ Examples: Database (shares connection),
       │                  EdgeExtractor (shares graph)
       │
       └─ NO → Use module-level functions
            └─ Examples: extract_symbols(), compute_pagerank()
```

---

### Flowchart 11.2: When to Use a Library vs Implement

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

**Example applications:**
- PageRank (NetworkX already in deps) → Use `nx.pagerank()`
- File watching (watchdog in deps) → Use `watchdog.Observer`
- Vector search (FAISS in deps) → Use `faiss.IndexFlatL2`
- AST parsing (stdlib) → Use `ast` module

---

### Flowchart 11.3: How to Handle Errors

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

## 12. Quick Reference Checklist

Use this checklist before marking a phase as "complete":

### Code Quality
- [ ] No design patterns unless PRD requires them
- [ ] All library methods verified to exist (Context7 check)
- [ ] No features outside PRD scope
- [ ] Type hints on all public functions
- [ ] Docstrings on all public APIs
- [ ] Error messages are actionable (tell user what to do)
- [ ] No magic numbers (all constants named)
- [ ] Module is <600 LOC

### Testing
- [ ] Unit tests written for new functionality
- [ ] All tests pass (`pytest tests/`)
- [ ] Type checks pass (`mypy src/`)
- [ ] Code formatted (`black --check src/ tests/`)

### Documentation
- [ ] Public functions have docstrings
- [ ] Complex logic has explanatory comments
- [ ] README updated if user-facing changes

### Performance
- [ ] No obvious O(n²) algorithms in hot paths
- [ ] Memory usage seems reasonable (no loading entire repo)
- [ ] If Phase 11.5 complete: benchmarks pass

### Git
- [ ] Committed with proper format: `feat(phase-N): description`
- [ ] Commit builds and tests pass

---

## Appendix A: Common Anti-Patterns to Avoid

### Anti-Pattern A.1: The God Class
❌ **Problem:**
```python
class CodeGrapher:
    """Does everything."""
    def parse(self): ...
    def index(self): ...
    def search(self): ...
    def rank(self): ...
    def format(self): ...
    # 800 lines later...
```

✅ **Solution:** Split by responsibility (see Rule 5.1)

---

### Anti-Pattern A.2: Configuration Overload
❌ **Problem:**
```python
class Config:
    enable_advanced_mode: bool
    use_gpu: bool
    cache_size: int
    log_level: str
    optimize_for_speed: bool
    # 20 more options...
```

✅ **Solution:** PRD specifies minimal config. Don't add options "just in case."

---

### Anti-Pattern A.3: Premature Generalization
❌ **Problem:**
```python
class LanguageParser(ABC):
    @abstractmethod
    def parse(self): ...

class PythonParser(LanguageParser):
    def parse(self): ...
    
# Only one implementation exists!
```

✅ **Solution:** Don't create abstractions for single use case. Wait for 3+ implementations.

---

### Anti-Pattern A.4: Exception Swallowing
❌ **Problem:**
```python
try:
    critical_operation()
except:
    pass  # Hope it works next time
```

✅ **Solution:** Always log, always handle specifically.

---

## Appendix B: Glossary of Terms

| Term | Definition | Example |
|------|------------|---------|
| **Symbol** | A code element (function, class, variable) | `def authenticate()` |
| **Import closure** | Set of files reachable via imports | All files imported by `main.py` |
| **PageRank** | Graph centrality algorithm | Scores symbols by call frequency |
| **FAISS** | Vector similarity search library | Finds nearest embedding neighbors |
| **Incremental index** | Update only changed symbols | Re-parse 1 file, not entire repo |
| **Token budget** | Max response size in tokens | 3500 tokens = ~900 words |
| **MCP** | Model Context Protocol | How Claude talks to tools |

---

## Appendix C: When to Consult Context7

Always use Context7 before writing code that uses these libraries:

1. **Before using any NetworkX method** (APIs change between versions)
2. **Before using FAISS** (C++ library with Python bindings - many gotchas)
3. **Before using transformers** (huge API surface, frequent changes)
4. **When implementing git hooks** (syntax varies by git version)

**Example workflow:**
```
Agent: About to implement Phase 4 (PageRank)
Action: Call Context7 for networkx API
Query: "How to compute PageRank with custom alpha in NetworkX 3.0"
Result: Verify signature is nx.pagerank(G, alpha=..., max_iter=...)
Then: Write code confidently
```

---

**End of Engineering Guidelines v2.0**

*This document should be read before Phase 1 and referenced throughout implementation.*
