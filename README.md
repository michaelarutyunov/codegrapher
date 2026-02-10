# CodeGrapher

**Local MCP server for token-efficient code search.**

CodeGrapher is a code understanding tool that helps AI coding agents work more efficiently with large codebases. By analyzing code structure, computing symbol importance, and using hybrid semantic + sparse search, it achieves **71.8% median token reduction** while finding relevant files for 65% of real-world tasks.

---

## What It Does

CodeGrapher solves the context window problem for AI-assisted development:

| Problem | Solution |
|---------|----------|
| LLMs have limited context windows | Returns only relevant code (71.8% median token reduction) |
| Finding related code is manual | Hybrid semantic + BM25 search with PageRank ranking |
| Indexes get stale | Automatic file watching & incremental updates (~0.7ms) |
| Sensitive data leaks | Secret detection before indexing |

**Key Features:**
- 🔍 **Hybrid Search** - Combines semantic embeddings (FAISS) + sparse BM25 search with Reciprocal Rank Fusion
- 🧩 **Smart Pairing** - Automatically includes test files when source files are found (7 bidirectional patterns)
- 📊 **PageRank Ranking** - Identifies high-utility functions that are called often
- 🔄 **Incremental Updates** - Only reindexes changed files (~0.7ms per update)
- 🎯 **Import Closure Pruning** - Limits results to files reachable from your cursor location
- 🔒 **Secret Detection** - Automatically excludes files with API keys, passwords, tokens
- 📡 **MCP Server** - Integrates with Claude Code and other MCP clients

---

## System Requirements

### Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.12 |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 500 MB free | 1 GB free |
| **OS** | Linux, macOS, WSL2 | Linux, macOS |

### Dependencies

CodeGrapher requires these Python packages (auto-installed):

- `transformers` ≥ 4.35.0 - Embedding model (jina-embeddings-v2-base-code)
- `faiss-cpu` ≥ 1.7.4 - Vector similarity search (dense)
- `rank-bm25` ≥ 0.2.2 - BM25 sparse search
- `torch` ≥ 2.1.0 - PyTorch backend for embeddings
- `networkx` ≥ 3.0 - Graph algorithms (PageRank)
- `scipy` ≥ 1.10.0 - Sparse linear algebra
- `watchdog` ≥ 3.0 - File watching
- `fastmcp` ≥ 0.2.0 - MCP server protocol
- `detect-secrets` ≥ 1.5.0 - Secret detection
- `numpy` ≥ 1.24.0 - Array operations
- `huggingface-hub` ≥ 0.19.0 - Model downloads

**Model Download:** On first run, CodeGrapher downloads the jina-embeddings-v2-base-code model (~307 MB) from Hugging Face.

---

## Installation

### Option 1: Install with UV (Recommended)

```bash
# Install UV if you don't have it
pip install uv

# Clone the repository
git clone https://github.com/michaelarutyunov/codegrapher.git
cd codegrapher

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Option 2: Install with pip

```bash
# Clone the repository
git clone https://github.com/michaelarutyunov/codegrapher.git
cd codegrapher

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install
pip install -e .
```

### Verify Installation

```bash
codegraph --help
```

---

## Quick Start

### 1. Initialize CodeGrapher in Your Repository

```bash
cd /path/to/your/repo
codegraph init
```

This creates `.codegraph/` directory and downloads the embedding model.

### 2. Build the Index

```bash
codegraph build --full
```

**First build time:** ~10-30 seconds for a 10k LOC project (varies by size).

### 3. Query CodeGrapher (CLI)

```bash
codegraph query "database connection pooling"
```

### 4. (Optional) Use with Claude Code via MCP

To make the `codegraph_query` tool available in Claude Code:

1. **From your project directory, start Claude Code:**
   ```bash
   cd /path/to/your/repo
   claude
   ```

2. **Open MCP manager:** Type `/mcp`

3. **Add server** with these settings:
   - Type: `stdio`
   - Command: `python3`
   - Args: `-m codegrapher.server`
   - Environment:
     - `PYTHONPATH` = path to your `src/` directory (e.g., `/path/to/codegrapher/src`)
     - `PROJECT_ROOT` = path to your project root

4. **Restart Claude Code** completely (`exit` or `Ctrl+D`)

See [MCP Integration](#mcp-integration) for detailed configuration options.

---

## Commands

| Command | Description |
|---------|-------------|
| `codegraph init` | Initialize repository, download model, install git hook |
| `codegraph build --full` | Build full index from scratch |
| `codegraph query <query>` | Test search from CLI |
| `codegraph update [file]` | Incremental update for changed files |
| `codegraph update --git-changed` | Update all files changed since last commit |
| `codegraph watch` | Watch for file changes and auto-update the index (foreground) |
| `codegraph callers <symbol>` | Find symbols that call the given symbol |
| `codegraph mcp-config` | Generate MCP server configuration |

### MCP Tools (Available in Claude Code)

| Tool | Description |
|------|-------------|
| `codegraph_query` | Search code index with token-efficient results |
| `codegraph_status` | Check index health, age, and staleness |
| `codegraph_refresh` | Update index (incremental or full rebuild) |

### Examples

```bash
# Initialize in a git repository
codegraph init

# Build the index
codegraph build --full

# Query for code related to authentication
codegraph query "authentication user login"

# Watch for changes (runs in foreground, press Ctrl+C to stop)
codegraph watch

# Update after editing files
codegraph update src/auth.py

# Update all changed files
codegraph update --git-changed

# Find callers of a specific symbol
codegraph callers extract_edges_from_file

# Find callers with JSON output for scripting
codegraph callers --json --limit 50 Database.get_all_symbols

# Generate MCP config for Claude Code
codegraph mcp-config > ~/.config/claude/mcp_config.json
```

---

## MCP Integration

CodeGrapher works as an MCP (Model Context Protocol) server. There are two ways to use it:

### Option 1: Claude Code CLI (Recommended for terminal use)

If you're using Claude Code from the terminal (`claude` command), configure it at user-level:

1. **Start Claude Code from anywhere** (no need to be in a specific project):
   ```bash
   claude
   ```

2. **Run the MCP manager:**
   ```
   /mcp
   ```

3. **Add the server** with these settings:
   - **Type:** stdio
   - **Command:** `python3`
   - **Args:** `-m codegrapher.server`
   - **Environment:** (empty - no longer needed!)

4. **Verify** with `/mcp` - you should see `codegrapher · ✔ connected`

**Or edit `~/.claude.json` directly:**

Add to the user-level `mcpServers` section:

```json
{
  "mcpServers": {
    "codegrapher": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "codegrapher.server"]
    }
  }
}
```

**How it works:** CodeGrapher automatically detects which project you're in by searching for `.codegraph` directories. No per-project configuration needed!

Then restart Claude Code completely (`exit` or `Ctrl+D`) and start a new session.

### Option 2: Claude Desktop App

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "codegrapher": {
      "command": "/path/to/codegrapher/.venv/bin/python",
      "args": ["-m", "codegrapher.server"],
      "cwd": "/path/to/codegrapher"
    }
  }
}
```

★ Insight ─────────────────────────────────────
MCP Config Key Updates (2026-01-15)
- **Auto-detection:** CodeGrapher now searches for `.codegraph` directories automatically
- **User-level config:** Configure once in `~/.claude.json`, works for all projects
- **No environment vars needed:** `PROJECT_ROOT` and `PYTHONPATH` are no longer required
- **New MCP tools:** `codegraph_status` (check index health) and `codegraph_refresh` (update index)
─────────────────────────────────────────────────

---

## How It Works

```
┌─────────────────┐
│  Python Source  │
└────────┬────────┘
         │
    ┌────▼─────┐
    │   AST    │ Extract symbols (functions, classes, imports)
    └────┬─────┘
         │
    ┌────▼────────┐
    │   Symbols    │ Function: database_connect()
    │   + Edges    │ └─► calls─► execute_query()
    └─────┬───────┘
          │
    ┌─────▼──────────────┐
    │  Dual Indexing     │
    ├────────────────────┤
    │ Dense: FAISS       │ jina-embeddings-v2-base-code (768-dim vectors)
    │ Sparse: BM25       │ Tokenized symbols (compound word splitting)
    │ Graph: PageRank    │ Call graph importance scores
    └─────┬──────────────┘
          │
    ┌─────▼──────────────┐
    │  Hybrid Search     │ Query → Dense + Sparse results
    │  (RRF Fusion)      │ → Reciprocal Rank Fusion
    └─────┬──────────────┘
          │
    ┌─────▼──────────────┐
    │  Smart Augment     │ + Test files (bidirectional pairing)
    │                    │ + Import closure pruning
    └─────┬──────────────┘
          │
    ┌─────▼──────────────┐
    │  Ranked Response   │ Top-K most relevant files
    └────────────────────┘
```

### Hybrid Search Pipeline

**1. Dual Search (Parallel)**
- **Dense (FAISS):** Semantic similarity via embeddings → top-k symbols
- **Sparse (BM25):** Keyword matching with compound word splitting → top-k symbols

**2. Reciprocal Rank Fusion (RRF)**
```
score(symbol) = Σ 1/(k + rank_in_results)
```
Where k=60 (constant), summed across both dense and sparse rankings.

**3. Augmentation**
- **Test-Source Pairing:** Auto-include test files for matched source files (7 patterns)
- **Import Closure:** Filter to files reachable from cursor position
- **Filename Matching:** Boost symbols from files mentioned in query

**4. Final Scoring**
```
score = 0.60 × cosine_similarity
      + 0.25 × pagerank_score
      + 0.10 × recency_score
      + 0.05 × is_test_file_penalty
```

- **Cosine similarity:** Semantic match to query (0-1)
- **PageRank:** Call graph importance (normalized 0-1)
- **Recency:** File modification recency (7d, 30d, older)
- **Test penalty:** Slight penalty to prioritize implementation

---

## Call Graph Features

### Named Fuzzy Edges (v1.5)

CodeGrapher now extracts call graph edges with contextual names instead of generic `<unknown>` prefixes:

| Before | After |
|--------|-------|
| `<unknown>.self.B1` | `c2.A2.B1` (contextual) |
| `<unknown>.self.B1` | `c1.A1.B1` (resolved, same-file inheritance) |

This enables queries like:
- "methods that call B1" → finds actual callers
- "what uses Database.get_all_symbols" → shows usage patterns

### 1-Level Inheritance Resolution

For classes in the same file, CodeGrapher resolves inherited method calls:

```python
# c1.py
class Base:
    def helper(self): pass

class Derived(Base):
    def method(self):
        self.helper()  # → Resolved to Base.helper!
```

**Limitation:** Cross-file inheritance falls back to contextual fuzzy (e.g., `c2.A2.B1` instead of `c1.A1.B1`). Full cross-file resolution requires import analysis and is planned for v2.

### Finding Callers with CLI

Use the `callers` command to find what calls a given symbol:

```bash
# Find callers of a function
codegraph callers extract_edges_from_file

# Output shows:
# - Caller symbol ID
# - File location
# - Line number
# - Signature
# - PageRank importance badge ([core], [important], [helper])

# JSON output for scripting
codegraph callers --json --limit 50 Database.get_all_symbols | jq '.callers[] | .caller_id'
```

---

## Comparison with Full Resolution Tools

| Feature | CodeGrapher v1.5 | Pyright/mypy | pyan |
|---------|-------------------|--------------|------|
| **Type inference** | None | Full | None |
| **Inheritance resolution** | 1-level, same-file | Full | Import-based |
| **Import resolution** | File-level only | Full | Full |
| **Call graph extraction** | AST + fuzzy | Semantic | AST |
| **Index build time** | ~3 min (1K files) | ~30 sec | ~10 sec |
| **Query speed** | ~50-100ms | Instant | Instant |
| **Use case** | Semantic search | Type checking | Visualization |

**When to use CodeGrapher:**
- Exploring unfamiliar codebases
- Semantic code search ("authentication logic")
- Understanding call patterns
- Quick navigation without IDE

**When to use full resolution tools:**
- Type checking before commits
- Finding type errors
- Complete accuracy required
- IDE integration (real-time analysis)

---

## Current Limitations

### Known Constraints

1. **Cross-file inheritance**
   - Falls back to contextual fuzzy (e.g., `Derived.parent_method`)
   - Full resolution planned for v2

2. **Multiple inheritance**
   - Only checks first parent for 1-level MRO
   - Covers ~80% of single-inheritance cases

3. **Nested classes**
   - Parser extracts top-level definitions only
   - Inner classes not indexed

4. **Dynamic features**
   - No runtime behavior analysis
   - Decorators like `@property` not specially handled
   - Metaclasses not understood

5. **`super()` calls**
   - Partially handled (extracted as `super.method` not fully resolved)

### Edge Case Behavior

| Pattern | Resolution |
|---------|------------|
| `self.method()` in same class | `Class.method` (resolved) |
| `self.method()` (inherited) | `BaseClass.method` (resolved, same-file) |
| `self.method()` (cross-file) | `CurrentClass.method` (fuzzy) |
| `super().method()` | `super().method` (as-is) |
| `obj.method()` | `obj.method` (unchanged) |
| `cls.method()` | `Class.method` (resolved) |

---

## Roadmap

### Planned Enhancements

#### Near-term (v1.6-v1.7)

- [ ] **`codegraph hierarchy` command** - Show class inheritance tree
- [ ] **`codegraph broken` command** - Find potential broken references
- [ ] **Import-aware edge resolution** - Resolve cross-file inheritance

#### Mid-term (v2.0)

- [ ] **Full cross-file resolution** - Resolve calls across files
- [ ] **Multiple inheritance MRO** - Follow Python's C3 linearization
- [ ] **`super()` call resolution** - Properly handle `super().method()`
- [ ] **Edge validation** - Detect calls to non-existent methods

#### Future Considerations

- [ ] **Type-aware resolution** - Use type hints for better accuracy
- [ ] **Decorator handling** - Properly handle `@property`, `@staticmethod`
- [ ] **Nested class support** - Index inner classes and methods
- [ ] **Incremental PageRank** - Update scores without full recomputation

---

## Development

### Running Tests

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=codegrapher
```

### Type Checking

```bash
mypy src/codegrapher
```

### Code Formatting

```bash
# Format code
black src/codegrapher

# Check formatting
black --check src/codegrapher
```

---

## Performance Benchmarks

On a repository with ~30,000 lines of Python code:

| Metric | Target | Actual |
|--------|--------|--------|
| Cold start | ≤ 2s | ~1.5s |
| Query latency | ≤ 500ms | ~100-300ms |
| Incremental update | ≤ 1s | ~0.7ms |
| Full index build | ≤ 30s | ~10-20s |
| RAM usage (idle) | ≤ 500MB | ~300-400MB |

**Token Savings:** 71.8% median (range: 43.3% - 94.7% across 23 real-world tasks)

---

## Evaluation Results

CodeGrapher was evaluated on 23 real-world tasks from 7 major Python projects (pytest, Flask, FastAPI, Pydantic, Click, Werkzeug, Jinja).

**Results:**

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Tasks Passing (≥85% recall)** | 15/23 (65%) | ≥20/23 (≥85%) | ⚠️ Below target |
| **Token Savings (Median)** | 71.8% | ≥30% | ✅ PASS |
| **Precision (Median)** | 14.3% | ≤40% | ✅ PASS |

**Status:** 2 of 3 criteria met. Token efficiency is excellent, but recall needs improvement (8 tasks still missing critical files).

**Documentation:**
- 📊 [Detailed Evaluation Report](fixtures/eval_report_before_after.md) - Complete before/after analysis with fix attribution
- 📈 [Implementation Progress](docs/PROGRESS.md) - Technical implementation details and decisions
- 🔬 [Evaluation Guide](docs/EVALUATION_GUIDE.md) - Methodology and ground truth dataset

---

## Configuration

CodeGrapher stores its index in `.codegraph/` directory:

```
.codegraph/
├── symbols.db       # SQLite database with symbols, edges, PageRank
├── index.faiss      # FAISS vector index
├── index.ids        # Symbol ID mappings
└── excluded_files.txt  # Files skipped due to secrets
```

This directory should **not** be committed to git (it's in `.gitignore`).

---

## Troubleshooting

### "codegraph: command not found"

Make sure your virtual environment is activated:
```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### "Model download failed"

The jina-embeddings-v2-base-code model (~307 MB) downloads on first run. If it fails:
```bash
# Check Hugging Face connectivity
ping huggingface.co

# Manually download if needed
export HF_HOME=/custom/cache/path
```

### High memory usage during indexing

For very large repositories (>100k LOC), CodeGrapher may use 1-2GB RAM during index building. This is temporary and freed after build completes.

### WSL2 Memory Issues

If running on WSL2 and experiencing crashes, add a `.wslconfig` file:
```ini
[wsl2]
memory=16GB
swap=4GB
```

Then restart WSL: `wsl --shutdown` from PowerShell.

### MCP Server Not Appearing in Claude Code

If the `codegraph_query` tool doesn't show up after configuring MCP:

1. **Verify you're editing the correct config file:**
   - Claude Code CLI uses: `~/.claude.json` (user-level under `mcpServers`)
   - Claude Desktop uses: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

2. **Ensure `type: "stdio"` is set** (required for Claude Code CLI)

3. **No environment variables needed** - auto-detection finds projects automatically

4. **Restart Claude Code completely** (`exit` or `Ctrl+D`, then start fresh)

5. **Run `codegraph build --full` first** - CodeGrapher needs an index to work

6. **Verify with `/mcp`** - should show `codegrapher · ✔ connected`

7. **Check you're in a git repository** - CodeGrapher searches for `.git` or `.codegraph` directories

### New MCP Tools Available

Once configured, you have access to three MCP tools:

**`codegraph_status`** - Check index health
```json
{
  "status": "success",
  "index_exists": true,
  "index_age_hours": 9.2,
  "total_symbols": 560,
  "is_stale": false,
  "suggestion": "Index is fresh."
}
```

**`codegraph_refresh`** - Update the index
```json
{
  "status": "success",
  "mode": "incremental",
  "files_updated": 3,
  "duration_seconds": 2.3
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

CodeGrapher uses:
- **jina-embeddings-v2-base-code** by Jina AI - Code-aware embedding model
- **FAISS** by Meta Research - Efficient similarity search
- **NetworkX** - Graph algorithms and PageRank
- **Watchdog** - File system event monitoring

---

## Contributing

Contributions are welcome! See the [Implementation Progress](docs/PROGRESS.md) document for technical details and architecture decisions.

**Key Resources:**
- 📊 [Evaluation Report](fixtures/eval_report_before_after.md) - Detailed performance analysis
- 🔬 [Evaluation Guide](docs/EVALUATION_GUIDE.md) - Testing methodology
- 📈 [Progress Log](docs/PROGRESS.md) - Implementation history

---

## Links

- **Repository:** https://github.com/michaelarutyunov/codegrapher
- **MCP Protocol:** https://modelcontextprotocol.io
