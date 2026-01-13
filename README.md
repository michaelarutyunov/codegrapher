# CodeGrapher

**Local MCP server for token-efficient code search.**

CodeGrapher is a code understanding tool that helps AI coding agents work more efficiently with large codebases. By analyzing code structure, computing symbol importance, and using semantic search, it reduces the amount of code context sent to LLMs by **87%+** while maintaining **100% recall** of relevant files.

---

## What It Does

CodeGrapher solves the context window problem for AI-assisted development:

| Problem | Solution |
|---------|----------|
| LLMs have limited context windows | Returns only relevant code (87% token reduction) |
| Finding related code is manual | Semantic search + PageRank ranking |
| Indexes get stale | Automatic file watching & incremental updates |
| Sensitive data leaks | Secret detection before indexing |

**Key Features:**
- 🧠 **Semantic Code Search** - Embedding-based search finds related code by meaning, not just keywords
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
- `faiss-cpu` ≥ 1.7.4 - Vector similarity search
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

### 3. Query CodeGrapher

```bash
codegraph query "database connection pooling"
```

Or use via MCP in Claude Code → the `codegraph_query` tool will be available.

---

## Commands

| Command | Description |
|---------|-------------|
| `codegraph init` | Initialize repository, download model, install git hook |
| `codegraph build --full` | Build full index from scratch |
| `codegraph query <query>` | Test search from CLI |
| `codegraph update [file]` | Incremental update for changed files |
| `codegraph update --git-changed` | Update all files changed since last commit |
| `codegraph mcp-config` | Generate MCP server configuration |

### Examples

```bash
# Initialize in a git repository
codegraph init

# Build the index
codegraph build --full

# Query for code related to authentication
codegraph query "authentication user login"

# Update after editing files
codegraph update src/auth.py

# Update all changed files
codegraph update --git-changed

# Generate MCP config for Claude Code
codegraph mcp-config > ~/.config/claude/mcp_config.json
```

---

## MCP Integration

CodeGrapher works as an MCP (Model Context Protocol) server. Configure it in your MCP client config:

### Claude Desktop

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "codegrapher": {
      "command": "/path/to/codegrapher/.venv/bin/python",
      "args": ["/path/to/codegrapher/src/codegrapher/server.py"]
    }
  }
}
```

Or generate the config automatically:

```bash
codegraph mcp-config
```

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
    └────┬────────┘
         │
    ┌────▼─────────────┐
    │  Embeddings      │ jina-embeddings-v2-base-code
    │  (768-dim vectors)│
    └────┬─────────────┘
         │
    ┌────▼─────────────┐
    │  FAISS Index     │ Vector similarity search
    │  + PageRank      │ + Importance scoring
    └────┬─────────────┘
         │
    ┌────▼─────────────┐
    │  Query Response  │ Top-K most relevant symbols
    └───────────────────┘
```

### Scoring Formula

Each symbol gets a composite score:

```
score = 0.60 × cosine_similarity
      + 0.25 × pagerank_score
      + 0.10 × recency_score
      + 0.05 × is_test_file_penalty
```

- **Cosine similarity:** Semantic match to your query (0-1)
- **PageRank:** How often this symbol is called by others (normalized 0-1)
- **Recency:** How recently the file was modified (7d, 30d, older)
- **Test penalty:** Slightly penalizes test files to prioritize implementation code

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

**Token Savings:** 87% average reduction (87,105 tokens → 704,104 baseline)

---

## Evaluation Results

CodeGrapher was evaluated on 20 real-world tasks across 7 major Python projects (pytest, Flask, FastAPI, Pydantic, Click, Werkzeug, Jinja).

**Acceptance Criteria:**

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Token Savings | ≥ 30% | 87.1% | ✅ PASS |
| Recall | ≥ 85% | 100.0% | ✅ PASS |
| Precision | ≤ 40% | 25.0% | ✅ PASS |

See `docs/PROGRESS.md` for detailed Phase 12 evaluation results.

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

## Links

- **Repository:** https://github.com/michaelarutyunov/codegrapher
- **MCP Protocol:** https://modelcontextprotocol.io
- **Documentation:** See `docs/PROGRESS.md` for implementation details
