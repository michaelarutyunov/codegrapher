"""MCP server for CodeGrapher.

This module implements the MCP server interface for CodeGrapher,
exposing the codegraph_query tool to Claude Code and other MCP clients.

Per PRD Section 6:
- Tool: codegraph_query
- Search: Vector search -> Import Closure Prune -> Weighted Score Rank
- Truncation: Token budget with file boundary breaks
- Error handling: Graceful fallback on index corruption/stale index
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastmcp import FastMCP

from codegrapher.models import Database, Symbol
from codegrapher.resolver import get_import_closure
from codegrapher.vector_store import EmbeddingModel, FAISSIndexManager


logger = logging.getLogger(__name__)

# Constants from PRD
DEFAULT_TOKEN_BUDGET = 3500
DEFAULT_MAX_DEPTH = 1
VECTOR_K = 20  # Number of results from vector search
STALE_INDEX_SECONDS = 3600  # 1 hour

# Initialize FastMCP server
mcp = FastMCP(name="codegrapher")


def find_repo_root() -> Optional[Path]:
    """Find the repository root by searching for .git directory.

    Returns:
        Repository root path, or None if not found
    """
    # Start from current directory
    current = Path.cwd()

    # Search up for .git directory
    for _ in range(20):  # Limit search depth
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    return None


def get_index_path(repo_root: Path) -> Path:
    """Get the path to the CodeGrapher index directory.

    Args:
        repo_root: Repository root path

    Returns:
        Path to .codegraph directory
    """
    return repo_root / ".codegraph"


def estimate_tokens(symbol: Symbol) -> int:
    """Estimate token count for a symbol.

    Uses a simple heuristic: 1 token per 4 characters for signature/doc,
    plus 2 tokens per line of code (based on line range).

    Args:
        symbol: Symbol to estimate

    Returns:
        Estimated token count
    """
    # Signature + docstring
    text_tokens = len(symbol.signature) + len(symbol.doc or "")
    text_tokens = max(1, text_tokens // 4)

    # Code lines (rough estimate)
    line_count = max(0, symbol.end_line - symbol.start_line)
    code_tokens = line_count * 2

    return text_tokens + code_tokens


def compute_weighted_scores(
    candidates: list[Symbol],
    query_embedding: np.ndarray,
    pagerank_scores: dict[str, float],
    git_log: dict[str, datetime],
) -> list[tuple[Symbol, float]]:
    """Compute weighted score for each candidate symbol.

    Per PRD Recipe 3:
        score(s) = 0.60 * norm_cosine + 0.25 * norm_pagerank
                  + 0.10 * norm_recency + 0.05 * is_test_file

    Args:
        candidates: List of candidate symbols
        query_embedding: Query vector for similarity
        pagerank_scores: Symbol ID -> PageRank score mapping
        git_log: File path -> last modified datetime

    Returns:
        List of (symbol, score) tuples, sorted by score descending
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

        # Component 4: Test file (0 or 1)
        is_test = 1.0 if "test" in symbol.file.lower() else 0.0

        # Weighted score
        score = (
            0.60 * norm_cosine
            + 0.25 * norm_pr
            + 0.10 * recency
            + 0.05 * is_test
        )

        scored.append((symbol, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def truncate_at_token_budget(
    scored_symbols: list[tuple[Symbol, float]],
    token_budget: int,
) -> list[Symbol]:
    """Truncate symbol list to fit within token budget.

    Always breaks at file boundaries (never mid-file).

    Args:
        scored_symbols: List of (symbol, score) tuples
        token_budget: Maximum tokens to include

    Returns:
        List of symbols that fit within budget
    """
    selected = []
    tokens_used = 0
    current_file = None
    file_buffer = []

    for symbol, _ in scored_symbols:
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


def get_git_log(repo_root: Path) -> dict[str, datetime]:
    """Get last modification time for each Python file.

    Args:
        repo_root: Repository root path

    Returns:
        Mapping of file path -> last modified datetime
    """
    git_log = {}

    try:
        # Get git log for all .py files
        result = subprocess.run(
            [
                "git",
                "log",
                "--name-only",
                "--pretty=format:%H %ct",
                "--",
                "*.py",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            current_time = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if " " in line and line.split()[0].startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    # Commit line: hash timestamp
                    try:
                        _, timestamp = line.rsplit(" ", 1)
                        current_time = datetime.fromtimestamp(float(timestamp))
                    except ValueError:
                        continue
                elif current_time and line.endswith(".py"):
                    # File line
                    git_log[line] = current_time

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning(f"Git log failed: {e}")

    # Fallback: use file modification times
    for py_file in repo_root.rglob("*.py"):
        if str(py_file) not in git_log:
            git_log[str(py_file)] = datetime.fromtimestamp(py_file.stat().st_mtime)

    return git_log


@mcp.tool
def codegraph_query(
    query: str,
    cursor_file: Optional[str] = None,
    _max_depth: int = DEFAULT_MAX_DEPTH,  # TODO: v2 graph expansion
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict:
    """Query the CodeGrapher index for relevant code symbols.

    Returns a ranked list of code symbols matching the query,
    optimized for token efficiency with import closure pruning.

    Args:
        query: Search query (symbol name, description, or keywords)
        cursor_file: Optional file path for import-closure pruning.
                   If provided, only returns symbols reachable from this file.
        max_depth: Graph hop depth for expansion (0-3, default 1).
                   Higher values include more related symbols.
        token_budget: Maximum tokens in response (default 3500).
                     Response is truncated at file boundaries if exceeded.

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - files: List of matching file entries (if successful)
        - tokens_used: Number of tokens used
        - total_symbols: Number of symbols returned
        - truncated: True if results were truncated due to token_budget
        - total_candidates: Total candidates before truncation
        - error_type: Error type code (if error)
        - message: Error message (if error)
    """
    # Find repository root
    repo_root = find_repo_root()
    if repo_root is None:
        return {
            "status": "error",
            "error_type": "repo_not_found",
            "message": "Could not find repository root (.git directory not found). "
            "Run from within a git repository.",
            "fallback_suggestion": "Use grep -r or find . -name '*.py' for search",
        }

    # Get index paths
    index_dir = get_index_path(repo_root)
    db_path = index_dir / "symbols.db"
    faiss_path = index_dir / "index.faiss"

    # Check if index exists
    if not db_path.exists() or not faiss_path.exists():
        return {
            "status": "error",
            "error_type": "index_not_found",
            "message": f"CodeGrapher index not found at {index_dir}. "
            f"Run 'codegraph init' to create the index.",
            "fallback_suggestion": "Use grep -r or find . -name '*.py' for search",
        }

    # Initialize components
    try:
        db = Database(db_path)
        faiss_manager = FAISSIndexManager(faiss_path)
        embedding_model = EmbeddingModel()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return {
            "status": "error",
            "error_type": "index_corrupt",
            "message": f"Failed to load index: {e}",
            "fallback_suggestion": "Run 'codegraph build --full' to rebuild the index",
        }

    # Check if index is stale
    try:
        last_indexed = db.get_meta("last_indexed")
        if last_indexed:
            last_indexed_time = datetime.fromisoformat(last_indexed)
            age_seconds = (datetime.now() - last_indexed_time).total_seconds()
            if age_seconds > STALE_INDEX_SECONDS:
                logger.warning(f"Index is stale: {age_seconds:.0f} seconds old")
                # Continue with stale index but flag it
    except Exception as e:
        logger.warning(f"Could not check index age: {e}")

    # Step 1: Generate query embedding
    try:
        query_embedding = embedding_model.embed_text(query)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        return {
            "status": "error",
            "error_type": "embedding_error",
            "message": f"Failed to process query: {e}",
            "fallback_suggestion": "Use text-based search instead",
        }

    # Step 2: Vector search
    try:
        vector_results = faiss_manager.search(query_embedding, k=VECTOR_K)
    except RuntimeError as e:
        logger.error(f"FAISS search failed: {e}")
        # Fallback to text search
        try:
            all_symbols = db.get_all_symbols()
            # Simple text search fallback
            query_lower = query.lower()
            scored = []
            for symbol in all_symbols:
                text = f"{symbol.id} {symbol.signature} {symbol.doc or ''}".lower()
                if query_lower in text:
                    scored.append((symbol, 0.5))  # Neutral score
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [s for s, _ in scored[:20]]
        except Exception as e2:
            logger.error(f"Text search fallback failed: {e2}")
            return {
                "status": "error",
                "error_type": "index_corrupt",
                "message": f"Vector search failed: {e}",
                "fallback_suggestion": "Run 'codegraph build --full' to rebuild the index",
            }
    else:
        # Get symbols from vector results
        candidate_ids = [sid for sid, _ in vector_results]
        candidates = []
        for symbol_id in candidate_ids:
            symbol = db.get_symbol(symbol_id)
            if symbol:
                candidates.append(symbol)

    if not candidates:
        return {
            "status": "success",
            "files": [],
            "tokens_used": 0,
            "total_symbols": 0,
            "truncated": False,
            "total_candidates": 0,
        }

    # Step 3: Import closure pruning (if cursor_file provided)
    if cursor_file:
        try:
            cursor_path = repo_root / cursor_file
            reachable_files = get_import_closure(cursor_path, repo_root, max_depth=10)

            # Filter to only symbols in reachable files
            pruned = []
            for symbol in candidates:
                symbol_file = repo_root / symbol.file
                if symbol_file in reachable_files:
                    pruned.append(symbol)

            if pruned:
                candidates = pruned
        except Exception as e:
            logger.warning(f"Import closure pruning failed: {e}")

    # Step 4: Expand graph (if max_depth > 0)
    # TODO: Implement graph expansion via edges table
    # For v1, we skip this step

    # Step 5: Compute weighted scores
    try:
        pagerank = {}  # TODO: Load cached PageRank scores
        git_log_data = get_git_log(repo_root)
        scored = compute_weighted_scores(candidates, query_embedding, pagerank, git_log_data)
    except Exception as e:
        logger.warning(f"Score computation failed: {e}, using vector order")
        scored = [(s, 0.5) for s in candidates]

    # Step 6: Truncate at token budget
    total_candidates = len(scored)
    selected = truncate_at_token_budget(scored, token_budget)
    truncated = len(selected) < total_candidates

    # Calculate tokens used
    tokens_used = sum(estimate_tokens(s) for s in selected)

    # Build response
    files = []
    for symbol in selected:
        files.append({
            "path": symbol.file,
            "line_range": [symbol.start_line, symbol.end_line],
            "symbol": symbol.id,
            "excerpt": symbol.signature,
        })

    return {
        "status": "success",
        "files": files,
        "tokens_used": tokens_used,
        "total_symbols": len(selected),
        "truncated": truncated,
        "total_candidates": total_candidates,
    }


def generate_mcp_config() -> str:
    """Generate the MCP server configuration template.

    Returns:
        JSON string with MCP server configuration
    """
    config = {
        "mcpServers": {
            "codegrapher": {
                "command": "python",
                "args": ["-m", "codegrapher.server"]
            }
        }
    }
    return json.dumps(config, indent=2)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
