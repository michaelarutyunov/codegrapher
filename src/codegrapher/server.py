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
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastmcp import FastMCP

from codegrapher.models import Database, Symbol
from codegrapher.resolver import get_import_closure
from codegrapher.sparse_index import (
    BM25Searcher,
    SparseIndex,
    augment_with_filename_matches,
    tokenize_compound_word,
)
from codegrapher.vector_store import EmbeddingModel, FAISSIndexManager


logger = logging.getLogger(__name__)

# Constants from PRD
DEFAULT_TOKEN_BUDGET = 3500
DEFAULT_MAX_DEPTH = 1
VECTOR_K = 20  # Number of results from vector search
STALE_INDEX_SECONDS = 3600  # 1 hour

# RRF constant (from research: k=60 works well for most cases)
RRF_K = 60

# Noise terms for query preprocessing (conservative list)
FILLER_TERMS = {
    'e.g', 'i.e', 'etc', 'eg', 'ie', 'aka',
    'please', 'thanks', 'help', 'github', 'issue'
}


def preprocess_query(query: str) -> str:
    """Remove noise terms while preserving technical keywords.

    Conservative filtering removes only non-technical filler terms that
    dilute search signal. Valid code terms like 'async', 'function',
    'class' are preserved.

    Args:
        query: Raw query string

    Returns:
        Cleaned query with noise terms removed

    Examples:
        >>> preprocess_query("e.g. fix async function KeyError in importlib")
        'fix async function KeyError importlib'
    """
    words = query.split()
    cleaned = []
    for w in words:
        # Strip trailing punctuation for comparison
        word_stripped = w.rstrip('.,;:!?)')
        word_lower = word_stripped.lower()
        if word_lower not in FILLER_TERMS:
            cleaned.append(w)
    return ' '.join(cleaned)

# PageRank cache: maps index_path -> (scores_dict, last_modified_time)
_pagerank_cache: dict[Path, tuple[dict[str, float], datetime]] = {}

# Initialize FastMCP server
mcp = FastMCP(name="codegrapher")


def _load_pagerank_scores(db: Database) -> dict[str, float]:
    """Load PageRank scores from database or cache.

    Computes PageRank scores on first call and caches them for subsequent calls.
    Cache is invalidated when the index is modified (checked via last_indexed time).

    Args:
        db: Database instance

    Returns:
        Dictionary mapping symbol_id to PageRank score (normalized 0-1)
    """
    global _pagerank_cache

    # Get database path for cache key
    db_path = db.db_path

    # Check if we have cached scores
    cached_scores, cache_time = _pagerank_cache.get(db_path, ({}, datetime.min))

    # Get last indexed time from database
    last_indexed_str = db.get_meta("last_indexed")
    last_indexed = datetime.fromisoformat(last_indexed_str) if last_indexed_str else datetime.min

    # Return cached scores if still valid
    if cached_scores and cache_time >= last_indexed:
        logger.debug(f"Using cached PageRank scores ({len(cached_scores)} symbols)")
        return cached_scores

    # Compute fresh PageRank scores
    from codegrapher.graph import compute_pagerank

    logger.info("Computing PageRank scores...")
    scores = compute_pagerank(db)

    if scores:
        logger.info(f"Computed PageRank for {len(scores)} symbols")
        # Cache the scores
        _pagerank_cache[db_path] = (scores, datetime.now())
    else:
        logger.warning("No PageRank scores computed (no call edges found)")

    return scores


def _get_pagerank_score(symbol_id: str, pagerank_scores: dict[str, float]) -> float:
    """Get PageRank score for a symbol with fallback to fuzzy lookup.

    Because v1 edge extraction uses fuzzy callee IDs like <unknown>.func_name,
    actual symbols may have low PageRank (isolated nodes) while their fuzzy
    counterparts have high scores. We take the maximum of both.

    Args:
        symbol_id: Full symbol ID like "src.module.Class.func"
        pagerank_scores: PageRank scores dict

    Returns:
        PageRank score (0.0 if not found)
    """
    # Try exact match
    exact_score = pagerank_scores.get(symbol_id, 0.0)

    # Also try fuzzy match <unknown>.<name>
    name = symbol_id.split(".")[-1]
    fuzzy_id = f"<unknown>.{name}"
    fuzzy_score = pagerank_scores.get(fuzzy_id, 0.0)

    # Return the maximum score (fuzzy edges may have higher centrality)
    return max(exact_score, fuzzy_score)


def find_repo_root() -> Optional[Path]:
    """Find the repository root by searching for .codegraph or .git directory.

    Searches from current directory upward, prioritizing .codegraph index.
    This allows codegrapher to work automatically in any indexed project.

    Returns:
        Repository root path, or None if not found
    """
    # Start from current directory
    current = Path.cwd()

    # Search up for .codegraph or .git directory
    for _ in range(20):  # Limit search depth
        # Prefer .codegraph (our index) over .git
        if (current / ".codegraph").exists():
            return current
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
        raw_pr = _get_pagerank_score(symbol.id, pagerank_scores)
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


def is_test_source_pair(test_file: str, source_file: str) -> bool:
    """Check if two files are test-source pairs.

    Supports seven common patterns:
    1. test_ prefix in same directory (test_mypath.py ↔ mypath.py)
    2. tests/ mirrors src/ structure (tests/mypath.py ↔ src/mypath.py)
    3. _test.py suffix (mypath_test.py ↔ mypath.py)
    4. Base filename match (handles nested paths like src/jinja2/compiler.py ↔ tests/test_compiler.py)
    5. Fuzzy matching (compiler.py ↔ test_compile.py - substring match)
    6. Parallel directory trees (src/_pytest/python/approx.py ↔ testing/python/approx.py)
    7. __init__.py handling (src/werkzeug/debug/__init__.py ↔ tests/test_debug.py)

    Args:
        test_file: Path to the test file
        source_file: Path to the source file

    Returns:
        True if files are test-source pairs, False otherwise

    Examples:
        >>> is_test_source_pair("test_pathlib.py", "pathlib.py")
        True
        >>> is_test_source_pair("tests/test_pathlib.py", "src/pathlib.py")
        True
        >>> is_test_source_pair("pathlib_test.py", "pathlib.py")
        True
        >>> is_test_source_pair("main.py", "utils.py")
        False
    """
    # Normalize paths for comparison
    test_norm = test_file.replace("\\", "/")
    source_norm = source_file.replace("\\", "/")

    # Pattern 1: test_ prefix in same directory
    if test_norm == f"test_{source_norm}":
        return True

    # Pattern 2: tests/ mirrors src/ structure
    if test_norm.startswith("tests/") and source_norm.startswith("src/"):
        test_suffix = test_norm.replace("tests/", "").replace("test_", "")
        source_suffix = source_norm.replace("src/", "")
        if test_suffix == source_suffix or test_suffix == f"test_{source_suffix}":
            return True

    # Pattern 3: _test.py suffix
    # Strip .py extension from source before adding _test
    if source_norm.endswith(".py"):
        source_base = source_norm[:-3]  # Remove ".py"
        if test_norm == f"{source_base}_test.py":
            return True

    # Pattern 4: Base filename match (for nested paths)
    # Extract base filenames (without path and .py extension)
    test_basename = test_norm.split("/")[-1].removesuffix(".py")
    source_basename = source_norm.split("/")[-1].removesuffix(".py")

    # Check if test name is test_ + source name or source name + _test
    if test_basename == f"test_{source_basename}" or test_basename == f"{source_basename}_test":
        return True
    # Handle case where test is named after a simplified source name
    # e.g., test_compiler.py pairs with compiler.py, even with nested paths
    if test_basename.replace("test_", "") == source_basename or source_basename == test_basename.replace("_test", ""):
        return True
    # Pattern 5: Fuzzy matching for real-world naming inconsistencies
    # e.g., compiler.py ↔ test_compile.py (one is substring of the other)
    test_core = test_basename.replace("test_", "").replace("_test", "")
    if test_core and source_basename:
        if test_core in source_basename or source_basename in test_core:
            return True

    # Pattern 6: Parallel directory trees
    # e.g., src/_pytest/python/approx.py ↔ testing/python/approx.py
    # Strip common prefixes and compare path suffixes
    test_stripped = test_norm
    source_stripped = source_norm
    for prefix in ["src/", "tests/", "testing/", "test/"]:
        if test_stripped.startswith(prefix):
            test_stripped = test_stripped[len(prefix):]
        if source_stripped.startswith(prefix):
            source_stripped = source_stripped[len(prefix):]

    # If the stripped paths match (same relative path), they're a pair
    if test_stripped == source_stripped:
        return True

    # Also check if test_stripped is "test_" + source_stripped
    if test_stripped.replace("test_", "", 1) == source_stripped:
        return True

    # Pattern 7: __init__.py handling
    # e.g., src/werkzeug/debug/__init__.py ↔ tests/test_debug.py
    # Use parent directory name for __init__.py files
    if source_norm.endswith("__init__.py"):
        # Get parent directory name: werkzeug/debug/__init__.py → debug
        source_parent = source_norm.replace("__init__.py", "").rstrip("/").split("/")[-1]
        if source_parent:
            # Check if test file matches parent directory name
            if test_basename == f"test_{source_parent}" or test_basename == f"{source_parent}_test":
                return True
            # Also check fuzzy match with parent directory
            if source_parent in test_basename or test_basename.replace("test_", "").replace("_test", "") in source_parent:
                return True

    return False


def augment_with_bidirectional_test_pairs(
    candidates: list[Symbol],
    all_symbols: list[Symbol],
) -> list[Symbol]:
    """Add bidirectional test-source pairs for all candidate files.

    For each source file in candidates, adds its corresponding test file(s).
    For each test file in candidates, adds its corresponding source file.

    This improves recall for tasks that require editing both implementation
    and tests, as test files often don't match search queries directly.

    Args:
        candidates: Current candidate symbols from search
        all_symbols: All symbols in the index

    Returns:
        Augmented list of symbols including test-source pairs

    Examples:
        >>> # candidates has symbols from compiler.py
        >>> # Adds symbols from test_compiler.py
        >>> augmented = augment_with_bidirectional_test_pairs(candidates, all_symbols)
    """
    # Build set of candidate files for quick lookup
    candidate_files = {s.file for s in candidates}
    candidate_ids = {s.id for s in candidates}
    augmented = list(candidates)

    # Separate test and source files from candidates
    test_files_in_candidates = [f for f in candidate_files if "test" in f.lower()]
    source_files_in_candidates = [f for f in candidate_files if "test" not in f.lower()]

    # Create a lookup map from file to all its symbols
    file_to_symbols: dict[str, list[Symbol]] = {}
    for symbol in all_symbols:
        if symbol.file not in file_to_symbols:
            file_to_symbols[symbol.file] = []
        file_to_symbols[symbol.file].append(symbol)

    # For each source file, find its test files (source → test pairing)
    for source_file in source_files_in_candidates:
        for potential_test in file_to_symbols:
            if potential_test in candidate_files:
                continue
            if "test" not in potential_test.lower():
                continue
            # Check if this is a test-source pair
            if is_test_source_pair(potential_test, source_file):
                # Add all symbols from the paired test file
                for symbol in file_to_symbols[potential_test]:
                    if symbol.id not in candidate_ids:
                        augmented.append(symbol)
                        candidate_ids.add(symbol.id)

    # For each test file, find its source files (test → source pairing)
    for test_file in test_files_in_candidates:
        for potential_source in file_to_symbols:
            if potential_source in candidate_files:
                continue
            if "test" in potential_source.lower():
                continue
            # Check if this is a test-source pair (parameters in test, source order)
            if is_test_source_pair(test_file, potential_source):
                # Add all symbols from the paired source file
                for symbol in file_to_symbols[potential_source]:
                    if symbol.id not in candidate_ids:
                        augmented.append(symbol)
                        candidate_ids.add(symbol.id)

    return augmented


def augment_with_test_source_pairs(
    cursor_file: str,
    candidates: list[Symbol],
    all_symbols: list[Symbol],
) -> list[Symbol]:
    """If cursor is in test file, include corresponding source files.

    Import closure is unidirectional (test → imports), missing the
    source → test relationship. This function adds source files when
    the cursor is in a test file.

    Args:
        cursor_file: Path to the file containing the cursor
        candidates: Current candidate symbols from import closure
        all_symbols: All symbols in the index

    Returns:
        Augmented list of symbols including source files

    Examples:
        >>> # Cursor in test_pathlib.py, candidates only have test symbols
        >>> augmented = augment_with_test_source_pairs("test_pathlib.py", candidates, all_symbols)
        >>> # Now includes symbols from src/pathlib.py
    """
    # Only augment if cursor is in a test file
    if "test" not in cursor_file.lower():
        return candidates

    candidate_files = {s.file for s in candidates}
    augmented = list(candidates)

    for symbol in all_symbols:
        if symbol.file in candidate_files:
            continue
        if is_test_source_pair(cursor_file, symbol.file):
            augmented.append(symbol)

    return augmented


def merge_dense_sparse_results(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge dense and sparse search results using Reciprocal Rank Fusion (RRF).

    RRF is rank-based and score-scale invariant:
        RRF_score(item) = Σ (k / (k + rank_position))

    Where k = 60 (RFF_K constant), rank = 1, 2, 3, ...

    Advantages over weighted averaging:
    - No score normalization needed
    - Works with different scoring systems
    - Handles items in only one ranking
    - Proven robust in research

    Args:
        dense_results: List of (symbol_id, cosine_sim) from FAISS, ranked by score
        sparse_results: List of (symbol_id, bm25_score) from BM25, ranked by score

    Returns:
        Merged and sorted list of (symbol_id, rrf_score) tuples
    """
    # Calculate RRF scores
    rrf_scores: dict[str, float] = {}

    # Add contributions from dense results (rank 1-based)
    for rank, (symbol_id, _score) in enumerate(dense_results, start=1):
        rrf_scores[symbol_id] = rrf_scores.get(symbol_id, 0.0) + (RRF_K / (RRF_K + rank))

    # Add contributions from sparse results (rank 1-based)
    for rank, (symbol_id, _score) in enumerate(sparse_results, start=1):
        rrf_scores[symbol_id] = rrf_scores.get(symbol_id, 0.0) + (RRF_K / (RRF_K + rank))

    # Sort by RRF score descending
    result = list(rrf_scores.items())
    result.sort(key=lambda x: x[1], reverse=True)
    return result


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

        # Build sparse index from database symbols
        sparse_index = SparseIndex()
        all_symbols = db.get_all_symbols()
        sparse_index.add_symbols(all_symbols)
        sparse_searcher = BM25Searcher(sparse_index)
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

    # Step 1: Preprocess query (remove noise terms)
    cleaned_query = preprocess_query(query)

    # DEBUG: Log query preprocessing for morphological gap analysis (task_028)
    query_tokens = cleaned_query.split()

    # Apply compound word splitting to query tokens
    # This enables "compile_templates" → ["compile_templates", "compile", "templates"]
    # Note: tokenize_compound_word() returns lowercase tokens for case-insensitive matching
    expanded_query_tokens = []
    for token in query_tokens:
        compound_tokens = tokenize_compound_word(token)
        expanded_query_tokens.extend(compound_tokens)

    # Ensure all query tokens are lowercase (tokenize_compound_word already does this)
    query_tokens = [t.lower() for t in expanded_query_tokens]

    logger.debug(f"[task_028-debug] Original query: '{query}'")
    logger.debug(f"[task_028-debug] Cleaned query: '{cleaned_query}'")
    logger.debug(f"[task_028-debug] Query tokens after compound splitting: {query_tokens}")

    # Step 2: Generate query embedding
    try:
        query_embedding = embedding_model.embed_text(cleaned_query)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        return {
            "status": "error",
            "error_type": "embedding_error",
            "message": f"Failed to process query: {e}",
            "fallback_suggestion": "Use text-based search instead",
        }

    # Step 3: Dense search (FAISS) + Sparse search (BM25) + Merge
    filename_matched_ids = set()  # Initialize for use in import closure pruning
    try:
        # Dense: Vector search
        dense_results = faiss_manager.search(query_embedding, k=VECTOR_K)

        # Sparse: BM25 search on preprocessed query tokens
        # Note: query_tokens already defined above for debug logging
        sparse_results = sparse_searcher.search(query_tokens, k=VECTOR_K)

        # DEBUG: Log sparse search results for morphological gap analysis (task_028)
        logger.debug(f"[task_028-debug] Sparse results (top 10): {sparse_results[:10]}")

        # Check if cursor file symbols are in sparse results
        if cursor_file:
            cursor_file_symbols = [s for s in all_symbols if cursor_file.endswith(s.file) or s.file.endswith(cursor_file)]
            cursor_file_ids = {s.id for s in cursor_file_symbols}
            sparse_ids = {sid for sid, _ in sparse_results}
            matched_cursor_symbols = cursor_file_ids & sparse_ids
            logger.debug(f"[task_028-debug] Cursor file: {cursor_file}")
            logger.debug(f"[task_028-debug] Cursor file symbols count: {len(cursor_file_symbols)}")
            logger.debug(f"[task_028-debug] Cursor file symbols in sparse results: {len(matched_cursor_symbols)}")
            if cursor_file_symbols:
                logger.debug(f"[task_028-debug] Sample cursor symbol: {cursor_file_symbols[0].id} | signature: {cursor_file_symbols[0].signature[:80]}")

        # Augment sparse results with filename matches
        sparse_ids = set(sid for sid, _ in sparse_results)
        filename_matched_ids = augment_with_filename_matches(
            query_tokens, all_symbols, sparse_ids
        )

        # Convert augmented IDs to sparse results format
        for aug_id in filename_matched_ids:
            if aug_id not in sparse_ids:
                sparse_results.append((aug_id, 0.1))  # Low score for filename matches

        # Merge dense and sparse results
        merged_results = merge_dense_sparse_results(dense_results, sparse_results)

        # Get symbols from merged results
        candidates = []
        for symbol_id, _ in merged_results:
            symbol = db.get_symbol(symbol_id)
            if symbol:
                candidates.append(symbol)

    except RuntimeError as e:
        logger.error(f"Hybrid search failed: {e}")
        # Fallback to text search
        try:
            query_lower = cleaned_query.lower()
            scored = []
            for symbol in all_symbols:
                text = f"{symbol.id} {symbol.signature} {symbol.doc or ''}".lower()
                # Check if any query token (including compound-split components) matches
                if any(token.lower() in text for token in query_tokens):
                    scored.append((symbol, 0.5))  # Neutral score
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [s for s, _ in scored[:20]]
            # Also perform filename matching for fallback (query_tokens already compound-split)
            filename_matched_ids = augment_with_filename_matches(
                query_tokens, all_symbols, set(s.id for s in candidates)
            )
        except Exception as e2:
            logger.error(f"Text search fallback failed: {e2}")
            return {
                "status": "error",
                "error_type": "index_corrupt",
                "message": f"Hybrid search failed: {e}",
                "fallback_suggestion": "Run 'codegraph build --full' to rebuild the index",
            }

    # Ensure cursor file symbols are included (defensive measure for semantic gaps)
    # When the cursor file itself should be included but wasn't found via search
    if cursor_file and candidates:
        cursor_symbols_included = any(
            cursor_file.endswith(s.file) or s.file.endswith(cursor_file)
            for s in candidates
        )
        if not cursor_symbols_included:
            # Add symbols from cursor file that weren't found via search
            for symbol in all_symbols:
                if (cursor_file.endswith(symbol.file) or symbol.file.endswith(cursor_file)):
                    if symbol not in candidates:
                        candidates.append(symbol)
                        filename_matched_ids.add(symbol.id)  # Mark as explicitly included

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
    # NOTE: Preserve filename-matched symbols as they are explicitly requested by query
    if cursor_file:
        try:
            cursor_path = repo_root / cursor_file
            reachable_files = get_import_closure(cursor_path, repo_root, max_depth=10)

            # Filter to only symbols in reachable files OR filename-matched
            pruned = []
            for symbol in candidates:
                symbol_file = repo_root / symbol.file
                # Preserve if in import closure OR was matched via filename
                if symbol_file in reachable_files or symbol.id in filename_matched_ids:
                    pruned.append(symbol)

            if pruned:
                candidates = pruned
        except Exception as e:
            logger.warning(f"Import closure pruning failed: {e}")

    # Step 3.5: Bidirectional test-source pairing augmentation
    # For each source file found, add its test files (and vice versa)
    # This improves recall when tests don't match search queries directly
    try:
        candidates = augment_with_bidirectional_test_pairs(candidates, all_symbols)
    except Exception as e:
        logger.warning(f"Test-source pairing failed: {e}")

    # Step 4: Expand graph (if max_depth > 0)
    # TODO: Implement graph expansion via edges table
    # For v1, we skip this step

    # Step 5: Compute weighted scores
    pagerank: dict[str, float] = {}  # Initialize in case score computation fails
    try:
        pagerank = _load_pagerank_scores(db)
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
        # Get PageRank score for this symbol (0.0 if not found)
        pr_score = _get_pagerank_score(symbol.id, pagerank)
        files.append({
            "path": symbol.file,
            "line_range": [symbol.start_line, symbol.end_line],
            "symbol": symbol.id,
            "excerpt": symbol.signature,
            "pagerank_score": round(pr_score, 6),
        })

    return {
        "status": "success",
        "files": files,
        "tokens_used": tokens_used,
        "total_symbols": len(selected),
        "truncated": truncated,
        "total_candidates": total_candidates,
    }


@mcp.tool
def codegraph_status() -> dict:
    """Check CodeGrapher index status and health.

    Returns information about the index including age, size,
    and whether it needs to be refreshed.

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - index_exists: bool
        - index_age_hours: float (hours since last update)
        - total_symbols: int
        - total_files: int
        - last_indexed: ISO timestamp or None
        - is_stale: bool (True if index is older than 24 hours)
        - suggestion: str (what to do if needed)
        - error_type: error code (if error)
        - message: error message (if error)
    """
    STALE_THRESHOLD_HOURS = 24

    # Find repository root
    repo_root = find_repo_root()
    if repo_root is None:
        return {
            "status": "error",
            "error_type": "repo_not_found",
            "message": "Not in a git repository (no .git directory found).",
            "suggestion": "Run from within a git repository.",
        }

    # Get index paths
    index_dir = get_index_path(repo_root)
    db_path = index_dir / "symbols.db"
    faiss_path = index_dir / "index.faiss"

    # Check if index exists
    index_exists = db_path.exists() and faiss_path.exists()

    if not index_exists:
        return {
            "status": "success",
            "index_exists": False,
            "is_stale": False,
            "suggestion": f"No index found. Run 'codegraph init' then 'codegraph build --full' to create the index.",
        }

    # Get index metadata
    try:
        db = Database(db_path)
        last_indexed_str = db.get_meta("last_indexed")
        total_symbols = db.get_meta("total_symbols")
        total_files = db.get_meta("total_files")

        # Calculate index age
        if last_indexed_str:
            last_indexed_time = datetime.fromisoformat(last_indexed_str)
            age_seconds = (datetime.now() - last_indexed_time).total_seconds()
            age_hours = age_seconds / 3600
        else:
            age_hours = float("inf")
            last_indexed_time = None

        is_stale = age_hours > STALE_THRESHOLD_HOURS

        # Build suggestion
        if is_stale:
            if age_hours > 168:  # 1 week
                suggestion = f"Index is {age_hours:.0f} hours ({age_hours/24:.0f} days) old. Run 'codegraph build --full' to rebuild."
            else:
                suggestion = f"Index is {age_hours:.0f} hours old. Run 'codegraph update --git-changed' to update."
        else:
            suggestion = "Index is fresh."

        return {
            "status": "success",
            "index_exists": True,
            "index_age_hours": round(age_hours, 1),
            "total_symbols": int(total_symbols) if total_symbols else 0,
            "total_files": int(total_files) if total_files else 0,
            "last_indexed": last_indexed_str,
            "is_stale": is_stale,
            "suggestion": suggestion,
        }

    except Exception as e:
        logger.error(f"Failed to get index status: {e}")
        return {
            "status": "error",
            "error_type": "index_read_error",
            "message": f"Failed to read index: {e}",
            "suggestion": "Run 'codegraph build --full' to rebuild the index.",
        }


@mcp.tool
def codegraph_refresh(mode: str = "incremental") -> dict:
    """Refresh the CodeGrapher index.

    Updates the index by re-indexing changed files (incremental)
    or rebuilding from scratch (full).

    Args:
        mode: "incremental" for git-changed files only, "full" for complete rebuild

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - mode: "incremental" or "full"
        - files_updated: int
        - symbols_added: int
        - symbols_removed: int
        - duration_seconds: float
        - index_age_hours: float (new age after refresh)
        - error_type: error code (if error)
        - message: error message (if error)
    """
    import subprocess

    if mode not in ("incremental", "full"):
        return {
            "status": "error",
            "error_type": "invalid_mode",
            "message": f"Invalid mode: {mode}. Use 'incremental' or 'full'.",
        }

    # Find repository root
    repo_root = find_repo_root()
    if repo_root is None:
        return {
            "status": "error",
            "error_type": "repo_not_found",
            "message": "Not in a git repository (no .git directory found).",
            "suggestion": "Run from within a git repository.",
        }

    # Get index paths
    index_dir = get_index_path(repo_root)
    db_path = index_dir / "symbols.db"
    faiss_path = index_dir / "index.faiss"

    # Check if index exists for incremental mode
    if mode == "incremental" and (not db_path.exists() or not faiss_path.exists()):
        return {
            "status": "error",
            "error_type": "index_not_found",
            "message": "No index found for incremental update.",
            "suggestion": "Run 'codegraph build --full' to create the index first.",
        }

    start_time = datetime.now()

    try:
        if mode == "full":
            # Full rebuild - use subprocess to call CLI
            result = subprocess.run(
                [sys.executable, "-m", "codegrapher.cli", "build", "--full"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "mode": mode,
                    "error_type": "build_failed",
                    "message": result.stderr or result.stdout,
                }

            # Parse output for stats
            # Look for lines like "Total symbols: 560"
            symbols_added = 0
            for line in result.stdout.split("\n"):
                if "Total symbols:" in line:
                    try:
                        symbols_added = int(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass

            return {
                "status": "success",
                "mode": mode,
                "files_updated": 0,  # CLI doesn't report this separately
                "symbols_added": symbols_added,
                "symbols_removed": 0,
                "duration_seconds": round((datetime.now() - start_time).total_seconds(), 1),
                "index_age_hours": 0.0,  # Fresh after full rebuild
            }

        else:  # incremental
            # Get git-changed files
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=M", "*.py"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "mode": mode,
                    "error_type": "git_failed",
                    "message": "Failed to get changed files from git.",
                }

            changed = result.stdout.strip().split("\n")
            files_to_update = [repo_root / f for f in changed if f and f.endswith(".py")]

            if not files_to_update:
                return {
                    "status": "success",
                    "mode": mode,
                    "files_updated": 0,
                    "symbols_added": 0,
                    "symbols_removed": 0,
                    "duration_seconds": 0.0,
                    "message": "No files changed since last update.",
                }

            # Run update command
            result = subprocess.run(
                [sys.executable, "-m", "codegrapher.cli", "update", "--git-changed"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "mode": mode,
                    "error_type": "update_failed",
                    "message": result.stderr or result.stdout,
                }

            # Parse "Files updated: X" from output
            files_updated = 0
            for line in result.stdout.split("\n"):
                if "Files updated:" in line:
                    try:
                        files_updated = int(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass

            return {
                "status": "success",
                "mode": mode,
                "files_updated": files_updated,
                "symbols_added": 0,  # CLI doesn't report this detail
                "symbols_removed": 0,
                "duration_seconds": round((datetime.now() - start_time).total_seconds(), 1),
                "index_age_hours": 0.0,  # Fresh after update
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "mode": mode,
            "error_type": "timeout",
            "message": "Index refresh timed out.",
        }
    except Exception as e:
        logger.error(f"Index refresh failed: {e}")
        return {
            "status": "error",
            "mode": mode,
            "error_type": "refresh_error",
            "message": f"Failed to refresh index: {e}",
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
