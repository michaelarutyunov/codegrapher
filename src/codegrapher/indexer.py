"""Incremental indexing logic for fast file change updates.

This module implements PRD Recipe 2: Incremental AST Diff and
PRD Recipe 5: Transaction Safety. It provides the IncrementalIndexer
class that maintains an LRU cache of parsed ASTs and computes minimal
diffs when files change, enabling <200ms update times.

Key features:
- LRU cache of 50 pickled ASTs
- SymbolDiff computation by comparing AST nodes
- Atomic transactions across SQLite + FAISS
- Crash-safe rollback on failure
"""

import ast
import logging
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from codegrapher.models import Database, Symbol
from codegrapher.sparse_index import tokenize_symbol
from codegrapher.vector_store import FAISSIndexManager


# Module logger
logger = logging.getLogger(__name__)


# Cache size from PRD Recipe 2
CACHE_SIZE = 50


@dataclass
class SymbolDiff:
    """Result of incremental AST comparison.

    Attributes:
        deleted: List of symbol IDs to remove from index
        added: List of new Symbol objects to add to index
        modified: List of Symbol objects with changed signatures/docs
    """
    deleted: List[str] = field(default_factory=list)
    added: List[Symbol] = field(default_factory=list)
    modified: List[Symbol] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if any changes exist."""
        return bool(self.deleted or self.added or self.modified)


class IncrementalIndexer:
    """Manages incremental index updates using AST caching.

    Maintains an LRU cache of pickled ASTs to compute minimal diffs
    when files change. This enables <200ms update times vs full rebuilds.

    Per PRD Recipe 2:
    - Cache size: 50 ASTs (FIFO eviction for simplicity in v1)
    - Performance: <50ms parsing + <150ms DB transaction = <200ms total
    """

    def __init__(self, cache_size: int = CACHE_SIZE):
        """Initialize the incremental indexer.

        Args:
            cache_size: Maximum number of ASTs to cache (default 50)
        """
        self._ast_cache: dict[Path, bytes] = {}
        self._cache_size = cache_size

    def update_file(
        self,
        file_path: Path,
        new_source: Optional[str] = None,
        repo_root: Optional[Path] = None
    ) -> SymbolDiff:
        """Compare new AST with cached old AST, return minimal diff.

        Computes a SymbolDiff by comparing top-level AST nodes between
        the cached old AST and newly parsed AST. Symbols are compared by
        (name, node_type) pairs to detect additions, deletions, and modifications.

        Args:
            file_path: Path to the file to update
            new_source: New source code (if None, reads from file_path)
            repo_root: Repository root for computing relative paths

        Returns:
            SymbolDiff containing changes

        Raises:
            SyntaxError: If the file contains invalid Python syntax
            FileNotFoundError: If file_path doesn't exist and new_source is None
        """
        # Step 1: Get source code
        if new_source is None:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            new_source = file_path.read_text(encoding="utf-8")

        # Step 2: Parse new AST
        try:
            new_ast = ast.parse(new_source, filename=str(file_path), mode='exec')
        except SyntaxError as e:
            logger.warning(
                f"Syntax error in {file_path}:{e.lineno}:{e.offset} - "
                f"skipping incremental update"
            )
            raise

        # Step 3: Load old AST from cache
        old_ast_bytes = self._ast_cache.get(file_path)
        if not old_ast_bytes:
            # First time seeing this file - all symbols are "added"
            new_symbols = self._build_symbol_map(new_ast)
            diff = SymbolDiff()
            for key, node in new_symbols.items():
                symbol = _node_to_symbol(node, file_path, repo_root)
                if symbol:
                    diff.added.append(symbol)
            self._update_cache(file_path, new_ast)
            return diff

        # Step 4: Build symbol multisets keyed by (name, type)
        old_ast = pickle.loads(old_ast_bytes)

        old_symbols = self._build_symbol_map(old_ast)
        new_symbols = self._build_symbol_map(new_ast)

        # Step 5: Compute diff
        deleted_keys = set(old_symbols.keys()) - set(new_symbols.keys())
        added_keys = set(new_symbols.keys()) - set(old_symbols.keys())
        common_keys = set(old_symbols.keys()) & set(new_symbols.keys())

        diff = SymbolDiff()

        # Deleted symbols
        # Note: Symbol IDs in the cache use full module path
        stem = file_path.stem
        diff.deleted = [f"{stem}.{key[0]}" for key in deleted_keys]

        # Added symbols
        for key in added_keys:
            node = new_symbols[key]
            symbol = _node_to_symbol(node, file_path, repo_root)
            if symbol:
                diff.added.append(symbol)

        # Modified symbols (same name/type but different content)
        for key in common_keys:
            old_node, new_node = old_symbols[key], new_symbols[key]

            if _needs_reembed(old_node, new_node):
                symbol = _node_to_symbol(new_node, file_path, repo_root)
                if symbol:
                    diff.modified.append(symbol)

        # Step 6: Update cache
        self._update_cache(file_path, new_ast)

        return diff

    def _build_symbol_map(self, tree: ast.Module) -> dict:
        """Build a map of (name, type) -> AST node for top-level symbols.

        Args:
            tree: Parsed AST module

        Returns:
            Dictionary mapping (name, type) tuples to AST nodes
        """
        symbol_map = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                symbol_map[(node.name, node.__class__.__name__)] = node
            elif isinstance(node, ast.Assign):
                # Module-level assignments (e.g., CONSTANT = ...)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbol_map[(target.id, ast.Assign.__name__)] = node
        return symbol_map

    def _update_cache(self, file_path: Path, tree: ast.Module) -> None:
        """Update the AST cache with a new tree.

        Implements FIFO eviction when cache exceeds cache_size.

        Args:
            file_path: Path to the file
            tree: Parsed AST to cache
        """
        # Add to cache
        self._ast_cache[file_path] = pickle.dumps(tree)

        # Evict oldest if over limit (FIFO for v1 simplicity)
        if len(self._ast_cache) > self._cache_size:
            # Get first key (oldest in insertion order)
            oldest = next(iter(self._ast_cache))
            del self._ast_cache[oldest]

    def invalidate(self, file_path: Path) -> None:
        """Remove a file from the AST cache.

        Useful when a file is deleted or when cache needs clearing.

        Args:
            file_path: Path to invalidate
        """
        self._ast_cache.pop(file_path, None)

    def clear(self) -> None:
        """Clear all cached ASTs."""
        self._ast_cache.clear()


def _node_to_symbol(
    node: ast.AST,
    file_path: Path,
    repo_root: Optional[Path] = None
) -> Optional[Symbol]:
    """Convert an AST node to a Symbol object.

    Args:
        node: AST node (FunctionDef, ClassDef, or Assign)
        file_path: Path to the file containing the node
        repo_root: Repository root for relative paths

    Returns:
        Symbol object, or None if node type is not supported
    """
    # Import here to avoid circular dependency
    from codegrapher.parser import (
        _format_function_signature,
        _format_class_signature,
        _extract_first_docstring_line,
        _get_end_line,
        EMBEDDING_DIM,
    )
    import numpy as np

    # Get module name for symbol ID
    if repo_root:
        try:
            rel_path = str(file_path.relative_to(repo_root))
        except ValueError:
            rel_path = str(file_path)
    else:
        rel_path = str(file_path)

    module_name = rel_path.replace(".py", "").replace("/", ".")

    # Placeholder embedding - will be replaced during indexing
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if isinstance(node, ast.FunctionDef):
        # Check if this is a method (has parent class)
        class_name = getattr(node, '_class_name', None)

        if class_name:
            symbol_id = f"{module_name}.{class_name}.{node.name}"
        else:
            symbol_id = f"{module_name}.{node.name}"

        return Symbol(
            id=symbol_id,
            file=rel_path,
            start_line=node.lineno,
            end_line=_get_end_line(node),
            signature=_format_function_signature(node),
            doc=_extract_first_docstring_line(node),
            mutates="",
            embedding=embedding
        )

    elif isinstance(node, ast.ClassDef):
        symbol_id = f"{module_name}.{node.name}"
        return Symbol(
            id=symbol_id,
            file=rel_path,
            start_line=node.lineno,
            end_line=_get_end_line(node),
            signature=_format_class_signature(node),
            doc=_extract_first_docstring_line(node),
            mutates="",
            embedding=embedding
        )

    elif isinstance(node, ast.Assign):
        # Module-level assignment
        for target in node.targets:
            if isinstance(target, ast.Name):
                symbol_id = f"{module_name}.{target.id}"
                # Simple signature for assignments
                try:
                    value_repr = ast.unparse(node.value)
                except Exception:
                    value_repr = "<complex expression>"

                return Symbol(
                    id=symbol_id,
                    file=rel_path,
                    start_line=node.lineno,
                    end_line=_get_end_line(node),
                    signature=f"{target.id} = {value_repr}",
                    doc="",
                    mutates="",
                    embedding=embedding
                )

    return None


def _needs_reembed(
    old_node: ast.FunctionDef | ast.ClassDef | ast.Assign,
    new_node: ast.FunctionDef | ast.ClassDef | ast.Assign
) -> bool:
    """Check if symbol changed in a way that requires re-embedding.

    Compares line numbers, docstrings, and signatures to determine if
    the semantic content changed enough to warrant re-embedding.

    Args:
        old_node: Old AST node
        new_node: New AST node

    Returns:
        True if re-embedding is needed, False otherwise
    """
    # Import helper functions
    from codegrapher.parser import (
        _format_function_signature,
        _format_class_signature,
        _extract_first_docstring_line,
    )

    # Compare line numbers (moved?)
    if old_node.lineno != new_node.lineno:
        return True

    # Compare end_lineno if available
    old_end = getattr(old_node, 'end_lineno', None)
    new_end = getattr(new_node, 'end_lineno', None)
    if old_end != new_end:
        return True

    # For assignments: compare values (they don't have docstrings)
    if isinstance(old_node, ast.Assign) and isinstance(new_node, ast.Assign):
        try:
            old_val = ast.unparse(old_node.value)
            new_val = ast.unparse(new_node.value)
            return old_val != new_val
        except Exception:
            # If unparsing fails, assume it changed
            return True

    # For functions/classes: compare docstrings
    # (Assign nodes are handled above and skip this)
    if not isinstance(old_node, ast.Assign) and not isinstance(new_node, ast.Assign):
        old_doc = _extract_first_docstring_line(old_node)  # type: ignore
        new_doc = _extract_first_docstring_line(new_node)  # type: ignore
        if old_doc != new_doc:
            return True

    # For functions: compare signature
    if isinstance(old_node, ast.FunctionDef) and isinstance(new_node, ast.FunctionDef):
        old_sig = _format_function_signature(old_node)
        new_sig = _format_function_signature(new_node)
        if old_sig != new_sig:
            return True

        # Compare decorators
        old_decs = [
            d.id if isinstance(d, ast.Name) else ast.unparse(d)
            for d in old_node.decorator_list
        ]
        new_decs = [
            d.id if isinstance(d, ast.Name) else ast.unparse(d)
            for d in new_node.decorator_list
        ]
        if old_decs != new_decs:
            return True

    # For classes: compare base classes
    elif isinstance(old_node, ast.ClassDef) and isinstance(new_node, ast.ClassDef):
        old_sig = _format_class_signature(old_node)
        new_sig = _format_class_signature(new_node)
        if old_sig != new_sig:
            return True

        # Compare bases
        old_bases = [ast.unparse(b) for b in old_node.bases]
        new_bases = [ast.unparse(b) for b in new_node.bases]
        if old_bases != new_bases:
            return True

    return False


@contextmanager
def atomic_update(db_path: Path, faiss_manager: FAISSIndexManager):
    """Context manager for atomic updates across SQLite + FAISS.

    Implements PRD Recipe 5: Transaction Safety. Uses BEGIN IMMEDIATE
    to lock the database, snapshots FAISS state, and rolls back both
    on any exception.

    Args:
        db_path: Path to SQLite database
        faiss_manager: FAISS index manager

    Yields:
        SQLite connection for making updates

    Example:
        ```python
        with atomic_update(db_path, faiss_mgr) as conn:
            # Apply updates
            conn.executemany("DELETE FROM symbols WHERE id = ?", [...])
            # Both commit atomically or rollback together
        ```
    """
    conn = sqlite3.connect(db_path)
    conn.execute("BEGIN IMMEDIATE")  # Lock database

    # Snapshot FAISS state for rollback
    faiss_backup = {
        'index': faiss_manager.index,
        'symbol_ids': faiss_manager.symbol_ids.copy()
    }

    try:
        yield conn  # Caller performs SQL + FAISS updates

        # Commit both atomically
        conn.commit()
        faiss_manager.save()  # Write FAISS to disk

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


def apply_diff(
    diff: SymbolDiff,
    db: Database,
    faiss_manager: FAISSIndexManager
) -> None:
    """Apply a SymbolDiff to the database and FAISS index.

    Deletes old symbols, inserts new symbols, and updates modified symbols.
    Also builds and stores sparse terms for BM25 search.
    Must be called within an atomic_update context.

    Args:
        diff: SymbolDiff to apply
        db: Database instance
        faiss_manager: FAISS index manager
    """
    # Delete old symbols
    if diff.deleted:
        db.delete_symbols_batch(diff.deleted)
        db.delete_sparse_terms_batch(diff.deleted)

    # Delete from FAISS
    if diff.deleted:
        faiss_manager.remove_symbols(diff.deleted)

    # Insert new symbols
    if diff.added:
        for symbol in diff.added:
            db.insert_symbol(symbol)

    # Build and store sparse terms for new symbols
    if diff.added:
        sparse_terms: dict[str, List[str]] = {}
        for symbol in diff.added:
            tokens = tokenize_symbol(symbol)
            sparse_terms[symbol.id] = tokens
        db.insert_sparse_terms_batch(sparse_terms)

    # Add to FAISS
    if diff.added:
        faiss_manager.add_symbols(diff.added)

    # Update modified symbols
    if diff.modified:
        for symbol in diff.modified:
            db.update_symbol(symbol)

    # Update sparse terms for modified symbols
    if diff.modified:
        sparse_terms: dict[str, List[str]] = {}
        for symbol in diff.modified:
            tokens = tokenize_symbol(symbol)
            sparse_terms[symbol.id] = tokens
        db.insert_sparse_terms_batch(sparse_terms)

    # Update in FAISS (requires remove + add for IndexFlatL2)
    if diff.modified:
        # FAISS IndexFlatL2 doesn't support in-place updates
        # Remove and re-add modified symbols
        modified_ids = [s.id for s in diff.modified]
        faiss_manager.remove_symbols(modified_ids)
        faiss_manager.add_symbols(diff.modified)
