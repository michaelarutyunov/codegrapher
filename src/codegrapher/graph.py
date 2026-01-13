"""Call graph construction and PageRank computation.

This module builds the directed call graph from extracted symbols and computes
PageRank scores for symbol importance ranking.

Per PRD Section 5:
- Call graph edges: function calls between symbols
- PageRank parameters: α=0.85, max_iter=100, tolerance=1e-6
- PageRank computed on directed call-graph edges only (imports/inheritance
  are stored but not used for PageRank in v1)
"""

import ast
from pathlib import Path
from typing import Dict, List

import networkx as nx

from codegrapher.models import Database, Edge, Symbol


# PageRank parameters from PRD Section 5
PAGERANK_ALPHA = 0.85
PAGERANK_MAX_ITER = 100
PAGERANK_TOLERANCE = 1e-6


def extract_edges_from_file(
    file_path: Path, repo_root: Path, symbols: List[Symbol]
) -> List[Edge]:
    """Extract call graph edges from a Python file.

    Analyzes the AST to find:
    - Function calls between symbols
    - Inheritance relationships (class -> base classes)
    - Import statements (module -> imported modules)

    Args:
        file_path: Path to the Python file
        repo_root: Repository root directory
        symbols: List of symbols already extracted from this file

    Returns:
        List of Edge objects representing relationships
    """
    if not file_path.exists():
        return []

    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    # Build symbol ID lookup for this file
    symbol_ids = {s.id: s for s in symbols}

    edges: List[Edge] = []

    # Walk through all nodes to find calls
    for node in ast.walk(tree):
        # Find function/class definitions to get their context
        current_symbol_id = None

        if isinstance(node, ast.FunctionDef):
            # Check if this is one of our symbols
            module_name = _path_to_module_name(
                file_path.relative_to(repo_root)
            )
            symbol_id = f"{module_name}.{node.name}"
            if symbol_id in symbol_ids:
                current_symbol_id = symbol_id

        elif isinstance(node, ast.ClassDef):
            module_name = _path_to_module_name(
                file_path.relative_to(repo_root)
            )
            class_id = f"{module_name}.{node.name}"
            if class_id in symbol_ids:
                current_symbol_id = class_id

                # Extract inheritance edges
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        # Simple base class like `class Foo(Bar):`
                        base_id = _resolve_base_class_name(
                            base.id, file_path, repo_root
                        )
                        if base_id:
                            edges.append(
                                Edge(
                                    caller_id=class_id,
                                    callee_id=base_id,
                                    type="inherit",
                                )
                            )
                    elif isinstance(base, ast.Attribute):
                        # Attribute base like `class Foo(module.Class):`
                        base_name = _unparse_attribute(base)
                        base_id = _resolve_base_class_name(
                            base_name, file_path, repo_root
                        )
                        if base_id:
                            edges.append(
                                Edge(
                                    caller_id=class_id,
                                    callee_id=base_id,
                                    type="inherit",
                                )
                            )

        # Find function calls within the current symbol
        if current_symbol_id:
            call_edges = _extract_function_calls(node, current_symbol_id)
            edges.extend(call_edges)

    # Extract import edges
    import_edges = _extract_import_edges(tree, file_path, repo_root)
    edges.extend(import_edges)

    return edges


def compute_pagerank(db: Database) -> Dict[str, float]:
    """Compute PageRank scores for all symbols in the database.

    Per PRD Section 5, PageRank is computed on directed call-graph
    edges only (type='call'). Imports and inheritance edges are stored
    but not used for PageRank in v1.

    Args:
        db: Database instance with symbols and edges

    Returns:
        Dictionary mapping symbol_id to PageRank score (normalized to 0-1)
    """
    # Get all edges from database
    all_edges = db.get_all_edges()

    # Filter to only call edges for PageRank
    call_edges = [(e.caller_id, e.callee_id) for e in all_edges if e.type == "call"]

    if not call_edges:
        return {}

    # Build directed graph
    graph = nx.DiGraph()
    graph.add_edges_from(call_edges)

    # Add isolated nodes (symbols with no calls)
    all_symbols = db.get_all_symbols()
    symbol_ids = {s.id for s in all_symbols}
    for symbol_id in symbol_ids:
        if symbol_id not in graph:
            graph.add_node(symbol_id)

    # Compute PageRank with PRD parameters
    raw_scores = nx.pagerank(
        graph,
        alpha=PAGERANK_ALPHA,
        max_iter=PAGERANK_MAX_ITER,
        tol=PAGERANK_TOLERANCE,
    )

    # Normalize to 0-1 range
    if raw_scores:
        max_score = max(raw_scores.values())
        if max_score > 0:
            return {k: v / max_score for k, v in raw_scores.items()}

    return raw_scores


def _extract_function_calls(
    node: ast.AST, caller_id: str
) -> List[Edge]:
    """Extract function call edges from an AST node.

    Finds direct function calls like `foo()` and method calls like
    `obj.method()`.

    Args:
        node: AST node to search for calls
        caller_id: Symbol ID of the caller

    Returns:
        List of Edge objects with type='call'
    """
    edges: List[Edge] = []

    if isinstance(node, ast.Call):
        # Get the function name being called
        callee_name = None

        if isinstance(node.func, ast.Name):
            # Direct call like `foo()`
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Method call like `obj.method()`
            callee_name = _unparse_attribute(node.func)

        if callee_name:
            # Create a simple symbol ID (may not match exact symbols,
            # but that's okay - edges are fuzzy in v1)
            callee_id = f"<unknown>.{callee_name}"
            edges.append(
                Edge(caller_id=caller_id, callee_id=callee_id, type="call")
            )

    # Recursively search for calls in child nodes
    for child in ast.iter_child_nodes(node):
        if child is node:
            continue
        edges.extend(_extract_function_calls(child, caller_id))

    return edges


def _extract_import_edges(
    tree: ast.AST, file_path: Path, repo_root: Path
) -> List[Edge]:
    """Extract import edges from an AST tree.

    Creates edges from the current module to imported modules.

    Args:
        tree: Parsed AST tree
        file_path: Path of the file being analyzed
        repo_root: Repository root

    Returns:
        List of Edge objects with type='import'
    """
    edges: List[Edge] = []

    # Get the module name for this file
    module_name = _path_to_module_name(file_path.relative_to(repo_root))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Create edge to imported module
                imported_module = alias.name
                edges.append(
                    Edge(
                        caller_id=module_name,
                        callee_id=imported_module,
                        type="import",
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Create edge to imported module
                edges.append(
                    Edge(
                        caller_id=module_name,
                        callee_id=node.module,
                        type="import",
                    )
                )

    return edges


def _path_to_module_name(file_path: Path) -> str:
    """Convert a file path to a Python module name.

    Examples:
        "src/mypackage/utils.py" -> "src.mypackage.utils"
        "mypackage/__init__.py" -> "mypackage"

    Args:
        file_path: File path (relative or absolute)

    Returns:
        Dotted module name
    """
    path_str = str(file_path).replace(".py", "").replace("/", ".")
    path_str = path_str.replace(".__init__", "")
    return path_str


def _unparse_attribute(node: ast.Attribute) -> str:
    """Convert an attribute AST node to a string.

    Examples:
        `obj.method` -> "obj.method"
        `module.Class.attr` -> "module.Class.attr"

    Args:
        node: AST Attribute node

    Returns:
        String representation of the attribute access
    """
    if isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    elif isinstance(node.value, ast.Attribute):
        return f"{_unparse_attribute(node.value)}.{node.attr}"
    else:
        # Fallback for complex expressions
        return "<unknown>"


def _resolve_base_class_name(
    base_name: str, file_path: Path, repo_root: Path
) -> str:
    """Resolve a base class name to a fully qualified symbol ID.

    For v1, this is a simplified implementation that returns a
    best-effort ID. Full resolution would require import analysis
    from Phase 6.

    Args:
        base_name: Simple name of the base class
        file_path: File containing the inheritance
        repo_root: Repository root

    Returns:
        Best-effort symbol ID for the base class
    """
    # In v1, we use a simple heuristic:
    # 1. Check if base class is in the same module
    # 2. Otherwise, return as-is (will be resolved during query time)

    module_name = _path_to_module_name(file_path.relative_to(repo_root))

    # Check if this is a same-module reference
    # (e.g., `class Foo(Bar):` where Bar is defined earlier)
    # For v1, we just return the module-prefixed version
    return f"{module_name}.{base_name}"
