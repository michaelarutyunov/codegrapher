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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx

from codegrapher.models import Database, Edge, Symbol


# PageRank parameters from PRD Section 5
PAGERANK_ALPHA = 0.85
PAGERANK_MAX_ITER = 100
PAGERANK_TOLERANCE = 1e-6


@dataclass
class ClassHierarchyMap:
    """In-memory mapping of class hierarchy for a single file.

    Built during edge extraction and used for 1-level MRO resolution.

    Attributes:
        class_methods: Dict mapping class_id → set of method names
                     Example: {"c2.A2": {"B2", "other"}, "c1.A1": {"B1"}}
        class_parents: Dict mapping class_id → list of parent class IDs
                     Example: {"c2.A2": ["c1.A1"], "c1.A1": []}
    """
    class_methods: Dict[str, Set[str]] = field(default_factory=dict)
    class_parents: Dict[str, List[str]] = field(default_factory=dict)

    def get_method_defining_class(self, class_id: str, method_name: str) -> Optional[str]:
        """Find which class defines a method using 1-level lookup.

        Search order:
        1. Check if method is in current class
        2. Check first parent (immediate base class only)
        3. Return None if not found

        Args:
            class_id: Full class ID (e.g., "c2.A2")
            method_name: Method name (e.g., "B1")

        Returns:
            Class ID that defines the method, or None if not found
        """
        # 1. Check current class
        if method_name in self.class_methods.get(class_id, set()):
            return class_id

        # 2. Check first parent only (1-level MRO)
        parents = self.class_parents.get(class_id, [])
        if parents and method_name in self.class_methods.get(parents[0], set()):
            return parents[0]

        # 3. Not found
        return None


def _build_class_hierarchy_map(
    symbols: List[Symbol],
    inherit_edges: List[Edge]
) -> ClassHierarchyMap:
    """Build a class hierarchy map for 1-level MRO resolution.

    Args:
        symbols: All symbols extracted from the current file
        inherit_edges: All inheritance edges extracted from the current file

    Returns:
        ClassHierarchyMap with class → methods and class → parents mappings

    Notes:
        - Only includes classes defined in THIS file
        - Cross-file parent references are kept in class_parents but won't
          have entries in class_methods (handled gracefully by get_method_defining_class)
    """
    hierarchy = ClassHierarchyMap()

    # Build class → methods mapping
    for symbol in symbols:
        parts = symbol.id.split(".")
        # Only methods have format: module.Class.method (3+ parts)
        if len(parts) >= 3:
            class_id = ".".join(parts[:-1])  # module.Class
            method_name = parts[-1]  # method

            if class_id not in hierarchy.class_methods:
                hierarchy.class_methods[class_id] = set()
            hierarchy.class_methods[class_id].add(method_name)

    # Build class → parents mapping from inheritance edges
    for edge in inherit_edges:
        if edge.type == "inherit":
            child_id = edge.caller_id
            parent_id = edge.callee_id

            if child_id not in hierarchy.class_parents:
                hierarchy.class_parents[child_id] = []
            hierarchy.class_parents[child_id].append(parent_id)

    return hierarchy


def _resolve_method_with_1level_mro(
    callee_name: str,
    caller_id: str,
    hierarchy_map: ClassHierarchyMap
) -> str:
    """Resolve a method call using 1-level inheritance lookup.

    Args:
        callee_name: Method call like "B1" (without self.) or "self.B1"
        caller_id: Full caller symbol ID (e.g., "c2.A2.B2")
        hierarchy_map: ClassHierarchyMap for the current file

    Returns:
        Resolved callee ID (e.g., "c1.A1.B1") or fallback fuzzy ID

    Examples:
        >>> _resolve_method_with_1level_mro("B1", "c2.A2.B2", hierarchy)
        "c1.A1.B1"  # Found in parent class

        >>> _resolve_method_with_1level_mro("Zebra", "c2.A2.B2", hierarchy)
        "c2.A2.Zebra"  # Not found, return contextual fuzzy
    """
    # Extract method name (strip self. or cls. prefix if present)
    if callee_name.startswith("self."):
        method_name = callee_name[5:]
    elif callee_name.startswith("cls."):
        method_name = callee_name[4:]
    else:
        method_name = callee_name

    # Parse caller_id to get class_id
    parts = caller_id.split(".")
    if len(parts) < 3:
        # Not a method (standalone function)
        return f"<unknown>.{callee_name}"

    class_id = ".".join(parts[:-1])  # e.g., "c2.A2"

    # Try 1-level resolution
    defining_class = hierarchy_map.get_method_defining_class(class_id, method_name)
    if defining_class is not None:
        return f"{defining_class}.{method_name}"

    # Fallback to contextual fuzzy
    return f"{class_id}.{method_name}"


def extract_edges_from_file(
    file_path: Path, repo_root: Path, symbols: List[Symbol]
) -> List[Edge]:
    """Extract call graph edges from a Python file.

    Analyzes the AST to find:
    - Function calls between symbols (with 1-level inheritance resolution)
    - Inheritance relationships (class -> base classes)
    - Import statements (module -> imported modules)

    Uses a two-pass approach:
    1. Extract inheritance edges and build class hierarchy map
    2. Extract call edges using 1-level MRO resolution

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
    module_name = _path_to_module_name(file_path.relative_to(repo_root))

    edges: List[Edge] = []

    # Pass 1: Extract inheritance edges and build class hierarchy map
    inherit_edges: List[Edge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_id = f"{module_name}.{node.name}"
            if class_id in symbol_ids:
                # Extract inheritance edges
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        # Simple base class like `class Foo(Bar):`
                        base_id = _resolve_base_class_name(
                            base.id, file_path, repo_root
                        )
                        if base_id:
                            inherit_edges.append(
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
                            inherit_edges.append(
                                Edge(
                                    caller_id=class_id,
                                    callee_id=base_id,
                                    type="inherit",
                                )
                            )

    # Add inheritance edges to result
    edges.extend(inherit_edges)

    # Build class hierarchy map for 1-level MRO
    hierarchy_map = _build_class_hierarchy_map(symbols, inherit_edges)

    # Pass 2: Extract call edges with 1-level MRO resolution
    # We manually traverse the AST to properly track class context
    def _extract_calls_from_node(node: ast.AST, class_context: Optional[str]) -> List[Edge]:
        """Recursively extract call edges, tracking class context."""
        call_edges: List[Edge] = []

        # Track if we're entering a class or function
        if isinstance(node, ast.ClassDef):
            class_id = f"{module_name}.{node.name}"
            if class_id in symbol_ids:
                # This is a tracked class, use it as context for children
                new_class_context = class_id
            else:
                new_class_context = class_context
        elif isinstance(node, ast.FunctionDef):
            # Determine the full symbol ID
            if class_context:
                class_name = class_context.split(".")[-1]
                symbol_id = f"{module_name}.{class_name}.{node.name}"
            else:
                symbol_id = f"{module_name}.{node.name}"

            if symbol_id in symbol_ids:
                # This is a tracked function, extract its calls
                call_edges = _extract_function_calls_with_mro(
                    node, symbol_id, hierarchy_map
                )
                # Don't recurse further - we already do that in _extract_function_calls_with_mro
                return call_edges
            else:
                # Not a tracked function, continue to children
                new_class_context = class_context
        else:
            new_class_context = class_context

        # Recurse into children
        for child in ast.iter_child_nodes(node):
            call_edges.extend(_extract_calls_from_node(child, new_class_context))

        return call_edges

    # Start extraction from the module level
    edges.extend(_extract_calls_from_node(tree, None))

    # Extract import edges
    import_edges = _extract_import_edges(tree, file_path, repo_root)
    edges.extend(import_edges)

    return edges


def _extract_function_calls_with_mro(
    node: ast.AST,
    caller_id: str,
    hierarchy_map: ClassHierarchyMap
) -> List[Edge]:
    """Extract function call edges with 1-level MRO resolution.

    Similar to _extract_function_calls but attempts to resolve self.method()
    calls using the class hierarchy map.

    Args:
        node: AST node to search for calls
        caller_id: Symbol ID of the caller
        hierarchy_map: ClassHierarchyMap for MRO resolution

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
            # For self.method() and cls.method(), try 1-level MRO resolution
            if callee_name.startswith("self.") or callee_name.startswith("cls."):
                callee_id = _resolve_method_with_1level_mro(
                    callee_name, caller_id, hierarchy_map
                )
            else:
                # Use contextual ID for direct calls and obj.method()
                callee_id = _make_contextual_callee_id(callee_name, caller_id)

            edges.append(
                Edge(caller_id=caller_id, callee_id=callee_id, type="call")
            )

    # Recursively search for calls in child nodes
    for child in ast.iter_child_nodes(node):
        if child is node:
            continue
        edges.extend(_extract_function_calls_with_mro(child, caller_id, hierarchy_map))

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
    `obj.method()`. Uses contextual callee IDs instead of fuzzy
    "<unknown>" prefixes where possible.

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
            # Use contextual callee ID instead of fuzzy <unknown>
            callee_id = _make_contextual_callee_id(callee_name, caller_id)
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


def _make_contextual_callee_id(callee_name: str, caller_id: str) -> str:
    """Generate a contextual callee ID from a call expression.

    Replaces fuzzy IDs like "<unknown>.self.B1" with contextual names
    like "c2.A2.B1" based on the caller's context.

    Args:
        callee_name: The extracted callee name (e.g., "self.B1", "foo", "obj.method")
        caller_id: Full symbol ID of caller (e.g., "c2.A2.B2", "c2.standalone_func")

    Returns:
        Contextual callee ID:
        - For "self.XXX": "{module}.{class}.{XXX}" (e.g., "c2.A2.B1")
        - For "cls.XXX": "{module}.{class}.{XXX}" (e.g., "c2.A2.factory")
        - For direct calls: "{XXX}" (e.g., "foo")
        - For obj.method: "obj.method" (unchanged)

    Examples:
        >>> _make_contextual_callee_id("self.B1", "c2.A2.B2")
        "c2.A2.B1"
        >>> _make_contextual_callee_id("foo", "c2.A2.B2")
        "foo"
        >>> _make_contextual_callee_id("obj.method", "c2.A2.B2")
        "obj.method"
    """
    # Handle self.method() calls
    if callee_name.startswith("self."):
        parts = caller_id.split(".")
        # Check if caller is a method (has at least 3 parts: module.class.method)
        if len(parts) >= 3:
            module = ".".join(parts[:-2])  # Everything except last two parts
            method_name = callee_name[5:]  # Strip "self."
            return f"{module}.{parts[-2]}.{method_name}"
        else:
            # self. used in non-method context (invalid Python)
            return f"<invalid>.{callee_name}"

    # Handle cls.method() calls (classmethods)
    if callee_name.startswith("cls."):
        parts = caller_id.split(".")
        if len(parts) >= 3:
            module = ".".join(parts[:-2])
            method_name = callee_name[4:]  # Strip "cls."
            return f"{module}.{parts[-2]}.{method_name}"
        else:
            return f"<invalid>.{callee_name}"

    # Direct calls and obj.method stay as-is
    return callee_name


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
