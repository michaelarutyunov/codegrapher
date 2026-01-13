"""Import resolution logic for mapping import strings to file paths.

This module implements PRD Recipe 1: Import Closure Pruning. It resolves
Python import statements to their corresponding file paths within the
repository, distinguishing between:
1. Relative imports (e.g., "..utils", ".config")
2. Absolute imports within the repo (e.g., "mypackage.foo")
3. External imports (stdlib and site-packages) - returns None

The resolver is used by the query engine to filter search results to only
files reachable from the user's cursor location.
"""

import sys
from pathlib import Path
from typing import List, Optional, Set


# Standard library modules cache - built once on first use
_STDLIB_MODULES: Optional[Set[str]] = None


def _get_stdlib_modules() -> Set[str]:
    """Get the set of standard library module names.

    Uses sys.stdlib_module_names (Python 3.10+) or falls back to
    sys.builtin_module_names with a hardcoded list of common stdlib modules.

    Returns:
        Set of standard library module names
    """
    global _STDLIB_MODULES
    if _STDLIB_MODULES is not None:
        return _STDLIB_MODULES

    stdlib: Set[str] = set()

    # Python 3.10+ has sys.stdlib_module_names
    if hasattr(sys, "stdlib_module_names"):
        stdlib = set(sys.stdlib_module_names)
    else:
        # Fallback for Python 3.8-3.9: use builtins + known stdlib
        stdlib = set(sys.builtin_module_names)
        # Common stdlib modules (not exhaustive, but covers most used)
        known_stdlib = {
            "argparse", "array", "asyncio", "base64", "bisect", "bool",
            "collections", "contextlib", "csv", "dataclasses", "datetime",
            "decimal", "dict", "enum", "errno", "filecmp", "float", "fnmatch",
            "fractions", "functools", "gc", "glob", "graphlib", "gzip",
            "hashlib", "heapq", "html", "int", "io", "ipaddress", "itertools",
            "json", "list", "logging", "math", "mmap", "numbers", "operator",
            "os", "pathlib", "pickle", "pprint", "random", "re", "select",
            "set", "shelve", "shutil", "signal", "socket", "sqlite3", "str",
            "string", "struct", "subprocess", "tarfile", "tempfile", "textwrap",
            "threading", "time", "timeit", "token", "traceback", "tuple",
            "types", "typing", "typing_extensions", "unicodedata", "unittest",
            "urllib", "uuid", "warnings", "weakref", "xml", "zipfile", "zoneinfo",
        }
        stdlib.update(known_stdlib)

    _STDLIB_MODULES = stdlib
    return stdlib


def _is_stdlib(module_name: str) -> bool:
    """Check if a module is part of the Python standard library.

    Args:
        module_name: Module name to check (e.g., "os.path", "sys")

    Returns:
        True if the module is in the standard library
    """
    stdlib = _get_stdlib_modules()

    # Check exact match
    if module_name in stdlib:
        return True

    # Check top-level module
    top_level = module_name.split(".")[0]
    return top_level in stdlib


def resolve_import_to_path(
    module_name: str,
    current_file: Path,
    repo_root: Path
) -> Optional[Path]:
    """Resolve an import string to an absolute file path within the repo.

    Handles both relative imports (e.g., '..utils', '.config') and absolute
    imports (e.g., 'mypackage.foo'). Returns None for external libraries.

    Relative imports are resolved from the directory containing current_file.
    Absolute imports are resolved by searching for the module under repo_root.

    Args:
        module_name: Import string from ast.Import or ast.ImportFrom.
                     May have '.' prefix for relative imports
                     (e.g., "..utils", ".config", "mypackage.foo").
        current_file: File containing the import statement (absolute path).
        repo_root: Root directory of the repository (absolute path).

    Returns:
        Absolute Path if import resolves to a repo file, None if
        import is external (stdlib or site-packages).

    Raises:
        ValueError: If module_name is empty or invalid
    """
    if not module_name:
        raise ValueError("module_name cannot be empty")

    if not current_file.is_absolute():
        raise ValueError(f"current_file must be absolute, got: {current_file}")

    if not repo_root.is_absolute():
        raise ValueError(f"repo_root must be absolute, got: {repo_root}")

    # Check if it's a relative import (starts with .)
    if module_name.startswith("."):
        return _resolve_relative_import(module_name, current_file, repo_root)

    # Absolute import - first check if it's stdlib
    if _is_stdlib(module_name):
        return None

    # Try to resolve absolute import within repo
    return _resolve_absolute_import(module_name, repo_root)


def _resolve_relative_import(
    module_name: str,
    current_file: Path,
    repo_root: Path
) -> Optional[Path]:
    """Resolve a relative import to a file path.

    Args:
        module_name: Import string with leading dots (e.g., "..utils")
        current_file: File containing the import
        repo_root: Repository root directory

    Returns:
        Absolute Path if found within repo, None otherwise
    """
    # Count leading dots to determine parent level
    level = 0
    for char in module_name:
        if char == '.':
            level += 1
        else:
            break

    # Get the module part (remove leading dots)
    module_part = module_name[level:] if level < len(module_name) else ""

    # Get the directory containing current_file
    current_dir = current_file.parent

    # Navigate up the directory hierarchy
    target_dir = current_dir
    for _ in range(level - 1):
        target_dir = target_dir.parent

    # If module_part is empty, it's a package import (e.g., "from . import foo")
    if not module_part:
        # Check for __init__.py in target_dir
        init_file = target_dir / "__init__.py"
        if init_file.exists():
            return init_file
        return None

    # Try to resolve the module
    # First, try as a file (module.py)
    module_file = target_dir / f"{module_part.replace('.', '/')}.py"
    if module_file.exists():
        # Check it's within repo
        try:
            return module_file.relative_to(repo_root)
        except ValueError:
            return None

    # Then, try as a package (module/__init__.py)
    package_dir = target_dir / module_part.replace('.', '/')
    init_file = package_dir / "__init__.py"
    if init_file.exists():
        try:
            return init_file.relative_to(repo_root)
        except ValueError:
            return None

    # Not found
    return None


def _resolve_absolute_import(
    module_name: str,
    repo_root: Path
) -> Optional[Path]:
    """Resolve an absolute import to a file path within the repo.

    Searches for the module under repo_root. Handles both simple modules
    (e.g., "utils") and dotted paths (e.g., "mypackage.submodule").

    Args:
        module_name: Absolute import string (e.g., "mypackage.foo")
        repo_root: Repository root directory

    Returns:
        Absolute Path if found within repo, None otherwise
    """
    # Split module name into parts
    parts = module_name.split(".")

    # Build all possible paths to check
    # e.g., for "mypackage.submodule", check:
    # 1. repo_root/mypackage/submodule.py
    # 2. repo_root/mypackage/submodule/__init__.py

    # First, try to find as Python file
    module_path = repo_root / "/".join(parts)
    module_path = module_path.with_suffix(".py")
    if module_path.exists():
        return module_path.absolute()

    # Then, try as package (last part is a directory)
    for i in range(len(parts), 0, -1):
        # Check for package/__init__.py
        package_path = repo_root / "/".join(parts[:i])
        init_file = package_path / "__init__.py"

        if init_file.exists():
            # If we have more parts (e.g., "pkg.sub.module"), check if submodule exists
            if i < len(parts):
                remaining = parts[i:]
                # Try as file
                sub_file = package_path / "/".join(remaining)
                sub_file = sub_file.with_suffix(".py")
                if sub_file.exists():
                    return sub_file.absolute()
                # Try as package
                sub_package = package_path / "/".join(remaining)
                sub_init = sub_package / "__init__.py"
                if sub_init.exists():
                    return sub_init.absolute()
            else:
                return init_file.absolute()

    # Also check common source directories
    for src_dir in ["src", "lib", "app"]:
        alt_root = repo_root / src_dir
        if alt_root.exists():
            result = _resolve_absolute_import(module_name, alt_root)
            if result:
                return result

    # Not found in repo - must be external
    return None


def build_import_graph(repo_root: Path) -> dict[str, List[str]]:
    """Build an import graph mapping files to their imported files.

    Scans all Python files in the repository and builds a mapping
    from file path to list of imported file paths (relative to repo_root).

    Args:
        repo_root: Root directory of the repository

    Returns:
        Dictionary mapping file paths to lists of imported file paths
    """
    from codegrapher.parser import extract_imports

    graph: dict[str, List[str]] = {}

    # Find all Python files
    py_files = list(repo_root.rglob("*.py"))

    for py_file in py_files:
        # Skip __pycache__ and virtual environments
        if "__pycache__" in str(py_file):
            continue
        if "venv" in str(py_file) or ".venv" in str(py_file):
            continue
        if "site-packages" in str(py_file):
            continue

        try:
            rel_file = str(py_file.relative_to(repo_root))
        except ValueError:
            # File not under repo_root
            continue

        imports = extract_imports(py_file)
        resolved: List[str] = []

        for imp in imports:
            resolved_path = resolve_import_to_path(imp, py_file, repo_root)
            if resolved_path:
                resolved.append(str(resolved_path))

        graph[rel_file] = resolved

    return graph


def get_import_closure(
    start_file: Path,
    repo_root: Path,
    max_depth: int = 10
) -> Set[Path]:
    """Get all files reachable from start_file via import graph.

    Implements the BFS traversal from PRD Recipe 1.

    Args:
        start_file: Starting file path (relative or absolute)
        repo_root: Repository root directory
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Set of reachable file paths (relative to repo_root)
    """
    # Normalize start_file to relative path
    if start_file.is_absolute():
        try:
            start_rel = str(start_file.relative_to(repo_root))
        except ValueError:
            # start_file not under repo_root
            return set()
    else:
        start_rel = str(start_file)

    # Build import graph
    graph = build_import_graph(repo_root)

    # BFS traversal
    visited: Set[str] = set()
    queue: List[str] = [start_rel]
    visited.add(start_rel)

    depth = 0
    while queue and depth < max_depth:
        level_size = len(queue)
        for _ in range(level_size):
            current = queue.pop(0)

            # Add imported files
            for imported in graph.get(current, []):
                if imported not in visited:
                    visited.add(imported)
                    queue.append(imported)

        depth += 1

    return {Path(f) for f in visited}
