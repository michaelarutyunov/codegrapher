"""AST parser for extracting symbols and imports from Python source code.

This module uses Python's built-in `ast` module to parse source code and
extract:
1. Symbols: functions, classes, and module-level assignments
2. Imports: both absolute and relative import statements

Per PRD Section 5, we extract top-level definitions only (not nested
functions/classes within functions) for incremental indexing performance.
"""

import ast
from pathlib import Path
from typing import List, Optional, Union, cast

import numpy as np
from codegrapher.models import Symbol, EMBEDDING_DIM


# Placeholder embedding for symbols that haven't been embedded yet
# Phase 5 will replace this with actual embeddings
_PLACEHOLDER_EMBEDDING = np.zeros(EMBEDDING_DIM, dtype=np.float32)


def extract_symbols(file_path: Path, repo_root: Optional[Path] = None) -> List[Symbol]:
    """Extract all symbols from a Python file.

    Parses the file and extracts top-level functions, classes, and
    module-level assignments. Each symbol gets a placeholder embedding
    that will be replaced during the embedding phase.

    Args:
        file_path: Path to the Python file to parse
        repo_root: Repository root for computing relative file paths.
                   If None, file_path is used as-is.

    Returns:
        List of Symbol objects found in the file

    Raises:
        SyntaxError: If the file contains invalid Python syntax
        FileNotFoundError: If the file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read source code
    source = file_path.read_text(encoding="utf-8")

    # Parse into AST
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in {file_path}:{e.lineno}: {e.msg}")

    # Compute relative path if repo_root provided
    if repo_root:
        try:
            rel_path = str(file_path.relative_to(repo_root))
        except ValueError:
            # file_path is not relative to repo_root
            rel_path = str(file_path)
    else:
        rel_path = str(file_path)

    # Extract module name from file path for symbol IDs
    module_name = _path_to_module_name(rel_path)

    symbols: List[Symbol] = []

    # Walk through top-level nodes only
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            symbol = _extract_function(node, module_name, rel_path)
            symbols.append(symbol)
        elif isinstance(node, ast.ClassDef):
            # Extract the class itself
            class_symbol = _extract_class(node, module_name, rel_path)
            symbols.append(class_symbol)

            # Extract methods from the class
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_symbol = _extract_function(
                        item, module_name, rel_path, class_name=node.name
                    )
                    symbols.append(method_symbol)
        elif isinstance(node, ast.Assign):
            # Extract module-level assignments
            for assign_symbol in _extract_assignments(node, module_name, rel_path):
                symbols.append(assign_symbol)

    return symbols


def extract_imports(file_path: Path) -> List[str]:
    """Extract all import statements from a Python file as strings.

    Returns raw import strings like "os.path", "sys", "..utils",
    "from .config import settings". These will be resolved to file paths
    by the import resolver in Phase 6.

    Args:
        file_path: Path to the Python file to parse

    Returns:
        List of import strings found in the file

    Raises:
        SyntaxError: If the file contains invalid Python syntax
        FileNotFoundError: If the file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source = file_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in {file_path}:{e.lineno}: {e.msg}")

    imports: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level  # 0=absolute, 1=., 2=..
            prefix = "." * level
            imports.append(f"{prefix}{module}")

    return imports


def _path_to_module_name(file_path: str) -> str:
    """Convert a file path to a Python module name.

    Examples:
        "src/mypackage/utils.py" -> "src.mypackage.utils"
        "mypackage/__init__.py" -> "mypackage"
        "mypackage/sub/module.py" -> "mypackage.sub.module"

    Args:
        file_path: File path (relative or absolute)

    Returns:
        Dotted module name
    """
    # Remove .py extension and replace / with .
    module = file_path.replace(".py", "").replace("/", ".")

    # Remove __init__ suffix (it represents the package itself)
    module = module.replace(".__init__", "")

    return module


def _format_function_signature(node: ast.FunctionDef) -> str:
    """Format a function signature as a string.

    Args:
        node: AST FunctionDef node

    Returns:
        Formatted signature like "def my_func(a: int, b: str) -> None:"
    """
    # Get function name
    name = node.name

    # Format arguments
    args = _format_arguments(node.args)

    # Format return annotation
    returns = ""
    if node.returns:
        returns = f" -> {_unparse(node.returns)}"

    return f"def {name}({args}){returns}:"


def _format_class_signature(node: ast.ClassDef) -> str:
    """Format a class signature as a string.

    Args:
        node: AST ClassDef node

    Returns:
        Formatted signature like "class MyClass(SuperClass, Generic[T]):"
    """
    name = node.name

    # Format base classes
    bases = [_unparse(base) for base in node.bases]
    if bases:
        bases_str = ", ".join(bases)
        return f"class {name}({bases_str}):"
    return f"class {name}:"


def _format_arguments(args_node: ast.arguments) -> str:
    """Format function arguments as a string.

    Args:
        args_node: AST arguments node

    Returns:
        Formatted arguments like "a: int, b: str, *args, **kwargs"
    """
    parts: List[str] = []

    # Positional args
    for arg in args_node.posonlyargs:
        parts.append(_format_arg(arg))

    for arg in args_node.args:
        parts.append(_format_arg(arg))

    # Varargs
    if args_node.vararg:
        parts.append(f"*{_format_arg(args_node.vararg)}")

    # Keyword-only args
    for arg in args_node.kwonlyargs:
        parts.append(_format_arg(arg))

    # Kwargs
    if args_node.kwarg:
        parts.append(f"**{_format_arg(args_node.kwarg)}")

    return ", ".join(parts)


def _format_arg(arg: ast.arg) -> str:
    """Format a single function argument.

    Args:
        arg: AST arg node

    Returns:
        Formatted argument like "x: int" or "x"
    """
    result = arg.arg
    if arg.annotation:
        result += f": {_unparse(arg.annotation)}"
    return result


def _unparse(node: ast.AST) -> str:
    """Convert an AST node back to source code string.

    Uses ast.unparse which is available in Python 3.9+.

    Args:
        node: AST node to unparse

    Returns:
        Source code string representation
    """
    return ast.unparse(node)


def _extract_function(
    node: ast.FunctionDef,
    module_name: str,
    file_path: str,
    class_name: Optional[str] = None,
) -> Symbol:
    """Extract a function or method symbol from an AST node.

    Args:
        node: AST FunctionDef node
        module_name: Dotted module name
        file_path: Relative file path
        class_name: Parent class name if this is a method

    Returns:
        Symbol object for the function
    """
    # Build symbol ID
    if class_name:
        symbol_id = f"{module_name}.{class_name}.{node.name}"
    else:
        symbol_id = f"{module_name}.{node.name}"

    # Get signature
    signature = _format_function_signature(node)

    # Get first line of docstring
    doc = _extract_first_docstring_line(node)

    # Detect mutated variables (look for assignments with walrus operator
    # or function calls that suggest mutation - simplified for v1)
    mutates = ""  # Phase 3: leave empty, can enhance later

    # Get line range
    start_line = node.lineno
    end_line = _get_end_line(node)

    return Symbol(
        id=symbol_id,
        file=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        doc=doc,
        mutates=mutates,
        embedding=_PLACEHOLDER_EMBEDDING.copy(),
    )


def _extract_class(node: ast.ClassDef, module_name: str, file_path: str) -> Symbol:
    """Extract a class symbol from an AST node.

    Args:
        node: AST ClassDef node
        module_name: Dotted module name
        file_path: Relative file path

    Returns:
        Symbol object for the class
    """
    symbol_id = f"{module_name}.{node.name}"
    signature = _format_class_signature(node)
    doc = _extract_first_docstring_line(node)

    start_line = node.lineno
    end_line = _get_end_line(node)

    return Symbol(
        id=symbol_id,
        file=file_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        doc=doc,
        mutates="",
        embedding=_PLACEHOLDER_EMBEDDING.copy(),
    )


def _extract_assignments(
    node: ast.Assign, module_name: str, file_path: str
) -> List[Symbol]:
    """Extract module-level assignment symbols from an AST node.

    Only captures assignments with simple names (not attributes, subscripts).
    Skips assignments that are part of type aliasing or imports.

    Args:
        node: AST Assign node
        module_name: Dotted module name
        file_path: Relative file path

    Returns:
        List of Symbol objects for the assignments
    """
    symbols: List[Symbol] = []

    for target in node.targets:
        # Only capture simple name assignments (x = ..., not x.y = ...)
        if isinstance(target, ast.Name):
            symbol_id = f"{module_name}.{target.id}"
            signature = f"{target.id} = <...>"
            doc = None  # Assignments don't have docstrings

            # Detect if this might be a constant (all caps name)
            mutates = ""
            if target.id.isupper():
                doc = f"Constant: {target.id}"

            start_line = node.lineno
            end_line = node.end_lineno or node.lineno

            symbols.append(
                Symbol(
                    id=symbol_id,
                    file=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature,
                    doc=doc,
                    mutates=mutates,
                    embedding=_PLACEHOLDER_EMBEDDING.copy(),
                )
            )

    return symbols


def _extract_first_docstring_line(
    node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module]
) -> Optional[str]:
    """Extract the first sentence from a node's docstring.

    Args:
        node: AST node (FunctionDef, ClassDef, AsyncFunctionDef, Module)

    Returns:
        First line of docstring, or None if no docstring
    """
    docstring = ast.get_docstring(node)
    if not docstring:
        return None

    # Get first line (or first sentence if split across lines)
    first_line = docstring.split("\n")[0].strip()

    # If first line ends with period, it's a complete sentence
    if first_line.endswith("."):
        return first_line

    # Otherwise, take first 80 chars
    if len(first_line) > 80:
        return first_line[:77] + "..."

    return first_line


def _get_end_line(node: ast.AST) -> int:
    """Get the end line number of an AST node.

    Handles both Python 3.8+ (end_lineno) and older versions.

    Args:
        node: AST node

    Returns:
        End line number (inclusive)
    """
    # Cast to access end_lineno attribute (exists on nodes with location info)
    located_node = cast(
        Union[
            ast.FunctionDef,
            ast.ClassDef,
            ast.AsyncFunctionDef,
            ast.Assign,
            ast.Expr,
        ],
        node,
    )
    if (
        hasattr(located_node, "end_lineno")
        and located_node.end_lineno is not None
    ):
        return located_node.end_lineno
    return located_node.lineno
