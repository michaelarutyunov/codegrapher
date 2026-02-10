"""CLI for CodeGrapher.

This module provides command-line interface commands for CodeGrapher.
Per Phase 11 requirements:
- codegraph init: Setup repo, download model, install git hook
- codegraph build --full: Full index build
- codegraph query: CLI testing
- codegraph update --git-changed: Helper for git hooks
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from codegrapher.indexer import IncrementalIndexer, apply_diff, atomic_update
from codegrapher.models import Database
from codegrapher.parser import extract_symbols
from codegrapher.secrets import scan_file
from codegrapher.server import (
    codegraph_query,
    find_repo_root,
    generate_mcp_config,
    get_index_path,
)
from codegrapher.vector_store import (
    EmbeddingModel,
    FAISSIndexManager,
    generate_symbol_embeddings,
)
from codegrapher.watcher import install_git_hook, FileWatcher
from codegrapher.graph import extract_edges_from_file, compute_pagerank

logger = logging.getLogger(__name__)

# Default constants
DEFAULT_TOKEN_BUDGET = 3500


def find_root() -> Path:
    """Find repository root, exit with error if not found."""
    root = find_repo_root()
    if root is None:
        print(
            "Error: Not in a git repository (no .git directory found)", file=sys.stderr
        )
        print("Run from within a git repository.", file=sys.stderr)
        sys.exit(1)
    return root


def _should_exclude_path(file_path: Path, repo_root: Path) -> bool:
    """Check if a file path should be excluded from indexing.

    Excludes common non-source directories and files:
    - Virtual environments: .venv, venv, .virtualenv, virtualenv, env
    - Cache directories: __pycache__, .pytest_cache, .tox, .mypy_cache, .ruff_cache
    - Build artifacts: build, dist, *.egg-info
    - IDE directories: .vscode, .idea, .DS_Store
    - Hidden files/directories starting with . (except specific allowed ones)

    Args:
        file_path: Path to check
        repo_root: Repository root for relative path computation

    Returns:
        True if the file should be excluded, False otherwise
    """
    # Get relative path from repo root
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        # File is not under repo_root (shouldn't happen with rglob)
        return True

    # Exclude common non-source directories
    exclude_patterns = [
        # Virtual environments
        ".venv",
        "venv",
        ".virtualenv",
        "virtualenv",
        "env",
        ".env",
        "ENV",
        "envs",
        # Cache directories
        "__pycache__",
        ".pytest_cache",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".coverage",
        # Build artifacts
        "build",
        "dist",
        "*.egg-info",
        "*.egg",
        # IDE directories
        ".vscode",
        ".idea",
        ".DS_Store",
        ".eclipse",
        # Node modules (if present)
        "node_modules",
    ]

    # Check if any part of the path matches exclude patterns
    parts = rel_path.parts
    for part in parts:
        for pattern in exclude_patterns:
            if pattern.endswith("/"):
                # Directory pattern (ends with /)
                if part == pattern.rstrip("/"):
                    return True
            elif pattern.startswith("*."):
                # Wildcard pattern (e.g., *.egg-info)
                suffix = pattern[1:]
                if part.endswith(suffix):
                    return True
            elif pattern == part or part.startswith(pattern + "/"):
                return True

    # Exclude hidden files/directories (starting with .)
    # But allow specific ones like .gitignore, .github, etc.
    allowed_hidden = {
        ".github",
        ".gitlab",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
        ".codegraph",
    }
    for part in parts:
        if part.startswith(".") and part not in allowed_hidden:
            # Only exclude if it's a directory or not explicitly allowed
            # Allow files like .gitignore
            if file_path.is_dir():
                return True
            # For files, only exclude common hidden files
            if part not in {".gitignore", ".gitattributes", ".editorconfig"}:
                return True

    return False


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize CodeGrapher in a repository."""
    root = find_root()
    index_dir = get_index_path(root)

    print(f"Repository root: {root}")
    print(f"Index directory: {index_dir}")

    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created index directory: {index_dir}")
    else:
        print(f"Index directory already exists: {index_dir}")

    if not args.no_model:
        print("\nDownloading embedding model (jina-embeddings-v2-base-code)...")
        print("This may take a while on first run (~300MB)...")
        try:
            model = EmbeddingModel()
            model._load_model()
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to download model: {e}", file=sys.stderr)
            print("Model will be downloaded on first use.", file=sys.stderr)

    if not args.no_hook:
        print("\nInstalling git hook...")
        if install_git_hook(root):
            print("Git hook installed successfully!")
            print("The index will be updated automatically after each commit.")
        else:
            print("Warning: Failed to install git hook", file=sys.stderr)

    print("\nInitialization complete!")
    print("Run 'codegraph build --full' to build the index.")


def cmd_build(args: argparse.Namespace) -> None:
    """Build the CodeGrapher index."""
    root = find_root()
    index_dir = get_index_path(root)
    db_path = index_dir / "symbols.db"
    faiss_path = index_dir / "index.faiss"

    if not args.full and not args.force:
        if db_path.exists() or faiss_path.exists():
            print("Error: Index already exists.", file=sys.stderr)
            print(
                "Use --full to rebuild the index, or 'codegraph update' for incremental updates.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Building index for repository: {root}")
    print(f"Index directory: {index_dir}")

    if args.full:
        if db_path.exists():
            db_path.unlink()
            print("Cleared existing database")
        if faiss_path.exists():
            faiss_path.unlink()
            print("Cleared existing FAISS index")

    print("\nInitializing components...")
    # Ensure index directory exists
    index_dir.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    db.initialize()  # Create tables
    faiss_manager = FAISSIndexManager(faiss_path)
    model = EmbeddingModel()

    print("\nFinding Python files...")
    py_files = list(root.rglob("*.py"))

    # Filter out excluded paths (virtual environments, caches, etc.)
    py_files = [f for f in py_files if not _should_exclude_path(f, root)]

    # Filter out manually excluded files from secrets scanning
    excluded_file = index_dir / "excluded_files.txt"
    if excluded_file.exists():
        excluded = set(excluded_file.read_text().strip().split("\n"))
        py_files = [f for f in py_files if str(f.relative_to(root)) not in excluded]

    print(f"Found {len(py_files)} Python files to index")
    if not py_files:
        print("No Python files found to index!")
        return

    all_symbols = []
    all_edges = []
    secret_count = 0
    syntax_error_count = 0

    for i, file_path in enumerate(py_files, 1):
        rel_path = file_path.relative_to(root)
        print(f"[{i}/{len(py_files)}] Processing {rel_path}...", end=" ")

        try:
            if scan_file(file_path, root):
                print("SKIPPED (secret detected)")
                secret_count += 1
                continue
        except Exception as e:
            print(f"WARNING: Secret scan failed: {e}")

        try:
            symbols = extract_symbols(file_path, root)
            edges = extract_edges_from_file(file_path, root, symbols)
            all_edges.extend(edges)
            all_symbols.extend(symbols)
            print(f"OK ({len(symbols)} symbols)")
        except SyntaxError:
            print("SKIPPED (syntax error)")
            syntax_error_count += 1
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_symbols:
        print("\nNo symbols extracted!")
        return

    print(f"\nGenerating embeddings for {len(all_symbols)} symbols...")
    all_symbols = generate_symbol_embeddings(all_symbols, model)
    print("Embeddings generated!")

    print("\nStoring symbols in database...")
    for symbol in all_symbols:
        db.insert_symbol(symbol)
    print(f"Stored {len(all_symbols)} symbols")

    print(f"\nStoring {len(all_edges)} edges in database...")
    for edge in all_edges:
        db.insert_edge(edge)
    print("Edges stored!")

    print("\nBuilding FAISS index...")
    faiss_manager.add_symbols(all_symbols)
    faiss_manager.save()
    print("FAISS index built!")

    print("\nComputing PageRank scores...")
    try:
        pagerank = compute_pagerank(db)
        db.set_meta("pagerank_computed", "true")
        db.set_meta("pagerank_count", str(len(pagerank)))
        print(f"PageRank computed for {len(pagerank)} symbols")
    except Exception as e:
        print(f"Warning: Failed to compute PageRank: {e}")

    db.set_meta("last_indexed", datetime.now().isoformat())
    db.set_meta("total_symbols", str(len(all_symbols)))
    db.set_meta("total_files", str(len(py_files)))

    print("\n" + "=" * 50)
    print("Index build complete!")
    print("=" * 50)
    print(f"Total files indexed: {len(py_files)}")
    print(f"Total symbols: {len(all_symbols)}")
    print(f"Total edges: {len(all_edges)}")
    if secret_count > 0:
        print(f"Files skipped (secrets): {secret_count}")
    if syntax_error_count > 0:
        print(f"Files skipped (syntax errors): {syntax_error_count}")
    print(f"\nIndex location: {index_dir}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query the CodeGrapher index."""
    result = codegraph_query.fn(
        query=args.query,
        cursor_file=args.cursor_file,
        _max_depth=args.max_depth,
        token_budget=args.token_budget,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if result["status"] == "error":
        print(f"Error: {result['error_type']}", file=sys.stderr)
        print(f"{result['message']}", file=sys.stderr)
        if "fallback_suggestion" in result:
            print(f"\nSuggestion: {result['fallback_suggestion']}", file=sys.stderr)
        sys.exit(1)

    print(f"Query: {args.query}")
    print(f"Tokens used: {result['tokens_used']}/{args.token_budget}")
    print(f"Symbols returned: {result['total_symbols']}")
    if result.get("truncated"):
        print("(results truncated by token budget)")

    # Group by file
    files_by_path = {}
    for entry in result["files"]:
        path = entry["path"]
        if path not in files_by_path:
            files_by_path[path] = []
        files_by_path[path].append(entry)

    print("\nResults:")
    print("-" * 50)
    for file_path, symbols in files_by_path.items():
        print(f"\n{file_path}:")
        for sym in symbols:
            print(
                f"  Lines {sym['line_range'][0]}-{sym['line_range'][1]}: {sym['symbol']}"
            )
            print(f"    {sym['excerpt'][:100]}...")


def cmd_update(args: argparse.Namespace) -> None:
    """Update the index for changed files."""
    root = find_root()
    index_dir = get_index_path(root)
    db_path = index_dir / "symbols.db"
    faiss_path = index_dir / "index.faiss"

    if not db_path.exists() or not faiss_path.exists():
        print(
            "Error: Index not found. Run 'codegraph build --full' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    files_to_update = []

    if args.git_changed:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=M", "*.py"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                changed = result.stdout.strip().split("\n")
                files_to_update = [root / f for f in changed if f and f.endswith(".py")]
        except Exception as e:
            print(f"Warning: Failed to get git changes: {e}", file=sys.stderr)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        files_to_update = [file_path]
    else:
        print("Error: Specify a file or use --git-changed", file=sys.stderr)
        sys.exit(1)

    if not files_to_update:
        print("No files to update.")
        return

    print(f"Updating {len(files_to_update)} file(s)...")

    db = Database(db_path)
    faiss_manager = FAISSIndexManager(faiss_path)
    model = EmbeddingModel()
    indexer = IncrementalIndexer()

    updated_count = 0
    skipped_count = 0

    for file_path in files_to_update:
        rel_path = file_path.relative_to(root)
        print(f"Updating {rel_path}...", end=" ")

        if not file_path.exists():
            print("DELETED")
            all_symbols = db.get_all_symbols()
            deleted = [s for s in all_symbols if s.file == str(rel_path)]
            if deleted:
                faiss_manager.remove_symbols([s.id for s in deleted])
                for sym in deleted:
                    db.delete_symbol(sym.id)
                print(f"  Removed {len(deleted)} symbols")
            updated_count += 1
            continue

        try:
            if scan_file(file_path, root):
                print("SKIPPED (secret detected)")
                skipped_count += 1
                continue
        except Exception as e:
            print(f"WARNING: Secret scan failed: {e}")

        try:
            diff = indexer.update_file(file_path, repo_root=root)
        except (SyntaxError, FileNotFoundError) as e:
            print(f"SKIPPED ({e})")
            skipped_count += 1
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            skipped_count += 1
            continue

        if not diff.has_changes:
            print("NO CHANGE")
            continue

        try:
            # Apply diff with atomic transaction
            with atomic_update(db_path, faiss_manager):
                apply_diff(diff, db, faiss_manager)
                # Generate embeddings for new/modified symbols
                for symbol in diff.added + diff.modified:
                    if symbol.embedding.sum() == 0:  # Placeholder embedding
                        symbol.embedding = model.embed_text(
                            symbol.signature + " " + (symbol.doc or "")
                        )
                        db.update_symbol(symbol)
            print(
                f"OK ({len(diff.added)} added, {len(diff.modified)} modified, {len(diff.deleted)} deleted)"
            )
            updated_count += 1
        except Exception as e:
            print(f"ERROR: {e}")
            skipped_count += 1

    db.set_meta("last_indexed", datetime.now().isoformat())

    print("\n" + "=" * 50)
    print("Update complete!")
    print("=" * 50)
    print(f"Files updated: {updated_count}")
    print(f"Files skipped: {skipped_count}")


def cmd_callers(args: argparse.Namespace) -> None:
    """Find symbols that call the given symbol.

    Queries the edges table for call edges where callee_id matches
    the given pattern, then enriches results with symbol information.

    Examples:
        codegraph callers extract_edges_from_file
        codegraph callers Database.get_all_symbols
        codegraph callers --json TokenValidator
    """
    root = find_root()
    index_dir = get_index_path(root)
    db_path = index_dir / "symbols.db"

    if not db_path.exists():
        print("Error: Index not found.", file=sys.stderr)
        print("Run 'codegraph build --full' first.", file=sys.stderr)
        sys.exit(1)

    db = Database(db_path)

    # Query for callers
    # First try exact match
    symbol_pattern = args.symbol
    all_callers = db.get_callers_for_symbol(symbol_pattern)

    if not all_callers:
        print(f"No callers found for '{args.symbol}'")
        print()
        print("Tips:")
        print("  - Use the full symbol ID (e.g., 'module.Class.method')")
        print("  - Symbols are case-sensitive")
        print("  - Use 'codegraph query' to search for symbols by name")
        return

    # Sort by PageRank score if available
    pagerank = {}
    try:
        # Try to load cached PageRank scores
        from codegrapher.graph import compute_pagerank

        pagerank = compute_pagerank(db)
    except Exception:
        pass  # PageRank not available, all scores will be 0

    # Enrich with symbol information
    enriched_callers = []
    for caller_id in all_callers:
        symbol = db.get_symbol(caller_id)
        if symbol:
            enriched_callers.append(
                {
                    "caller_id": caller_id,
                    "file": symbol.file,
                    "start_line": symbol.start_line,
                    "signature": symbol.signature,
                    "pagerank": pagerank.get(caller_id, 0.0),
                }
            )

    # Sort by PageRank score (descending)
    enriched_callers.sort(key=lambda x: x["pagerank"], reverse=True)

    # Apply limit
    limited_callers = enriched_callers[: args.limit]

    # Output
    if args.json:
        result = {
            "symbol_id": args.symbol,
            "callers": limited_callers,
            "total_callers": len(enriched_callers),
            "truncated": len(enriched_callers) > args.limit,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Callers of {args.symbol} (found {len(enriched_callers)} total)")
        if len(enriched_callers) > args.limit:
            print(f"  (showing top {args.limit}, use --limit N to see more)")
        print()

        for i, caller in enumerate(limited_callers, 1):
            pagerank_note = ""
            if caller["pagerank"] > 0:
                if caller["pagerank"] >= 0.1:
                    pagerank_note = " [core]"
                elif caller["pagerank"] >= 0.05:
                    pagerank_note = " [important]"
                elif caller["pagerank"] >= 0.01:
                    pagerank_note = " [helper]"

            print(f"{i}. {caller['caller_id']} {pagerank_note}")
            print(f"   Location: {caller['file']}:{caller['start_line']}")
            print(f"   Signature: {caller['signature'][:80]}")
            print()


def cmd_mcp_config(args: argparse.Namespace) -> None:
    """Generate MCP server configuration."""
    config = generate_mcp_config()
    if args.output:
        Path(args.output).write_text(config)
        print(f"Configuration written to: {args.output}")
    else:
        print(config)


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch for file changes and auto-update the index."""
    root = find_root()
    index_path = get_index_path(root)

    # Check if index exists
    if not index_path.exists():
        print(f"Error: Index not found at {index_path}", file=sys.stderr)
        print(
            "Run 'codegraph init' followed by 'codegraph build --full' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize components
    print("Loading index...")
    db = Database(index_path / "symbols.db")
    faiss_manager = FAISSIndexManager(index_path / "index.faiss")
    indexer = IncrementalIndexer(cache_size=50)

    print(f"Watching for changes in: {root}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    # Create and start watcher
    watcher = FileWatcher(
        repo_root=root,
        db=db,
        faiss_manager=faiss_manager,
        indexer=indexer,
        bulk_callback=None,  # No auto-rebuild for watch mode
    )

    try:
        watcher.start()
        # Keep the main thread alive
        import signal

        signal.pause()  # Wait for signal (Ctrl+C)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
    finally:
        watcher.stop()
        stats = watcher.get_stats()
        print("=" * 50)
        print("Watcher stopped")
        print(f"  Changes processed: {stats['changes_processed']}")
        print(f"  Bulk rebuilds triggered: {stats['bulk_rebuilds_triggered']}")


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="CodeGrapher: Token-efficient code search for Python repositories",
        prog="codegraph",
    )
    parser.add_argument("--version", action="version", version="CodeGrapher 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init subcommand
    init_parser = subparsers.add_parser(
        "init", help="Initialize CodeGrapher in a repository"
    )
    init_parser.add_argument(
        "--no-model", action="store_true", help="Skip model download"
    )
    init_parser.add_argument(
        "--no-hook", action="store_true", help="Skip git hook installation"
    )
    init_parser.set_defaults(func=cmd_init)

    # build subcommand
    build_parser = subparsers.add_parser("build", help="Build the CodeGrapher index")
    build_parser.add_argument(
        "--full", action="store_true", help="Perform a full rebuild"
    )
    build_parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if index exists"
    )
    build_parser.set_defaults(func=cmd_build)

    # query subcommand
    query_parser = subparsers.add_parser("query", help="Query the CodeGrapher index")
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument(
        "--cursor-file", help="File path for import-closure pruning"
    )
    query_parser.add_argument(
        "--max-depth", type=int, default=1, help="Graph hop depth (default: 1)"
    )
    query_parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Maximum tokens in response",
    )
    query_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    query_parser.set_defaults(func=cmd_query)

    # update subcommand
    update_parser = subparsers.add_parser(
        "update", help="Update the index for changed files"
    )
    update_parser.add_argument("file", nargs="?", help="Single file to update")
    update_parser.add_argument(
        "--git-changed", action="store_true", help="Update all git-changed Python files"
    )
    update_parser.set_defaults(func=cmd_update)

    # mcp-config subcommand
    mcp_config_parser = subparsers.add_parser(
        "mcp-config", help="Generate MCP server configuration"
    )
    mcp_config_parser.add_argument("--output", "-o", help="Write config to file")
    mcp_config_parser.set_defaults(func=cmd_mcp_config)

    # watch subcommand
    watch_parser = subparsers.add_parser(
        "watch", help="Watch for file changes and auto-update the index"
    )
    watch_parser.set_defaults(func=cmd_watch)

    # callers subcommand
    callers_parser = subparsers.add_parser(
        "callers", help="Find symbols that call the given symbol"
    )
    callers_parser.add_argument(
        "symbol", help="Symbol to find callers for (e.g., extract_edges_from_file)"
    )
    callers_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum number of callers to return (default: 20)",
    )
    callers_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    callers_parser.set_defaults(func=cmd_callers)

    return parser


# Standalone entry points for individual commands (for pyproject.toml)
def init_command() -> None:
    """Entry point for codegraph-init."""
    parser = argparse.ArgumentParser(
        prog="codegraph-init", description="Initialize CodeGrapher in a repository"
    )
    parser.add_argument("--no-model", action="store_true", help="Skip model download")
    parser.add_argument(
        "--no-hook", action="store_true", help="Skip git hook installation"
    )
    args = parser.parse_args()
    cmd_init(args)


def build_command() -> None:
    """Entry point for codegraph-build."""
    parser = argparse.ArgumentParser(
        prog="codegraph-build", description="Build the CodeGrapher index"
    )
    parser.add_argument("--full", action="store_true", help="Perform a full rebuild")
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if index exists"
    )
    args = parser.parse_args()
    cmd_build(args)


def query_command() -> None:
    """Entry point for codegraph-query."""
    parser = argparse.ArgumentParser(
        prog="codegraph-query", description="Query the CodeGrapher index"
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--cursor-file", help="File path for import-closure pruning")
    parser.add_argument(
        "--max-depth", type=int, default=1, help="Graph hop depth (default: 1)"
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Maximum tokens in response",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    cmd_query(args)


def update_command() -> None:
    """Entry point for codegraph-update."""
    parser = argparse.ArgumentParser(
        prog="codegraph-update", description="Update the CodeGrapher index"
    )
    parser.add_argument("file", nargs="?", help="Single file to update")
    parser.add_argument(
        "--git-changed", action="store_true", help="Update all git-changed Python files"
    )
    args = parser.parse_args()
    cmd_update(args)


def mcp_config_command() -> None:
    """Entry point for codegraph-mcp-config."""
    parser = argparse.ArgumentParser(
        prog="codegraph-mcp-config", description="Generate MCP server configuration"
    )
    parser.add_argument("--output", "-o", help="Write config to file")
    args = parser.parse_args()
    cmd_mcp_config(args)


def watch_command() -> None:
    """Entry point for codegraph-watch."""
    parser = argparse.ArgumentParser(
        prog="codegraph-watch",
        description="Watch for file changes and auto-update the CodeGrapher index",
    )
    args = parser.parse_args()
    cmd_watch(args)


def callers_command() -> None:
    """Entry point for codegraph-callers."""
    parser = argparse.ArgumentParser(
        prog="codegraph-callers", description="Find symbols that call the given symbol"
    )
    parser.add_argument(
        "symbol", help="Symbol to find callers for (e.g., extract_edges_from_file)"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum number of callers to return (default: 20)",
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    cmd_callers(args)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
