"""Sparse index for BM25 keyword search.

This module implements sparse retrieval using BM25 ranking over tokenized
symbol signatures and filenames. Complements dense vector search for
exact symbol matching and rare term queries.
"""

import re
from collections import defaultdict
from typing import List, Optional, Set

from rank_bm25 import BM25Okapi

from codegrapher.models import Database, Symbol


# BM25 parameters (defaults work well for code)
BM25_K1 = 1.5  # Term frequency saturation
BM25_B = 0.75  # Length normalization
BM25_EPSILON = 0.25  # Floor for IDF values


def tokenize_symbol(symbol: Symbol) -> List[str]:
    """Extract searchable tokens from a symbol.

    Tokenizes symbol signature and filename into multiple patterns:
    1. Dotted module patterns (e.g., "os.path", "HTTPServer.request")
    2. CamelCase symbols split into components (e.g., "FloatOperation" -> ["Float", "Operation"])
    3. ALL_CAPS identifiers (e.g., "HTTP", "API", "DEFAULT_PORT")
    4. snake_case symbols (e.g., "import_module")
    5. Type annotations (e.g., "List[str]", "Optional[Symbol]")
    6. Decorators (e.g., "@property", "@staticmethod")
    7. Underscore-prefixed modules from filename (e.g., "_utils")
    8. Alphanumeric words (3+ chars)
    9. Docstring tokens (first 200 chars)

    Args:
        symbol: Symbol to tokenize

    Returns:
        List of lowercase tokens, filtered to 3+ characters
    """
    tokens = []

    # 1. Signature dotted.module.pattern (case-insensitive)
    # Matches: os.path, HTTPServer.RequestHandler, urllib.parse.urlencode
    tokens.extend(re.findall(r"\b[a-zA-Z_][a-zA-Z_0-9]*\.[a-zA-Z_][a-zA-Z_0-9]*", symbol.signature))
    tokens.extend(re.findall(r"\b[a-zA-Z_][a-zA-Z_0-9]*\.[a-zA-Z_][a-zA-Z_0-9]*\.[a-zA-Z_][a-zA-Z_0-9]*", symbol.signature))

    # 2. CamelCase symbols - split into components and add both full and parts
    # "FloatOperation" -> "FloatOperation", "Float", "Operation"
    # "XMLParser" -> "XMLParser", "XML", "Parser"
    camel_case_full = re.findall(r"\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b", symbol.signature)
    tokens.extend(camel_case_full)
    for cc in camel_case_full:
        # Split CamelCase into components: "FloatOperation" -> ["Float", "Operation"]
        components = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)', cc)
        tokens.extend(components)

    # 3. ALL_CAPS identifiers (constants, acronyms)
    # Matches: HTTP, API, DEFAULT_PORT, MAX_RETRIES
    tokens.extend(re.findall(r"\b[A-Z]{2,}[A-Z0-9_]*\b", symbol.signature))

    # 4. snake_case symbols
    tokens.extend(re.findall(r"\b[a-z][a-z_0-9]*_[a-z][a-z_0-9]*\b", symbol.signature))

    # 5. Type annotations - extract generic types and inner types
    # "List[str]" -> "List", "str"
    # "Optional[Symbol]" -> "Optional", "Symbol"
    # "dict[str, int]" -> "dict", "str", "int"
    type_patterns = re.findall(r"\b([A-Z][a-zA-Z0-9]*)\[[^\]]+\]", symbol.signature)
    for pattern in type_patterns:
        tokens.append(pattern)
        # Extract inner types from brackets
        inner = re.search(r"\[([^\]]+)\]", pattern)
        if inner:
            # Split by comma and extract individual types
            inner_types = re.findall(r'\b([A-Z][a-zA-Z0-9]*|[a-z_][a-z_0-9]*)\b', inner.group(1))
            tokens.extend(inner_types)

    # 6. Decorators
    # @property, @staticmethod, @contextmanager, @lru_cache
    decorators = re.findall(r"@([a-zA-Z_][a-zA-Z0-9_]*)", symbol.signature)
    tokens.extend(decorators)

    # 7. Underscore-prefixed modules (internal/private)
    # Matches: _utils, _client, _internal
    # Skips: __init__, __name__ (double underscore pattern)
    tokens.extend(re.findall(r"\b_[a-z][a-z_0-9]*\b", symbol.file))

    # 8. Alphanumeric words (3+ chars)
    tokens.extend(re.findall(r"\b[a-zA-Z]{3,}\b", symbol.signature))

    # 9. Docstring tokens (if present)
    if symbol.doc:
        tokens.extend(re.findall(r"\b[a-zA-Z]{3,}\b", symbol.doc[:200]))

    # Deduplicate and filter
    seen = set()
    unique_tokens = []
    for t in tokens:
        t_lower = t.lower()
        if len(t) > 2 and t_lower not in seen:
            seen.add(t_lower)
            unique_tokens.append(t)

    return unique_tokens


class SparseIndex:
    """In-memory inverted index for sparse BM25 retrieval.

    Stores tokenized symbols and provides filename-based augmentation.
    Used by BM25Searcher for keyword-based code search.

    Example:
        >>> index = SparseIndex()
        >>> index.add_symbols([symbol1, symbol2])
        >>> assert len(index) == 2
    """

    def __init__(self) -> None:
        """Initialize empty sparse index."""
        self._symbol_ids: List[str] = []
        self._symbol_map: dict[str, Symbol] = {}
        self._file_to_symbols: dict[str, List[str]] = defaultdict(list)

    def add_symbols(self, symbols: List[Symbol]) -> None:
        """Add symbols to the sparse index.

        Args:
            symbols: List of symbols to index
        """
        for symbol in symbols:
            if symbol.id not in self._symbol_map:
                self._symbol_ids.append(symbol.id)
            self._symbol_map[symbol.id] = symbol
            self._file_to_symbols[symbol.file].append(symbol.id)

    def get_symbol(self, symbol_id: str) -> Optional[Symbol]:
        """Retrieve a symbol by ID.

        Args:
            symbol_id: Fully qualified symbol name

        Returns:
            Symbol if found, None otherwise
        """
        return self._symbol_map.get(symbol_id)

    def get_all_symbols(self) -> List[Symbol]:
        """Get all symbols in the index.

        Returns:
            List of all symbols
        """
        return [self._symbol_map[sid] for sid in self._symbol_ids]

    def get_symbols_in_files(self, file_paths: Set[str]) -> List[Symbol]:
        """Get all symbols from the specified files.

        Args:
            file_paths: Set of file paths to filter by

        Returns:
            List of symbols in the specified files
        """
        symbols = []
        for file_path in file_paths:
            symbols.extend(
                self._symbol_map[sid]
                for sid in self._file_to_symbols.get(file_path, [])
            )
        return symbols

    def __len__(self) -> int:
        """Return number of symbols in index."""
        return len(self._symbol_ids)

    def load_from_database(self, db: Database) -> int:
        """Load symbols and their sparse terms from database.

        Populates the index with all symbols from the database, using
        pre-computed sparse terms for faster BM25 indexing.

        Args:
            db: Database instance to load symbols from

        Returns:
            Number of symbols loaded
        """
        # Get all symbols from database
        symbols = db.get_all_symbols()

        # Add symbols to index
        self.add_symbols(symbols)

        return len(symbols)


class BM25Searcher:
    """BM25 keyword search over symbol signatures.

    Uses rank-bm25 library for sparse retrieval. Tokenizes symbols
    using multiple patterns (dotted modules, CamelCase, snake_case)
    and ranks by BM25 score.

    Example:
        >>> searcher = BM25Searcher(index)
        >>> results = searcher.search(["import", "module"])
        >>> assert len(results) > 0
    """

    def __init__(self, index: SparseIndex) -> None:
        """Initialize BM25 searcher with sparse index.

        Args:
            index: SparseIndex containing symbols to search
        """
        self._index = index
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_docs: List[List[str]] = []

    def build_index(self) -> None:
        """Build BM25 index from symbol tokens.

        Tokenizes all symbols and initializes BM25 model.
        Called automatically on first search if not already built.
        """
        symbols = self._index.get_all_symbols()

        # Tokenize all symbols
        self._tokenized_docs = [tokenize_symbol(s) for s in symbols]

        # Initialize BM25
        if self._tokenized_docs:
            self._bm25 = BM25Okapi(
                self._tokenized_docs,
                k1=BM25_K1,
                b=BM25_B,
                epsilon=BM25_EPSILON
            )

    def search(
        self,
        query_tokens: List[str],
        k: int = 20
    ) -> List[tuple[str, float]]:
        """Search for symbols matching query tokens.

        Args:
            query_tokens: List of query tokens (already preprocessed)
            k: Maximum number of results to return

        Returns:
            List of (symbol_id, score) tuples, sorted by score descending
        """
        # Build index if not already built
        if self._bm25 is None:
            self.build_index()

        if not self._bm25 or not query_tokens:
            return []

        # BM25 search returns scores
        scores = self._bm25.get_scores(query_tokens)

        # Get symbol IDs
        symbol_ids = self._index._symbol_ids

        # Create (id, score) pairs and sort by score
        results = list(zip(symbol_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]


def augment_with_filename_matches(
    query_tokens: List[str],
    all_symbols: List[Symbol],
    initial_results: Set[str],
) -> Set[str]:
    """Add symbols from files mentioned in query.

    Filename-only queries (e.g., ["_utils.py"]) won't match symbols
    because filenames aren't tokenized. This function adds all symbols
    from .py files found in the query tokens.

    Args:
        query_tokens: Tokenized query (may contain .py filenames)
        all_symbols: All symbols in the index
        initial_results: Initial set of symbol IDs from BM25

    Returns:
        Set of symbol IDs including filename matches

    Examples:
        >>> query = ["_utils.py", "_client.py"]
        >>> symbols = [symbol1, symbol2]  # from _utils.py
        >>> results = augment_with_filename_matches(query, symbols, set())
        >>> assert len(results) > 0
    """
    result_ids = set(initial_results)

    # Check if any token is a filename (case-insensitive for matching)
    for token in query_tokens:
        if token.endswith('.py'):
            token_lower = token.lower()
            # Find all symbols in this file
            for symbol in all_symbols:
                if token_lower in symbol.file.lower():
                    result_ids.add(symbol.id)

    return result_ids
