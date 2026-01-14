"""Tests for sparse index module.

Tests BM25 tokenization, compound word splitting, and search functionality.
"""

import numpy as np
import pytest

from codegrapher.models import Symbol
from codegrapher.sparse_index import (
    BM25Searcher,
    SparseIndex,
    augment_with_filename_matches,
    tokenize_compound_word,
    tokenize_symbol,
)


class TestCompoundWordSplitting:
    """Test compound word tokenization."""

    def test_underscore_separated(self):
        """Underscore-separated identifiers split correctly."""
        result = tokenize_compound_word("compile_templates")
        assert "compile_templates" in result  # Original preserved
        assert "compile" in result
        assert "templates" in result

        result = tokenize_compound_word("stream_with_context")
        assert "stream_with_context" in result
        assert "stream" in result
        assert "with" in result
        assert "context" in result

        result = tokenize_compound_word("import_module_using_spec")
        assert "import_module_using_spec" in result
        assert "import" in result
        assert "module" in result
        assert "using" in result
        assert "spec" in result

    def test_camelcase_splitting(self):
        """CamelCase identifiers split correctly."""
        result = tokenize_compound_word("FloatOperation")
        assert "FloatOperation" in result
        assert "Float" in result
        assert "Operation" in result

        result = tokenize_compound_word("TestClient")
        assert "TestClient" in result
        assert "Test" in result
        assert "Client" in result

        result = tokenize_compound_word("UnicodeDecodeError")
        assert "UnicodeDecodeError" in result
        assert "Unicode" in result
        assert "Decode" in result
        assert "Error" in result

    def test_mixed_identifiers(self):
        """Mixed underscore + CamelCase splits correctly."""
        result = tokenize_compound_word("root_render_func")
        assert "root_render_func" in result
        assert "root" in result
        assert "render" in result
        assert "func" in result

        result = tokenize_compound_word("generate_async")
        assert "generate_async" in result
        assert "generate" in result
        assert "async" in result

        result = tokenize_compound_word("http_exception")
        assert "http_exception" in result
        assert "http" in result
        assert "exception" in result

    def test_simple_identifiers_unchanged(self):
        """Simple identifiers without compound patterns are unchanged."""
        result = tokenize_compound_word("simple")
        assert result == ["simple"]

        result = tokenize_compound_word("compile")
        assert "compile" in result
        assert len(result) == 1  # No splitting

        result = tokenize_compound_word("templates")
        assert "templates" in result
        assert len(result) == 1

    def test_edge_cases(self):
        """Edge cases handled correctly."""
        # Single character
        result = tokenize_compound_word("a")
        assert result == ["a"]

        # Two characters (below threshold)
        result = tokenize_compound_word("ab")
        assert result == ["ab"]

        # Underscore only
        result = tokenize_compound_word("_")
        assert result == ["_"]

        # All caps abbreviation
        result = tokenize_compound_word("HTTP")
        assert "HTTP" in result

        # Numbers in identifiers
        result = tokenize_compound_word("test_123")
        assert "test_123" in result
        assert "test" in result
        # Numbers filtered by 3-char threshold (123 stays, single digits go)

    def test_deduplication(self):
        """Duplicate tokens are removed."""
        result = tokenize_compound_word("test_test")
        assert result.count("test") == 1  # Only one "test"
        assert "test_test" in result  # Original preserved

    def test_case_preservation(self):
        """Original case preserved in result."""
        result = tokenize_compound_word("Test")
        assert "Test" in result
        # Case-insensitive deduplication means "test" won't be added if "Test" exists

    def test_hyphen_separated(self):
        """Hyphen-separated identifiers split correctly."""
        result = tokenize_compound_word("chunk-boundary")
        assert "chunk-boundary" in result
        assert "chunk" in result
        assert "boundary" in result

        result = tokenize_compound_word("carriage-return")
        assert "carriage-return" in result
        assert "carriage" in result
        assert "return" in result


class TestTokenizeSymbol:
    """Test symbol tokenization with compound word splitting."""

    def test_tokenize_symbol_with_compound_words(self):
        """Symbol tokenization includes compound word splitting."""
        symbol = Symbol(
            id="jinja.Compiler.compile_templates",
            file="src/jinja2/compiler.py",
            signature="def compile_templates(self, context: Context) -> str:",
            start_line=100,
            end_line=150,
            doc="Compile templates with context.",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32),
        )

        tokens = tokenize_symbol(symbol)

        # Original identifier should be present
        assert "compile_templates" in tokens

        # Component tokens from compound splitting
        assert "compile" in tokens
        assert "templates" in tokens

        # Other expected tokens
        assert "context" in tokens
        assert "def" in tokens

    def test_tokenize_symbol_with_camelcase(self):
        """Symbol tokenization splits CamelCase identifiers."""
        symbol = Symbol(
            id="decimal.FloatOperation",
            file="decimal.py",
            signature="class FloatOperation(Exception):",
            start_line=1,
            end_line=10,
            doc="Float operation exception.",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32),
        )

        tokens = tokenize_symbol(symbol)

        # Original and components
        assert "FloatOperation" in tokens
        assert "Float" in tokens
        assert "Operation" in tokens

    def test_tokenize_symbol_with_mixed_patterns(self):
        """Symbol tokenization handles mixed patterns."""
        symbol = Symbol(
            id="flask.helpers.stream_with_context",
            file="src/flask/helpers.py",
            signature="def stream_with_context(generator, context):",
            start_line=50,
            end_line=100,
            doc="Stream generator with context.",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32),
        )

        tokens = tokenize_symbol(symbol)

        # Compound word splitting
        assert "stream_with_context" in tokens
        assert "stream" in tokens
        assert "with" in tokens
        assert "context" in tokens


class TestSparseIndex:
    """Test SparseIndex class."""

    def test_init(self):
        """Initialize empty sparse index."""
        index = SparseIndex()
        assert len(index) == 0

    def test_add_symbols(self):
        """Add symbols to sparse index."""
        index = SparseIndex()

        symbol1 = Symbol(
            id="test.function1",
            file="test.py",
            signature="def function1():",
            start_line=1,
            end_line=5,
            doc="",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32),
        )

        symbol2 = Symbol(
            id="test.function2",
            file="test.py",
            signature="def function2():",
            start_line=10,
            end_line=15,
            doc="",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32),
        )

        index.add_symbols([symbol1, symbol2])
        assert len(index) == 2

    def test_get_symbol(self):
        """Retrieve symbol by ID."""
        index = SparseIndex()

        symbol = Symbol(
            id="test.function",
            file="test.py",
            signature="def function():",
            start_line=1,
            end_line=5,
            doc="",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32),
        )

        index.add_symbols([symbol])
        retrieved = index.get_symbol("test.function")
        assert retrieved is not None
        assert retrieved.id == "test.function"

    def test_get_all_symbols(self):
        """Get all symbols from index."""
        index = SparseIndex()

        symbols = [
            Symbol(
                id=f"test.function{i}",
                file="test.py",
                signature=f"def function{i}():",
                start_line=i + 1,  # start_line must be >= 1
                end_line=i + 6,
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            )
            for i in range(3)
        ]

        index.add_symbols(symbols)
        all_symbols = index.get_all_symbols()
        assert len(all_symbols) == 3


class TestBM25Searcher:
    """Test BM25Searcher class."""

    def test_init(self):
        """Initialize BM25 searcher with index."""
        index = SparseIndex()
        searcher = BM25Searcher(index)
        assert searcher is not None

    def test_search_empty_index(self):
        """Search on empty index returns empty results."""
        index = SparseIndex()
        searcher = BM25Searcher(index)
        results = searcher.search(["test", "query"])
        assert results == []

    def test_search_with_symbols(self):
        """Search with symbols returns results."""
        index = SparseIndex()

        symbols = [
            Symbol(
                id="test.compile_templates",
                file="compiler.py",
                signature="def compile_templates(context):",
                start_line=1,
                end_line=10,
                doc="Compile templates with context.",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            ),
            Symbol(
                id="test.render_function",
                file="render.py",
                signature="def render_function(data):",
                start_line=20,
                end_line=30,
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            ),
        ]

        index.add_symbols(symbols)
        searcher = BM25Searcher(index)

        # Search for "compile" - should match compile_templates
        results = searcher.search(["compile"], k=5)
        assert len(results) > 0

        # After compound splitting, "compile" should match "compile_templates"
        result_ids = [sid for sid, _ in results]
        assert "test.compile_templates" in result_ids


class TestFilenameMatching:
    """Test filename matching augmentation."""

    def test_augment_with_filename_matches(self):
        """Filename matching adds symbols from .py files."""
        symbols = [
            Symbol(
                id="test.function1",
                file="compiler.py",
                signature="def function1():",
                start_line=1,
                end_line=5,
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            ),
            Symbol(
                id="test.function2",
                file="test.py",
                signature="def function2():",
                start_line=10,
                end_line=15,
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            ),
        ]

        query_tokens = ["compiler.py", "compile"]
        initial_results = set()

        result_ids = augment_with_filename_matches(query_tokens, symbols, initial_results)

        # Should include symbol from compiler.py
        assert "test.function1" in result_ids

    def test_filename_matching_case_insensitive(self):
        """Filename matching is case-insensitive."""
        symbols = [
            Symbol(
                id="test.function",
                file="TestClient.py",
                signature="def function():",
                start_line=1,
                end_line=5,
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            ),
        ]

        query_tokens = ["testclient.py"]
        initial_results = set()

        result_ids = augment_with_filename_matches(query_tokens, symbols, initial_results)

        # Should match despite case difference
        assert "test.function" in result_ids
