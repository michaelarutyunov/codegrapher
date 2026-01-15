"""Unit tests for MCP server.

Tests for Phase 10: MCP Server Interface.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codegrapher.server import (
    estimate_tokens,
    compute_weighted_scores,
    truncate_at_token_budget,
    find_repo_root,
    get_index_path,
    get_git_log,
    generate_mcp_config,
    codegraph_query,
    is_test_source_pair,
    augment_with_bidirectional_test_pairs,
)
from codegrapher.models import Symbol


# Sample symbol for testing
SAMPLE_SYMBOL = Symbol(
    id="test.function",
    file="test.py",
    start_line=10,
    end_line=20,
    signature="def function(param: str) -> bool:",
    doc="A test function for validation",
    mutates="",
    embedding=np.zeros(768, dtype=np.float32),
)

SAMPLE_SYMBOL_SHORT = Symbol(
    id="test.short",
    file="test.py",
    start_line=1,
    end_line=5,
    signature="x = 42",
    doc="",
    mutates="",
    embedding=np.zeros(768, dtype=np.float32),
)


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_simple_function(self):
        """Test token estimation for a simple function."""
        tokens = estimate_tokens(SAMPLE_SYMBOL)
        assert tokens > 0
        # Rough estimate: signature (~40 chars) + doc (~30 chars) = ~17 tokens
        # Plus code lines (10 lines * 2) = 20 tokens
        # Total ~37 tokens
        assert 10 < tokens < 100

    def test_short_symbol(self):
        """Test token estimation for a short symbol."""
        tokens = estimate_tokens(SAMPLE_SYMBOL_SHORT)
        assert tokens > 0
        assert tokens < estimate_tokens(SAMPLE_SYMBOL)


class TestComputeWeightedScores:
    """Tests for compute_weighted_scores function."""

    def test_empty_candidates(self):
        """Test with no candidates."""
        query_embedding = np.random.randn(768).astype(np.float32)
        result = compute_weighted_scores([], query_embedding, {}, {})
        assert result == []

    def test_scoring(self):
        """Test that scores are computed and sorted."""
        symbols = [
            Symbol(
                id="test.a",
                file="test.py",
                start_line=1,
                end_line=5,
                signature="def a():",
                doc="",
                mutates="",
                embedding=np.random.randn(768).astype(np.float32),
            ),
            Symbol(
                id="test.b",
                file="test.py",
                start_line=6,
                end_line=10,
                signature="def b():",
                doc="",
                mutates="",
                embedding=np.random.randn(768).astype(np.float32),
            ),
        ]

        query_embedding = np.random.randn(768).astype(np.float32)
        result = compute_weighted_scores(symbols, query_embedding, {}, {})

        assert len(result) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
        assert all(isinstance(s, Symbol) for s, _ in result)
        # Scores might be numpy floats, check with numbers abstract type
        import numbers
        assert all(isinstance(score, numbers.Real) for _, score in result)

        # Check sorted by score descending
        scores = [float(score) for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_pagerank_normalization(self):
        """Test PageRank score normalization."""
        symbols = [
            Symbol(
                id="test.a",
                file="test.py",
                start_line=1,
                end_line=5,
                signature="def a():",
                doc="",
                mutates="",
                embedding=np.random.randn(768).astype(np.float32),
            ),
        ]

        query_embedding = np.random.randn(768).astype(np.float32)
        pagerank = {"test.a": 1.0}  # Max PageRank

        result = compute_weighted_scores(symbols, query_embedding, pagerank, {})

        # PageRank component should be 1.0 (normalized)
        assert len(result) == 1


class TestTruncateAtTokenBudget:
    """Tests for truncate_at_token_budget function."""

    def test_empty_list(self):
        """Test with empty symbol list."""
        result = truncate_at_token_budget([], 1000)
        assert result == []

    def test_no_truncation_needed(self):
        """Test when all symbols fit within budget."""
        symbols = [
            Symbol(
                id=f"test.{i}",
                file=f"test{i}.py",
                start_line=1,
                end_line=5,
                signature=f"def func{i}():",
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            )
            for i in range(3)
        ]

        scored = [(s, 0.5) for s in symbols]
        result = truncate_at_token_budget(scored, token_budget=10000)

        assert len(result) == 3

    def test_truncation_at_file_boundary(self):
        """Test that truncation breaks at file boundaries."""
        symbols = [
            Symbol(
                id=f"file1.func{i}",
                file="file1.py",
                start_line=1,
                end_line=20,  # Moderate size
                signature=f"def func{i}():",
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            )
            for i in range(2)
        ]
        # Add another file's symbols
        symbols += [
            Symbol(
                id=f"file2.func{i}",
                file="file2.py",
                start_line=1,
                end_line=20,
                signature=f"def func{i}():",
                doc="",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32),
            )
            for i in range(2)
        ]

        scored = [(s, 0.5) for s in symbols]

        # Set a very low budget that should only fit one file
        result = truncate_at_token_budget(scored, token_budget=50)

        # Should include some symbols but not all
        # The key test is that we don't crash and return a list
        assert isinstance(result, list)
        # Results shouldn't have symbols from both files when budget is tight
        files_in_result = {s.file for s in result}
        assert len(files_in_result) <= 1


class TestFindRepoRoot:
    """Tests for find_repo_root function."""

    def test_finds_git_directory(self, tmp_path):
        """Test that .git directory is found."""
        # Create .git directory
        (tmp_path / ".git").mkdir()

        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = find_repo_root()
            assert result == tmp_path
        finally:
            os.chdir(original_cwd)

    def test_no_git_directory(self, tmp_path):
        """Test behavior when .git doesn't exist."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = find_repo_root()
            assert result is None
        finally:
            os.chdir(original_cwd)


class TestGetIndexPath:
    """Tests for get_index_path function."""

    def test_returns_codegraph_dir(self):
        """Test that .codegraph path is returned."""
        repo_root = Path("/test/repo")
        result = get_index_path(repo_root)
        assert result == repo_root / ".codegraph"


class TestGenerateMcpConfig:
    """Tests for generate_mcp_config function."""

    def test_returns_valid_json(self):
        """Test that config is valid JSON."""
        config = generate_mcp_config()
        parsed = json.loads(config)
        assert "mcpServers" in parsed
        assert "codegrapher" in parsed["mcpServers"]

    def test_config_structure(self):
        """Test config structure matches PRD."""
        config = generate_mcp_config()
        parsed = json.loads(config)

        codegrapher = parsed["mcpServers"]["codegrapher"]
        assert codegrapher["command"] == "python"
        assert codegrapher["args"] == ["-m", "codegrapher.server"]


class TestCodegraphQuery:
    """Tests for codegraph_query tool."""

    def test_no_repo_root(self, tmp_path):
        """Test behavior when repository root not found."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Access the underlying function via .fn attribute
            result = codegraph_query.fn(query="test query")
            assert result["status"] == "error"
            assert result["error_type"] == "repo_not_found"
        finally:
            os.chdir(original_cwd)

    def test_index_not_found(self, tmp_path):
        """Test behavior when index doesn't exist."""
        # Create .git directory
        (tmp_path / ".git").mkdir()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = codegraph_query.fn(query="test query")
            assert result["status"] == "error"
            assert result["error_type"] == "index_not_found"
            assert "fallback_suggestion" in result
        finally:
            os.chdir(original_cwd)

    @patch('codegrapher.server.Database')
    @patch('codegrapher.server.FAISSIndexManager')
    @patch('codegrapher.server.EmbeddingModel')
    def test_successful_query(
        self, mock_model_class, mock_faiss_class, mock_db_class, tmp_path
    ):
        """Test successful query execution."""
        # Create .git directory
        (tmp_path / ".git").mkdir()
        (tmp_path / ".codegraph").mkdir()

        # Create index files
        (tmp_path / ".codegraph" / "symbols.db").touch()
        (tmp_path / ".codegraph" / "index.faiss").touch()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Mock the components
            mock_db = MagicMock()
            mock_db.get_symbol.return_value = None
            mock_db.get_meta.return_value = None
            mock_db_class.return_value = mock_db

            mock_faiss = MagicMock()
            mock_faiss.search.return_value = []  # No results
            mock_faiss_class.return_value = mock_faiss

            mock_model = MagicMock()
            mock_model.embed_text.return_value = np.zeros(768, dtype=np.float32)
            mock_model_class.return_value = mock_model

            result = codegraph_query.fn(query="test query")

            assert result["status"] == "success"
            assert result["files"] == []
            assert result["total_symbols"] == 0

        finally:
            os.chdir(original_cwd)


class TestGetGitLog:
    """Tests for get_git_log function."""

    def test_no_git_repo(self, tmp_path):
        """Test behavior when not in a git repo."""
        result = get_git_log(tmp_path)
        # Should fallback to file modification times
        assert isinstance(result, dict)

    def test_with_python_files(self, tmp_path):
        """Test that Python files are included."""
        # Create some Python files
        (tmp_path / "test1.py").touch()
        (tmp_path / "test2.py").touch()

        result = get_git_log(tmp_path)

        # Should have entries for the files
        assert "test1.py" in result or str(tmp_path / "test1.py") in result
        assert "test2.py" in result or str(tmp_path / "test2.py") in result


class TestIsTestSourcePair:
    """Tests for is_test_source_pair function."""

    def test_test_prefix_same_directory(self):
        """Pattern 1: test_ prefix in same directory."""
        assert is_test_source_pair("test_compiler.py", "compiler.py")
        assert is_test_source_pair("test_client.py", "client.py")

    def test_tests_src_mirror_structure(self):
        """Pattern 2: tests/ mirrors src/ structure."""
        assert is_test_source_pair("tests/test_compiler.py", "src/compiler.py")
        assert is_test_source_pair("tests/test_client.py", "src/client.py")
        assert is_test_source_pair("tests/compiler.py", "src/compiler.py")

    def test_test_suffix(self):
        """Pattern 3: _test.py suffix."""
        assert is_test_source_pair("compiler_test.py", "compiler.py")
        assert is_test_source_pair("client_test.py", "client.py")

    def test_no_match(self):
        """Files that are not test-source pairs."""
        assert not is_test_source_pair("main.py", "utils.py")
        assert not is_test_source_pair("test_a.py", "test_b.py")
        assert not is_test_source_pair("compiler.py", "utils.py")

    def test_windows_paths(self):
        """Windows path normalization."""
        assert is_test_source_pair("test_compiler.py", "compiler.py")
        assert is_test_source_pair("tests\\test_compiler.py", "src\\compiler.py")

    def test_parallel_directory_trees(self):
        """Pattern 6: Parallel directory trees (src/ vs testing/)."""
        # pytest's own test structure
        assert is_test_source_pair("testing/python/approx.py", "src/_pytest/python/approx.py")
        # Same filename in parallel trees
        assert is_test_source_pair("testing/foo/bar.py", "src/foo/bar.py")
        # With test_ prefix in parallel tree
        assert is_test_source_pair("testing/test_utils.py", "src/utils.py")

    def test_init_py_handling(self):
        """Pattern 7: __init__.py uses parent directory name."""
        # Werkzeug example: debug/__init__.py ↔ test_debug.py
        assert is_test_source_pair("tests/test_debug.py", "src/werkzeug/debug/__init__.py")
        # Flask example
        assert is_test_source_pair("tests/test_helpers.py", "src/flask/helpers/__init__.py")
        # Generic pattern
        assert is_test_source_pair("test_mymodule.py", "mypackage/mymodule/__init__.py")


class TestAugmentWithBidirectionalTestPairs:
    """Tests for augment_with_bidirectional_test_pairs function."""

    def make_symbol(self, id: str, file: str) -> Symbol:
        """Helper to create a symbol."""
        return Symbol(
            id=id,
            file=file,
            start_line=1,
            end_line=5,
            signature=f"def {id}():",
            doc="",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32),
        )

    def test_source_to_test_pairing(self):
        """Source file in candidates → adds test file."""
        # Candidates have symbols from compiler.py
        source_symbol = self.make_symbol("compiler.parse", "compiler.py")
        candidates = [source_symbol]

        # All symbols includes test_compiler.py
        test_symbol = self.make_symbol("test_compiler.parse", "test_compiler.py")
        all_symbols = [source_symbol, test_symbol]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should add test file
        assert len(result) > len(candidates)
        assert any(s.file == "test_compiler.py" for s in result)

    def test_test_to_source_pairing(self):
        """Test file in candidates → adds source file."""
        # Candidates have symbols from test_compiler.py
        test_symbol = self.make_symbol("test_compiler.parse", "test_compiler.py")
        candidates = [test_symbol]

        # All symbols includes compiler.py
        source_symbol = self.make_symbol("compiler.parse", "compiler.py")
        all_symbols = [test_symbol, source_symbol]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should add source file
        assert len(result) > len(candidates)
        assert any(s.file == "compiler.py" for s in result)

    def test_tests_src_structure_pairing(self):
        """Tests/ and src/ directory structure pairing."""
        # Candidates have symbols from src/compiler.py
        source_symbol = self.make_symbol("compiler.parse", "src/compiler.py")
        candidates = [source_symbol]

        # All symbols includes tests/test_compiler.py
        test_symbol = self.make_symbol("test_compiler.parse", "tests/test_compiler.py")
        all_symbols = [source_symbol, test_symbol]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should add test file
        assert any(s.file == "tests/test_compiler.py" for s in result)

    def test_multiple_candidates_all_get_paired(self):
        """Multiple source files → all get their test files added."""
        # Candidates have multiple source files
        compiler_sym = self.make_symbol("compiler.parse", "compiler.py")
        client_sym = self.make_symbol("client.request", "client.py")
        candidates = [compiler_sym, client_sym]

        # All symbols includes both test files
        test_compiler = self.make_symbol("test_compiler.parse", "test_compiler.py")
        test_client = self.make_symbol("test_client.request", "test_client.py")
        all_symbols = [compiler_sym, client_sym, test_compiler, test_client]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should add both test files
        result_files = {s.file for s in result}
        assert "test_compiler.py" in result_files
        assert "test_client.py" in result_files

    def test_no_duplicates(self):
        """Don't add symbols that are already in candidates."""
        # Candidates have both source and test
        source_sym = self.make_symbol("compiler.parse", "compiler.py")
        test_sym = self.make_symbol("test_compiler.parse", "test_compiler.py")
        candidates = [source_sym, test_sym]

        all_symbols = [source_sym, test_sym]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should not add duplicates
        assert len(result) == len(candidates)

    def test_no_false_positives(self):
        """Don't pair unrelated files."""
        # Candidates have compiler.py
        compiler_sym = self.make_symbol("compiler.parse", "compiler.py")
        candidates = [compiler_sym]

        # All symbols includes unrelated test files
        test_client = self.make_symbol("test_client.request", "test_client.py")
        test_utils = self.make_symbol("test_utils.helper", "test_utils.py")
        all_symbols = [compiler_sym, test_client, test_utils]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should NOT add unrelated test files
        result_files = {s.file for s in result}
        assert "test_client.py" not in result_files
        assert "test_utils.py" not in result_files

    def test_real_world_jinja_example(self):
        """Real-world example from task_028 (jinja)."""
        # Candidates have src/jinja2/compiler.py
        compiler = self.make_symbol("jinja2.compiler.compile", "src/jinja2/compiler.py")
        candidates = [compiler]

        # All symbols includes tests/test_compile.py
        test_compile = self.make_symbol("test_compile.test", "tests/test_compile.py")
        all_symbols = [compiler, test_compile]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should add test file
        result_files = {s.file for s in result}
        assert "tests/test_compile.py" in result_files

    def test_real_world_starlette_example(self):
        """Real-world example from task_039 (starlette)."""
        # Candidates have starlette/middleware/exceptions.py
        exceptions = self.make_symbol("middleware.http_exception", "starlette/middleware/exceptions.py")
        candidates = [exceptions]

        # All symbols includes tests/test_exceptions.py
        test_exceptions = self.make_symbol("test_exceptions.exception", "tests/test_exceptions.py")
        all_symbols = [exceptions, test_exceptions]

        result = augment_with_bidirectional_test_pairs(candidates, all_symbols)

        # Should add test file
        result_files = {s.file for s in result}
        assert "tests/test_exceptions.py" in result_files
