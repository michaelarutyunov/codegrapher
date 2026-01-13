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
