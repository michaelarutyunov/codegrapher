"""Unit tests for CLI.

Tests for Phase 11: CLI & Build Tools.
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codegrapher.cli import (
    create_parser,
    cmd_init,
    cmd_build,
    cmd_query,
    cmd_update,
    cmd_mcp_config,
    init_command,
    build_command,
    query_command,
    update_command,
    mcp_config_command,
)


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_exists(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "codegraph"

    def test_has_all_subcommands(self):
        """Test that all subcommands are registered."""
        parser = create_parser()
        # Check subcommands exist by testing parse_args for each
        subcommands = ["init", "build", "query", "update", "mcp-config"]
        for cmd in subcommands:
            # For query, we need to provide a query argument
            if cmd == "query":
                args = parser.parse_args([cmd, "test"])
            else:
                args = parser.parse_args([cmd])
            assert args.command == cmd


class TestCmdInit:
    """Tests for cmd_init function."""

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    @patch("codegrapher.cli.EmbeddingModel")
    @patch("codegrapher.cli.install_git_hook")
    def test_init_creates_index_dir(
        self, mock_hook, mock_model_class, mock_get_index, mock_find_root, tmp_path
    ):
        """Test that init creates the index directory."""
        mock_find_root.return_value = tmp_path
        mock_get_index.return_value = tmp_path / ".codegraph"
        mock_hook.return_value = True

        args = argparse.Namespace(no_model=True, no_hook=True)
        cmd_init(args)

        assert (tmp_path / ".codegraph").exists()

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    @patch("codegrapher.cli.EmbeddingModel")
    @patch("codegrapher.cli.install_git_hook")
    def test_init_with_model_download(
        self, mock_hook, mock_model_class, mock_get_index, mock_find_root, tmp_path
    ):
        """Test init with model download."""
        mock_find_root.return_value = tmp_path
        mock_get_index.return_value = tmp_path / ".codegraph"

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_hook.return_value = True

        args = argparse.Namespace(no_model=False, no_hook=True)
        cmd_init(args)

        # Model should be loaded
        mock_model._load_model.assert_called_once()

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    @patch("codegrapher.cli.EmbeddingModel")
    @patch("codegrapher.cli.install_git_hook")
    def test_init_with_hook_install(
        self, mock_hook, mock_model_class, mock_get_index, mock_find_root, tmp_path
    ):
        """Test init with git hook installation."""
        mock_find_root.return_value = tmp_path
        mock_get_index.return_value = tmp_path / ".codegraph"
        mock_hook.return_value = True

        args = argparse.Namespace(no_model=True, no_hook=False)
        cmd_init(args)

        # Hook should be installed
        mock_hook.assert_called_once_with(tmp_path)


class TestCmdBuild:
    """Tests for cmd_build function."""

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    @patch("codegrapher.cli.Database")
    @patch("codegrapher.cli.FAISSIndexManager")
    @patch("codegrapher.cli.EmbeddingModel")
    def test_build_requires_full_or_force(
        self, mock_model_class, mock_faiss_class, mock_db_class, mock_get_index, mock_find_root, tmp_path
    ):
        """Test that build requires --full or --force when index exists."""
        mock_find_root.return_value = tmp_path
        mock_get_index.return_value = tmp_path / ".codegraph"

        # Create existing index
        index_dir = tmp_path / ".codegraph"
        index_dir.mkdir(parents=True)
        (index_dir / "symbols.db").touch()

        args = argparse.Namespace(full=False, force=False)
        with pytest.raises(SystemExit):
            cmd_build(args)

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    @patch("codegrapher.cli.Database")
    @patch("codegrapher.cli.FAISSIndexManager")
    @patch("codegrapher.cli.EmbeddingModel")
    def test_build_clears_existing_index(
        self, mock_model_class, mock_faiss_class, mock_db_class, mock_get_index, mock_find_root, tmp_path
    ):
        """Test that build --full clears existing index."""
        mock_find_root.return_value = tmp_path
        index_dir = tmp_path / ".codegraph"
        mock_get_index.return_value = index_dir

        # Create existing index
        index_dir.mkdir(parents=True)
        db_path = index_dir / "symbols.db"
        faiss_path = index_dir / "index.faiss"
        db_path.touch()
        faiss_path.touch()

        args = argparse.Namespace(full=True, force=False)
        cmd_build(args)

        # Files should be deleted
        assert not db_path.exists()
        assert not faiss_path.exists()


class TestCmdQuery:
    """Tests for cmd_query function."""

    @patch("codegrapher.cli.codegraph_query")
    def test_query_with_json_output(self, mock_query):
        """Test query with --json flag."""
        mock_query.fn.return_value = {
            "status": "success",
            "files": [
                {
                    "path": "test.py",
                    "line_range": [1, 10],
                    "symbol": "test.function",
                    "excerpt": "def test():",
                    "pagerank_score": 0.001234,
                }
            ],
            "tokens_used": 100,
            "total_symbols": 1,
            "truncated": False,
        }

        args = argparse.Namespace(
            query="test query",
            cursor_file=None,
            max_depth=1,
            token_budget=3500,
            json=True
        )

        with patch("builtins.print") as mock_print:
            cmd_query(args)
            # Check that JSON output was printed
            printed_args = [call.args[0] for call in mock_print.call_args_list]
            assert any("test.py" in arg for arg in printed_args)

    @patch("codegrapher.cli.codegraph_query")
    def test_query_with_error(self, mock_query):
        """Test query with error response."""
        mock_query.fn.return_value = {
            "status": "error",
            "error_type": "index_not_found",
            "message": "Index not found",
            "fallback_suggestion": "Run 'codegraph build --full'"
        }

        args = argparse.Namespace(
            query="test query",
            cursor_file=None,
            max_depth=1,
            token_budget=3500,
            json=False
        )

        with pytest.raises(SystemExit):
            cmd_query(args)


class TestCmdUpdate:
    """Tests for cmd_update function."""

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    def test_update_requires_index(self, mock_get_index, mock_find_root, tmp_path):
        """Test that update fails when index doesn't exist."""
        mock_find_root.return_value = tmp_path
        mock_get_index.return_value = tmp_path / ".codegraph"

        args = argparse.Namespace(git_changed=False, file="test.py")
        with pytest.raises(SystemExit):
            cmd_update(args)

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    @patch("codegrapher.cli.Database")
    @patch("codegrapher.cli.FAISSIndexManager")
    @patch("codegrapher.cli.EmbeddingModel")
    @patch("codegrapher.cli.IncrementalIndexer")
    def test_update_with_file(
        self, mock_indexer_class, mock_model_class, mock_faiss_class, mock_db_class,
        mock_get_index, mock_find_root, tmp_path
    ):
        """Test update with a specific file."""
        mock_find_root.return_value = tmp_path
        index_dir = tmp_path / ".codegraph"
        mock_get_index.return_value = index_dir

        # Create existing index
        index_dir.mkdir(parents=True)
        (index_dir / "symbols.db").touch()
        (index_dir / "index.faiss").touch()

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        # Mock database
        mock_db = MagicMock()
        mock_db.get_all_symbols.return_value = []
        mock_db_class.return_value = mock_db

        # Mock diff
        from codegrapher.indexer import SymbolDiff
        mock_diff = SymbolDiff()
        mock_indexer = MagicMock()
        mock_indexer.update_file.return_value = mock_diff
        mock_indexer_class.return_value = mock_indexer

        args = argparse.Namespace(git_changed=False, file=str(test_file))
        with patch("codegrapher.cli.apply_diff"):
            with patch("codegrapher.cli.atomic_update"):
                cmd_update(args)

        # update_file should have been called
        mock_indexer.update_file.assert_called_once()


class TestCmdMcpConfig:
    """Tests for cmd_mcp_config function."""

    @patch("codegrapher.cli.generate_mcp_config")
    def test_mcp_config_prints_to_stdout(self, mock_generate):
        """Test that mcp-config prints to stdout by default."""
        mock_generate.return_value = '{"test": "config"}'

        args = argparse.Namespace(output=None)

        with patch("builtins.print") as mock_print:
            cmd_mcp_config(args)
            mock_print.assert_called_once_with('{"test": "config"}')

    @patch("codegrapher.cli.generate_mcp_config")
    def test_mcp_config_writes_to_file(self, mock_generate, tmp_path):
        """Test that mcp-config writes to file when --output is specified."""
        mock_generate.return_value = '{"test": "config"}'
        output_path = tmp_path / "config.json"

        args = argparse.Namespace(output=str(output_path))

        cmd_mcp_config(args)

        assert output_path.exists()
        assert output_path.read_text() == '{"test": "config"}'


class TestStandaloneCommands:
    """Tests for standalone command entry points."""

    def test_init_command_entry_point(self):
        """Test init_command entry point exists and is callable."""
        assert callable(init_command)

    def test_build_command_entry_point(self):
        """Test build_command entry point exists and is callable."""
        assert callable(build_command)

    def test_query_command_entry_point(self):
        """Test query_command entry point exists and is callable."""
        assert callable(query_command)

    def test_update_command_entry_point(self):
        """Test update_command entry point exists and is callable."""
        assert callable(update_command)

    def test_mcp_config_command_entry_point(self):
        """Test mcp_config_command entry point exists and is callable."""
        assert callable(mcp_config_command)

    @patch("codegrapher.cli.find_repo_root")
    @patch("codegrapher.cli.get_index_path")
    @patch("codegrapher.cli.EmbeddingModel")
    @patch("codegrapher.cli.install_git_hook")
    def test_init_command_calls_cmd_init(
        self, mock_hook, mock_model_class, mock_get_index, mock_find_root, tmp_path
    ):
        """Test that init_command wrapper calls cmd_init."""
        mock_find_root.return_value = tmp_path
        mock_get_index.return_value = tmp_path / ".codegraph"
        mock_hook.return_value = True

        with patch("sys.argv", ["codegraph-init", "--no-model", "--no-hook"]):
            init_command()

        # Index directory should be created
        assert (tmp_path / ".codegraph").exists()
