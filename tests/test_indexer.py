"""Unit tests for incremental indexing logic.

Tests for SymbolDiff, IncrementalIndexer, atomic_update, and helpers.
"""

import ast
import pickle
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codegrapher.indexer import (
    SymbolDiff,
    IncrementalIndexer,
    atomic_update,
    apply_diff,
    _needs_reembed,
)
from codegrapher.models import Symbol, Database
from codegrapher.vector_store import FAISSIndexManager


# Sample source code for testing
SAMPLE_SOURCE = """
def foo():
    '''A simple function.'''
    pass

class Bar:
    '''A simple class.'''
    pass

CONSTANT = 42
"""

MODIFIED_SOURCE = """
def foo():
    '''A modified function.'''
    pass

class Bar:
    '''A simple class.'''
    pass

CONSTANT = 42
"""

MODIFIED_SIGNATURE = """
def foo(x: int) -> int:
    '''A simple function.'''
    return x

class Bar:
    '''A simple class.'''
    pass

CONSTANT = 42
"""

DELETED_FUNCTION = """
class Bar:
    '''A simple class.'''
    pass

CONSTANT = 42
"""

ADDED_FUNCTION = """
def foo():
    '''A simple function.'''
    pass

class Bar:
    '''A simple class.'''
    pass

def baz():
    '''A new function.'''
    pass

CONSTANT = 42
"""


class TestSymbolDiff:
    """Tests for SymbolDiff dataclass."""

    def test_empty_diff_has_no_changes(self):
        """Test that an empty diff reports has_changes=False."""
        diff = SymbolDiff()
        assert diff.has_changes is False

    def test_diff_with_deleted_has_changes(self):
        """Test that diff with deleted symbols reports has_changes=True."""
        diff = SymbolDiff(deleted=["module.function"])
        assert diff.has_changes is True

    def test_diff_with_added_has_changes(self):
        """Test that diff with added symbols reports has_changes=True."""
        diff = SymbolDiff(added=[Symbol(
            id="test.function",
            file="test.py",
            start_line=1,
            end_line=5,
            signature="def function():",
            doc="A function",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32)
        )])
        assert diff.has_changes is True

    def test_diff_with_modified_has_changes(self):
        """Test that diff with modified symbols reports has_changes=True."""
        diff = SymbolDiff(modified=[Symbol(
            id="test.function",
            file="test.py",
            start_line=1,
            end_line=5,
            signature="def function():",
            doc="A function",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32)
        )])
        assert diff.has_changes is True


class TestIncrementalIndexer:
    """Tests for IncrementalIndexer class."""

    def test_initialization(self):
        """Test that IncrementalIndexer initializes correctly."""
        indexer = IncrementalIndexer(cache_size=50)
        assert indexer._cache_size == 50
        assert len(indexer._ast_cache) == 0

    def test_custom_cache_size(self):
        """Test that custom cache size is respected."""
        indexer = IncrementalIndexer(cache_size=10)
        assert indexer._cache_size == 10

    def test_update_file_first_time(self, temp_repo):
        """Test update_file when seeing a file for the first time."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        diff = indexer.update_file(test_file, repo_root=temp_repo)

        # All symbols should be marked as added
        assert len(diff.deleted) == 0
        assert len(diff.modified) == 0
        # Should have 3 symbols: foo, Bar, CONSTANT
        assert len(diff.added) == 3

        # Verify cache was populated
        assert test_file in indexer._ast_cache

    def test_update_file_with_no_changes(self, temp_repo):
        """Test update_file when file hasn't changed."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        # First update - adds all symbols
        diff1 = indexer.update_file(test_file, repo_root=temp_repo)
        assert len(diff1.added) == 3

        # Second update with same content - no changes
        diff2 = indexer.update_file(test_file, repo_root=temp_repo)
        assert len(diff2.added) == 0
        assert len(diff2.deleted) == 0
        assert len(diff2.modified) == 0

    def test_update_file_detects_docstring_change(self, temp_repo):
        """Test update_file detects docstring changes."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        # First update
        diff1 = indexer.update_file(test_file, repo_root=temp_repo)

        # Modify docstring
        test_file.write_text(MODIFIED_SOURCE)
        diff2 = indexer.update_file(test_file, repo_root=temp_repo)

        # Should have one modified function
        assert len(diff2.modified) == 1
        assert "foo" in diff2.modified[0].id
        assert len(diff2.added) == 0
        assert len(diff2.deleted) == 0

    def test_update_file_detects_signature_change(self, temp_repo):
        """Test update_file detects signature changes."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        # First update
        diff1 = indexer.update_file(test_file, repo_root=temp_repo)

        # Modify signature
        test_file.write_text(MODIFIED_SIGNATURE)
        diff2 = indexer.update_file(test_file, repo_root=temp_repo)

        # Should have one modified function
        assert len(diff2.modified) == 1
        assert "foo" in diff2.modified[0].id

    def test_update_file_detects_deletion(self, temp_repo):
        """Test update_file detects deleted symbols."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        # First update
        diff1 = indexer.update_file(test_file, repo_root=temp_repo)
        assert len(diff1.added) == 3

        # Remove function
        test_file.write_text(DELETED_FUNCTION)
        diff2 = indexer.update_file(test_file, repo_root=temp_repo)

        # Should have one deleted symbol
        assert len(diff2.deleted) == 1
        assert "test.foo" in diff2.deleted

    def test_update_file_detects_addition(self, temp_repo):
        """Test update_file detects added symbols."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        # First update
        diff1 = indexer.update_file(test_file, repo_root=temp_repo)
        initial_count = len(diff1.added)

        # Add new function
        test_file.write_text(ADDED_FUNCTION)
        diff2 = indexer.update_file(test_file, repo_root=temp_repo)

        # Should have one added symbol
        assert len(diff2.added) == 1
        assert "test.baz" in diff2.added[0].id

    def test_cache_fifo_eviction(self, temp_repo):
        """Test that cache evicts oldest entries when full."""
        indexer = IncrementalIndexer(cache_size=3)

        # Add 4 files to cache of size 3
        for i in range(4):
            test_file = temp_repo / f"test{i}.py"
            test_file.write_text(f"def func{i}(): pass")
            indexer.update_file(test_file, repo_root=temp_repo)

        # Only 3 files should be in cache
        assert len(indexer._ast_cache) == 3

        # First file should be evicted (FIFO)
        test0 = temp_repo / "test0.py"
        assert test0 not in indexer._ast_cache

        # Last file should still be in cache
        test3 = temp_repo / "test3.py"
        assert test3 in indexer._ast_cache

    def test_invalidate(self, temp_repo):
        """Test that invalidate removes a file from cache."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        # Populate cache
        indexer.update_file(test_file, repo_root=temp_repo)
        assert test_file in indexer._ast_cache

        # Invalidate
        indexer.invalidate(test_file)
        assert test_file not in indexer._ast_cache

    def test_clear(self, temp_repo):
        """Test that clear empties the cache."""
        indexer = IncrementalIndexer()

        # Add multiple files
        for i in range(3):
            test_file = temp_repo / f"test{i}.py"
            test_file.write_text(f"def func{i}(): pass")
            indexer.update_file(test_file, repo_root=temp_repo)

        assert len(indexer._ast_cache) == 3

        # Clear cache
        indexer.clear()
        assert len(indexer._ast_cache) == 0

    def test_update_file_with_new_source_param(self, temp_repo):
        """Test update_file with new_source parameter (doesn't read file)."""
        indexer = IncrementalIndexer()

        # File doesn't need to exist
        non_existent = temp_repo / "does_not_exist.py"

        diff = indexer.update_file(
            non_existent,
            new_source=SAMPLE_SOURCE,
            repo_root=temp_repo
        )

        # Should still work
        assert len(diff.added) == 3
        assert non_existent in indexer._ast_cache

    def test_update_file_syntax_error(self, temp_repo):
        """Test that SyntaxError is raised for invalid Python."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "test.py"
        test_file.write_text("this is not valid python !!!")

        with pytest.raises(SyntaxError):
            indexer.update_file(test_file, repo_root=temp_repo)

    def test_update_file_not_found(self, temp_repo):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        indexer = IncrementalIndexer()
        test_file = temp_repo / "does_not_exist.py"

        with pytest.raises(FileNotFoundError):
            indexer.update_file(test_file, repo_root=temp_repo)


class TestNeedsReembed:
    """Tests for _needs_reembed helper function."""

    def test_function_no_change(self):
        """Test that unchanged function returns False."""
        old = ast.parse("def foo(): pass").body[0]
        new = ast.parse("def foo(): pass").body[0]
        assert _needs_reembed(old, new) is False

    def test_function_lineno_change(self):
        """Test that line number change returns True."""
        old = ast.parse("def foo(): pass").body[0]
        new = ast.parse("\n\ndef foo(): pass").body[0]
        assert _needs_reembed(old, new) is True

    def test_function_docstring_change(self):
        """Test that docstring change returns True."""
        old = ast.parse("def foo(): '''old'''").body[0]
        new = ast.parse("def foo(): '''new'''").body[0]
        assert _needs_reembed(old, new) is True

    def test_function_signature_change(self):
        """Test that signature change returns True."""
        old = ast.parse("def foo(): pass").body[0]
        new = ast.parse("def foo(x: int) -> int: pass").body[0]
        assert _needs_reembed(old, new) is True

    def test_function_decorator_change(self):
        """Test that decorator change returns True."""
        old = ast.parse("def foo(): pass").body[0]
        new = ast.parse("@decorator\ndef foo(): pass").body[0]
        assert _needs_reembed(old, new) is True

    def test_class_no_change(self):
        """Test that unchanged class returns False."""
        old = ast.parse("class Foo: pass").body[0]
        new = ast.parse("class Foo: pass").body[0]
        assert _needs_reembed(old, new) is False

    def test_class_base_change(self):
        """Test that base class change returns True."""
        old = ast.parse("class Foo: pass").body[0]
        new = ast.parse("class Foo(Bar): pass").body[0]
        assert _needs_reembed(old, new) is True

    def test_assignment_no_change(self):
        """Test that unchanged assignment returns False."""
        old = ast.parse("X = 42").body[0]
        new = ast.parse("X = 42").body[0]
        assert _needs_reembed(old, new) is False

    def test_assignment_value_change(self):
        """Test that value change returns True."""
        old = ast.parse("X = 42").body[0]
        new = ast.parse("X = 43").body[0]
        assert _needs_reembed(old, new) is True


class TestAtomicUpdate:
    """Tests for atomic_update context manager."""

    def test_successful_update(self):
        """Test that successful update commits both SQLite and FAISS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"
            index_path = tmpdir / "test.faiss"

            # Create database and FAISS manager
            db = Database(db_path)
            db.initialize()

            faiss_mgr = FAISSIndexManager(index_path)

            # Perform atomic update
            with atomic_update(db_path, faiss_mgr) as conn:
                conn.execute("INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                    "test.func", "test.py", 1, 5, "def func():", "doc", "", b"\x00" * 3072
                ))

            # Verify database was committed
            result = db.get_symbol("test.func")
            assert result is not None

    def test_rollback_on_exception(self):
        """Test that exception rolls back both SQLite and FAISS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"
            index_path = tmpdir / "test.faiss"

            # Create database and FAISS manager
            db = Database(db_path)
            db.initialize()

            faiss_mgr = FAISSIndexManager(index_path)

            # Try to perform update that fails
            with pytest.raises(ValueError):
                with atomic_update(db_path, faiss_mgr) as conn:
                    conn.execute("INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                        "test.func", "test.py", 1, 5, "def func():", "doc", "", b"\x00" * 3072
                    ))
                    raise ValueError("Intentional error")

            # Verify database was rolled back
            result = db.get_symbol("test.func")
            assert result is None


class TestApplyDiff:
    """Tests for apply_diff function."""

    def test_apply_diff_with_deletions(self):
        """Test apply_diff with deleted symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"
            index_path = tmpdir / "test.faiss"

            db = Database(db_path)
            db.initialize()

            # Add a symbol
            symbol = Symbol(
                id="test.func",
                file="test.py",
                start_line=1,
                end_line=5,
                signature="def func():",
                doc="A function",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32)
            )
            db.insert_symbol(symbol)

            # Create diff with deletion
            diff = SymbolDiff(deleted=["test.func"])

            # Create mock FAISS manager
            faiss_mgr = MagicMock(spec=FAISSIndexManager)

            # Apply diff
            apply_diff(diff, db, faiss_mgr)

            # Verify deletion
            assert db.get_symbol("test.func") is None
            faiss_mgr.remove_symbols.assert_called_once_with(["test.func"])

    def test_apply_diff_with_additions(self):
        """Test apply_diff with added symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"
            index_path = tmpdir / "test.faiss"

            db = Database(db_path)
            db.initialize()

            # Create diff with addition
            symbol = Symbol(
                id="test.func",
                file="test.py",
                start_line=1,
                end_line=5,
                signature="def func():",
                doc="A function",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32)
            )
            diff = SymbolDiff(added=[symbol])

            # Create mock FAISS manager
            faiss_mgr = MagicMock(spec=FAISSIndexManager)

            # Apply diff
            apply_diff(diff, db, faiss_mgr)

            # Verify addition
            result = db.get_symbol("test.func")
            assert result is not None
            faiss_mgr.add_symbols.assert_called_once_with([symbol])

    def test_apply_diff_with_modifications(self):
        """Test apply_diff with modified symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"
            index_path = tmpdir / "test.faiss"

            db = Database(db_path)
            db.initialize()

            # Add original symbol
            old_symbol = Symbol(
                id="test.func",
                file="test.py",
                start_line=1,
                end_line=5,
                signature="def func():",
                doc="Old doc",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32)
            )
            db.insert_symbol(old_symbol)

            # Create diff with modification
            new_symbol = Symbol(
                id="test.func",
                file="test.py",
                start_line=1,
                end_line=5,
                signature="def func():",
                doc="New doc",
                mutates="",
                embedding=np.zeros(768, dtype=np.float32)
            )
            diff = SymbolDiff(modified=[new_symbol])

            # Create mock FAISS manager
            faiss_mgr = MagicMock(spec=FAISSIndexManager)

            # Apply diff
            apply_diff(diff, db, faiss_mgr)

            # Verify modification
            result = db.get_symbol("test.func")
            assert result is not None
            assert result.doc == "New doc"
            faiss_mgr.remove_symbols.assert_called_once_with(["test.func"])
            faiss_mgr.add_symbols.assert_called_once_with([new_symbol])

    def test_apply_diff_empty(self):
        """Test apply_diff with empty diff (no changes)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"
            index_path = tmpdir / "test.faiss"

            db = Database(db_path)
            db.initialize()

            # Create empty diff
            diff = SymbolDiff()

            # Create mock FAISS manager
            faiss_mgr = MagicMock(spec=FAISSIndexManager)

            # Apply diff (should not call anything)
            apply_diff(diff, db, faiss_mgr)

            # Verify no changes
            faiss_mgr.remove_symbols.assert_not_called()
            faiss_mgr.add_symbols.assert_not_called()


# Fixture for temp_repo
@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
