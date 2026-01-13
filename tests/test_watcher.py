"""Unit tests for file watching and automatic index updates.

Tests for Phase 9: File Watching & Auto-Update.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codegrapher.watcher import (
    PendingChange,
    IndexUpdateHandler,
    FileWatcher,
    get_git_hook_template,
    install_git_hook,
)
from codegrapher.indexer import IncrementalIndexer
from codegrapher.models import Database, Symbol
from codegrapher.vector_store import FAISSIndexManager


# Sample Python source code
SAMPLE_SOURCE = """
def hello():
    '''A simple function.'''
    pass

CONSTANT = 42
"""


MODIFIED_SOURCE = """
def hello():
    '''A modified function.'''
    pass

CONSTANT = 42
"""


class TestPendingChange:
    """Tests for PendingChange dataclass."""

    def test_creation(self):
        """Test PendingChange creation."""
        change = PendingChange(
            file_path=Path("test.py"),
            change_type="modified"
        )
        assert change.file_path == Path("test.py")
        assert change.change_type == "modified"
        assert isinstance(change.timestamp, float)

    def test_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        change = PendingChange(
            file_path=Path("test.py"),
            change_type="created"
        )
        after = time.time()
        assert before <= change.timestamp <= after


class TestIndexUpdateHandler:
    """Tests for IndexUpdateHandler class."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            yield repo_root

    @pytest.fixture
    def setup_components(self, temp_repo):
        """Set up database, FAISS manager, and indexer."""
        db_path = temp_repo / "test.db"
        index_path = temp_repo / "test.faiss"

        db = Database(db_path)
        db.initialize()

        faiss_manager = FAISSIndexManager(index_path)
        indexer = IncrementalIndexer()

        return db, faiss_manager, indexer

    def test_initialization(self, temp_repo, setup_components):
        """Test IndexUpdateHandler initialization."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        assert handler.repo_root == temp_repo
        assert handler.db == db
        assert handler.faiss_manager == faiss_manager
        assert handler.indexer == indexer
        assert handler.bulk_callback is None
        assert len(handler._pending_changes) == 0
        assert handler._changes_processed == 0

    def test_queue_change(self, temp_repo, setup_components):
        """Test queuing a file change."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        test_file = temp_repo / "test.py"
        handler._queue_change(test_file, "modified")

        assert len(handler._pending_changes) == 1
        assert test_file in handler._pending_changes
        assert handler._pending_changes[test_file].change_type == "modified"

    def test_queue_change_updates_existing(self, temp_repo, setup_components):
        """Test that queuing same file updates the existing change."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        test_file = temp_repo / "test.py"
        handler._queue_change(test_file, "created")
        first_timestamp = handler._pending_changes[test_file].timestamp

        # Wait a bit to ensure different timestamp
        time.sleep(0.01)
        handler._queue_change(test_file, "modified")

        assert len(handler._pending_changes) == 1
        assert handler._pending_changes[test_file].change_type == "modified"
        assert handler._pending_changes[test_file].timestamp > first_timestamp

    def test_on_created_py_file(self, temp_repo, setup_components):
        """Test on_created event for .py file."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        # Create mock event
        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_repo / "test.py")

        handler.on_created(event)

        assert len(handler._pending_changes) == 1
        assert Path(event.src_path) in handler._pending_changes

    def test_on_created_non_py_file(self, temp_repo, setup_components):
        """Test on_created event for non-.py file (should be ignored)."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        # Create mock event for .txt file
        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_repo / "test.txt")

        handler.on_created(event)

        assert len(handler._pending_changes) == 0

    def test_on_created_directory(self, temp_repo, setup_components):
        """Test on_created event for directory (should be ignored)."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        # Create mock event for directory
        event = Mock()
        event.is_directory = True
        event.src_path = str(temp_repo / "test_dir")

        handler.on_created(event)

        assert len(handler._pending_changes) == 0

    def test_on_modified(self, temp_repo, setup_components):
        """Test on_modified event."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_repo / "test.py")

        handler.on_modified(event)

        assert len(handler._pending_changes) == 1
        assert handler._pending_changes[Path(event.src_path)].change_type == "modified"

    def test_on_deleted(self, temp_repo, setup_components):
        """Test on_deleted event."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_repo / "test.py")

        handler.on_deleted(event)

        assert len(handler._pending_changes) == 1
        assert handler._pending_changes[Path(event.src_path)].change_type == "deleted"

    def test_on_moved(self, temp_repo, setup_components):
        """Test on_moved event."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        event = Mock()
        event.is_directory = False
        event.src_path = str(temp_repo / "old.py")
        event.dest_path = str(temp_repo / "new.py")

        handler.on_moved(event)

        # Should have both deletion and creation
        assert len(handler._pending_changes) == 2
        assert handler._pending_changes[Path(event.src_path)].change_type == "deleted"
        assert handler._pending_changes[Path(event.dest_path)].change_type == "created"

    def test_get_stats(self, temp_repo, setup_components):
        """Test getting watcher statistics."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        stats = handler.get_stats()
        assert stats["changes_processed"] == 0
        assert stats["bulk_rebuilds_triggered"] == 0
        assert stats["pending_changes"] == 0

        # Add some pending changes
        handler._queue_change(temp_repo / "test1.py", "modified")
        handler._queue_change(temp_repo / "test2.py", "created")

        stats = handler.get_stats()
        assert stats["pending_changes"] == 2

    def test_handle_deletion_removes_symbols(self, temp_repo, setup_components):
        """Test that file deletion removes symbols from index."""
        db, faiss_manager, indexer = setup_components

        handler = IndexUpdateHandler(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        # Create a test file and index it
        test_file = temp_repo / "test.py"
        test_file.write_text(SAMPLE_SOURCE)

        # Index the file
        import numpy as np
        symbol = Symbol(
            id="test.function",
            file=str(test_file),
            start_line=2,
            end_line=5,
            signature="def hello():",
            doc="A simple function.",
            mutates="",
            embedding=np.zeros(768, dtype=np.float32)
        )
        db.insert_symbol(symbol)

        # Verify symbol exists
        assert db.get_symbol("test.function") is not None

        # Handle deletion
        with patch('codegrapher.watcher.atomic_update'):
            handler._handle_deletion(test_file)

        # Verify symbol was removed
        assert db.get_symbol("test.function") is None


class TestFileWatcher:
    """Tests for FileWatcher class."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            yield repo_root

    @pytest.fixture
    def setup_components(self, temp_repo):
        """Set up database, FAISS manager, and indexer."""
        db_path = temp_repo / "test.db"
        index_path = temp_repo / "test.faiss"

        db = Database(db_path)
        db.initialize()

        faiss_manager = FAISSIndexManager(index_path)
        indexer = IncrementalIndexer()

        return db, faiss_manager, indexer

    def test_initialization(self, temp_repo, setup_components):
        """Test FileWatcher initialization."""
        db, faiss_manager, indexer = setup_components

        watcher = FileWatcher(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        assert watcher.repo_root == temp_repo
        assert watcher.db == db
        assert watcher.faiss_manager == faiss_manager
        assert watcher.indexer == indexer
        assert watcher.event_handler is not None
        assert watcher.observer is not None

    def test_initialization_with_bulk_callback(self, temp_repo, setup_components):
        """Test FileWatcher initialization with bulk callback."""
        db, faiss_manager, indexer = setup_components

        callback = Mock()
        watcher = FileWatcher(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
            bulk_callback=callback,
        )

        assert watcher.event_handler.bulk_callback == callback

    def test_get_stats(self, temp_repo, setup_components):
        """Test getting statistics from watcher."""
        db, faiss_manager, indexer = setup_components

        watcher = FileWatcher(
            repo_root=temp_repo,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
        )

        stats = watcher.get_stats()
        assert "changes_processed" in stats
        assert "bulk_rebuilds_triggered" in stats
        assert "pending_changes" in stats


class TestGitHook:
    """Tests for git hook functionality."""

    def test_get_git_hook_template(self):
        """Test git hook template generation."""
        template = get_git_hook_template()

        assert "#!/bin/bash" in template
        assert "CodeGrapher post-commit hook" in template
        assert "git diff-tree" in template
        assert "codegraph" in template
        assert "build --full" in template

    def test_install_git_hook_no_git_dir(self, tmp_path):
        """Test installing hook when .git directory doesn't exist."""
        result = install_git_hook(tmp_path)
        assert result is False

    def test_install_git_hook_creates_hook(self, tmp_path):
        """Test that install_git_hook creates the hook file."""
        # Create .git/hooks directory
        hooks_dir = tmp_path / ".git" / "hooks"
        hooks_dir.mkdir(parents=True)

        result = install_git_hook(tmp_path)
        assert result is True

        # Check hook file exists
        hook_path = hooks_dir / "post-commit"
        assert hook_path.exists()

        # Check hook is executable
        import stat
        st = hook_path.stat()
        assert st.st_mode & stat.S_IXUSR

    def test_install_git_hook_content(self, tmp_path):
        """Test that installed hook has correct content."""
        hooks_dir = tmp_path / ".git" / "hooks"
        hooks_dir.mkdir(parents=True)

        install_git_hook(tmp_path)

        hook_path = hooks_dir / "post-commit"
        content = hook_path.read_text()

        assert "#!/bin/bash" in content
        assert "CodeGrapher" in content

    def test_install_git_hook_already_exists(self, tmp_path):
        """Test that existing hook is not overwritten."""
        hooks_dir = tmp_path / ".git" / "hooks"
        hooks_dir.mkdir(parents=True)

        # Create existing hook
        hook_path = hooks_dir / "post-commit"
        hook_path.write_text("# Existing hook")

        result = install_git_hook(tmp_path)
        assert result is False

        # Verify hook wasn't changed
        assert hook_path.read_text() == "# Existing hook"
