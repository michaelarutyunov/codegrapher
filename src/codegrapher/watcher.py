"""File watching and automatic index updates.

This module implements PRD Phase 9: File Watching & Auto-Update.
Monitors Python source files for changes and triggers incremental
index updates automatically.

Key features:
- watchdog-based monitoring of *.py files
- Secret detection before indexing
- Incremental updates via IncrementalIndexer
- Bulk change queuing for background rebuild
- Debouncing to avoid redundant updates
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from watchdog.observers import Observer

from codegrapher.indexer import IncrementalIndexer, atomic_update, apply_diff
from codegrapher.models import Database
from codegrapher.secrets import scan_file
from codegrapher.vector_store import FAISSIndexManager


logger = logging.getLogger(__name__)

# Constants from PRD Phase 9
BULK_CHANGE_THRESHOLD = 20  # Files changed before triggering full rebuild
DEBOUNCE_SECONDS = 0.5  # Wait before processing changes (debounce)
WATCHDOG_POLL_INTERVAL = 1.0  # Observer polling interval


@dataclass
class PendingChange:
    """A pending file change waiting to be processed.

    Attributes:
        file_path: Path to the changed file
        change_type: Type of change (created, modified, deleted, moved)
        timestamp: When the change was detected
    """

    file_path: Path
    change_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: float = field(default_factory=time.time)


class IndexUpdateHandler(FileSystemEventHandler):
    """Handles file system events and triggers index updates.

    Watches for *.py file changes and queues them for processing.
    Implements debouncing to avoid redundant updates for rapid changes.
    """

    def __init__(
        self,
        repo_root: Path,
        db: Database,
        faiss_manager: FAISSIndexManager,
        indexer: IncrementalIndexer,
        bulk_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the file event handler.

        Args:
            repo_root: Repository root path
            db: Database instance
            faiss_manager: FAISS index manager
            indexer: Incremental indexer instance
            bulk_callback: Optional callback for bulk rebuilds
        """
        super().__init__()
        self.repo_root = repo_root
        self.db = db
        self.faiss_manager = faiss_manager
        self.indexer = indexer
        self.bulk_callback = bulk_callback

        # Pending changes (debouncing)
        self._pending_changes: Dict[Path, PendingChange] = {}
        self._pending_lock = threading.Lock()

        # Background thread for processing changes
        self._stop_event = threading.Event()
        self._processor_thread: Optional[threading.Thread] = None

        # Statistics
        self._changes_processed = 0
        self._bulk_rebuilds_triggered = 0

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        """Handle file creation event."""
        if event.is_directory:
            return
        src_path = event.src_path if isinstance(event.src_path, str) else event.src_path.decode()
        if not src_path.endswith(".py"):
            return

        file_path = Path(src_path)
        self._queue_change(file_path, "created")

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        """Handle file modification event."""
        if event.is_directory:
            return
        src_path = event.src_path if isinstance(event.src_path, str) else event.src_path.decode()
        if not src_path.endswith(".py"):
            return

        file_path = Path(src_path)
        self._queue_change(file_path, "modified")

    def on_deleted(self, event: FileDeletedEvent) -> None:  # type: ignore[override]
        """Handle file deletion event."""
        if event.is_directory:
            return
        src_path = event.src_path if isinstance(event.src_path, str) else event.src_path.decode()
        if not src_path.endswith(".py"):
            return

        file_path = Path(src_path)
        self._queue_change(file_path, "deleted")

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        """Handle file move/rename event."""
        if event.is_directory:
            return

        src_path_str = event.src_path if isinstance(event.src_path, str) else event.src_path.decode()
        dest_path_str = event.dest_path if isinstance(event.dest_path, str) else event.dest_path.decode()
        src_path = Path(src_path_str)
        dest_path = Path(dest_path_str)

        # Handle source deletion
        if src_path.suffix == ".py":
            self._queue_change(src_path, "deleted")

        # Handle destination creation
        if dest_path.suffix == ".py":
            self._queue_change(dest_path, "created")

    def _queue_change(self, file_path: Path, change_type: str) -> None:
        """Queue a change for processing (with debouncing).

        Args:
            file_path: Path to the changed file
            change_type: Type of change
        """
        with self._pending_lock:
            # Update or add the pending change
            self._pending_changes[file_path] = PendingChange(
                file_path=file_path, change_type=change_type
            )
            logger.debug(f"Queued {change_type}: {file_path}")

    def start(self) -> None:
        """Start the background processor thread."""
        if self._processor_thread is not None:
            logger.warning("Processor thread already running")
            return

        self._stop_event.clear()
        self._processor_thread = threading.Thread(
            target=self._process_changes_loop,
            name="IndexUpdateProcessor",
            daemon=True,
        )
        self._processor_thread.start()
        logger.info("Index update processor started")

    def stop(self) -> None:
        """Stop the background processor thread."""
        if self._processor_thread is None:
            return

        self._stop_event.set()
        self._processor_thread.join(timeout=5.0)
        self._processor_thread = None
        logger.info("Index update processor stopped")

    def _process_changes_loop(self) -> None:
        """Background thread loop for processing pending changes."""
        while not self._stop_event.is_set():
            try:
                # Sleep for debounce interval
                time.sleep(DEBOUNCE_SECONDS)

                # Check if we should stop
                if self._stop_event.is_set():
                    break

                # Process pending changes
                self._process_pending_changes()

            except Exception as e:
                logger.error(f"Error processing changes: {e}", exc_info=True)

        # Final flush before exit
        self._process_pending_changes()

    def _process_pending_changes(self) -> None:
        """Process all pending changes.

        Implements bulk change logic: if >20 files changed,
        triggers background rebuild instead of incremental updates.
        """
        with self._pending_lock:
            if not self._pending_changes:
                return

            # Get snapshot of pending changes
            changes = dict(self._pending_changes)
            self._pending_changes.clear()

        # Check for bulk change threshold
        if len(changes) > BULK_CHANGE_THRESHOLD:
            logger.info(
                f"Bulk change detected ({len(changes)} files), "
                "triggering background rebuild"
            )
            self._bulk_rebuilds_triggered += 1
            if self.bulk_callback:
                self.bulk_callback()
            return

        # Process incremental updates
        logger.debug(f"Processing {len(changes)} pending changes")
        for file_path, change in changes.items():
            try:
                self._process_single_change(file_path, change.change_type)
                self._changes_processed += 1
            except Exception as e:
                logger.error(
                    f"Error processing {file_path}: {e}", exc_info=True
                )

    def _process_single_change(self, file_path: Path, change_type: str) -> None:
        """Process a single file change.

        Args:
            file_path: Path to the changed file
            change_type: Type of change (created, modified, deleted, moved)
        """
        # Skip if file doesn't exist (deleted files)
        if change_type == "deleted" or not file_path.exists():
            self._handle_deletion(file_path)
            return

        # Scan for secrets
        try:
            has_secret = scan_file(file_path, self.repo_root)
            if has_secret:
                logger.warning(
                    f"Skipping {file_path}: secret detected"
                )
                return
        except Exception as e:
            logger.warning(f"Secret scan failed for {file_path}: {e}")

        # Update index incrementally
        with atomic_update(self.db.db_path, self.faiss_manager) as conn:
            diff = self.indexer.update_file(file_path, repo_root=self.repo_root)

            if diff.has_changes:
                apply_diff(diff, self.db, self.faiss_manager)
                logger.debug(
                    f"Updated index for {file_path}: "
                    f"+{len(diff.added)} -{len(diff.deleted)} "
                    f"~{len(diff.modified)}"
                )

    def _handle_deletion(self, file_path: Path) -> None:
        """Handle file deletion by removing symbols from index.

        Args:
            file_path: Path to the deleted file
        """
        # Get all symbols from this file
        symbols = self.db.connect().execute(
            "SELECT id FROM symbols WHERE file = ?", (str(file_path),)
        ).fetchall()

        if not symbols:
            return

        symbol_ids = [row[0] for row in symbols]

        # Remove from database and FAISS
        with atomic_update(self.db.db_path, self.faiss_manager) as conn:
            self.db.delete_symbols_batch(symbol_ids)
            self.faiss_manager.remove_symbols(symbol_ids)
            logger.debug(f"Removed {len(symbol_ids)} symbols from {file_path}")

    def get_stats(self) -> Dict[str, int]:
        """Get watcher statistics.

        Returns:
            Dictionary with stats (processed, bulk_rebuilds, pending)
        """
        with self._pending_lock:
            pending_count = len(self._pending_changes)

        return {
            "changes_processed": self._changes_processed,
            "bulk_rebuilds_triggered": self._bulk_rebuilds_triggered,
            "pending_changes": pending_count,
        }


class FileWatcher:
    """Manages file watching for automatic index updates.

    Wraps watchdog.Observer and provides a simple API for
    starting/stopping the watcher.

    Example:
        >>> watcher = FileWatcher(repo_root, db, faiss_manager, indexer)
        >>> watcher.start()
        >>> # ... make changes to files ...
        >>> watcher.stop()
        >>> stats = watcher.get_stats()
    """

    def __init__(
        self,
        repo_root: Path,
        db: Database,
        faiss_manager: FAISSIndexManager,
        indexer: IncrementalIndexer,
        bulk_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the file watcher.

        Args:
            repo_root: Repository root path to watch
            db: Database instance
            faiss_manager: FAISS index manager
            indexer: Incremental indexer instance
            bulk_callback: Optional callback for bulk rebuilds
        """
        self.repo_root = repo_root
        self.db = db
        self.faiss_manager = faiss_manager
        self.indexer = indexer

        # Create event handler
        self.event_handler = IndexUpdateHandler(
            repo_root=repo_root,
            db=db,
            faiss_manager=faiss_manager,
            indexer=indexer,
            bulk_callback=bulk_callback,
        )

        # Create observer
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            str(repo_root),
            recursive=True,
        )

    def start(self) -> None:
        """Start watching for file changes."""
        self.event_handler.start()
        self.observer.start()
        logger.info(f"File watcher started for {self.repo_root}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        self.event_handler.stop()
        self.observer.stop()
        self.observer.join(timeout=5.0)
        logger.info("File watcher stopped")

    def get_stats(self) -> Dict[str, int]:
        """Get watcher statistics.

        Returns:
            Dictionary with stats
        """
        return self.event_handler.get_stats()


def get_git_hook_template() -> str:
    """Generate the post-commit git hook template.

    This hook ensures index consistency even if the watcher is dead.
    It runs after each git commit and updates the index for any
    committed Python files.

    Returns:
        Shell script content for post-commit hook
    """
    return """#!/bin/bash
# CodeGrapher post-commit hook
# Updates the index after each commit to ensure consistency

# Get list of committed Python files
CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD | grep '\\.py$' || true)

if [ -z "$CHANGED_FILES" ]; then
    # No Python files changed
    exit 0
fi

# Run codegraph update for changed files
echo "CodeGrapher: Updating index for committed files..."

# Count files
FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l)

if [ "$FILE_COUNT" -gt 20 ]; then
    # Bulk change - trigger full rebuild
    echo "  Bulk change detected ($FILE_COUNT files), triggering full rebuild..."
    codegraph build --full
else:
    # Incremental update
    for file in $CHANGED_FILES; do
        if [ -f "$file" ]; then
            echo "  Updating: $file"
            codegraph update "$file" 2>/dev/null || true
        fi
    done
fi

echo "CodeGrapher: Index update complete"
"""


def install_git_hook(repo_root: Path) -> bool:
    """Install the post-commit git hook.

    Args:
        repo_root: Repository root path

    Returns:
        True if hook was installed, False otherwise
    """
    hooks_dir = repo_root / ".git" / "hooks"
    hook_path = hooks_dir / "post-commit"

    # Check if .git directory exists
    if not hooks_dir.exists():
        logger.warning(f".git/hooks not found in {repo_root}")
        return False

    # Check if hook already exists
    if hook_path.exists():
        logger.warning(f"post-commit hook already exists at {hook_path}")
        # Could backup existing hook here
        return False

    # Write hook
    try:
        hook_content = get_git_hook_template()
        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)  # Make executable
        logger.info(f"Installed post-commit hook at {hook_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to install git hook: {e}")
        return False
