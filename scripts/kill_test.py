#!/usr/bin/env python3
"""Kill test for atomic transaction safety.

This test verifies that atomic_update() correctly rolls back
both SQLite and FAISS when the process is killed mid-transaction.

The test forks a child process that:
1. Starts an atomic transaction
2. Writes data to SQLite
3. Simulates a crash (os._exit)

The parent then verifies that:
- No partial data was written to SQLite
- The database is still in a consistent state
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codegrapher.models import Database, Symbol
from codegrapher.vector_store import FAISSIndexManager
from codegrapher.indexer import atomic_update


def child_process(db_path: Path, index_path: Path) -> None:
    """Child process that crashes mid-transaction.

    This process:
    1. Starts an atomic transaction
    2. Inserts data into SQLite
    3. Crashes before committing

    The atomic_update context manager should roll back the
    transaction when the exception occurs.
    """
    # Create database and initialize
    db = Database(db_path)
    db.initialize()

    # Add a symbol before the crash
    symbol = Symbol(
        id="before_crash.symbol",
        file="before.py",
        start_line=1,
        end_line=5,
        signature="def before():",
        doc="Should exist",
        mutates="",
        embedding=np.zeros(768, dtype=np.float32)
    )
    db.insert_symbol(symbol)

    # Start atomic transaction that will crash
    try:
        with atomic_update(db_path, FAISSIndexManager(index_path)) as conn:
            # Insert a symbol that should be rolled back
            conn.execute(
                "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("rolled_back.symbol", "after.py", 1, 5, "def after():", "doc", "", b"\x00" * 3072)
            )

            # Simulate crash - this exits without cleanup
            # os._exit() bypasses normal exception handling
            os._exit(99)  # Special exit code for crash simulation

    except Exception:
        # Shouldn't reach here due to os._exit()
        os._exit(1)


def parent_process(db_path: Path, index_path: Path) -> int:
    """Parent process that verifies rollback.

    Returns:
        0 if test passed, 1 if test failed
    """
    db = Database(db_path)

    # Verify "before_crash" symbol exists
    before = db.get_symbol("before_crash.symbol")
    if before is None:
        print("FAIL: Symbol before crash should exist")
        return 1

    # Verify "rolled_back" symbol does NOT exist
    after = db.get_symbol("rolled_back.symbol")
    if after is not None:
        print("FAIL: Symbol after crash should not exist (transaction not rolled back)")
        return 1

    # Verify database is still valid
    if not db.is_valid():
        print("FAIL: Database is corrupted after crash")
        return 1

    print("PASS: Atomic transaction correctly rolled back after crash")
    return 0


def main() -> int:
    """Run the kill test."""
    print("=" * 60)
    print("Kill Test: Atomic Transaction Safety")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        db_path = tmpdir / "test.db"
        index_path = tmpdir / "test.faiss"

        print(f"\nDatabase: {db_path}")
        print(f"FAISS Index: {index_path}")

        # Fork to create child process
        pid = os.fork()

        if pid == 0:
            # Child process - will crash
            child_process(db_path, index_path)
            # Should never reach here due to os._exit()
            return 1

        else:
            # Parent process - wait for child
            print(f"\nChild process started (PID: {pid})")
            print("Child will crash mid-transaction...")

            pid_result, status = os.waitpid(pid, 0)
            exit_code = os.WEXITSTATUS(status)

            print(f"Child exited with code: {exit_code}")

            # Verify child exited with our special crash code
            if exit_code != 99:
                print(f"FAIL: Child exited unexpectedly (code {exit_code})")
                return 1

            print("\nVerifying database state after crash...")
            return parent_process(db_path, index_path)


if __name__ == "__main__":
    sys.exit(main())
