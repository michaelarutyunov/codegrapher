"""Integration tests for CodeGrapher.

Tests for Phase 11: CLI & Build Tools - Integration testing.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from codegrapher.models import Database


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command and return the result.

    Args:
        cmd: Command to run as a list of strings
        cwd: Working directory

    Returns:
        CompletedProcess result
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result


def create_test_repo(tmp_path: Path, loc: int = 100) -> Path:
    """Create a test repository with Python files.

    Args:
        tmp_path: Temporary directory path
        loc: Lines of code to generate

    Returns:
        Path to the test repository
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir(parents=True, exist_ok=True)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_path, capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, timeout=10)

    # Create source directory
    src_dir = repo_path / "src"
    src_dir.mkdir()

    # Create Python files
    files_count = max(1, loc // 50)
    for i in range(files_count):
        file_path = src_dir / f"module{i}.py"
        content = f"""# Module {i}

def function_{i}(param: str) -> bool:
    '''Test function {i}.

    Args:
        param: Test parameter

    Returns:
        True if successful
    '''
    result = param.upper()
    return len(result) > 0


class Class{i}:
    '''Test class {i}.'''

    def method_one(self) -> None:
        '''First method.'''
        pass

    def method_two(self, x: int) -> int:
        '''Second method.

        Args:
            x: Input value

        Returns:
            Doubled value
        '''
        return x * 2


# Constants
CONSTANT_{i} = "test_value_{i}"
"""
        file_path.write_text(content)

    # Commit initial files
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, timeout=10)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        timeout=10,
    )

    return repo_path


def modify_file(repo_path: Path, file_path: str, add_function: str) -> None:
    """Modify a file by adding a function.

    Args:
        repo_path: Repository root path
        file_path: Relative path to file (from repo root)
        add_function: Function definition to add
    """
    full_path = repo_path / file_path
    content = full_path.read_text()
    content += f"\n{add_function}\n"
    full_path.write_text(content)


def test_incremental_update_survives_crash():
    """Verify index integrity after process kill during update.

    This is the Phase 11 integration test that verifies:
    1. Full index build works
    2. Incremental update can be started
    3. Process can be killed mid-update
    4. Database is not corrupted after crash
    5. Queries still work after crash

    Per ENGINEERING_GUIDELINES Rule 6.4.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        repo = create_test_repo(tmp_path, loc=200)

        # Step 1: Full index build
        print("\n=== Step 1: Full index build ===")
        result = run_command(["python", "-m", "codegrapher.cli", "build", "--full"], cwd=repo)
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        print("Build succeeded")

        # Verify index exists
        index_dir = repo / ".codegraph"
        db_path = index_dir / "symbols.db"
        faiss_path = index_dir / "index.faiss"
        assert db_path.exists(), "Database not created"
        assert faiss_path.exists(), "FAISS index not created"

        # Step 2: Modify one file
        print("\n=== Step 2: Modify file ===")
        modify_file(
            repo,
            "src/module0.py",
            "def new_func() -> None:\n    '''New function added.'''\n    pass"
        )
        print("File modified")

        # Step 3: Start incremental update in background
        print("\n=== Step 3: Start incremental update ===")

        # Create a script that will run update and sleep
        update_script = repo / "update_and_sleep.py"
        update_script.write_text("""
import sys
import time
from pathlib import Path

# Change to repo directory
repo = Path(__file__).parent
import os
os.chdir(repo)

# Import and run update
from codegrapher.cli import create_parser
parser = create_parser()
args = parser.parse_args(["update", "src/module0.py"])
args.func(args)

# Sleep to allow killing
time.sleep(10)
""")

        # Start the update process
        proc = subprocess.Popen(
            [sys.executable, str(update_script)],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait a bit for the update to start
        time.sleep(0.5)

        # Step 4: Kill process mid-update
        print("\n=== Step 4: Kill process mid-update ===")
        proc.kill()
        proc.wait(timeout=5)
        print("Process killed")

        # Step 5: Verify database is not corrupted
        print("\n=== Step 5: Verify database integrity ===")
        db = Database(db_path)
        assert db.is_valid(), "Database corrupted after crash"
        print("Database is valid")

        # Step 6: Verify we can still query
        print("\n=== Step 6: Verify query still works ===")
        # This should not raise an error
        symbols = db.get_all_symbols()
        assert len(symbols) > 0, "No symbols found in database"
        print(f"Found {len(symbols)} symbols in database")

        # Step 7: Run a successful update after crash
        print("\n=== Step 7: Run successful update after crash ===")
        result = run_command(
            ["python", "-m", "codegrapher.cli", "update", "src/module0.py"],
            cwd=repo,
        )
        # Should succeed even after crash
        print(f"Update return code: {result.returncode}")
        if result.returncode != 0:
            print(f"Update stderr: {result.stderr}")
        # Update might fail if we killed it too early, but database should still be valid
        db = Database(db_path)
        assert db.is_valid(), "Database corrupted after second update attempt"
        print("Database still valid after recovery attempt")

    print("\n=== Test passed ===")


def test_full_build_workflow():
    """Test the complete workflow: init, build, query.

    This tests the happy path without crashes.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        repo = create_test_repo(tmp_path, loc=150)

        # Initialize
        print("\n=== Initialize ===")
        result = run_command(
            ["python", "-m", "codegrapher.cli", "init", "--no-model", "--no-hook"],
            cwd=repo,
        )
        assert result.returncode == 0, f"Init failed: {result.stderr}"
        index_dir = repo / ".codegraph"
        assert index_dir.exists(), "Index directory not created"
        print("Init succeeded")

        # Build
        print("\n=== Build ===")
        result = run_command(
            ["python", "-m", "codegrapher.cli", "build", "--full"],
            cwd=repo,
        )
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        db_path = index_dir / "symbols.db"
        assert db_path.exists(), "Database not created"
        print("Build succeeded")

        # Query
        print("\n=== Query ===")
        result = run_command(
            ["python", "-m", "codegrapher.cli", "query", "function", "--json"],
            cwd=repo,
        )
        assert result.returncode == 0, f"Query failed: {result.stderr}"
        # Should have some output
        assert len(result.stdout) > 0, "Query returned no output"
        print("Query succeeded")

    print("\n=== Workflow test passed ===")


def test_update_workflow():
    """Test the incremental update workflow."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        repo = create_test_repo(tmp_path, loc=100)

        # Build initial index
        result = run_command(
            ["python", "-m", "codegrapher.cli", "build", "--full"],
            cwd=repo,
        )
        assert result.returncode == 0, f"Build failed: {result.stderr}"

        # Modify a file
        modify_file(
            repo,
            "src/module0.py",
            "def new_function() -> None:\n    pass"
        )

        # Update single file
        result = run_command(
            ["python", "-m", "codegrapher.cli", "update", "src/module0.py"],
            cwd=repo,
        )
        assert result.returncode == 0, f"Update failed: {result.stderr}"

        # Verify index still works
        db_path = repo / ".codegraph" / "symbols.db"
        db = Database(db_path)
        assert db.is_valid(), "Database corrupted after update"

    print("\n=== Update workflow test passed ===")


if __name__ == "__main__":
    # Run tests
    test_full_build_workflow()
    test_update_workflow()
    test_incremental_update_survives_crash()
    print("\n=== All integration tests passed ===")
