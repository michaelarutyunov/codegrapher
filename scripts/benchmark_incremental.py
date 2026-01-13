#!/usr/bin/env python3
"""Performance benchmark for incremental indexing.

Verifies Phase 8 acceptance criteria:
- Incremental update <200ms for small changes

This benchmark:
1. Creates a test Python file with multiple symbols
2. Runs initial indexing (cache miss)
3. Makes a small change (docstring)
4. Runs incremental update (cache hit)
5. Reports timing
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codegrapher.indexer import IncrementalIndexer


# Sample source code with multiple symbols
SAMPLE_CODE = """
'''Module docstring.'''
import os
from typing import List


def helper_function(x: int, y: int) -> int:
    '''A helper function for calculations.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    '''
    return x + y


class DataProcessor:
    '''Processes data from various sources.

    This class provides methods for data validation and transformation.
    '''

    def __init__(self, name: str):
        '''Initialize the processor.

        Args:
            name: Name of this processor
        '''
        self.name = name
        self.count = 0

    def process(self, data: List[str]) -> int:
        '''Process a list of data items.

        Args:
            data: List of strings to process

        Returns:
            Number of items processed
        '''
        self.count += len(data)
        return self.count

    def reset(self) -> None:
        '''Reset the counter to zero.'''
        self.count = 0


class Validator:
    '''Validates input data.

    Provides static methods for common validation tasks.
    '''

    @staticmethod
    def is_valid_email(email: str) -> bool:
        '''Check if email is valid.

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        '''
        return "@" in email and "." in email


# Module-level constants
MAX_ITEMS = 1000
DEFAULT_TIMEOUT = 30.0
VERSION = "1.0.0"
"""


MODIFIED_CODE = """
'''Module docstring.'''
import os
from typing import List


def helper_function(x: int, y: int) -> int:
    '''A helper function for calculations.

    This is the MODIFIED docstring.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    '''
    return x + y


class DataProcessor:
    '''Processes data from various sources.

    This class provides methods for data validation and transformation.
    '''

    def __init__(self, name: str):
        '''Initialize the processor.

        Args:
            name: Name of this processor
        '''
        self.name = name
        self.count = 0

    def process(self, data: List[str]) -> int:
        '''Process a list of data items.

        Args:
            data: List of strings to process

        Returns:
            Number of items processed
        '''
        self.count += len(data)
        return self.count

    def reset(self) -> None:
        '''Reset the counter to zero.'''
        self.count = 0


class Validator:
    '''Validates input data.

    Provides static methods for common validation tasks.
    '''

    @staticmethod
    def is_valid_email(email: str) -> bool:
        '''Check if email is valid.

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        '''
        return "@" in email and "." in email


# Module-level constants
MAX_ITEMS = 1000
DEFAULT_TIMEOUT = 30.0
VERSION = "1.0.0"
"""


def format_ms(seconds: float) -> str:
    """Format seconds as milliseconds."""
    ms = seconds * 1000
    return f"{ms:.2f}ms"


def benchmark_incremental_update() -> None:
    """Run the incremental update benchmark."""
    print("=" * 60)
    print("Performance Benchmark: Incremental Indexing")
    print("=" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / "sample.py"
        test_file.write_text(SAMPLE_CODE)

        indexer = IncrementalIndexer(cache_size=50)

        # Benchmark 1: Initial indexing (cache miss)
        print("1. Initial indexing (cache miss)...")
        start = time.perf_counter()
        diff1 = indexer.update_file(test_file, repo_root=tmpdir)
        initial_time = time.perf_counter() - start

        print(f"   - Symbols added: {len(diff1.added)}")
        print(f"   - Time: {format_ms(initial_time)}")
        print()

        # Benchmark 2: Incremental update (cache hit, docstring change)
        print("2. Incremental update (cache hit, docstring change)...")
        test_file.write_text(MODIFIED_CODE)

        start = time.perf_counter()
        diff2 = indexer.update_file(test_file, repo_root=tmpdir)
        incremental_time = time.perf_counter() - start

        print(f"   - Symbols modified: {len(diff2.modified)}")
        print(f"   - Time: {format_ms(incremental_time)}")
        print()

        # Benchmark 3: Incremental update (cache hit, no change)
        print("3. Incremental update (cache hit, no change)...")
        start = time.perf_counter()
        diff3 = indexer.update_file(test_file, repo_root=tmpdir)
        no_change_time = time.perf_counter() - start

        print(f"   - Symbols changed: {len(diff3.added) + len(diff3.deleted) + len(diff3.modified)}")
        print(f"   - Time: {format_ms(no_change_time)}")
        print()

        # Results
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print()

        # Phase 8 requirement: <200ms for incremental updates
        TARGET_MS = 200

        print(f"Target: <{TARGET_MS}ms for incremental updates")
        print()

        if incremental_time < (TARGET_MS / 1000):
            print(f"✓ PASS: Docstring change took {format_ms(incremental_time)}")
        else:
            print(f"✗ FAIL: Docstring change took {format_ms(incremental_time)} (target: {TARGET_MS}ms)")

        if no_change_time < (TARGET_MS / 1000):
            print(f"✓ PASS: No-change update took {format_ms(no_change_time)}")
        else:
            print(f"✗ FAIL: No-change update took {format_ms(no_change_time)} (target: {TARGET_MS}ms)")

        print()
        print("Summary:")
        print(f"  - Initial indexing: {format_ms(initial_time)}")
        print(f"  - Docstring change: {format_ms(incremental_time)}")
        print(f"  - No-change update: {format_ms(no_change_time)}")

        # Exit with appropriate code
        if incremental_time < (TARGET_MS / 1000) and no_change_time < (TARGET_MS / 1000):
            print()
            print("All performance targets met!")
            return 0
        else:
            print()
            print("Some performance targets NOT met!")
            return 1


if __name__ == "__main__":
    sys.exit(benchmark_incremental_update())
