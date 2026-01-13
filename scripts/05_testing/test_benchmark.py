#!/usr/bin/env python3
"""Smoke test for benchmark.py infrastructure.

Tests benchmark utilities without requiring hyperfine.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import benchmark utilities
sys.path.insert(0, str(Path(__file__).parent))
import benchmark


def test_create_test_repo():
    """Test repository creation."""
    print("Testing: create_test_repo()")

    # Create small repo (1000 LOC)
    repo_path = benchmark.create_test_repo(1000)

    try:
        # Verify structure
        assert repo_path.exists(), "Repo directory not created"
        assert (repo_path / "src" / "testpkg").exists(), "Package directory not created"

        # Count LOC
        loc = sum(
            len(f.read_text().splitlines())
            for f in (repo_path / "src").rglob("*.py")
        )

        print(f"  ✓ Created repo with {loc} LOC")

        # Verify files
        py_files = list((repo_path / "src").rglob("*.py"))
        print(f"  ✓ Created {len(py_files)} Python files")

        assert loc >= 700, f"LOC too low: {loc} < 700"
        assert loc <= 1300, f"LOC too high: {loc} > 1300"

        print("  ✅ PASS")
        return True

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(repo_path)


def test_check_dependencies():
    """Test dependency checking."""
    print("\nTesting: check_dependencies()")

    import shutil

    has_hyperfine = shutil.which("hyperfine") is not None
    has_psrecord = shutil.which("psrecord") is not None

    print(f"  - hyperfine: {'✓ installed' if has_hyperfine else '✗ not installed'}")
    print(f"  - psrecord: {'✓ installed' if has_psrecord else '✗ not installed'}")

    if has_hyperfine and has_psrecord:
        print("  ✅ All dependencies available")
        return True
    else:
        print("  ⚠️  Some dependencies missing (this is OK for smoke test)")
        return False


def test_disk_overhead():
    """Test disk overhead measurement (doesn't require hyperfine)."""
    print("\nTesting: benchmark_disk_overhead()")

    repo_path = benchmark.create_test_repo(500)

    try:
        # Initialize git repo
        import subprocess
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Initialize index
        result = subprocess.run(
            [sys.executable, "-m", "codegrapher.cli", "init"],
            cwd=repo_path,
            capture_output=True,
        )

        if result.returncode != 0:
            print(f"  ✗ Init failed: {result.stderr.decode()}")
            return False

        result = subprocess.run(
            [sys.executable, "-m", "codegrapher.cli", "build", "--full"],
            cwd=repo_path,
            capture_output=True,
        )

        if result.returncode != 0:
            print(f"  ✗ Build failed: {result.stderr.decode()}")
            return False

        # Measure disk overhead
        overhead_ratio, passed = benchmark.benchmark_disk_overhead(repo_path)

        if passed:
            print(f"  ✅ PASS: Overhead {overhead_ratio:.2f}× (target: ≤1.5×)")
        else:
            print(f"  ✗ FAIL: Overhead {overhead_ratio:.2f}× (target: ≤1.5×)")

        return passed

    finally:
        import shutil
        shutil.rmtree(repo_path)


def main():
    """Run smoke tests."""
    print("=" * 60)
    print("Benchmark Infrastructure Smoke Test")
    print("=" * 60)
    print()

    results = []

    # Test 1: Repository creation
    try:
        results.append(("create_test_repo", test_create_test_repo()))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("create_test_repo", False))

    # Test 2: Dependencies
    try:
        results.append(("check_dependencies", test_check_dependencies()))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append(("check_dependencies", False))

    # Test 3: Disk overhead (doesn't need hyperfine)
    try:
        results.append(("disk_overhead", test_disk_overhead()))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("disk_overhead", False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print()

    if passed == total:
        print("✅ All smoke tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed (may need full dependencies)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
