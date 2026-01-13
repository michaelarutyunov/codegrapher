#!/usr/bin/env python3
"""Performance benchmarking for CodeGrapher v1.0.

This script verifies all performance targets from PRD Section 10:
- Cold start: ≤ 2s
- Query latency: ≤ 500ms
- Incremental index (1 file): ≤ 1s
- Full index 30k LOC: ≤ 30s
- RAM idle: ≤ 500MB
- Disk overhead: ≤ 1.5× repo size

Usage:
    python scripts/benchmark.py [--repo-path PATH] [--output FILE]

Requirements:
    - hyperfine (install: cargo install hyperfine or apt-get install hyperfine)
    - psrecord (install: pip install psrecord)
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


def check_dependencies() -> None:
    """Verify required tools are installed."""
    required = {
        "hyperfine": "Install: cargo install hyperfine OR apt-get install hyperfine",
        "psrecord": "Install: pip install psrecord",
    }

    missing = []
    for tool, install_cmd in required.items():
        # Check system PATH
        if shutil.which(tool) is None:
            # Also check .venv/bin for tools like psrecord
            venv_tool = Path.cwd() / ".venv" / "bin" / tool
            if not venv_tool.exists():
                missing.append(f"  - {tool}: {install_cmd}")

    if missing:
        print("❌ Missing required tools:")
        print("\n".join(missing))
        sys.exit(1)


def get_tool_path(tool: str) -> str:
    """Get path to a tool, checking both system PATH and .venv/bin.

    Args:
        tool: Tool name (e.g., "psrecord", "hyperfine")

    Returns:
        Full path to the tool
    """
    # Check system PATH first
    system_path = shutil.which(tool)
    if system_path:
        return system_path

    # Check .venv/bin
    venv_path = Path.cwd() / ".venv" / "bin" / tool
    if venv_path.exists():
        return str(venv_path)

    # Fallback to tool name (will likely fail)
    return tool


def create_test_repo(loc_target: int = 30000) -> Path:
    """Create a test repository with approximately LOC_TARGET lines of code.

    Args:
        loc_target: Target lines of code (default: 30000)

    Returns:
        Path to the created test repository
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="codegraph_bench_"))

    # Calculate number of files to create
    # Each file will have ~200 LOC
    lines_per_file = 200
    num_files = loc_target // lines_per_file

    print(f"Creating test repository with ~{loc_target} LOC ({num_files} files)...")

    # Create package structure
    pkg_dir = tmpdir / "src" / "testpkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text('"""Test package."""\n')

    # Template for a Python file (~200 LOC)
    file_template = '''"""Module {module_num}: Auto-generated for benchmarking."""

import os
import sys
from typing import List, Dict, Optional


class DataProcessor{module_num}:
    """Processes data for module {module_num}.

    This class handles various data transformation operations.
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        """Initialize the processor.

        Args:
            name: Name of this processor
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {{}}
        self.count = 0
        self.cache = {{}}

    def process(self, data: List[str]) -> Dict[str, int]:
        """Process a list of data items.

        Args:
            data: List of strings to process

        Returns:
            Dictionary mapping items to their counts
        """
        result = {{}}
        for item in data:
            if item in self.cache:
                result[item] = self.cache[item]
            else:
                processed = self._transform(item)
                self.cache[item] = processed
                result[item] = processed
        self.count += len(data)
        return result

    def _transform(self, item: str) -> int:
        """Transform a single item.

        Args:
            item: Item to transform

        Returns:
            Transformed value
        """
        return len(item) * 2 + self.count

    def reset(self) -> None:
        """Reset the processor state."""
        self.count = 0
        self.cache.clear()


class Validator{module_num}:
    """Validates input data for module {module_num}.

    Provides static methods for common validation tasks.
    """

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Check if email is valid.

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        """
        return "@" in email and "." in email

    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """Check if phone number is valid.

        Args:
            phone: Phone number to validate

        Returns:
            True if valid, False otherwise
        """
        digits = "".join(c for c in phone if c.isdigit())
        return len(digits) >= 10

    @classmethod
    def validate_all(cls, data: Dict[str, str]) -> Dict[str, bool]:
        """Validate all fields in data dictionary.

        Args:
            data: Dictionary of field values

        Returns:
            Dictionary mapping field names to validation results
        """
        results = {{}}
        if "email" in data:
            results["email"] = cls.is_valid_email(data["email"])
        if "phone" in data:
            results["phone"] = cls.is_valid_phone(data["phone"])
        return results


def helper_function_{module_num}(x: int, y: int) -> int:
    """A helper function for calculations in module {module_num}.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    return x + y


def advanced_calculator_{module_num}(values: List[int], operation: str = "sum") -> float:
    """Perform calculations on a list of values.

    Args:
        values: List of integers
        operation: Operation to perform (sum, mean, max, min)

    Returns:
        Result of the calculation

    Raises:
        ValueError: If operation is not recognized
    """
    if not values:
        return 0.0

    if operation == "sum":
        return float(sum(values))
    elif operation == "mean":
        return sum(values) / len(values)
    elif operation == "max":
        return float(max(values))
    elif operation == "min":
        return float(min(values))
    else:
        raise ValueError(f"Unknown operation: {{operation}}")


# Module-level constants
MAX_ITEMS_{module_num} = 1000
DEFAULT_TIMEOUT_{module_num} = 30.0
VERSION_{module_num} = "1.0.0"
'''

    # Create files
    for i in range(num_files):
        module_file = pkg_dir / f"module_{i:04d}.py"
        module_file.write_text(file_template.format(module_num=i))

    # Count actual LOC
    actual_loc = sum(
        len(f.read_text().splitlines())
        for f in pkg_dir.rglob("*.py")
    )

    print(f"✓ Created {num_files} files with {actual_loc} LOC at {tmpdir}")
    return tmpdir


def benchmark_cold_start(repo_path: Path) -> Tuple[float, bool]:
    """Benchmark cold start time.

    Args:
        repo_path: Path to test repository

    Returns:
        Tuple of (time in seconds, pass/fail boolean)
    """
    print("\n" + "="*60)
    print("Benchmarking: Cold Start")
    print("="*60)

    # Run codegraph init and build to set up the index
    subprocess.run(
        ["python", "-m", "codegrapher.cli", "init"],
        cwd=repo_path,
        capture_output=True,
        check=False,
    )

    subprocess.run(
        ["python", "-m", "codegrapher.cli", "build", "--full"],
        cwd=repo_path,
        capture_output=True,
        check=False,
    )

    # Measure server startup time
    cmd = [
        "python", "-m", "codegrapher.server"
    ]

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready (max 5 seconds)
    for _ in range(50):
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    elapsed = time.perf_counter() - start

    # Kill the server
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

    target = 2.0
    passed = elapsed <= target

    print(f"Target: ≤ {target}s")
    print(f"Actual: {elapsed:.2f}s")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return elapsed, passed


def benchmark_query_latency(repo_path: Path) -> Tuple[float, bool]:
    """Benchmark query latency using hyperfine.

    Args:
        repo_path: Path to test repository

    Returns:
        Tuple of (mean time in ms, pass/fail boolean)
    """
    print("\n" + "="*60)
    print("Benchmarking: Query Latency")
    print("="*60)

    # Run hyperfine with query command
    cmd = [
        get_tool_path("hyperfine"),
        "--warmup", "3",
        "--min-runs", "10",
        "--export-json", "/tmp/codegraph_query_bench.json",
        f"cd {repo_path} && python -m codegrapher.cli query 'DataProcessor'"
    ]

    result = subprocess.run(
        cmd,
        shell=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ hyperfine failed: {result.stderr}")
        return 0.0, False

    # Parse JSON output
    with open("/tmp/codegraph_query_bench.json") as f:
        data = json.load(f)

    # Extract mean time in milliseconds
    mean_ms = data["results"][0]["mean"] * 1000

    target_ms = 500
    passed = mean_ms <= target_ms

    print(f"Target: ≤ {target_ms}ms")
    print(f"Actual: {mean_ms:.0f}ms")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return mean_ms, passed


def benchmark_incremental_update(repo_path: Path) -> Tuple[float, bool]:
    """Benchmark incremental update time using hyperfine.

    Args:
        repo_path: Path to test repository

    Returns:
        Tuple of (mean time in seconds, pass/fail boolean)
    """
    print("\n" + "="*60)
    print("Benchmarking: Incremental Update (1 file)")
    print("="*60)

    # Pick a file to modify
    test_files = list((repo_path / "src" / "testpkg").glob("module_*.py"))
    if not test_files:
        print("❌ No test files found")
        return 0.0, False

    test_file = test_files[0]
    original_content = test_file.read_text()

    # Create a modified version with a docstring change
    modified_content = original_content.replace(
        "Auto-generated for benchmarking.",
        "Auto-generated for benchmarking. MODIFIED."
    )

    # Write a shell script that modifies the file and runs update
    script_path = repo_path / "bench_update.sh"
    script_path.write_text(f"""#!/bin/bash
cat > {test_file} << 'EOF'
{modified_content}
EOF
cd {repo_path} && python -m codegrapher.cli update {test_file}
cat > {test_file} << 'EOF'
{original_content}
EOF
""")
    script_path.chmod(0o755)

    # Run hyperfine
    cmd = [
        get_tool_path("hyperfine"),
        "--warmup", "2",
        "--min-runs", "5",
        "--export-json", "/tmp/codegraph_update_bench.json",
        str(script_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ hyperfine failed: {result.stderr}")
        return 0.0, False

    # Parse JSON output
    with open("/tmp/codegraph_update_bench.json") as f:
        data = json.load(f)

    # Extract mean time in seconds
    mean_s = data["results"][0]["mean"]

    target_s = 1.0
    passed = mean_s <= target_s

    print(f"Target: ≤ {target_s}s")
    print(f"Actual: {mean_s:.2f}s")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")

    # Clean up
    script_path.unlink()

    return mean_s, passed


def benchmark_full_build(repo_path: Path) -> Tuple[float, bool]:
    """Benchmark full index build time using hyperfine.

    Args:
        repo_path: Path to test repository

    Returns:
        Tuple of (mean time in seconds, pass/fail boolean)
    """
    print("\n" + "="*60)
    print("Benchmarking: Full Index Build")
    print("="*60)

    # Count LOC
    loc = sum(
        len(f.read_text().splitlines())
        for f in (repo_path / "src").rglob("*.py")
    )

    print(f"Repository size: {loc} LOC")

    # Create a script that clears index and rebuilds
    script_path = repo_path / "bench_build.sh"
    script_path.write_text(f"""#!/bin/bash
rm -rf {repo_path}/.codegraph
cd {repo_path} && python -m codegrapher.cli init
cd {repo_path} && python -m codegrapher.cli build --full
""")
    script_path.chmod(0o755)

    # Run hyperfine
    cmd = [
        get_tool_path("hyperfine"),
        "--warmup", "1",
        "--min-runs", "3",
        "--export-json", "/tmp/codegraph_build_bench.json",
        str(script_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ hyperfine failed: {result.stderr}")
        return 0.0, False

    # Parse JSON output
    with open("/tmp/codegraph_build_bench.json") as f:
        data = json.load(f)

    # Extract mean time in seconds
    mean_s = data["results"][0]["mean"]

    # Scale target based on actual LOC
    # PRD target: 30s for 30k LOC = 1ms per LOC
    target_s = (loc / 30000) * 30

    passed = mean_s <= target_s

    print(f"Target: ≤ {target_s:.0f}s (scaled for {loc} LOC)")
    print(f"Actual: {mean_s:.0f}s")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")

    # Clean up
    script_path.unlink()

    return mean_s, passed


def benchmark_ram_usage(repo_path: Path) -> Tuple[float, bool]:
    """Benchmark RAM usage using psrecord.

    Args:
        repo_path: Path to test repository

    Returns:
        Tuple of (max RAM in MB, pass/fail boolean)
    """
    print("\n" + "="*60)
    print("Benchmarking: RAM Usage (Idle)")
    print("="*60)

    # Start server in background
    proc = subprocess.Popen(
        ["python", "-m", "codegrapher.server"],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to stabilize
    time.sleep(2)

    # Record memory for 10 seconds
    log_file = "/tmp/codegraph_memory.txt"
    psrecord_proc = subprocess.Popen(
        [
            get_tool_path("psrecord"),
            str(proc.pid),
            "--duration", "10",
            "--interval", "0.5",
            "--log", log_file,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for psrecord to finish
    psrecord_proc.wait()

    # Kill the server
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Parse memory log
    max_ram_mb = 0.0
    with open(log_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                # Second column is RAM in MB
                ram_mb = float(parts[1])
                max_ram_mb = max(max_ram_mb, ram_mb)

    target_mb = 500
    passed = max_ram_mb <= target_mb

    print(f"Target: ≤ {target_mb}MB")
    print(f"Actual: {max_ram_mb:.0f}MB")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return max_ram_mb, passed


def benchmark_disk_overhead(repo_path: Path) -> Tuple[float, bool]:
    """Benchmark disk overhead using du.

    Args:
        repo_path: Path to test repository

    Returns:
        Tuple of (overhead ratio, pass/fail boolean)
    """
    print("\n" + "="*60)
    print("Benchmarking: Disk Overhead")
    print("="*60)

    # Measure repository size (source code only)
    result = subprocess.run(
        ["du", "-sb", str(repo_path / "src")],
        capture_output=True,
        text=True,
        check=True,
    )
    repo_size_bytes = int(result.stdout.split()[0])

    # Measure index size
    index_path = repo_path / ".codegraph"
    if not index_path.exists():
        print("❌ Index not found")
        return 0.0, False

    result = subprocess.run(
        ["du", "-sb", str(index_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    index_size_bytes = int(result.stdout.split()[0])

    # Calculate overhead ratio
    overhead_ratio = index_size_bytes / repo_size_bytes

    target_ratio = 1.5
    passed = overhead_ratio <= target_ratio

    print(f"Repository size: {repo_size_bytes / 1024 / 1024:.1f}MB")
    print(f"Index size: {index_size_bytes / 1024 / 1024:.1f}MB")
    print(f"Target: ≤ {target_ratio}× repo size")
    print(f"Actual: {overhead_ratio:.2f}×")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")

    return overhead_ratio, passed


def generate_report(results: Dict[str, Tuple[float, bool]], output_file: Optional[str] = None) -> None:
    """Generate markdown report of benchmark results.

    Args:
        results: Dictionary mapping metric names to (value, passed) tuples
        output_file: Optional path to write report to
    """
    report_lines = [
        "# CodeGrapher v1.0 Performance Benchmark Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Results",
        "",
        "| Scenario | Target | Actual | Status |",
        "|----------|--------|--------|--------|",
    ]

    # Format each result
    metric_formats = {
        "cold_start": ("Cold start", "≤ 2s", lambda v: f"{v:.2f}s"),
        "query_latency": ("Query latency", "≤ 500ms", lambda v: f"{v:.0f}ms"),
        "incremental_update": ("Incremental update (1 file)", "≤ 1s", lambda v: f"{v:.2f}s"),
        "full_build": ("Full index build", "≤ 30s (30k LOC)", lambda v: f"{v:.0f}s"),
        "ram_usage": ("RAM idle", "≤ 500MB", lambda v: f"{v:.0f}MB"),
        "disk_overhead": ("Disk overhead", "≤ 1.5× repo size", lambda v: f"{v:.2f}×"),
    }

    all_passed = True
    for key, (label, target, formatter) in metric_formats.items():
        if key in results:
            value, passed = results[key]
            all_passed = all_passed and passed
            status = "✅" if passed else "❌"
            report_lines.append(
                f"| {label} | {target} | {formatter(value)} | {status} |"
            )

    report_lines.extend([
        "",
        "## Summary",
        "",
    ])

    if all_passed:
        report_lines.append("✅ **All performance targets met!**")
    else:
        report_lines.append("❌ **Some performance targets NOT met!**")
        report_lines.append("")
        report_lines.append("Failed metrics:")
        for key, (label, _, _) in metric_formats.items():
            if key in results:
                _, passed = results[key]
                if not passed:
                    report_lines.append(f"- {label}")

    report = "\n".join(report_lines)

    # Print to stdout
    print("\n" + "="*60)
    print(report)
    print("="*60)

    # Optionally write to file
    if output_file:
        Path(output_file).write_text(report)
        print(f"\n✓ Report written to {output_file}")


def main() -> int:
    """Run all benchmarks and generate report."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for CodeGrapher v1.0"
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        help="Path to existing test repository (default: create temporary repo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write markdown report",
    )
    parser.add_argument(
        "--loc",
        type=int,
        default=30000,
        help="Lines of code for test repository (default: 30000)",
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Create or use existing test repository
    if args.repo_path:
        repo_path = args.repo_path
        print(f"Using existing repository: {repo_path}")
    else:
        repo_path = create_test_repo(args.loc)

    # Run benchmarks
    results = {}

    try:
        results["cold_start"] = benchmark_cold_start(repo_path)
        results["query_latency"] = benchmark_query_latency(repo_path)
        results["incremental_update"] = benchmark_incremental_update(repo_path)
        results["full_build"] = benchmark_full_build(repo_path)
        results["ram_usage"] = benchmark_ram_usage(repo_path)
        results["disk_overhead"] = benchmark_disk_overhead(repo_path)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up temporary repository
        if not args.repo_path:
            print(f"\nCleaning up temporary repository: {repo_path}")
            shutil.rmtree(repo_path)

    # Generate report
    generate_report(results, args.output)

    # Exit with appropriate code
    all_passed = all(passed for _, passed in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
