#!/usr/bin/env python3
"""Add query_category field to ground truth test cases.

This script categorizes test cases based on their query_terms:
- symbol: Query contains class/function names (CamelCase, snake_case)
- description: Natural language describing behavior
- error: Contains error keywords (KeyError, Exception, crash)
- agent-error: Full traceback format (file paths, line numbers)
- dependency: Refactoring/structural queries ("find callers", "what uses")
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def count_symbol_terms(query_terms: List[str]) -> int:
    """Count CamelCase and snake_case identifiers in query terms.

    Args:
        query_terms: List of query term strings

    Returns:
        Count of symbol-like terms
    """
    count = 0
    for term in query_terms:
        # CamelCase pattern (e.g., UserAuth, StreamContext)
        if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)+$", term):
            count += 1
        # snake_case pattern (e.g., import_module, sys_modules)
        elif re.match(r"^[a-z_]+_[a-z_]+$", term):
            count += 1
        # Module.attribute pattern (e.g., pytest.approx, sys.modules)
        elif re.match(r"^[a-z_]+\.[a-z_]+$", term):
            count += 1

    return count


def has_error_keywords(query_terms: List[str]) -> bool:
    """Check if query contains error-related keywords.

    Args:
        query_terms: List of query term strings

    Returns:
        True if error keywords found
    """
    error_keywords = [
        "error", "exception", "crash", "fails", "failure", "bug",
        "keyerror", "attributeerror", "valueerror", "typeerror",
        "importerror", "runtimeerror"
    ]

    text = " ".join(query_terms).lower()

    for keyword in error_keywords:
        if keyword in text:
            return True

    # Check for specific Error/Exception class names
    for term in query_terms:
        if re.search(r"(Error|Exception)$", term, re.IGNORECASE):
            return True

    return False


def has_dependency_keywords(query_terms: List[str], description: str) -> bool:
    """Check if query is about dependencies/refactoring.

    Args:
        query_terms: List of query term strings
        description: Task description

    Returns:
        True if dependency/refactoring query
    """
    dependency_keywords = [
        "callers", "uses", "imports", "depends", "dependency",
        "refactor", "rename", "move", "restructure"
    ]

    text = " ".join(query_terms + [description]).lower()

    for keyword in dependency_keywords:
        if keyword in text:
            return True

    return False


def has_traceback_format(query_terms: List[str]) -> bool:
    """Check if query contains traceback-like format.

    Args:
        query_terms: List of query term strings

    Returns:
        True if looks like a traceback
    """
    text = " ".join(query_terms)

    # Look for file path patterns
    if re.search(r"/[a-z_/]+\.py:\d+", text):
        return True

    # Look for "Traceback" keyword
    if "Traceback" in text or "traceback" in text:
        return True

    return False


def categorize_test_case(task: Dict[str, Any]) -> str:
    """Categorize a test case based on its query_terms and description.

    Args:
        task: Test case dict

    Returns:
        Category string: "symbol", "description", "error", "agent-error", or "dependency"
    """
    query_terms = task.get("query_terms", [])
    description = task.get("description", "")

    # Check for traceback format first (most specific)
    if has_traceback_format(query_terms):
        return "agent-error"

    # Check for dependency queries
    if has_dependency_keywords(query_terms, description):
        return "dependency"

    # Check for error keywords
    has_errors = has_error_keywords(query_terms)

    # Count symbol-like terms
    symbol_count = count_symbol_terms(query_terms)

    # Categorization logic
    if has_errors and symbol_count >= 2:
        # Error with specific symbols (e.g., "pytest.approx FloatOperation crash")
        return "error"
    elif symbol_count >= 3:
        # Mostly symbols (e.g., "import_module_using_spec sys.modules importlib")
        return "symbol"
    elif has_errors:
        # Error without many symbols
        return "error"
    elif symbol_count >= 1:
        # Some symbols, not dominated by natural language
        return "symbol"
    else:
        # Mostly natural language descriptions
        return "description"


def add_categories(input_path: Path, output_path: Path) -> None:
    """Add query_category field to all tasks.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file
    """
    with open(input_path, "r") as f:
        tasks = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"Loaded {len(tasks)} tasks from {input_path}")
    logger.info("")

    # Category counts for summary
    category_counts: Dict[str, int] = {}

    for i, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        category = categorize_test_case(task)

        task["query_category"] = category
        category_counts[category] = category_counts.get(category, 0) + 1

        logger.info(
            f"[{i:2d}/{len(tasks)}] {task_id}: {category:12s} | "
            f"{', '.join(task['query_terms'][:3])}"
        )

    # Write output
    with open(output_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info("")
    logger.info("Category distribution:")

    total = len(tasks)
    for category in sorted(category_counts.keys()):
        count = category_counts[category]
        percentage = (count / total) * 100
        logger.info(f"  {category:12s}: {count:2d} ({percentage:5.1f}%)")

    logger.info("")
    logger.info(f"✅ Categorized tasks saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add query_category field to ground truth test cases"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "ground_truth_categorized.jsonl",
        help="Output JSONL file"
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    add_categories(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
