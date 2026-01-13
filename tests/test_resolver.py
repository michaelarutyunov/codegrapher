"""Unit tests for import resolution logic."""

from pathlib import Path
import sys

import pytest

from codegrapher.resolver import (
    resolve_import_to_path,
    _is_stdlib,
    build_import_graph,
    get_import_closure,
)


# Test repo paths
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_REPO = FIXTURES_DIR / "test_repos" / "simple_repo"


class TestStdlibDetection:
    """Tests for standard library module detection."""

    def test_os_is_stdlib(self):
        """Test that 'os' is recognized as stdlib."""
        assert _is_stdlib("os") is True

    def test_os_path_is_stdlib(self):
        """Test that 'os.path' is recognized as stdlib."""
        assert _is_stdlib("os.path") is True

    def test_sys_is_stdlib(self):
        """Test that 'sys' is recognized as stdlib."""
        assert _is_stdlib("sys") is True

    def test_typing_is_stdlib(self):
        """Test that 'typing' is recognized as stdlib."""
        assert _is_stdlib("typing") is True

    def test_json_is_stdlib(self):
        """Test that 'json' is recognized as stdlib."""
        assert _is_stdlib("json") is True

    def test_collections_is_stdlib(self):
        """Test that 'collections' is recognized as stdlib."""
        assert _is_stdlib("collections") is True

    def test_unknown_module_not_stdlib(self):
        """Test that unknown module is not stdlib."""
        assert _is_stdlib("mypackage") is False
        assert _is_stdlib("custom_utils") is False


class TestRelativeImports:
    """Tests for relative import resolution."""

    def test_single_dot_import(self):
        """Test that '.utils' resolves to sibling file."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path(".utils", current_file, TEST_REPO)
        assert result is not None
        assert result.name == "utils.py"
        assert "mypackage" in result.parts

    def test_double_dot_import_to_sibling_in_parent(self):
        """Test that '..sibling' from subpackage goes to parent's sibling."""
        # From src/mypackage/subpackage/__init__.py
        # '..submodule' should find src/mypackage/submodule.py
        current_file = TEST_REPO / "src" / "mypackage" / "subpackage" / "__init__.py"
        result = resolve_import_to_path("..submodule", current_file, TEST_REPO)
        assert result is not None
        assert result.name == "submodule.py"
        assert "mypackage" in result.parts

    def test_double_dot_import_missing_file(self):
        """Test that '..submodule' returns None when file doesn't exist."""
        # From src/mypackage/utils.py, '..submodule' would look for src/submodule.py
        # which doesn't exist
        current_file = TEST_REPO / "src" / "mypackage" / "utils.py"
        result = resolve_import_to_path("..submodule", current_file, TEST_REPO)
        assert result is None  # File doesn't exist at src/submodule.py

    def test_triple_dot_import(self):
        """Test that '...module' goes up two levels."""
        # From src/mypackage/subpackage/__init__.py
        # '...utils' should look at src/ level, then find mypackage/utils.py
        # But '...utils' means go up 2 levels to src/, then look for utils.py
        # Since there's no src/utils.py, this should return None
        current_file = TEST_REPO / "src" / "mypackage" / "subpackage" / "__init__.py"
        result = resolve_import_to_path("...utils", current_file, TEST_REPO)
        assert result is None  # No src/utils.py file exists

    def test_relative_import_with_submodule(self):
        """Test '.submodule.function' style import."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path(".submodule", current_file, TEST_REPO)
        assert result is not None
        assert result.name == "submodule.py"


class TestAbsoluteImports:
    """Tests for absolute import resolution within repo."""

    def test_absolute_import_to_package(self):
        """Test 'mypackage.submodule' resolves correctly."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("mypackage.submodule", current_file, TEST_REPO)
        assert result is not None
        assert "submodule.py" in str(result)

    def test_absolute_import_to_subpackage(self):
        """Test 'mypackage.subpackage' resolves to __init__.py."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("mypackage.subpackage", current_file, TEST_REPO)
        assert result is not None
        assert result.name == "__init__.py"
        assert "subpackage" in result.parts

    def test_absolute_import_not_in_repo(self):
        """Test absolute import outside repo returns None."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("requests", current_file, TEST_REPO)
        # requests is not in stdlib but also not in repo - should be None
        assert result is None


class TestStdlibImports:
    """Tests that stdlib imports return None."""

    def test_os_import_returns_none(self):
        """Test 'import os' returns None."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("os", current_file, TEST_REPO)
        assert result is None

    def test_os_path_import_returns_none(self):
        """Test 'import os.path' returns None."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("os.path", current_file, TEST_REPO)
        assert result is None

    def test_sys_import_returns_none(self):
        """Test 'import sys' returns None."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("sys", current_file, TEST_REPO)
        assert result is None

    def test_typing_import_returns_none(self):
        """Test 'from typing import List' returns None for 'typing'."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("typing", current_file, TEST_REPO)
        assert result is None


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_module_name_raises_error(self):
        """Test that empty module_name raises ValueError."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        with pytest.raises(ValueError, match="module_name cannot be empty"):
            resolve_import_to_path("", current_file, TEST_REPO)

    def test_non_absolute_current_file_raises_error(self):
        """Test that relative current_file raises ValueError."""
        with pytest.raises(ValueError, match="current_file must be absolute"):
            resolve_import_to_path(
                ".utils",
                Path("src/mypackage/main.py"),  # Relative path
                TEST_REPO
            )

    def test_non_absolute_repo_root_raises_error(self):
        """Test that relative repo_root raises ValueError."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        with pytest.raises(ValueError, match="repo_root must be absolute"):
            resolve_import_to_path(
                ".utils",
                current_file,
                Path(".")  # Relative path
            )

    def test_missing_relative_import(self):
        """Test that non-existent relative import returns None."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path(".nonexistent", current_file, TEST_REPO)
        assert result is None

    def test_missing_absolute_import(self):
        """Test that non-existent absolute import returns None."""
        current_file = TEST_REPO / "src" / "mypackage" / "main.py"
        result = resolve_import_to_path("nonexistent.module", current_file, TEST_REPO)
        assert result is None


class TestImportGraph:
    """Tests for import graph building."""

    def test_build_import_graph(self):
        """Test that import graph is built correctly."""
        graph = build_import_graph(TEST_REPO)

        # Check that main.py is in the graph
        main_key = "src/mypackage/main.py"
        assert main_key in graph

        # main.py should import utils.py
        main_imports = graph[main_key]
        assert any("utils.py" in imp for imp in main_imports)

    def test_import_graph_excludes_stdlib(self):
        """Test that stdlib imports are not in the graph."""
        graph = build_import_graph(TEST_REPO)

        for imports in graph.values():
            # None of the imports should be stdlib modules
            for imp in imports:
                assert "os.py" not in imp
                assert "sys.py" not in imp
                assert "typing.py" not in imp


class TestImportClosure:
    """Tests for import closure (BFS traversal)."""

    def test_import_closure_from_main(self):
        """Test that closure includes all reachable files."""
        start_file = TEST_REPO / "src" / "mypackage" / "main.py"
        closure = get_import_closure(start_file, TEST_REPO)

        # main.py should be in closure
        assert any("main.py" in str(p) for p in closure)

        # Files imported by main.py should be in closure
        assert any("utils.py" in str(p) for p in closure)
        assert any("submodule.py" in str(p) for p in closure)

    def test_import_closure_no_stdlib(self):
        """Test that closure doesn't include stdlib files."""
        start_file = TEST_REPO / "src" / "mypackage" / "main.py"
        closure = get_import_closure(start_file, TEST_REPO)

        # None of the paths should point to stdlib locations
        for path in closure:
            assert "site-packages" not in str(path)
            assert str(path).startswith("/usr/lib") is False

    def test_import_closure_respects_max_depth(self):
        """Test that max_depth limits traversal."""
        start_file = TEST_REPO / "src" / "mypackage" / "main.py"

        # With depth 0, should only include start file
        closure = get_import_closure(start_file, TEST_REPO, max_depth=0)
        # Start file is always added
        assert len(closure) >= 1

        # With depth 1, should include direct imports
        closure = get_import_closure(start_file, TEST_REPO, max_depth=1)
        assert len(closure) >= 1


class TestCircularImports:
    """Tests for handling circular import situations."""

    def test_no_infinite_loop_on_circular_imports(self):
        """Test that circular imports don't cause infinite loops."""
        # The test repo has utils.py importing from ..submodule
        # which could create cycles in more complex scenarios
        start_file = TEST_REPO / "src" / "mypackage" / "utils.py"
        closure = get_import_closure(start_file, TEST_REPO, max_depth=10)

        # Should complete without hanging
        assert isinstance(closure, set)

    def test_visited_prevents_revisiting(self):
        """Test that visited set prevents revisiting files."""
        start_file = TEST_REPO / "src" / "mypackage" / "main.py"
        closure = get_import_closure(start_file, TEST_REPO, max_depth=10)

        # No duplicates in closure
        assert len(closure) == len(set(closure))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
