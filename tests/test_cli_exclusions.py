"""Tests for CLI path exclusion logic."""

from pathlib import Path
import pytest

from codegrapher.cli import _should_exclude_path


class TestShouldExcludePath:
    """Tests for _should_exclude_path function."""

    def test_excludes_virtual_environment_directories(self, tmp_path):
        """Test that virtual environment directories are excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        # Test various venv directory names
        venv_names = [
            ".venv",
            "venv",
            ".virtualenv",
            "virtualenv",
            "env",
            "ENV",
        ]

        for venv_name in venv_names:
            venv_dir = repo_root / venv_name
            venv_dir.mkdir()
            test_file = venv_dir / "lib" / "python3.12" / "site-packages" / "foo.py"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.touch()

            assert _should_exclude_path(test_file, repo_root), f"Should exclude {venv_name}"

    def test_excludes_cache_directories(self, tmp_path):
        """Test that cache directories are excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        cache_dirs = [
            "__pycache__",
            ".pytest_cache",
            ".tox",
            ".mypy_cache",
            ".ruff_cache",
            ".hypothesis",
        ]

        for cache_name in cache_dirs:
            cache_dir = repo_root / "src" / cache_name
            cache_dir.mkdir(parents=True)
            test_file = cache_dir / "test.py"
            test_file.touch()

            assert _should_exclude_path(test_file, repo_root), f"Should exclude {cache_name}"

    def test_excludes_build_artifacts(self, tmp_path):
        """Test that build artifact directories are excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        build_dirs = ["build", "dist"]

        for build_name in build_dirs:
            build_dir = repo_root / build_name
            build_dir.mkdir()
            test_file = build_dir / "test.py"
            test_file.touch()

            assert _should_exclude_path(test_file, repo_root), f"Should exclude {build_name}"

    def test_excludes_egg_info(self, tmp_path):
        """Test that .egg-info directories are excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        egg_info_dir = repo_root / "my_package.egg-info"
        egg_info_dir.mkdir()
        test_file = egg_info_dir / "test.py"
        test_file.touch()

        assert _should_exclude_path(test_file, repo_root), "Should exclude .egg-info"

    def test_excludes_ide_directories(self, tmp_path):
        """Test that IDE directories are excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        ide_dirs = [".vscode", ".idea", ".eclipse"]

        for ide_name in ide_dirs:
            ide_dir = repo_root / ide_name
            ide_dir.mkdir()
            test_file = ide_dir / "settings.json"
            test_file.touch()

            assert _should_exclude_path(test_file, repo_root), f"Should exclude {ide_name}"

    def test_excludes_node_modules(self, tmp_path):
        """Test that node_modules directory is excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        node_modules = repo_root / "node_modules"
        node_modules.mkdir()
        test_file = node_modules / "package" / "index.js"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        assert _should_exclude_path(test_file, repo_root), "Should exclude node_modules"

    def test_allows_source_files(self, tmp_path):
        """Test that source files are not excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        # Create typical source directory structure
        src_dir = repo_root / "src"
        src_dir.mkdir()

        test_file = src_dir / "mypackage" / "__init__.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        assert not _should_exclude_path(test_file, repo_root), "Should allow source files"

    def test_allows_common_hidden_files(self, tmp_path):
        """Test that common hidden files are allowed."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        allowed_hidden = [
            ".gitignore",
            ".gitattributes",
            ".editorconfig",
        ]

        for filename in allowed_hidden:
            test_file = repo_root / filename
            test_file.touch()

            assert not _should_exclude_path(test_file, repo_root), f"Should allow {filename}"

    def test_allows_dot_github(self, tmp_path):
        """Test that .github directory is allowed."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        github_dir = repo_root / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        test_file = github_dir / "ci.yml"
        test_file.touch()

        assert not _should_exclude_path(test_file, repo_root), "Should allow .github directory"

    def test_excludes_other_hidden_directories(self, tmp_path):
        """Test that other hidden directories are excluded."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        hidden_dir = repo_root / ".hidden_dir"
        hidden_dir.mkdir()
        test_file = hidden_dir / "test.py"
        test_file.touch()

        assert _should_exclude_path(test_file, repo_root), "Should exclude .hidden_dir"

    def test_allows_dot_codegraph(self, tmp_path):
        """Test that .codegraph directory is allowed."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        codegraph_dir = repo_root / ".codegraph"
        codegraph_dir.mkdir()
        test_file = codegraph_dir / "symbols.db"
        test_file.touch()

        assert not _should_exclude_path(test_file, repo_root), "Should allow .codegraph directory"

    def test_excludes_nested_venv(self, tmp_path):
        """Test that .venv is excluded even when nested."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        # Create nested structure with .venv
        project_dir = repo_root / "projects" / "myproject"
        project_dir.mkdir(parents=True)
        venv_dir = project_dir / ".venv"
        venv_dir.mkdir()
        test_file = venv_dir / "lib" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.touch()

        assert _should_exclude_path(test_file, repo_root), "Should exclude nested .venv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
