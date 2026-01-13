"""Unit tests for secret detection logic."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from codegrapher.secrets import (
    scan_file,
    is_excluded,
    get_excluded_files,
    clear_excluded_file,
    SecretFoundError,
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        yield repo_root


@pytest.fixture
def sample_file(temp_repo):
    """Create a sample Python file for testing."""
    test_file = temp_repo / "test.py"
    test_file.write_text('''
def hello():
    print("Hello, world")
''')
    return test_file


class TestScanFile:
    """Tests for scan_file function."""

    def test_clean_file_returns_false(self, sample_file, temp_repo):
        """Test that a clean file returns False (no secrets)."""
        # Mock detect-secrets to return no secrets
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "results": {},
            "plugins_used": []
        })

        with patch('subprocess.run', return_value=mock_result):
            has_secrets = scan_file(sample_file, temp_repo)
            assert has_secrets is False

    def test_file_with_secret_returns_true(self, sample_file, temp_repo):
        """Test that a file with secrets returns True."""
        # Mock detect-secrets to return secrets
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "results": {
                str(sample_file): [
                    {
                        "type": "Base64 High Entropy String",
                        "hashed_secret": "abc123",
                        "line_number": 1
                    }
                ]
            },
            "plugins_used": []
        })

        with patch('subprocess.run', return_value=mock_result):
            has_secrets = scan_file(sample_file, temp_repo)
            assert has_secrets is True

        # Check that file was added to excluded list
        assert is_excluded(sample_file, temp_repo)

    def test_nonexistent_file_raises_error(self, temp_repo):
        """Test that scanning a nonexistent file raises FileNotFoundError."""
        nonexistent = temp_repo / "does_not_exist.py"

        with pytest.raises(FileNotFoundError):
            scan_file(nonexistent, temp_repo)

    def test_detect_secrets_timeout_returns_false(self, sample_file, temp_repo):
        """Test that timeout returns False (treat as clean)."""
        import subprocess

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('cmd', 10)):
            has_secrets = scan_file(sample_file, temp_repo)
            assert has_secrets is False

    def test_detect_secrets_not_installed_returns_false(self, sample_file, temp_repo):
        """Test that missing detect-secrets returns False."""
        with patch('subprocess.run', side_effect=FileNotFoundError):
            has_secrets = scan_file(sample_file, temp_repo)
            assert has_secrets is False

    def test_invalid_json_returns_false(self, sample_file, temp_repo):
        """Test that invalid JSON output returns False."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json"

        with patch('subprocess.run', return_value=mock_result):
            has_secrets = scan_file(sample_file, temp_repo)
            assert has_secrets is False

    def test_scan_with_baseline(self, sample_file, temp_repo):
        """Test that baseline file is used when present."""
        # Create baseline file
        baseline = temp_repo / ".secrets.baseline"
        baseline.write_text(json.dumps({
            "results": {},
            "plugins_used": []
        }))

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"results": {}, "plugins_used": []})

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            scan_file(sample_file, temp_repo)

            # Check that baseline was passed to detect-secrets
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--baseline" in cmd
            assert str(baseline) in cmd


class TestExcludedFiles:
    """Tests for excluded files tracking."""

    def test_is_excluded_returns_false_when_no_excluded_file(self, sample_file, temp_repo):
        """Test that is_excluded returns False when excluded files don't exist."""
        assert is_excluded(sample_file, temp_repo) is False

    def test_is_excluded_returns_true_when_excluded(self, sample_file, temp_repo):
        """Test that is_excluded returns True for excluded files."""
        # Manually add file to excluded list
        excluded_path = temp_repo / ".codegraph/excluded_files.txt"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_path.write_text("test.py\n")

        assert is_excluded(sample_file, temp_repo) is True

    def test_get_excluded_files_returns_list(self, temp_repo):
        """Test that get_excluded_files returns list of files."""
        excluded_path = temp_repo / ".codegraph/excluded_files.txt"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_path.write_text("file1.py\nfile2.py\nfile3.py\n")

        excluded = get_excluded_files(temp_repo)
        assert excluded == ["file1.py", "file2.py", "file3.py"]

    def test_get_excluded_files_returns_empty_when_no_file(self, temp_repo):
        """Test that get_excluded_files returns [] when excluded files don't exist."""
        assert get_excluded_files(temp_repo) == []

    def test_clear_excluded_file_removes_from_list(self, temp_repo):
        """Test that clear_excluded_file removes file from excluded list."""
        excluded_path = temp_repo / ".codegraph/excluded_files.txt"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_path.write_text("file1.py\nfile2.py\nfile3.py\n")

        file_to_clear = temp_repo / "file2.py"
        result = clear_excluded_file(temp_repo, file_to_clear)

        assert result is True
        excluded = get_excluded_files(temp_repo)
        assert excluded == ["file1.py", "file3.py"]
        assert "file2.py" not in excluded

    def test_clear_excluded_file_returns_false_when_not_excluded(self, temp_repo):
        """Test that clear_excluded_file returns False when file isn't in list."""
        excluded_path = temp_repo / ".codegraph/excluded_files.txt"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_path.write_text("file1.py\nfile2.py\n")

        file_to_clear = temp_repo / "file3.py"
        result = clear_excluded_file(temp_repo, file_to_clear)

        assert result is False

    def test_excluded_files_are_sorted(self, temp_repo):
        """Test that excluded files are maintained in sorted order when added."""
        # Create multiple files and add them in random order
        files = [
            temp_repo / "zebra.py",
            temp_repo / "apple.py",
            temp_repo / "middle.py"
        ]

        # Add each file (simulating secret detection)
        for file_path in files:
            from codegrapher.secrets import _add_to_excluded_files
            _add_to_excluded_files(file_path, temp_repo)

        # Read back - should be sorted
        excluded = get_excluded_files(temp_repo)
        assert excluded == ["apple.py", "middle.py", "zebra.py"]


class TestScanIntegration:
    """Integration tests for scanning workflow."""

    def test_scan_adds_to_excluded_list(self, sample_file, temp_repo):
        """Test that scanning a file with secrets adds it to excluded list."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "results": {
                str(sample_file): [
                    {"type": "Secret", "line_number": 1}
                ]
            },
            "plugins_used": []
        })

        with patch('subprocess.run', return_value=mock_result):
            scan_file(sample_file, temp_repo)

        assert is_excluded(sample_file, temp_repo)

    def test_scan_does_not_duplicate_excluded_files(self, sample_file, temp_repo):
        """Test that scanning the same file twice doesn't duplicate entries."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "results": {
                str(sample_file): [
                    {"type": "Secret", "line_number": 1}
                ]
            },
            "plugins_used": []
        })

        with patch('subprocess.run', return_value=mock_result):
            scan_file(sample_file, temp_repo)
            scan_file(sample_file, temp_repo)  # Scan again

        # Should only appear once
        excluded = get_excluded_files(temp_repo)
        assert excluded.count(str(sample_file.relative_to(temp_repo))) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
