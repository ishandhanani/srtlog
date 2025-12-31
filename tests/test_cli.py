"""
Tests for CLI module
"""

import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """Test that CLI help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "srtlog.cli.main", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent / "src",
    )

    assert result.returncode == 0
    assert "srtlog" in result.stdout or "usage" in result.stdout.lower()


def test_cli_version():
    """Test that CLI version command works."""
    result = subprocess.run(
        [sys.executable, "-m", "srtlog.cli.main", "--version"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent / "src",
    )

    assert result.returncode == 0
    assert "0.1.0" in result.stdout
