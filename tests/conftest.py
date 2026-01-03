"""
Pytest fixtures for srtlog tests
"""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def job_4401453_dir(fixtures_dir):
    """Path to job 4401453 test fixture (aggregated mode)."""
    return fixtures_dir / "4401453"


@pytest.fixture
def job_4401453_logs_dir(job_4401453_dir):
    """Path to logs directory for job 4401453 (aggregated mode)."""
    return job_4401453_dir / "logs"


@pytest.fixture
def job_4401463_dir(fixtures_dir):
    """Path to job 4401463 test fixture (disaggregated mode)."""
    return fixtures_dir / "4401463"


@pytest.fixture
def job_4401463_logs_dir(job_4401463_dir):
    """Path to logs directory for job 4401463 (disaggregated mode)."""
    return job_4401463_dir / "logs"
