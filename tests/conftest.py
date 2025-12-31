"""
Pytest fixtures for srtlog tests
"""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def job_4401453_dir(fixtures_dir):
    """Path to job 4401453 test fixture."""
    return fixtures_dir / "4401453"


@pytest.fixture
def job_4401453_logs_dir(job_4401453_dir):
    """Path to logs directory for job 4401453."""
    return job_4401453_dir / "logs"
