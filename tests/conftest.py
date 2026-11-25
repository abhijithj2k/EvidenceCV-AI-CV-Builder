from pathlib import Path

import pytest

from src.data_loader import load_jobs, load_test_profiles


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def jobs():
    return load_jobs(ROOT / "data" / "jobs.jsonl")


@pytest.fixture(scope="session")
def profiles():
    return load_test_profiles(ROOT / "data" / "student_profiles.jsonl")
