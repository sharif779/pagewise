from pathlib import Path

import pytest

SAMPLE_TXT = Path(__file__).parent / "sample.txt"


@pytest.fixture
def sample_path() -> str:
    return str(SAMPLE_TXT)
