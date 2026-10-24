"""Shared pytest fixtures.

Every test runs on FAKE data generated here, never on the real competition data. The
rules forbid redistributing that data, and this repo is public, so the tests must be
able to run anywhere (your PC, GitHub Actions) without it.
"""

import pandas as pd
import pytest

from name_that_llm.data import LABELS, load_train


@pytest.fixture
def fake_data_dir(tmp_path):
    """A tiny FAKE competition folder: 6 prompts x 18 models, plus 3 test rows.

    Each model gets its own marker word, so classifiers have something to learn. The ids
    look numeric on purpose ("000000"), to catch pandas dropping leading zeros.
    """
    rows = [
        {
            "id": f"{p:03d}{m:03d}",
            "prompt_id": f"prompt{p}",
            "text": f"topic{p} words about subject {p}. marker{m} marker{m} — style{m}",
            "model": label,
        }
        for p in range(6)
        for m, label in enumerate(LABELS)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        {"id": ["000001", "a1b2c3", "999999"], "text": ["marker3 text", None, "marker5 text"]}
    ).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


@pytest.fixture
def fake_train(fake_data_dir):
    return load_train(fake_data_dir)
