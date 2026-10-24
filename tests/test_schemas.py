import pandas as pd
import pytest
from pandera.errors import SchemaError

from name_that_llm.schemas import validate_submission

TEST_IDS = pd.Series(["a", "b", "c"])


def _sub(ids, models):
    return pd.DataFrame({"id": ids, "model": models})


def test_validate_submission_accepts_a_valid_file():
    sub = _sub(["a", "b", "c"], ["gpt-5.5", "kimi-k3", "glm-5.3"])
    assert validate_submission(sub, TEST_IDS).equals(sub)


@pytest.mark.parametrize(
    "ids,models",
    [
        (["a", "b"], ["gpt-5.5", "gpt-5.5"]),  # a test id is missing
        (["a", "a", "b", "c"], ["gpt-5.5"] * 4),  # duplicate id
        (["a", "b", "c"], ["gpt-5.5", "gpt-9", "gpt-5.5"]),  # unknown label
        (["a", "b", "c"], ["gpt-5.5", None, "gpt-5.5"]),  # null label
    ],
)
def test_validate_submission_rejects_bad_files(ids, models):
    with pytest.raises((SchemaError, ValueError)):
        validate_submission(_sub(ids, models), TEST_IDS)
