"""Tests for the demo's "Do you know who wrote it?" answer (P6): app/demo/feedback.js.

The visitor's optional answer turns the public site into a live test set, so it must be
counted once, and against the right result. These tests run the page's own module under
Node (cli.mjs `feedback`), and skip where Node isn't installed.
"""

import shutil

import pytest

from name_that_llm.browser import run_engine
from name_that_llm.data import LABELS

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="these tests need Node")


def answers(results: list[dict], steps: list[tuple[int, str]]) -> list[dict | None]:
    """Create `results` in order, then answer them as `steps` (result index, choice) say."""
    request = {"results": results, "steps": steps, "labels": LABELS}
    return run_engine("feedback", request)["fields"]


def test_feedback_belongs_to_its_result():
    # Review Focus 4: an answer carries ITS result's labels, and only the first one counts,
    # even when the visitor answers an older result after a new one appeared.
    first = {"label": LABELS[0], "challenger_label": None, "band": "Unsure"}
    second = {"label": LABELS[1], "challenger_label": LABELS[2], "band": "Fairly sure"}
    steps = [(0, "a person"), (0, "another model"), (1, LABELS[5]), (1, LABELS[6])]
    fields = answers([first, second], steps)
    assert fields[0] == {"actual": "a person", **first}
    assert fields[1] is None  # the same result, answered twice
    assert fields[2] == {"actual": LABELS[5], **second}
    assert fields[3] is None


def test_feedback_accepts_only_known_choices():
    result = {"label": LABELS[0], "challenger_label": None, "band": "Unsure"}
    fields = answers(
        [result, result], [(0, "a robot"), (0, ""), (0, LABELS[3]), (1, "another model")]
    )
    assert fields[:2] == [None, None]  # refused, and the result can still be answered
    assert fields[2]["actual"] == LABELS[3]
    assert fields[3]["actual"] == "another model"
