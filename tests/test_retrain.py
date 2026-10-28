"""Tests for the weekly retraining's decision (P6): when to propose a new challenger.

The retraining job never changes the site: it only opens a pull request that makes a
better model the demo's challenger, and a person merges it. These tests pin when it may
propose one: only on a significant win over the champion, and never below the current
challenger.
"""

import numpy as np

from name_that_llm.labels import LABELS
from name_that_llm.retrain import pr_body, propose

# 20 prompts x 18 models: each prompt answered once by every model, as in our data.
Y = np.array(LABELS * 20)
GROUPS = np.repeat([f"prompt{i}" for i in range(20)], 18)
RNG = np.random.default_rng(0)


def _pred(accuracy: float) -> np.ndarray:
    """Predictions right on about `accuracy` of the rows, wrong (another label) elsewhere."""
    right = RNG.random(len(Y)) < accuracy
    wrong = np.roll(Y, 1)  # a different label on every row
    return np.where(right, Y, wrong)


def test_propose_needs_a_significant_win():
    champion = _pred(0.30)
    yes, numbers = propose(Y, GROUPS, champion, _pred(0.60), candidate_cv=0.60, challenger_cv=None)
    assert yes and numbers["lo"] > 0
    # The same predictions: no difference, so the interval contains 0.
    no, numbers = propose(Y, GROUPS, champion, champion, candidate_cv=0.30, challenger_cv=None)
    assert not no and numbers["lo"] <= 0 <= numbers["hi"]


def test_propose_without_a_challenger():
    # Review Focus 5: no challenger yet (an empty CHALLENGER_RUN) must not block the first.
    yes, numbers = propose(Y, GROUPS, _pred(0.3), _pred(0.6), candidate_cv=0.6, challenger_cv=None)
    assert yes and numbers["challenger_cv"] is None


def test_propose_refuses_a_candidate_below_the_challenger():
    yes, _ = propose(Y, GROUPS, _pred(0.3), _pred(0.6), candidate_cv=0.58, challenger_cv=0.61)
    assert not yes  # it beats the champion, but the current challenger is better


def test_the_body_reports_both_scores_and_the_interval():
    _, numbers = propose(Y, GROUPS, _pred(0.3), _pred(0.6), candidate_cv=0.6, challenger_cv=0.5)
    body = pr_body(numbers, run_id="abc123")
    for value in (numbers["champion_cv"], numbers["candidate_cv"], numbers["lo"], numbers["hi"]):
        assert f"{value:.4f}" in body
    assert "abc123" in body and "0.5000" in body  # the run, and the current challenger
    assert "merging" in body.lower()  # what merging does, for the person who decides
