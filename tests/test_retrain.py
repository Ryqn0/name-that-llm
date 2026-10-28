"""Tests for the weekly retraining's decision (P6): when to propose a new challenger.

The retraining job never changes the site: it only opens a pull request that makes a
better model the demo's challenger, and a person merges it. These tests pin when it may
propose one: only on a significant win over the champion, and never below the current
challenger.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

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
    yes, numbers = propose(Y, GROUPS, champion, _pred(0.60), candidate_cv=0.60, pinned_cvs=[])
    assert yes and numbers["lo"] > 0
    # The same predictions: no difference, so the interval contains 0.
    no, numbers = propose(Y, GROUPS, champion, champion, candidate_cv=0.30, pinned_cvs=[])
    assert not no and numbers["lo"] <= 0 <= numbers["hi"]


def test_propose_without_a_challenger():
    # Review Focus 5: no challenger yet (an empty CHALLENGER_RUN) must not block the first.
    yes, numbers = propose(Y, GROUPS, _pred(0.3), _pred(0.6), candidate_cv=0.6, pinned_cvs=[])
    assert yes and numbers["pinned_cvs"] == []


def test_propose_refuses_a_candidate_below_the_challenger():
    yes, _ = propose(Y, GROUPS, _pred(0.3), _pred(0.6), candidate_cv=0.58, pinned_cvs=[0.61])
    assert not yes  # it beats the champion, but the current challenger is better


def test_the_body_reports_both_scores_and_the_interval():
    _, numbers = propose(Y, GROUPS, _pred(0.3), _pred(0.6), candidate_cv=0.6, pinned_cvs=[0.5])
    body = pr_body(numbers, run_id="abc123")
    for value in (numbers["champion_cv"], numbers["candidate_cv"], numbers["lo"], numbers["hi"]):
        assert f"{value:.4f}" in body
    assert "abc123" in body and "0.5000" in body  # the run, and the current challenger
    assert "merging" in body.lower()  # what merging does, for the person who decides


def test_propose_needs_a_margin_over_every_pinned_model():
    # The final review's finding 1: retraining the same recipe gives the same CV, give or
    # take noise (0.4205 on the PC, 0.4186 on GitHub), and a tie must not open a pull request
    # every week. The candidate must beat BOTH pinned models (the demo's champion and
    # challenger, by the CVs their MLflow runs logged) by at least 0.005.
    champion, candidate = _pred(0.3), _pred(0.6)
    no, _ = propose(
        Y, GROUPS, champion, candidate, candidate_cv=0.4186, pinned_cvs=[0.3212, 0.4205]
    )
    assert not no
    yes, numbers = propose(
        Y, GROUPS, champion, candidate, candidate_cv=0.43, pinned_cvs=[0.3212, 0.4205]
    )
    assert yes and numbers["pinned_cvs"] == [0.3212, 0.4205]
    # After a promotion, the champion is the pool model and there's no challenger: the same
    # recipe is not proposed again.
    no, _ = propose(Y, GROUPS, champion, candidate, candidate_cv=0.4190, pinned_cvs=[0.4186])
    assert not no


ROOT = Path(__file__).resolve().parents[1]


def test_the_retraining_workflow_keeps_its_write_token_in_one_step():
    # The final review's finding 2: the token that can push and open pull requests must not
    # sit in .git/config for the whole job (the checkout's default), where every step could
    # read it, including packages installed at run time. Those are pinned too.
    workflow = yaml.safe_load((ROOT / ".github/workflows/retrain.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["retrain"]["steps"]
    checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["persist-credentials"] is False and "token" not in checkout["with"]
    assert [s for s in steps if "RETRAIN_PR_TOKEN" in json.dumps(s)] == [steps[-1]]
    commands = " ".join(s.get("run", "") for s in steps)
    assert "kaggle==" in commands and "kaggle>=" not in commands


def test_the_pull_request_step_sets_up_git_credentials_before_pushing(monkeypatch, tmp_path):
    # Without credentials kept by the checkout, `git push` gets them from gh (GH_TOKEN).
    spec = importlib.util.spec_from_file_location(
        "retrain_challenger", ROOT / "scripts" / "retrain_challenger.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setattr(module, "DEMO_DIR", tmp_path)
    _, numbers = propose(Y, GROUPS, _pred(0.3), _pred(0.6), candidate_cv=0.6, pinned_cvs=[])
    module.open_pull_request("run123", numbers)
    setup = commands.index(["gh", "auth", "setup-git"])
    push = next(i for i, c in enumerate(commands) if c[:2] == ["git", "push"])
    assert setup < push and (tmp_path / "CHALLENGER_RUN").read_text().strip() == "run123"
