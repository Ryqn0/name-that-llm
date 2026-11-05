"""The gate script's own logic (P8 Task 4): its LLM loader and its blend form.

scripts/ isn't a package, so the script is loaded as a module from its path.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from name_that_llm.data import LABELS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ensemble_gate.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("ensemble_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fold_file(runs: Path, job: str, k: int, n_rows: int, value: float) -> None:
    """A fold file as notebook 12 writes it: runs/<job>/fold<k>.npz, holding key fold<k>."""
    path = runs / job / f"fold{k}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"fold{k}": np.full((n_rows, len(LABELS)), value)})


def test_llm_folds_fill_their_rows_and_leave_the_rest_unknown(gate, tmp_path):
    folds = np.array([0, 1, 0, 2])
    _fold_file(tmp_path, "seed42-f01", 0, 2, 0.25)
    _fold_file(tmp_path, "seed42-f01", 1, 1, 0.5)
    llm = gate.load_llm_folds([tmp_path], folds)
    assert (llm[folds == 0] == 0.25).all()
    assert (llm[folds == 1] == 0.5).all()
    assert np.isnan(llm[folds == 2]).all()  # a fold not done stays unknown, never zero


@pytest.mark.parametrize("job", ["ep1-seed7", "seed420-f0", "final-f0"])
def test_llm_folds_refuse_any_job_but_our_seeds(gate, tmp_path, job):
    # Fold 0 of another seed holds other rows: same shape, silently the wrong predictions.
    _fold_file(tmp_path, job, 0, 1, 0.25)
    with pytest.raises(SystemExit, match=job):
        gate.load_llm_folds([tmp_path], np.array([0, 1]))


def test_llm_folds_refuse_a_fold_found_twice(gate, tmp_path):
    # Without the check, whichever file sorts last would win without a word.
    _fold_file(tmp_path, "seed42-f01", 0, 1, 0.25)
    _fold_file(tmp_path, "seed42-rerun", 0, 1, 0.5)
    with pytest.raises(SystemExit, match="fold 0"):
        gate.load_llm_folds([tmp_path], np.array([0, 1]))


@pytest.mark.parametrize(
    ("flags", "baseline_log", "candidate_log"),
    [([], False, False), (["--candidate-log"], False, True), (["--log"], True, True)],
)
def test_the_blend_form_reaches_each_side(
    gate, monkeypatch, fake_train, flags, baseline_log, candidate_log
):
    rng = np.random.default_rng(0)
    members = {m: rng.dirichlet(np.ones(len(LABELS)), len(fake_train)) for m in ("a", "b")}
    monkeypatch.setattr(gate, "load_train", lambda: fake_train)
    monkeypatch.setattr(gate, "_members", lambda *_: members)
    seen, real = [], gate.nested_blend_oof

    def spy(*args, log=False, **kwargs):
        seen.append(log)
        return real(*args, log=log, **kwargs)

    monkeypatch.setattr(gate, "nested_blend_oof", spy)
    gate.main(["--baseline", "a", "--candidate", "a,b", *flags])
    assert seen == [baseline_log, candidate_log]
