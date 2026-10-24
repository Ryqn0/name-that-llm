import numpy as np
import pytest

from name_that_llm.data import LABELS
from name_that_llm.metrics import lab_accuracy, per_class_recall, prompt_bootstrap_ci, score


def test_score_is_mean_per_class_recall():
    y, p = np.array(["a", "a", "a", "b"]), np.array(["a", "a", "b", "b"])
    assert score(y, p) == pytest.approx((2 / 3 + 1) / 2)


def test_lab_accuracy_counts_sibling_models_as_correct():
    y = np.array(["gpt-5.5", "claude-opus-5"])
    p = np.array(["gpt-6-astra", "gemini-3.6-flash"])
    assert lab_accuracy(y, p) == 0.5


def test_per_class_recall_is_indexed_by_all_labels():
    r = per_class_recall(np.array(["gpt-5.5"]), np.array(["gpt-5.5"]))
    assert list(r.index) == LABELS and r["gpt-5.5"] == 1.0 and np.isnan(r["kimi-k3"])


def _toy(n_prompts=20):
    return np.tile(np.array(LABELS), n_prompts), np.repeat(np.arange(n_prompts), 18)


def test_bootstrap_identical_predictions_gives_zero():
    y, g = _toy()
    assert prompt_bootstrap_ci(y, y, y, g, n_boot=200) == (0.0, 0.0, 0.0)


def test_bootstrap_clearly_better_model_has_interval_above_zero():
    y, g = _toy()
    pred_a = y.copy()
    pred_a[g % 2 == 0] = "kimi-k3"  # model A is wrong on half the prompts
    diff, lo, hi = prompt_bootstrap_ci(y, pred_a, y, g, n_boot=500, seed=0)
    assert diff > 0 and lo > 0 and lo <= diff <= hi


def test_bootstrap_is_reproducible():
    y, g = _toy()
    pred_a = y.copy()
    pred_a[::3] = "kimi-k3"
    assert prompt_bootstrap_ci(y, pred_a, y, g, seed=1) == prompt_bootstrap_ci(
        y, pred_a, y, g, seed=1
    )
