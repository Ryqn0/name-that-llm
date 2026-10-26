"""Tests for the app's predictor (P4): the full champion pipeline, on tiny random models.

The pipeline is checked for its plumbing, not its accuracy: tiny random models can't tell
18 authors apart, but they can prove that every part is computed, weighted, flattened and
protected against non-finite numbers exactly as the design says.
"""

import numpy as np
import pytest

from name_that_llm.champion import Champion, describe_confidence, fit_confidence_bands
from name_that_llm.config import load_config
from name_that_llm.data import LABELS, flatten_text
from name_that_llm.models import build_dense_model, build_model
from name_that_llm.submit import _aligned_proba

TEXTS = ["marker3 words about subject 1", "marker5 marker5 — style5", "topic2 marker7"]


@pytest.fixture
def parts(fake_train, tiny_qwen, tiny_encoder, tiny_modernbert):
    """Everything a Champion needs, fitted on the fake data like the registration does."""
    from name_that_llm.encoder import load_encoder
    from name_that_llm.lm import load_lm_layers, pooled_hidden_states, select_layers

    texts, y = fake_train["text"].tolist(), fake_train["model"]
    lm, tokenizer = load_lm_layers(tiny_qwen, n_layers=10)
    X = select_layers(pooled_hidden_states(texts, lm, tokenizer, layers=range(6, 10)), 0, 4)
    return {
        "text_model": build_model(load_config("lr")).fit(texts, y),
        "lm_head": build_dense_model(0.1).fit(X, y),
        "lm_train_mean": X.mean(axis=0),
        "lm": (lm, tokenizer),
        # Two DIFFERENT fold models, so averaging can't be confused with using only one.
        "encoders": [load_encoder(tiny_encoder), load_encoder(tiny_modernbert)],
    }


def make_champion(parts, weights=(0.7, 0.15, 0.15), **overrides):
    # The tiny models have 64 positions: keep every input within them.
    settings = {"lm_max_length": 48, "encoder_max_length": 48}
    return Champion(**{**parts, "weights": weights, **settings, **overrides})


def poison(module, args, out):
    """A forward hook that turns a block's output into NaN, like a float16 overflow."""
    if isinstance(out, tuple):
        return (out[0] * float("nan"), *out[1:])
    return out * float("nan")


def test_predict_proba_rows_are_distributions_in_labels_order(parts):
    proba = make_champion(parts).predict_proba(TEXTS)
    assert proba.shape == (3, len(LABELS)) and np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-9)


def test_predict_proba_follows_the_weights(parts):
    from name_that_llm.encoder import predict_proba

    flat = [flatten_text(t) for t in TEXTS]
    text_only = make_champion(parts, weights=(1, 0, 0)).predict_proba(TEXTS)
    np.testing.assert_allclose(text_only, _aligned_proba(parts["text_model"], flat), atol=1e-9)
    # The encoder part is the MEAN of the fold models, each reading at most 48 tokens here.
    each = [predict_proba(m, t, flat, max_length=48) for m, t in parts["encoders"]]
    encoder_only = make_champion(parts, weights=(0, 0, 1)).predict_proba(TEXTS)
    np.testing.assert_allclose(encoder_only, (each[0] + each[1]) / 2, atol=1e-9)


def test_predict_proba_flattens_markdown_first(parts):
    champion = make_champion(parts)
    markdown = "**marker3** words\n\n- about subject 1"
    np.testing.assert_allclose(
        champion.predict_proba([markdown]),
        champion.predict_proba([flatten_text(markdown)]),
        atol=1e-12,
    )


def test_predict_proba_falls_back_when_lm_features_are_not_finite(parts):
    lm, tokenizer = parts["lm"]
    # Block 6 produces hidden state 6; the NaN then spreads to 7, 8 and 9, all read here.
    handle = lm.layers[5].register_forward_hook(poison)
    try:
        lm_only = make_champion(parts, weights=(0, 1, 0)).predict_proba(TEXTS[:1])
    finally:
        handle.remove()
    expected = _aligned_proba(parts["lm_head"], parts["lm_train_mean"][None, :])
    np.testing.assert_allclose(lm_only, expected, atol=1e-9)


def test_predict_proba_falls_back_when_an_encoder_is_not_finite(parts, tiny_encoder):
    from name_that_llm.encoder import load_encoder

    nan_model, nan_tokenizer = load_encoder(tiny_encoder)
    nan_model.classifier.register_forward_hook(poison)
    champion = make_champion(parts, weights=(0, 0, 1), encoders=[(nan_model, nan_tokenizer)])
    np.testing.assert_allclose(champion.predict_proba(TEXTS[:2]), 1 / len(LABELS), atol=1e-12)


def test_predict_proba_handles_unusual_text(parts):
    # Review Focus 1: shapes a visitor might paste. A valid distribution, never a crash.
    unusual = ["🙂 " * 30, "日本語の文章 " * 25, "https://x.org/" + "a" * 5000, "!? " * 30]
    proba = make_champion(parts).predict_proba(unusual)
    assert proba.shape == (4, len(LABELS)) and np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-9)


def _rows(top: float, n_right: int, n: int = 100):
    """n prediction rows whose top probability is `top` on LABELS[0]; n_right are right."""
    row = np.full(len(LABELS), (1 - top) / (len(LABELS) - 1))
    row[0] = top
    return [row] * n, [LABELS[0]] * n_right + [LABELS[1]] * (n - n_right)


def test_fit_confidence_bands_uses_terciles_and_measures_accuracy():
    # Hand-built: 100 rows each at top probability 0.2, 0.5 and 0.8, right 3, 5 and 8
    # times in 10. The bands must find exactly those groups and those accuracies.
    proba, y = [], []
    for top, n_right in [(0.2, 30), (0.5, 50), (0.8, 80)]:
        rows, labels = _rows(top, n_right)
        proba += rows
        y += labels
    bands = fit_confidence_bands(np.array(proba), np.array(y))
    assert [b.label for b in bands] == ["Fairly sure", "Unsure", "Very unsure"]
    assert [b.min_top for b in bands] == pytest.approx([0.8, 0.5, 0.0])
    assert [b.accuracy for b in bands] == pytest.approx([0.8, 0.5, 0.3])


def test_describe_confidence_picks_the_band():
    proba, y = [], []
    for top, n_right in [(0.2, 30), (0.5, 50), (0.8, 80)]:
        rows, labels = _rows(top, n_right)
        proba += rows
        y += labels
    bands = fit_confidence_bands(np.array(proba), np.array(y))
    assert describe_confidence(0.95, bands).label == "Fairly sure"
    assert describe_confidence(0.5, bands).label == "Unsure"  # exactly at the edge
    assert describe_confidence(0.1, bands).label == "Very unsure"
    assert describe_confidence(0.0, bands).label == "Very unsure"
