"""Tests for the browser demo's engine (P4 revision): the TF-IDF model in JavaScript.

The public demo runs `lr` inside the visitor's browser, so app/demo/ntl.js must compute
exactly what scikit-learn computes, from the files export_browser_model writes. These
tests run the JavaScript with Node and compare it with Python on the same inputs. They
skip where Node isn't installed (it is on GitHub's CI machines).
"""

import json
import shutil

import numpy as np
import pytest

from name_that_llm.browser import (
    export_browser_model,
    run_engine,
    unknown_share,
    unusual_thresholds,
)
from name_that_llm.champion import Band
from name_that_llm.config import load_config
from name_that_llm.data import LABELS, flatten_text
from name_that_llm.models import build_model
from name_that_llm.submit import _aligned_proba

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="the browser engine's tests need Node"
)
BANDS = [Band("Fairly sure", 0.2, 0.6), Band("Unsure", 0.1, 0.4), Band("Very unsure", 0.0, 0.2)]

# Text a visitor might paste: markdown of every kind, training-style text that must
# survive, odd whitespace, decomposed accents, emoji and other scripts, nothing at all.
TRICKY = [
    "# Title\n\nSome **bold** and *italic* text.\n\n- first point\n- second point\n\n"
    "1. step one\n2) step two\n\n> a quote\n\n```python\nx = 1\n```\n\n"
    "A [link](https://example.com) and `code`.\n\n| a | b |\n|---|---|\n| 1 | 2 |",
    "It costs $5. Then *this* matters — as it did in 2023. Done (see 1. above).",
    "  a\n\nb\t c   d  ",
    "caf" + "é",
    "🙂 émoji, 日本語の文章 and ünïcödé",
    "Tabs\tand non-breaking spaces",
    "> quote\n>> nested\n- [link](http://x.y) ![img](a.png)",
    "",
]
TEXTS = [
    "marker3 words about subject 1",
    "**marker5** style5\n\n- topic2",
    "unknownword anotherone",
    "marker7 — marker7, style7 TOPIC4",
    "Ünïcödé marker1 日本",
    "x",
]


@pytest.fixture
def exported(fake_train, tmp_path):
    """`lr` fit on the fake data, exported the way the demo's model is: (model, folder)."""
    lr = build_model(load_config("lr")).fit(fake_train["text"].tolist(), fake_train["model"])
    export_browser_model(lr, BANDS, tmp_path / "model", meta={"source": "test"})
    return lr, tmp_path / "model"


@needs_node
def test_js_flatten_matches_python():
    assert run_engine("flatten", {"texts": TRICKY})["texts"] == [flatten_text(t) for t in TRICKY]


@needs_node
def test_js_probabilities_match_sklearn(exported):
    lr, folder = exported
    js = np.array(run_engine("predict", {"model": str(folder), "texts": TEXTS})["probabilities"])
    expected = _aligned_proba(lr, [flatten_text(t) for t in TEXTS])
    np.testing.assert_allclose(js, expected, atol=1e-5)


def test_export_writes_the_model_files(exported):
    lr, folder = exported
    model = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    vectorizers = dict(lr.named_steps["features"].transformer_list)
    sizes = [len(vectorizers["words"].vocabulary_), len(vectorizers["chars"].vocabulary_)]
    assert model["labels"] == LABELS
    assert [len(model["words"]["vocabulary"]), len(model["chars"]["vocabulary"])] == sizes
    n = sum(sizes)  # float32: both idf vectors, the 18 x n coefficients, the 18 intercepts
    assert (folder / "weights.bin").stat().st_size == 4 * (n + 18 * n + 18)
    assert [band["label"] for band in model["bands"]] == ["Fairly sure", "Unsure", "Very unsure"]


def test_export_rejects_unsupported_settings(fake_train, tmp_path):
    # The engine implements exactly the settings we use; anything else must fail loudly
    # at export, not silently give different numbers in the browser.
    lr = build_model(load_config("lr"))
    lr.named_steps["features"].transformer_list[1][1].set_params(analyzer="char_wb")
    lr.fit(fake_train["text"].tolist(), fake_train["model"])
    with pytest.raises(ValueError, match="char_wb"):
        export_browser_model(lr, BANDS, tmp_path / "model", meta={})


# ---------- unusual texts (P6 Task 2) ----------
LIMITS = {"min_words": 36, "max_unknown_share": 0.5, "max_non_latin_share": 0.2}


def _word_unigrams(lr) -> set[str]:
    """The exported model's word vocabulary, unigrams only."""
    vocabulary = dict(lr.named_steps["features"].transformer_list)["words"].vocabulary_
    return {term for term in vocabulary if " " not in term}


@needs_node
def test_js_text_stats_match_python(exported):
    lr, folder = exported
    stats = run_engine("stats", {"model": str(folder), "texts": TEXTS})["stats"]
    expected = unknown_share(TEXTS, _word_unigrams(lr))
    np.testing.assert_allclose([s["unknown_share"] for s in stats], expected, atol=1e-9)
    assert [s["words"] for s in stats] == [len(flatten_text(t).split()) for t in TEXTS]


def test_thresholds_come_from_held_out_passages(fake_train):
    limits = unusual_thresholds(load_config("lr"), fake_train)
    # Each fold's vocabulary misses its held-out prompts' topic words, so some share of
    # held-out words is unknown: a vocabulary fitted on ALL rows would show 0 here.
    assert limits["max_unknown_share"] > 0
    assert limits["min_words"] == min(len(flatten_text(t).split()) for t in fake_train["text"])
    assert limits["max_non_latin_share"] == 0.2


@needs_node
def test_unusual_reasons(fake_train, tmp_path):
    lr = build_model(load_config("lr")).fit(fake_train["text"].tolist(), fake_train["model"])
    export_browser_model(lr, BANDS, tmp_path / "model", meta={"unusual": LIMITS})
    known = "marker1 words about subject topic2 style1 "
    texts = [
        known * 5,  # 25 words, fewer than min_words
        "日本語の文章です。" * 40,  # Japanese: non-Latin letters
        "zorblax quintophane " * 20,  # 40 words the model has never seen
        known * 8,  # 48 training-like words
    ]
    reasons = run_engine("stats", {"model": str(tmp_path / "model"), "texts": texts})["reasons"]
    assert reasons[0] == ["short"]
    assert "non_latin" in reasons[1]
    assert reasons[2] == ["unknown_words"]
    assert reasons[3] == []


@needs_node
def test_no_thresholds_means_no_reasons(exported):
    _, folder = exported  # exported without meta["unusual"]
    assert run_engine("stats", {"model": str(folder), "texts": ["two words"]})["reasons"] == [[]]
