"""Tests for the browser demo's engine (P4 revision): the TF-IDF model in JavaScript.

The public demo runs `lr` inside the visitor's browser, so app/demo/ntl.js must compute
exactly what scikit-learn computes, from the files export_browser_model writes. These
tests run the JavaScript with Node and compare it with Python on the same inputs. They
skip where Node isn't installed (it is on GitHub's CI machines).
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from name_that_llm.browser import export_browser_model
from name_that_llm.champion import Band
from name_that_llm.config import load_config
from name_that_llm.data import LABELS, flatten_text
from name_that_llm.models import build_model
from name_that_llm.submit import _aligned_proba

CLI = Path(__file__).resolve().parents[1] / "app" / "demo" / "cli.mjs"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="the browser engine's tests need Node")
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


def run_js(command: str, payload: dict) -> dict:
    """Call the Node harness: JSON in on stdin, JSON out on stdout."""
    done = subprocess.run(
        [NODE, str(CLI), command],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(done.stdout)


@pytest.fixture
def exported(fake_train, tmp_path):
    """`lr` fit on the fake data, exported the way the demo's model is: (model, folder)."""
    lr = build_model(load_config("lr")).fit(fake_train["text"].tolist(), fake_train["model"])
    export_browser_model(lr, BANDS, tmp_path / "model", meta={"source": "test"})
    return lr, tmp_path / "model"


@needs_node
def test_js_flatten_matches_python():
    assert run_js("flatten", {"texts": TRICKY})["texts"] == [flatten_text(t) for t in TRICKY]


@needs_node
def test_js_probabilities_match_sklearn(exported):
    lr, folder = exported
    js = np.array(run_js("predict", {"model": str(folder), "texts": TEXTS})["probabilities"])
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
