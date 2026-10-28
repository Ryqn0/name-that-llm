"""The browser demo's model (P4 revision): `lr`, exported for JavaScript.

Why: Hugging Face now charges for Docker Spaces, so the free public demo is a STATIC page,
and its model runs inside the visitor's browser (app/demo/ntl.js). The full champion needs
Qwen and a GPU-class amount of memory, so the demo runs the TF-IDF model `lr` alone.

What gets exported, from a fitted `lr` pipeline (TF-IDF word and character n-grams, then a
logistic regression):
  model.json   the 18 labels and their labs, each vectorizer's settings and vocabulary (the
               term of every column, in column order), the "How sure?" bands, and `meta`
  weights.bin  float32 little-endian, in this order: the words' idf, the characters' idf,
               the 18 x n coefficients (row by row), the 18 intercepts

The JavaScript engine re-implements scikit-learn's computation for EXACTLY the settings we
use; tests/test_browser.py checks it number for number. Any other setting is refused here,
loudly, rather than silently giving different numbers in the browser.
"""

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer

from name_that_llm.champion import Band
from name_that_llm.config import ExperimentConfig
from name_that_llm.cv import make_folds
from name_that_llm.data import flatten_text
from name_that_llm.labels import LAB_NAMES, LAB_OF, LABELS
from name_that_llm.models import build_model

# The demo's folder in the repo: the engine (ntl.js), its Node harness (cli.mjs), the page,
# and MODEL_RUN. Only checks run from the repo use it: an installed wheel has no app/.
DEMO_DIR = Path(__file__).resolve().parents[2] / "app" / "demo"

# What the engine implements. Every other parameter must have scikit-learn's default.
_EXPECTED = {
    "words": {"analyzer": "word", "lowercase": True, "token_pattern": r"(?u)\b\w\w+\b"},
    "chars": {"analyzer": "char", "lowercase": False},
}
_SHARED = {
    "sublinear_tf": True,
    "norm": "l2",
    "use_idf": True,
    "binary": False,
    "stop_words": None,
    "strip_accents": None,
    "preprocessor": None,
    "tokenizer": None,
}


def export_browser_model(
    text_model, bands: Sequence[Band], out_dir: Path, meta: dict
) -> list[Path]:
    """Write model.json and weights.bin for the browser engine; return their paths."""
    union, clf = text_model.named_steps["features"], text_model.named_steps["clf"]
    vectorizers = dict(union.transformer_list)
    _check(union, vectorizers, clf)

    blocks, arrays = {}, []
    for name in ("words", "chars"):  # the same order as the coefficient columns
        vectorizer = vectorizers[name]
        params = vectorizer.get_params()
        blocks[name] = {
            "analyzer": params["analyzer"],
            "lowercase": params["lowercase"],
            "ngram_range": list(params["ngram_range"]),
            # The term of each column, in column order: a list is half the size of a dict.
            "vocabulary": sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get),
        }
        arrays.append(vectorizer.idf_)
    arrays += [clf.coef_, clf.intercept_]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "labels": LABELS,
        "labs": {label: LAB_NAMES[LAB_OF[label]] for label in LABELS},
        **blocks,
        "bands": [vars(band) for band in bands],
        "meta": meta,
    }
    (out_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    weights = np.concatenate([np.asarray(a, dtype="<f4").ravel() for a in arrays])
    weights.tofile(out_dir / "weights.bin")
    return [out_dir / "model.json", out_dir / "weights.bin"]


def _check(union, vectorizers: dict, clf) -> None:
    """Refuse anything the JavaScript engine doesn't reproduce."""
    names = [name for name, _ in union.transformer_list]
    if names != ["words", "chars"]:
        raise ValueError(f"expected the FeatureUnion parts ['words', 'chars'], got {names}")
    for name, expected in _EXPECTED.items():
        params = vectorizers[name].get_params()
        for key, value in {**_SHARED, **expected}.items():
            if params[key] != value:
                raise ValueError(
                    f"the browser engine doesn't implement {name}.{key}={params[key]!r} "
                    f"(it expects {value!r})"
                )
    if list(clf.classes_) != LABELS or clf.coef_.shape[0] != len(LABELS):
        raise ValueError("expected a multinomial logistic regression over the 18 LABELS")


def unknown_share(texts: Sequence[str], vocabulary: set[str]) -> np.ndarray:
    """The share of each text's words that `vocabulary` doesn't know (P6, I-025).

    Words as `lr`'s word analyzer sees them, unigrams only: the text flattened (as the page
    does), lowercased, then runs of 2 or more word characters. ntl.js's textStats computes
    the same number. A text with no word at all scores 0: it's flagged as short instead.
    """
    analyze = TfidfVectorizer(**_EXPECTED["words"], ngram_range=(1, 1)).build_analyzer()
    out = np.zeros(len(texts))
    for i, text in enumerate(texts):
        tokens = analyze(flatten_text(str(text)))
        if tokens:
            out[i] = sum(token not in vocabulary for token in tokens) / len(tokens)
    return out


def unusual_thresholds(cfg: ExperimentConfig, df: pd.DataFrame, quantile: float = 0.99) -> dict:
    """The demo's limits for an "unusual" text, measured on our own passages (P6, I-025).

      - min_words: the shortest training passage, in words after flattening (as the page
        counts them);
      - max_unknown_share: the `quantile` of HELD-OUT unknown-word shares. Each passage is
        measured against the vocabulary of its fold's training rows only, the way a
        visitor's text meets a model that never saw it. Against the full vocabulary, every
        training word would be known, and the limit would be 0;
      - max_non_latin_share: 0.2, a fixed rule, since our passages are English.

    With quantile 0.99, about 1% of our own held-out passages exceed max_unknown_share.
    """
    texts = df["text"].astype(str).tolist()
    folds = make_folds(df, cfg.n_splits, cfg.seed).to_numpy()
    words = dict(build_model(cfg).named_steps["features"].transformer_list)["words"]
    shares = np.zeros(len(texts))
    for k in np.unique(folds):
        fitted = clone(words).fit([texts[i] for i in np.flatnonzero(folds != k)])
        vocabulary = {term for term in fitted.vocabulary_ if " " not in term}
        rows = np.flatnonzero(folds == k)
        shares[rows] = unknown_share([texts[i] for i in rows], vocabulary)
    return {
        "min_words": min(len(flatten_text(t).split()) for t in texts),
        "max_unknown_share": float(np.quantile(shares, quantile)),
        "max_non_latin_share": 0.2,
    }


def run_engine(command: str, payload: dict) -> dict:
    """Run the JavaScript engine under Node (app/demo/cli.mjs): JSON in, JSON out.

    The page's own engine, so Python can check it against scikit-learn: the tests, and the
    export's parity check on real passages (scripts/export_browser_model.py).
    """
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("running the browser engine needs Node.js (https://nodejs.org)")
    done = subprocess.run(
        [node, str(DEMO_DIR / "cli.mjs"), command],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(done.stdout)
