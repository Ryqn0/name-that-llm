"""The app's predictor (P4): the leaderboard champion, run on one passage at a time.

It computes exactly what the offline submission computes (notebooks/90-submit), step by
step, so the live app and the leaderboard score describe the same model:

  submission notebook                         here
  ------------------------------------------  -----------------------------------------
  (passages arrive already flattened)         flatten_text: chatbot markdown -> one paragraph
  predict_proba_test(lr)                      _aligned_proba(text_model, ...)
  Qwen 3 4B hidden states 6-9, mean-pooled    the same, from a Qwen cut to 10 blocks (lm.py)
  LM head (scaled logistic regression)        the same fitted head
  predict_proba_folds: mean of 5 fold models  the same mean, with the models kept in memory
  blend(..., weights 0.70 / 0.15 / 0.15)      the same blend and weights (ensemble.yaml)

The one structural difference: the submission loads each fold model, predicts the whole
test set, then frees it. A web app answers one passage at a time, and reloading 5 models
per request would cost seconds, so the Champion keeps them loaded.

It also turns the champion's confidence into plain words: the "How sure?" bands, measured
on out-of-fold predictions (fit_confidence_bands).

torch and transformers are only needed by the models handed in, never imported here
(ADR 0002), so this module loads fine wherever the text-only code does.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from name_that_llm.blend import blend
from name_that_llm.data import flatten_text
from name_that_llm.encoder import predict_proba
from name_that_llm.labels import LABELS
from name_that_llm.lm import pooled_hidden_states, select_layers
from name_that_llm.submit import _aligned_proba

# From the most confident band to the least. The wording is fixed; how often each band's
# best guess is RIGHT comes from data (fit_confidence_bands), not from guesswork.
BAND_LABELS = ("Fairly sure", "Unsure", "Very unsure")


@dataclass(frozen=True)
class Band:
    """One "How sure?" band: its words, where it starts, and how often it's right."""

    label: str
    min_top: float  # the lowest top probability inside this band
    accuracy: float  # share of out-of-fold rows in this band whose best guess was right


def fit_confidence_bands(proba: np.ndarray, y: np.ndarray) -> list[Band]:
    """Three bands from the out-of-fold predictions, the most confident first.

    The rows are sorted by their top probability and split into three equal groups
    (terciles). Each band records its lowest top probability and how often its best guess
    was right: that's what lets the page say "at this level, it's right about N times in
    10" honestly, instead of trusting the raw percentages (which aren't calibrated).
    """
    top = proba.max(axis=1)
    right = np.asarray(LABELS)[proba.argmax(axis=1)] == np.asarray(y)
    order = np.argsort(-top, kind="stable")  # the most confident rows first
    bands = [
        Band(label, float(top[rows].min()), float(right[rows].mean()))
        for label, rows in zip(BAND_LABELS, np.array_split(order, len(BAND_LABELS)), strict=True)
    ]
    # The last band starts at 0, so ANY confidence, however low, lands in a band.
    bands[-1] = Band(bands[-1].label, 0.0, bands[-1].accuracy)
    return bands


def describe_confidence(top: float, bands: Sequence[Band]) -> Band:
    """The band a top probability falls into: the first (most confident) it reaches."""
    return next(band for band in bands if top >= band.min_top)


@dataclass
class Champion:
    """The full champion, ready to predict: three fitted models and their blend weights.

    `lm` is (model, tokenizer) from lm.load_lm_layers; `encoders` lists (model, tokenizer)
    pairs from encoder.load_encoder, in fold order. `lm_train_mean` is the mean of the
    training features: the neutral input when a passage's LM features aren't finite.
    """

    text_model: object
    lm_head: object
    lm_train_mean: np.ndarray
    lm: tuple
    encoders: list
    weights: tuple[float, float, float]  # text model, LM head, encoder (ensemble.yaml)
    lm_layer: int = 6
    lm_window: int = 4
    lm_max_length: int = 512
    encoder_max_length: int = 512

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """(n, 18) probabilities, columns in LABELS order; each row sums to 1."""
        flat = [flatten_text(t) for t in texts]
        p_text = _aligned_proba(self.text_model, flat)
        return blend([p_text, self._lm_proba(flat), self._encoder_proba(flat)], self.weights)

    def _lm_proba(self, flat: list[str]) -> np.ndarray:
        model, tokenizer = self.lm
        layers = range(self.lm_layer, self.lm_layer + self.lm_window)
        features = pooled_hidden_states(
            flat, model, tokenizer, layers=layers, batch_size=8, max_length=self.lm_max_length
        )
        X = select_layers(features, 0, self.lm_window)  # the mean of layers 6 to 9
        # LENIENT, like the submission: an overflow on an unusual passage costs that
        # passage's LM opinion, not the request. The training mean scales to all zeros,
        # so the LM head gives a near-neutral answer and the other two models decide.
        X[~np.isfinite(X).all(axis=1)] = self.lm_train_mean
        return _aligned_proba(self.lm_head, X)

    def _encoder_proba(self, flat: list[str]) -> np.ndarray:
        p = np.mean(
            [
                predict_proba(
                    model, tokenizer, flat, batch_size=8, max_length=self.encoder_max_length
                )
                for model, tokenizer in self.encoders
            ],
            axis=0,
        )
        # LENIENT too: a uniform row adds the same amount to all 18 labels, so it can't
        # change which label wins. The other two models decide that passage.
        p[~np.isfinite(p).all(axis=1)] = 1 / p.shape[1]
        return p
