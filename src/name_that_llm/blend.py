"""Blending: combine the probabilities of several models into one prediction (P3b).

Two models that make DIFFERENT mistakes can beat both: where the TF-IDF model hesitates,
the LM-feature model may be sure, and vice versa. A weighted average of their probabilities
is the simplest blend, and it's robust: there's just one number (the weight) to choose.

The weight is chosen on out-of-fold predictions, so it reflects how the models do on
prompts they never saw. Anything chosen this way is slightly optimistic, which is why the
plan picks it on seed-7 folds and reports it on seeds 42, 43 and 44.
"""

from collections.abc import Sequence

import numpy as np

from name_that_llm.labels import LABELS
from name_that_llm.metrics import score


def blend(probas: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted arithmetic mean of (n, 18) probability matrices; weights are normalized."""
    w = np.asarray(weights, dtype=float)
    if len(w) != len(probas) or (w < 0).any() or w.sum() == 0:
        raise ValueError(f"need one non-negative weight per model, not all zero; got {weights}")
    w = w / w.sum()
    return sum(wi * np.asarray(p, dtype=float) for wi, p in zip(w, probas, strict=True))


def best_blend_weight(
    p_text: np.ndarray,
    p_lm: np.ndarray,
    y: np.ndarray,
    grid: Sequence[float] = tuple(np.round(np.linspace(0, 1, 21), 2)),
) -> tuple[float, float]:
    """The LM weight in `grid` with the best balanced accuracy: (weight, score).

    Ties go to the SMALLER weight: if leaning on the LM model doesn't help, the simpler
    blend (more text model) wins. `grid` is scanned in increasing order for that reason.
    """
    labels = np.asarray(LABELS)
    best_weight, best_score = 0.0, -1.0
    for w in sorted(grid):
        s = score(y, labels[((1 - w) * p_text + w * p_lm).argmax(axis=1)])
        if s > best_score + 1e-12:  # strictly better only, so ties keep the smaller weight
            best_weight, best_score = float(w), s
    return best_weight, best_score
