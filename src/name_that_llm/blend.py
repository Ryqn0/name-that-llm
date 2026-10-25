"""Blending: combine the probabilities of several models into one prediction (P3b).

Two models that make DIFFERENT mistakes can beat both: where the TF-IDF model hesitates,
the LM-feature model may be sure, and vice versa. A weighted average of their probabilities
is the simplest blend, and it's robust: there's just one number (the weight) to choose.

The weight is chosen on out-of-fold predictions, so it reflects how the models do on
prompts they never saw. Anything chosen this way is slightly optimistic, which is why the
plan picks it on seed-7 folds and reports it on seeds 42, 43 and 44.
"""

import itertools
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


def best_blend_weights(
    probas: Sequence[np.ndarray], y: np.ndarray, step: float = 0.05
) -> tuple[tuple[float, ...], float]:
    """The weights, one per model, with the best balanced accuracy: (weights, score).

    The search covers every combination on a grid of `step` whose weights sum to 1 (the
    "simplex"). Three models at step 0.05 make 231 combinations, each one argmax away, so
    an exact search is cheap and needs no optimizer.

    Ties go to the SMALLEST weight on the LATER models: list the incumbents first and the
    newcomer last, and a newcomer that adds nothing gets weight 0. For two models this is
    exactly best_blend_weight's rule.
    """
    n = round(1 / step)
    if n < 1 or abs(n * step - 1) > 1e-9:
        raise ValueError(f"step must divide 1 evenly, got {step}")
    labels = np.asarray(LABELS)
    best_weights, best_score = None, -1.0
    # idx holds the later models' weights in grid steps, idx[0] being the LAST model's.
    # product() varies its last position fastest, so the last model's weight grows
    # slowest: the first combination to reach a score is the one leaning least on it.
    for idx in itertools.product(range(n + 1), repeat=len(probas) - 1):
        rest = n - sum(idx)
        if rest < 0:
            continue  # the weights would sum to more than 1
        w = np.array([rest, *reversed(idx)]) / n
        s = score(y, labels[blend(probas, w).argmax(axis=1)])
        if s > best_score + 1e-12:  # strictly better only, so ties keep the earlier one
            best_weights, best_score = w, s
    return tuple(float(round(x, 4)) for x in best_weights), best_score
