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
import pandas as pd
from scipy.special import softmax

from name_that_llm.labels import LABELS
from name_that_llm.metrics import score


def blend(probas: Sequence[np.ndarray], weights: Sequence[float], log: bool = False) -> np.ndarray:
    """Weighted mean of (n, 18) probability matrices; weights are normalized.

    log=False: the arithmetic mean, our blend since P3b.
    log=True (I-015, P7 Task 7): the weighted GEOMETRIC mean, renormalized so each row sums
    to 1. A member sure that a label is wrong then vetoes it, where the plain mean only
    dilutes it. Probabilities are floored at 1e-12, so a single 0 can't wipe a row out.
    """
    w = np.asarray(weights, dtype=float)
    if len(w) != len(probas) or (w < 0).any() or w.sum() == 0:
        raise ValueError(f"need one non-negative weight per model, not all zero; got {weights}")
    w = w / w.sum()
    if not log:
        return sum(wi * np.asarray(p, dtype=float) for wi, p in zip(w, probas, strict=True))
    logs = sum(
        wi * np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1.0))
        for wi, p in zip(w, probas, strict=True)
    )
    # Subtracting each row's max before exp keeps the numbers in range; it cancels out.
    out = np.exp(logs - logs.max(axis=1, keepdims=True))
    return out / out.sum(axis=1, keepdims=True)


def center(proba: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    """Take `alpha` of each group's mean log-probability out of its rows (P8 Task 7).

    A topic pulls every passage about it toward the same labels, and the group's mean is that
    pull, so removing it leaves what is particular to each passage. That only holds if a group
    has many passages: in a small one, the mean is mostly the row's own signal. So alpha stays
    below 1, where a group of one row would lose everything and become uniform.
    """
    if not 0 <= alpha < 1:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    logs = pd.DataFrame(np.log(np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)))
    return softmax((logs - alpha * logs.groupby(groups).transform("mean")).to_numpy(), axis=1)


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
    probas: Sequence[np.ndarray], y: np.ndarray, step: float = 0.05, log: bool = False
) -> tuple[tuple[float, ...], float]:
    """The weights, one per model, with the best balanced accuracy: (weights, score).

    The search covers every combination on a grid of `step` whose weights sum to 1 (the
    "simplex"). Three models at step 0.05 make 231 combinations, each one argmax away, so
    an exact search is cheap and needs no optimizer.

    Ties go to the SMALLEST weight on the LATER models: list the incumbents first and the
    newcomer last, and a newcomer that adds nothing gets weight 0. For two models this is
    exactly best_blend_weight's rule.

    log=True searches the log blend's weights (blend(log=True)): a member's confidence
    counts differently there, so its best weights needn't be the probability blend's.
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
        s = score(y, labels[blend(probas, w, log=log).argmax(axis=1)])
        if s > best_score + 1e-12:  # strictly better only, so ties keep the earlier one
            best_weights, best_score = w, s
    return tuple(float(round(x, 4)) for x in best_weights), best_score


def nested_blend_oof(
    probas: Sequence[np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    rows: np.ndarray | None = None,
    step: float = 0.05,
    log: bool = False,
) -> tuple[np.ndarray, dict[int, tuple[float, ...]]]:
    """The blend's out-of-fold predictions, its weights chosen by NESTED cross-validation.

    For each fold k among `rows`, the weights are chosen (best_blend_weights) on the OTHER
    folds' rows only, then applied to fold k's rows. So no row is ever scored with weights
    fitted on it: the honest way to weigh a member that exists on one seed only (P8's pool
    members and the LLM), where weights can't be chosen on another seed as before.

    log=True blends in log space, its weights chosen for it (best_blend_weights).
    Returns the blended probabilities (zeros outside `rows`) and each fold's weights.
    """
    y, folds = np.asarray(y), np.asarray(folds)
    rows = np.ones(len(y), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    probas = [np.asarray(p, dtype=float) for p in probas]
    out = np.zeros_like(probas[0])
    weights = {}
    for k in sorted(int(k) for k in np.unique(folds[rows])):
        fit, target = rows & (folds != k), rows & (folds == k)
        w, _ = best_blend_weights([p[fit] for p in probas], y[fit], step, log)
        out[target] = blend([p[target] for p in probas], w, log=log)
        weights[k] = tuple(w)
    return out, weights
