"""Per-prompt assignment: each model answers each prompt (at most) once, so use that.

A classifier labels passages one at a time. Inside one prompt group, it might call two
passages "gpt-5.5". But each model answered each prompt only once, so one of those two
labels must be wrong. Solving the whole group at once fixes that. We choose the
combination of labels, one per passage, with each model used at most once, that is
jointly the most probable. That's a classic ASSIGNMENT PROBLEM (like matching workers to
jobs), solved exactly by the Hungarian algorithm: scipy's `linear_sum_assignment`.

Why log-probabilities as costs: the probability of a whole assignment is the PRODUCT of
each passage's probability for its label. Minimizing the sum of -log(p) maximizes that
product, and the solver minimizes sums.

Spike on train (2026-10-24, `lr` OOF probabilities): 0.321 plain -> 0.406 with groups
rebuilt from topic, and 0.413 with the true prompt groups.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from name_that_llm.labels import LABELS


def assign_labels(proba: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Assign one label per row, using each label at most ceil(size / 18) times per group.

    proba:  (n, 18) probabilities, columns in LABELS order.
    groups: (n,) group id per row (e.g. from grouping.topic_groups).
    Returns n label strings in the INPUT row order (Review Focus 4).
    """
    # Clip before the log. A probability of exactly 0 (a label a fold never saw) would cost
    # +inf and break the solver. 1e-12 makes it merely "very unlikely" (Review Focus 3).
    cost = -np.log(np.clip(proba, 1e-12, None))
    labels = np.asarray(LABELS)
    out = np.empty(len(proba), dtype=object)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        # "Slots": a group of 18 or fewer passages gets each model once. A larger group
        # (two prompts merged by the clustering) gets each model ceil(n/18) times. Tiling
        # the cost columns k times gives the solver k copies of every label to choose from
        # (Review Focus 2).
        k = -(-len(rows) // len(LABELS))  # ceiling division
        row_idx, col_idx = linear_sum_assignment(np.tile(cost[rows], (1, k)))
        # row_idx indexes `rows` (the solver returns them sorted), so writing through
        # rows[row_idx] puts each label back at the passage's original position.
        out[rows[row_idx]] = labels[col_idx % len(LABELS)]
    return out.astype(str)
