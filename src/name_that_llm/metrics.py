"""Scoring: the competition metric, diagnostics, and an honest A/B comparison.

- score:               the official metric (balanced accuracy)
- per_class_recall:    which of the 18 models we recognize well, and which we miss
- lab_accuracy:        are our mistakes "near misses" (right lab, wrong model)?
- prompt_bootstrap_ci: is model B REALLY better than model A, or did it just get lucky?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from name_that_llm.labels import LAB_OF, LABELS


def score(y_true, y_pred) -> float:
    """Balanced accuracy: the average over classes of each class's recall.

    Recall of class c is the share of c's passages we labelled c. Averaging per class means
    every model counts equally. Our classes are balanced, so this is close to plain
    accuracy, but we always use the official metric.
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def per_class_recall(y_true, y_pred) -> pd.Series:
    """Recall for each of the 18 labels, in LABELS order. NaN where a label never occurs."""
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=np.nan)
    return pd.Series(recalls, index=LABELS, name="recall")


def lab_accuracy(y_true, y_pred) -> float:
    """Share of passages whose predicted model comes from the RIGHT LAB.

    If lab accuracy is much higher than model accuracy, our errors are mostly confusions
    between siblings (e.g. gpt-5.6-luna vs gpt-5.6-sol). That points toward a two-step
    model: first pick the lab, then the model within the lab (spec §6.7, M3).
    """
    labs_true = np.array([LAB_OF[m] for m in y_true])
    labs_pred = np.array([LAB_OF[m] for m in y_pred])
    return float((labs_true == labs_pred).mean())


def prompt_bootstrap_ci(
    y_true, pred_a, pred_b, groups, n_boot=2000, seed=0, alpha=0.05
) -> tuple[float, float, float]:
    """How much better is model B than model A, with a confidence interval?

    Inputs are row-aligned arrays: the true labels, A's and B's predictions, and each
    row's group (its prompt_id). Returns three floats:
        (observed difference, lower bound, upper bound)
    where difference = score(B) - score(A). If the lower bound is above 0, B is better
    with about (1 - alpha) confidence. If the interval contains 0, the "improvement" may
    just be luck.

    THE BOOTSTRAP IDEA: we only have one dataset, so we can't re-run the world to see how
    much the difference would vary. Instead, we simulate many alternative datasets by
    resampling the data we have WITH REPLACEMENT, recomputing the difference each time.
    The spread of those differences shows how uncertain the observed one is.

    WHY RESAMPLE PROMPTS, NOT ROWS: the 18 answers to one prompt are linked. They share a
    topic, and a hard prompt is hard for both models. Resampling individual rows would
    pretend we have 2,699 independent data points and make the interval too narrow
    (overconfident). Resampling whole prompts keeps each prompt's 18 rows together. It
    also keeps all 18 classes in every resample, which balanced accuracy needs.
    """
    # (Body written by Ryqn0 as a learning contribution, then reviewed.)
    # Work on plain NumPy arrays. Indexing a pandas Series with an integer array looks the
    # integers up as index LABELS, which silently picks the wrong rows (or raises KeyError)
    # once the index isn't 0..n-1, e.g. after filtering.
    y_true, pred_a, pred_b, groups = (np.asarray(a) for a in (y_true, pred_a, pred_b, groups))

    # PAIRED bootstrap: in each round, both models are scored on the SAME resampled rows,
    # so "this resample happened to be easy" cancels out of the difference.
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    differences = []

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[g] for g in sampled_groups])
        diff = score(y_true[sampled_indices], pred_b[sampled_indices]) - score(
            y_true[sampled_indices], pred_a[sampled_indices]
        )
        differences.append(diff)

    lo, hi = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    observed_diff = score(y_true, pred_b) - score(y_true, pred_a)
    return float(observed_diff), float(lo), float(hi)
