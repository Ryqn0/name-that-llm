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
    # TODO(you): write the body here, in 5-10 lines. Suggested steps:
    #   1. rng = np.random.default_rng(seed). The seed makes results reproducible.
    #   2. Find the unique groups, and precompute the row indices of each group
    #      (a dict {group: np.flatnonzero(groups == group)}), so the loop stays fast.
    #   3. Repeat n_boot times: draw len(unique groups) groups WITH replacement
    #      (rng.choice(..., replace=True)), concatenate their row indices, and store
    #      score(y_true[idx], pred_b[idx]) - score(y_true[idx], pred_a[idx]).
    #   4. lo, hi = the alpha/2 and 1 - alpha/2 quantiles of the stored differences
    #      (np.quantile).
    #   5. Return (observed difference on ALL rows, lo, hi), each as a float.
    # Check your work with:  uv run --all-extras pytest tests/test_metrics.py -q
    raise NotImplementedError("prompt_bootstrap_ci: your turn! See the TODO above.")
