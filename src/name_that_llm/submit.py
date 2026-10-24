"""Batch inference: turn a config + labelled training data into a validated submission.

This is what the OFFLINE Kaggle submission notebook calls. Kaggle reruns that notebook on
~4,000 hidden passages, with no internet and at most 9 hours. So this module:
  - fits on ALL of train.csv. Cross-validation was only for measuring; the final model
    should learn from every prompt we have.
  - never imports wandb (Review Focus 5): the scoring machine may not have it.
  - validates before returning. A malformed file would waste one of the 5 daily
    submissions, so validate_submission raises a precise error here instead.
"""

import numpy as np
import pandas as pd

from name_that_llm.assign import assign_labels
from name_that_llm.blend import blend
from name_that_llm.config import BlendConfig, ExperimentConfig
from name_that_llm.grouping import topic_groups
from name_that_llm.labels import LABELS
from name_that_llm.models import build_dense_model, build_model
from name_that_llm.schemas import validate_submission


def _aligned_proba(model, X) -> np.ndarray:
    """predict_proba with columns in LABELS order. Map them explicitly, as run_cv does,
    instead of trusting that model.classes_ happens to match."""
    proba = np.zeros((len(X), len(LABELS)))
    proba[:, [LABELS.index(c) for c in model.classes_]] = model.predict_proba(X)
    return proba


def predict_proba_test(
    cfg: ExperimentConfig, train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    """Fit the `cfg` text pipeline on all of `train`; (n_test, 18) probabilities for `test`."""
    model = build_model(cfg).fit(train["text"], train["model"])
    return _aligned_proba(model, test["text"])


def predict_submission(
    cfg: ExperimentConfig, train: pd.DataFrame, test: pd.DataFrame, assign: bool = False
) -> pd.DataFrame:
    """Fit the `cfg` pipeline on all of `train`, then predict `test` (id, text).

    Returns a DataFrame (id, model) with the rows in `test` order, already validated.
    Unusual test texts (empty, very long, emoji, non-English) still get a label: the
    TF-IDF vectorizer maps unknown characters to "no features", and the classifier falls
    back to its learned class priors (Review Focus 2).

    assign=True: rebuild the hidden prompts from topic (grouping.topic_groups), then give
    each group a one-model-per-passage assignment (assign.assign_labels). It's OFF by
    default: it only helps if the hidden test shares train's "18 answers per prompt" shape,
    which the leaderboard A/B test checks.
    """
    proba = predict_proba_test(cfg, train, test)
    if assign:
        labels = assign_labels(proba, topic_groups(test["text"].tolist()))
    else:
        labels = np.asarray(LABELS)[proba.argmax(axis=1)]
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "model": labels})
    return validate_submission(sub, test["id"])


def predict_submission_blend(
    text_cfg: ExperimentConfig,
    blend_cfg: BlendConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    lm_train: np.ndarray,
    lm_test: np.ndarray,
) -> pd.DataFrame:
    """The P3b submission: the TF-IDF text model blended with an LM-feature model.

    `lm_train` / `lm_test` are the LM features (one row per passage, same order as `train`
    and `test`), computed in the notebook with lm.pooled_hidden_states + lm.select_layers.
    This module stays torch-free: it only sees the resulting arrays.
    """
    if len(lm_train) != len(train) or len(lm_test) != len(test):
        # A length mismatch means rows can't be matched to passages: refuse rather than
        # silently pair features with the wrong passages.
        raise ValueError(
            f"LM features don't match the data: {len(lm_train)} vs {len(train)} train rows, "
            f"{len(lm_test)} vs {len(test)} test rows"
        )
    p_text = predict_proba_test(text_cfg, train, test)
    head = build_dense_model(blend_cfg.C).fit(lm_train, train["model"])
    p_lm = _aligned_proba(head, lm_test)
    mixed = blend([p_text, p_lm], [1 - blend_cfg.weight, blend_cfg.weight])
    sub = pd.DataFrame(
        {"id": test["id"].to_numpy(), "model": np.asarray(LABELS)[mixed.argmax(axis=1)]}
    )
    return validate_submission(sub, test["id"])
