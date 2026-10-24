"""Batch inference: turn a config + labelled training data into a validated submission.

This is what the OFFLINE Kaggle submission notebook calls. Kaggle reruns that notebook on
~4,000 hidden passages, with no internet and at most 9 hours. So this module:
  - fits on ALL of train.csv. Cross-validation was only for measuring; the final model
    should learn from every prompt we have.
  - never imports wandb (Review Focus 5): the scoring machine may not have it.
  - validates before returning. A malformed file would waste one of the 5 daily
    submissions, so validate_submission raises a precise error here instead.
"""

import pandas as pd

from name_that_llm.config import ExperimentConfig
from name_that_llm.models import build_model
from name_that_llm.schemas import validate_submission


def predict_submission(
    cfg: ExperimentConfig, train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Fit the `cfg` pipeline on all of `train`, then predict `test` (id, text).

    Returns a DataFrame (id, model) with the rows in `test` order, already validated.
    Unusual test texts (empty, very long, emoji, non-English) still get a label: the
    TF-IDF vectorizer maps unknown characters to "no features", and the classifier falls
    back to its learned class priors (Review Focus 2).
    """
    model = build_model(cfg).fit(train["text"], train["model"])
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "model": model.predict(test["text"])})
    return validate_submission(sub, test["id"])
