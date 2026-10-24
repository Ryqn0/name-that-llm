"""Training with honest validation: prompt-grouped cross-validation for one config.

`run_cv` is our backtest. Each fold's model trains on 4/5 of the prompts and predicts
the other 1/5, prompts it has never seen, just like the hidden test. Doing that for every
fold yields an OUT-OF-FOLD (OOF) prediction for every training row. OOF predictions are
the most useful artifact in the project. They give:
  - an honest score: no row was ever predicted by a model that trained on it
  - fair A/B tests: every model is scored on identical folds (see metrics.prompt_bootstrap_ci)
  - ensembling material later: OOF probabilities are what ensembles are trained on

Command line:
    uv run python -m name_that_llm.train --config lr [--data-dir DIR] [--track]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.config import ExperimentConfig, load_config
from name_that_llm.cv import make_folds
from name_that_llm.data import LABELS, load_train
from name_that_llm.metrics import score
from name_that_llm.models import build_model


@dataclass
class CVResult:
    oof: np.ndarray  # (n_rows, 18) out-of-fold probabilities, columns in LABELS order
    pred: np.ndarray  # (n_rows,) predicted label = the most probable column
    folds: np.ndarray  # (n_rows,) the fold each row was validated in
    fold_scores: list[float]  # balanced accuracy of each fold (shows the variance)
    score: float  # balanced accuracy over all OOF predictions


def run_cv(cfg: ExperimentConfig, df: pd.DataFrame) -> CVResult:
    """Cross-validate `cfg` on `df` (id, prompt_id, text, model) with prompt-grouped folds."""
    texts = df["text"].to_numpy()
    y = df["model"].to_numpy()
    folds = make_folds(df, cfg.n_splits, cfg.seed).to_numpy()

    oof = np.zeros((len(df), len(LABELS)))
    for k in np.unique(folds):
        train_rows, valid_rows = folds != k, folds == k
        # A FRESH, untrained pipeline per fold: the vectorizer's vocabulary is learned
        # from the training prompts only, so nothing leaks in from the held-out prompts.
        model = build_model(cfg).fit(texts[train_rows], y[train_rows])
        # Review Focus 3: never assume the model saw all 18 labels. If a fold's training
        # data lacked one, `classes_` has only 17 entries, and writing the probabilities
        # into columns 0..16 would silently shift every column after the gap. Mapping
        # each class to its LABELS position keeps the columns aligned. A label the model
        # never saw keeps probability 0.
        cols = [LABELS.index(c) for c in model.classes_]
        oof[np.ix_(np.flatnonzero(valid_rows), cols)] = model.predict_proba(texts[valid_rows])

    pred = np.array(LABELS)[oof.argmax(axis=1)]
    fold_scores = [score(y[folds == k], pred[folds == k]) for k in np.unique(folds)]
    return CVResult(oof=oof, pred=pred, folds=folds, fold_scores=fold_scores, score=score(y, pred))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-validate one experiment config with prompt-grouped folds."
    )
    parser.add_argument("--config", required=True, help="config name (e.g. lr) or YAML path")
    parser.add_argument("--data-dir", type=Path, default=None, help="folder with train.csv")
    parser.add_argument("--track", action="store_true", help="log the run to Weights & Biases")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df = load_train(args.data_dir)
    result = run_cv(cfg, df)
    print(f"{cfg.name}: balanced accuracy: {result.score:.4f}")
    print("per fold:", ", ".join(f"{s:.4f}" for s in result.fold_scores))

    if args.track:
        # Imported only when asked for. The offline submission path must never need
        # wandb installed or reachable (Review Focus 5).
        from name_that_llm.tracking import log_cv_run

        print("W&B run id:", log_cv_run(cfg, result, df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
