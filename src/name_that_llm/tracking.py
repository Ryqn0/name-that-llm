"""Experiment tracking with Weights & Biases (W&B): the project's "lab notebook".

Every cross-validation run gets logged with its exact config, its scores, and two versioned
artifacts:
  - `train-data` (input): which data version the run used. That's lineage. W&B
    deduplicates by content, so identical data is stored only once.
  - `oof-<config name>` (output): the out-of-fold probabilities plus the fold of each row.
    Later A/B tests and ensembles start from these, without re-training anything.

The W&B project must stay PRIVATE: the `train-data` artifact contains competition data,
which the rules forbid redistributing.

wandb is imported INSIDE the function, so importing this module (or the training code)
never requires wandb (Review Focus 5).
"""

import tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss

from name_that_llm.config import ExperimentConfig
from name_that_llm.data import LABELS
from name_that_llm.metrics import lab_accuracy, per_class_recall

PROJECT = "name-that-llm"


def log_cv_run(cfg: ExperimentConfig, result, df: pd.DataFrame) -> str:
    """Log one CVResult (from train.run_cv) to W&B, and return the run id."""
    import wandb

    run = wandb.init(project=PROJECT, job_type="cv", name=cfg.name, config=cfg.model_dump())
    y = df["model"].to_numpy()

    # Summary metrics: one number each, shown in the W&B runs table (sortable, comparable).
    run.summary["cv_balanced_accuracy"] = result.score
    run.summary["lab_accuracy"] = lab_accuracy(y, result.pred)
    # Log loss judges the PROBABILITIES, not just the top pick: confident mistakes cost a
    # lot. It's the number to watch for calibration (it matters for the I-001 assignment).
    run.summary["oof_log_loss"] = log_loss(y, result.oof, labels=LABELS)
    run.summary["fold_scores"] = result.fold_scores

    recall = per_class_recall(y, result.pred).rename_axis("model").reset_index()
    label_index = {label: i for i, label in enumerate(LABELS)}
    run.log(
        {
            "per_class_recall": wandb.Table(dataframe=recall),
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=[label_index[m] for m in y],
                preds=[label_index[m] for m in result.pred],
                class_names=LABELS,
            ),
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "train.parquet"
        df.to_parquet(data_file, index=False)
        data_artifact = wandb.Artifact("train-data", type="dataset")
        data_artifact.add_file(str(data_file))
        if run.offline:
            # W&B can't declare inputs in offline mode (tests, CI, no network), so log the
            # data as an output instead. Its version is still attached to the run, so the
            # lineage survives.
            run.log_artifact(data_artifact)
        else:
            run.use_artifact(data_artifact)  # recorded as this run's INPUT

        oof = pd.DataFrame(result.oof, columns=LABELS)
        oof.insert(0, "fold", result.folds)
        oof.insert(0, "id", df["id"].to_numpy())
        oof_file = Path(tmp) / f"oof-{cfg.name}.parquet"
        oof.to_parquet(oof_file, index=False)
        oof_artifact = wandb.Artifact(f"oof-{cfg.name}", type="oof")
        oof_artifact.add_file(str(oof_file))
        run.log_artifact(oof_artifact)  # recorded as this run's OUTPUT

        run.finish()  # inside the `with`, while the temporary files still exist
    return run.id
