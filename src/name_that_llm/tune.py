"""Hyperparameter tuning of the LM blend with Optuna (P3b Task 6).

What gets tuned: which Qwen size, which layer (and how many neighboring layers to average),
the regularization C of the LM-side logistic regression, and the blend weight. Five knobs
that interact, which is why a joint search beats tweaking them one at a time.

Why Optuna's TPE sampler instead of a grid: a full grid over 5 knobs is thousands of CVs.
TPE ("Tree-structured Parzen Estimator") learns from finished trials which regions look
good and samples more there, so ~60 trials explore the space far better than 60 grid points.

Why seed-7 folds: every trial is scored on the SAME folds, so trials are comparable. The
final config is then re-measured on different folds (seeds 42/43/44) in the notebook. The
best of 60 trials on one split is partly luck, and fresh folds reveal how much.

No pruning: one trial is just 5 dense fits (seconds to a minute), so stopping bad trials
early would save little. Add optuna's MedianPruner if trials ever get expensive.

P5 adds a second study, for a fine-tuned encoder (run_encoder_study). Its trials are GPU
fine-tuning runs of several minutes each, so that study warm-starts from the default config
and prunes bad trials early.

This module imports optuna at the top on purpose: only the tuning notebooks import it. The
text-only offline path never does (tests/test_train.py checks that).
"""

import numpy as np
import optuna
import pandas as pd

from name_that_llm.blend import blend
from name_that_llm.config import BlendConfig, EncoderConfig, load_config
from name_that_llm.cv import make_folds
from name_that_llm.encoder import fit_predict_fold
from name_that_llm.labels import LABELS
from name_that_llm.lm import select_layers
from name_that_llm.metrics import score
from name_that_llm.train import run_cv, run_cv_features


def suggest_blend(trial: optuna.Trial, n_layers: dict[str, int]) -> BlendConfig:
    """Sample one blend config. n_layers maps each LM variation to its number of layers."""
    variation = trial.suggest_categorical("lm_variation", sorted(n_layers))
    total = n_layers[variation]
    # One layer parameter PER variation: their valid ranges differ (29 vs 37 layers), and
    # a consistent range per name lets TPE learn each one properly.
    layer = trial.suggest_int(f"layer_{variation}", 0, total - 1)
    window = trial.suggest_categorical("layer_window", [1, 2, 4])
    window = min(window, total - layer)  # clip so the window fits after `layer`
    return BlendConfig(
        name="trial",
        lm_variation=variation,
        layer=layer,
        layer_window=window,
        C=trial.suggest_float(
            "C", 0.01, 10.0, log=True
        ),  # log scale: 0.01..0.1 matters as much as 1..10
        weight=trial.suggest_float("weight", 0.0, 1.0),
    )


def run_study(
    features: dict[str, np.ndarray],
    df: pd.DataFrame,
    n_trials: int,
    seed: int = 7,
    text_config: str = "lr",
) -> optuna.Study:
    """Search blend configs on seed-`seed` folds; the objective is the blend's balanced accuracy.

    `features` maps variation -> (layers, n_rows, hidden), rows aligned with `df`.
    """
    n_layers = {variation: len(F) for variation, F in features.items()}
    y = df["model"].to_numpy()
    labels = np.asarray(LABELS)
    # The text side doesn't change between trials: compute its OOF once, on the same folds.
    p_text = run_cv(load_config(text_config).model_copy(update={"seed": seed}), df).oof

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_blend(trial, n_layers)
        X = select_layers(features[cfg.lm_variation], cfg.layer, cfg.layer_window)
        lm = run_cv_features(X, df, C=cfg.C, n_splits=cfg.n_splits, seed=seed)
        trial.set_user_attr("config", cfg.model_dump())  # the resolved config (window clipped)
        trial.set_user_attr("lm_cv", lm.score)
        mixed = blend([p_text, lm.oof], [1 - cfg.weight, cfg.weight])
        return score(y, labels[mixed.argmax(axis=1)])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def best_blend_config(study: optuna.Study, name: str = "lm_blend_tuned") -> BlendConfig:
    """The best trial's config, renamed, with the standard evaluation seed (42)."""
    return BlendConfig(**{**study.best_trial.user_attrs["config"], "name": name, "seed": 42})


def suggest_encoder(trial: optuna.Trial, base: EncoderConfig) -> EncoderConfig:
    """Sample the three knobs that matter most when fine-tuning; keep the rest of `base`."""
    return base.model_copy(
        update={
            # log scale: 5e-6 -> 1e-5 matters as much as 5e-5 -> 1e-4
            "lr": trial.suggest_float("lr", 5e-6, 1e-4, log=True),
            "epochs": trial.suggest_int("epochs", 2, 5),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }
    )


def run_encoder_study(
    base: EncoderConfig,
    df: pd.DataFrame,
    n_trials: int,
    seed: int = 7,
    tune_folds: tuple[int, ...] = (0, 1),
) -> optuna.Study:
    """Tune lr, epochs and warmup on `tune_folds` of the seed-`seed` folds.

    A trial is expensive (a GPU fine-tuning run per fold), hence three choices:
      - Proxy folds: each trial trains on only `tune_folds` (2 of the 5 folds by default).
        The winner is re-measured on full, fresh CV later (the final notebook).
      - Warm start: trial 0 is ALWAYS `base`'s own values, so the result can't end below the
        default on these folds. The P3b blend tuning lacked this, and its 60 trials never
        reached the hand-picked config.
      - Fold-level pruning: after each fold, the running mean is reported. A trial below
        the median of earlier trials at the same point stops there (MedianPruner), which
        halves the cost of a bad trial. Pruning per EPOCH instead would compare learning
        rates midway through schedules of different lengths, which isn't fair.
    The epochs are a tuned, FIXED number, never early-stopped on a validation fold: that
    would leak the fold into its own score.
    """
    texts, y = df["text"].tolist(), df["model"].to_numpy()
    folds = make_folds(df, base.n_splits, seed).to_numpy()
    labels = np.asarray(LABELS)

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_encoder(trial, base).model_copy(update={"seed": seed})
        trial.set_user_attr("config", cfg.model_dump())  # before training: pruned ones too
        scores = []
        for step, k in enumerate(tune_folds):
            proba, _ = fit_predict_fold(cfg, texts, y, folds, k)
            scores.append(score(y[folds == k], labels[proba.argmax(axis=1)]))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )
    study.enqueue_trial({"lr": base.lr, "epochs": base.epochs, "warmup_ratio": base.warmup_ratio})
    study.optimize(objective, n_trials=n_trials)
    return study


def best_encoder_config(study: optuna.Study, name: str = "encoder") -> EncoderConfig:
    """The best trial's config, renamed, with the standard evaluation seed (42)."""
    return EncoderConfig(**{**study.best_trial.user_attrs["config"], "name": name, "seed": 42})
