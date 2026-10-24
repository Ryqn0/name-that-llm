"""Experiment configuration: every setting that defines a run, in one typed object.

Why a config object instead of loose function arguments? Each experiment then IS its config.
The config gets logged with the results (W&B, from Task 8), so any score can be reproduced
later from exactly the settings that produced it. Pydantic validates the values when they're
loaded, so a typo in a YAML file fails immediately instead of silently using a default.
"""

from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class ExperimentConfig(BaseModel):
    # extra="forbid": an unknown key (e.g. "c" instead of "C") is an error, not ignored.
    model_config = ConfigDict(extra="forbid")

    name: str
    model: Literal["mnb", "cnb", "lr", "svm", "nbsvm"]

    # Features: word n-grams (vocabulary habits) + character n-grams (punctuation,
    # spacing, word endings, casing). Both kinds are TF-IDF weighted.
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (1, 5)
    min_df: int = 2  # ignore n-grams seen in fewer documents (mostly noise)
    max_features: int | None = 200_000  # per vectorizer; caps memory and training time

    # Model hyperparameters (each model reads only the ones it uses):
    C: float = 1.0  # inverse regularization strength: bigger = trust the data more
    alpha: float = 0.1  # Naive Bayes smoothing: avoids zero probabilities for unseen n-grams

    # Validation (backtesting): grouped folds on prompt_id, fixed seed (spec §6.6).
    n_splits: int = 5
    seed: int = 42


def load_config(name_or_path: str | Path) -> ExperimentConfig:
    """Load a config by NAME (a file shipped inside the package) or by PATH (any YAML file).

    The named configs live inside the package (`name_that_llm/configs/`), not at the repo
    root. That way the wheel used by the offline Kaggle notebook carries them along
    (ADR 0001).
    """
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} or path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        resource = files("name_that_llm") / "configs" / f"{name_or_path}.yaml"
        if not resource.is_file():
            available = sorted(p.name.removesuffix(".yaml") for p in resource.parent.iterdir())
            raise FileNotFoundError(f"no config named {name_or_path!r}; available: {available}")
        text = resource.read_text(encoding="utf-8")
    return ExperimentConfig(**yaml.safe_load(text))
