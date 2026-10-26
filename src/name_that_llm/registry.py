"""Packaging and the model registry for the app's champion (P4): MLflow on DagsHub.

Why a registry, rather than files copied into the app: the app asks for
"models:/name-that-llm@champion", not for a file path. Promoting a better model later
(P6: the online A/B test, continuous training) then means moving the `champion` alias in
the registry. The app picks the new version up on its next start, with no code change, and
every version keeps its history.

Why MLflow's `pyfunc` format: it wraps any model behind one `predict()` interface, so the
app never needs to know that the champion is three models (TF-IDF, an LLM's hidden states
and a fine-tuned encoder) blended together.

What goes into a registered version (save_champion_artifacts):
  text_model.joblib    the fitted TF-IDF + logistic regression (`lr`)
  lm_head.joblib       the fitted LM head (scaler + logistic regression)
  lm_train_mean.npy    the training features' mean: the fallback for non-finite features
  encoders/fold0..     the fine-tuned encoder's fold models (notebook 09's output)
  meta.json            the blend weights, the "How sure?" bands, the LM settings (which
                       Qwen, pinned to a revision), the package version and git commit
Qwen's own weights are NOT stored: they're public (Apache 2.0) and come from the Hugging
Face Hub at the pinned revision.

mlflow is imported at the top on purpose, like optuna in tune.py: only the app and the
registration script import this module, never the offline submission.
"""

import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from name_that_llm import __version__
from name_that_llm.champion import Band, Champion
from name_that_llm.labels import LABELS

REGISTERED_NAME = "name-that-llm"


def save_champion_artifacts(
    out_dir: Path,
    *,
    text_model,
    lm_head,
    lm_train_mean: np.ndarray,
    encoder_dirs: Sequence[Path],
    meta: dict,
) -> Path:
    """Write the champion's parts into `out_dir`, in the layout load_champion reads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(text_model, out_dir / "text_model.joblib")
    joblib.dump(lm_head, out_dir / "lm_head.joblib")
    np.save(out_dir / "lm_train_mean.npy", np.asarray(lm_train_mean, dtype=np.float32))
    for k, folder in enumerate(encoder_dirs):
        shutil.copytree(folder, out_dir / "encoders" / f"fold{k}", dirs_exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_champion(artifact_dir: Path, n_encoder_folds: int | None = None) -> tuple[Champion, dict]:
    """Rebuild the Champion from a saved folder: (champion, meta).

    `n_encoder_folds` keeps only the first folds: the design's fallback (§10) if the
    Space's CPU proves too slow. `meta["encoder_folds"]` records how many are in use, so
    the app can say so.
    """
    from name_that_llm.encoder import load_encoder
    from name_that_llm.lm import load_lm_layers

    folder = Path(artifact_dir)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    lm = meta["lm"]
    fold_dirs = sorted(
        (folder / "encoders").glob("fold*"), key=lambda d: int(d.name.removeprefix("fold"))
    )[:n_encoder_folds]
    champion = Champion(
        text_model=joblib.load(folder / "text_model.joblib"),
        lm_head=joblib.load(folder / "lm_head.joblib"),
        lm_train_mean=np.load(folder / "lm_train_mean.npy"),
        # Only the blocks the LM head reads, plus one (the 9-block trap, see lm.py).
        lm=load_lm_layers(lm["source"], lm["layer"] + lm["window"], lm.get("revision")),
        encoders=[load_encoder(d) for d in fold_dirs],
        weights=tuple(meta["weights"]),
        lm_layer=lm["layer"],
        lm_window=lm["window"],
        lm_max_length=lm["max_length"],
        encoder_max_length=meta["encoder_max_length"],
    )
    return champion, {**meta, "encoder_folds": len(fold_dirs)}


class ChampionModel(mlflow.pyfunc.PythonModel):
    """The champion as an MLflow pyfunc: a DataFrame of texts in, 18 probabilities out."""

    def load_context(self, context):
        folds = os.environ.get("ENCODER_FOLDS")  # unset: every fold model
        self.champion, self.meta = load_champion(
            context.artifacts["champion"], int(folds) if folds else None
        )

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        texts = model_input["text"].tolist()
        return pd.DataFrame(self.champion.predict_proba(texts), columns=LABELS)


def log_champion(artifact_dir: Path, registered_name: str = REGISTERED_NAME) -> str:
    """Log the saved champion as a new version of `registered_name`; returns the version."""
    with mlflow.start_run(run_name="register-champion"):
        info = mlflow.pyfunc.log_model(
            name="champion",
            python_model=ChampionModel(),
            artifacts={"champion": str(artifact_dir)},
            registered_model_name=registered_name,
            # Explicit, so MLflow doesn't infer them by LOADING the model in a subprocess,
            # which would download Qwen. The app's Docker image provides these.
            pip_requirements=[
                f"name-that-llm=={__version__}",
                "torch==2.11.0",
                "transformers==5.16.1",
            ],
        )
    return str(info.registered_model_version)


def set_alias(name: str, alias: str, version: str) -> None:
    """Point `alias` (e.g. "champion") at `version` of the registered model `name`."""
    MlflowClient().set_registered_model_alias(name, alias, version)


@dataclass
class RegisteredPredictor:
    """What the app serves: probabilities for texts, and the facts about the loaded model."""

    model: mlflow.pyfunc.PyFuncModel
    info: dict = field(default_factory=dict)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """(n, 18) probabilities, columns in LABELS order."""
        return self.model.predict(pd.DataFrame({"text": list(texts)}))[LABELS].to_numpy()


def load_registered(
    name: str = REGISTERED_NAME, alias: str = "champion"
) -> tuple[RegisteredPredictor, dict]:
    """Load the version `alias` points to: (predictor, info).

    The alias is resolved to a version number ONCE, then that exact version is loaded. So
    `info["model_version"]` always names what's really running, even if the alias moves
    while the app is up (Review Focus 5). The new version arrives with the next restart.
    """
    version = MlflowClient().get_model_version_by_alias(name, alias).version
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    meta = model.unwrap_python_model().meta
    info = {
        "model_name": name,
        "model_version": str(version),
        "alias": alias,
        "encoder_folds": meta["encoder_folds"],
        "qwen_revision": meta["lm"].get("revision"),
        "bands": [Band(**band) for band in meta["bands"]],
    }
    return RegisteredPredictor(model, info), info
