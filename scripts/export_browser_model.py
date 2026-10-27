"""Export the browser demo's model and pin it (P4 revision, run in Task 8b).

The free public demo (design §12) runs `lr`, the TF-IDF model, inside the visitor's
browser. Run by hand, once per new demo model. In order, it:
  1. loads `lr` from the registered champion's files: data/kaggle-outputs/champion is the
     folder register_champion.py logged as name-that-llm version 1;
  2. measures lr's OWN "How sure?" bands on out-of-fold predictions (seed 42): alone, lr
     is less sure than the champion, so the champion's thresholds would mislabel it;
  3. exports model.json and weights.bin for the browser engine (name_that_llm.browser);
  4. checks PARITY on 50 training passages: the JavaScript engine, run by Node, against
     scikit-learn. The same top label on all 50 and probabilities within 1e-5, or it stops;
  5. unless --dry-run, logs the folder as an MLflow run (experiment "browser-demo") and
     writes the run's id to app/demo/MODEL_RUN. The deploy downloads exactly that run, so
     the public model changes only through a pull request that changes this file.

The MLflow server comes from the environment, as for register_champion.py. Nothing here
prints a passage: only counts and scores.

Usage (on Windows add -X utf8, since MLflow prints an emoji that cp1252 can't encode):
  python scripts/export_browser_model.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np

from name_that_llm import __version__
from name_that_llm.browser import DEMO_DIR, export_browser_model, run_engine
from name_that_llm.champion import fit_confidence_bands
from name_that_llm.config import load_config
from name_that_llm.data import flatten_text, load_train
from name_that_llm.submit import _aligned_proba
from name_that_llm.train import run_cv

# The registered champion whose `lr` the demo runs (its files are --champion-dir).
SOURCE = "models:/name-that-llm/1"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--champion-dir", type=Path, default=Path("data/kaggle-outputs/champion"))
    parser.add_argument("--out", type=Path, default=Path("data/kaggle-outputs/browser-demo"))
    parser.add_argument("--parity-rows", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="every check, but no MLflow run")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    train = load_train()
    texts, y = train["text"].tolist(), train["model"].to_numpy()

    print(f"1. lr, from the registered champion's files ({SOURCE})", flush=True)
    text_model = joblib.load(args.champion_dir / "text_model.joblib")

    print("2. lr's own 'How sure?' bands, from out-of-fold predictions (seed 42)", flush=True)
    # "lr" is the config the champion's lr was fit with (lm_blend.yaml's text_config).
    cv = run_cv(load_config("lr").model_copy(update={"seed": 42}), train)
    bands = fit_confidence_bands(cv.oof, y)
    print(f"   balanced accuracy {cv.score:.4f}")
    for band in bands:
        print(f"   {band.label:12s} top >= {band.min_top:.3f}: right {band.accuracy:.1%}")
    if not all(a.accuracy >= b.accuracy for a, b in zip(bands, bands[1:], strict=False)):
        print("   WARNING: a more confident band is not more often right")

    print("3. export for the browser engine", flush=True)
    # The page reads cv_balanced_accuracy for its "right about N times in 10".
    meta = {"source": SOURCE, "cv_balanced_accuracy": cv.score, "package_version": __version__}
    for path in export_browser_model(text_model, bands, args.out, meta):
        print(f"   {path} ({path.stat().st_size / 1e6:.1f} MB)")

    print("4. parity: the JavaScript engine (Node) vs scikit-learn", flush=True)
    rows = np.random.default_rng(0).choice(len(train), args.parity_rows, replace=False)
    sample = [texts[i] for i in rows]
    js = np.array(run_engine("predict", {"model": str(args.out), "texts": sample})["probabilities"])
    # The engine flattens what it's given; so does the champion before its lr.
    py = _aligned_proba(text_model, [flatten_text(t) for t in sample])
    agree = (js.argmax(1) == py.argmax(1)).sum()
    diff = np.abs(js - py).max()
    print(f"   same top label: {agree} of {len(rows)} | max probability difference {diff:.1e}")
    if agree < len(rows) or diff > 1e-5:
        sys.exit("STOP: the browser engine disagrees with scikit-learn")

    if args.dry_run:
        print("dry run: every check passed; nothing was logged")
        return
    print("5. log the files as an MLflow run, and pin it", flush=True)
    mlflow.set_experiment("browser-demo")
    with mlflow.start_run(run_name="export-browser-model") as run:
        mlflow.log_params({"source": SOURCE, "parity_rows": len(rows)})
        mlflow.log_metrics({"cv_balanced_accuracy": cv.score, "parity_max_diff": diff})
        mlflow.log_artifacts(str(args.out))
    (DEMO_DIR / "MODEL_RUN").write_text(f"{run.info.run_id}\n", encoding="utf-8")
    print(f"   run {run.info.run_id} logged; app/demo/MODEL_RUN pins it")


if __name__ == "__main__":
    main()
