"""Weekly retraining that may propose a new challenger for the demo (P6): run by retrain.yml.

The demo runs `lr` (TF-IDF + logistic regression) trained on our 2,699 passages. P7 Task 3
showed that the pool of public generated passages lifts that recipe a lot. Every week, this
script retrains it with the latest pool, and proposes the result if it wins:
  1. it reads the competition data (--data-dir) and the pool (--pool, from build_pool.py);
  2. it cross-validates, on the same seed-42 folds, the champion's recipe (`lr` on our
     passages) and the candidate (`lr` plus the pool, capped at 30,000 passages; no fold
     ever trains on a pool passage about one of its validation prompts);
  3. it reads the current challenger's CV from its MLflow run (app/demo/CHALLENGER_RUN);
  4. it asks name_that_llm.retrain.propose. If the answer is yes:
     - it fits the candidate on all our passages plus the capped pool;
     - it exports the candidate for the browser, with its own "How sure?" bands and the
       unusual-text limits, and checks parity under Node on 50 passages;
     - it logs the files as an MLflow run (experiment browser-demo);
     - it writes CHALLENGER_RUN on a new branch, pushes it, and opens a pull request. A
       person decides by merging it.
It prints scores and counts only, never a passage: the repo is public, and so are its
Actions logs. With --dry-run it stops before the MLflow run and the pull request.

Usage:  python scripts/retrain_challenger.py --pool PATH [--data-dir DIR] [--dry-run]
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm import __version__
from name_that_llm.browser import DEMO_DIR, export_browser_model, run_engine, unusual_thresholds
from name_that_llm.champion import fit_confidence_bands
from name_that_llm.config import load_config
from name_that_llm.data import flatten_text, load_train
from name_that_llm.models import build_model
from name_that_llm.pool import pool_for_fold
from name_that_llm.retrain import pr_body, propose
from name_that_llm.submit import _aligned_proba
from name_that_llm.train import run_cv_pool

POOL_CAP = 30_000  # as in P7 Task 3's check, which measured +0.099 for lr with it


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, required=True, help="pool.parquet (build_pool.py)")
    parser.add_argument("--data-dir", type=Path, help="the folder with train.csv")
    parser.add_argument("--out", type=Path, default=Path("data/kaggle-outputs/retrain"))
    parser.add_argument("--dry-run", action="store_true", help="no MLflow run, no pull request")
    return parser.parse_args(argv)


def pinned_cvs() -> list[float]:
    """The CVs logged by the models the demo pins now: MODEL_RUN's, then CHALLENGER_RUN's.

    A missing or empty pin, or a run that logged no CV, is skipped: no challenger yet is the
    usual case, and the bootstrap against the champion's recipe still decides.
    """
    import mlflow

    cvs = []
    for name in ("MODEL_RUN", "CHALLENGER_RUN"):
        path = DEMO_DIR / name
        run = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        cv = mlflow.get_run(run).data.metrics.get("cv_balanced_accuracy") if run else None
        if cv is not None:
            cvs.append(cv)
    return cvs


def open_pull_request(run_id: str, numbers: dict) -> None:
    """Pin the new challenger on a branch, push it, and open the pull request (gh, GH_TOKEN)."""
    branch = f"retrain/{datetime.date.today():%Y-%m-%d}"
    (DEMO_DIR / "CHALLENGER_RUN").write_text(f"{run_id}\n", encoding="utf-8")
    bot = ["-c", "user.name=github-actions[bot]"]
    bot += ["-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com"]
    title = f"feat(demo): a new challenger, CV {numbers['candidate_cv']:.4f} (weekly retraining)"
    for command in (
        # The checkout keeps no credentials (persist-credentials: false), so `git push` gets
        # them from gh, which reads GH_TOKEN: only this step ever holds the write token.
        ["gh", "auth", "setup-git"],
        ["git", "switch", "-c", branch],
        ["git", "add", str(DEMO_DIR / "CHALLENGER_RUN")],
        ["git", *bot, "commit", "-m", title],
        ["git", "push", "origin", branch],
        [
            "gh",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            title,
            "--body",
            pr_body(numbers, run_id),
        ],
    ):
        subprocess.run(command, check=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    train = load_train(args.data_dir)
    texts, y = train["text"].tolist(), train["model"].to_numpy()
    groups = train["prompt_id"].to_numpy()
    pool = pd.read_parquet(args.pool)
    cfg = load_config("lr").model_copy(update={"seed": 42})
    print(f"our passages: {len(train):,} | the pool: {len(pool):,} passages", flush=True)

    print("1. the champion's recipe (lr) and the candidate (lr + the pool), same folds", flush=True)
    champion = run_cv_pool(cfg, train)
    candidate = run_cv_pool(cfg, train, pool, max_pool=POOL_CAP)
    current = pinned_cvs()
    yes, numbers = propose(y, groups, champion.pred, candidate.pred, candidate.score, current)
    print(
        f"   champion's recipe {numbers['champion_cv']:.4f} | candidate "
        f"{numbers['candidate_cv']:.4f} | difference {numbers['diff']:+.4f}, 95% interval "
        f"[{numbers['lo']:+.4f}, {numbers['hi']:+.4f}] | pinned models' CVs {current}"
    )
    if not yes:
        print("no proposal: the candidate doesn't clearly beat the champion's recipe, or")
        print("doesn't beat the models the demo pins by 0.005")
        return

    print("2. the candidate, fit on all our passages plus the capped pool", flush=True)
    extra = pool_for_fold(pool, set(), POOL_CAP, seed=cfg.seed)  # no held-out prompts here
    fit_texts = texts + extra["text"].astype(str).tolist()
    model = build_model(cfg).fit(fit_texts, np.concatenate([y, extra["model"].to_numpy()]))
    bands = fit_confidence_bands(candidate.oof, y)  # the candidate's own held-out confidence
    meta = {
        "source": "weekly retraining: lr + the pool",
        "cv_balanced_accuracy": candidate.score,
        "package_version": __version__,
        "pool_passages": len(extra),
        # Measured with our passages' vocabularies. The candidate knows the pool's words
        # too, so it meets fewer unknown words: this limit flags a little less, never more.
        "unusual": unusual_thresholds(cfg, train),
    }
    export_browser_model(model, bands, args.out, meta)

    print("3. parity: the JavaScript engine (Node) vs scikit-learn", flush=True)
    rows = np.random.default_rng(0).choice(len(train), 50, replace=False)
    sample = [texts[i] for i in rows]
    js = np.array(run_engine("predict", {"model": str(args.out), "texts": sample})["probabilities"])
    py = _aligned_proba(model, [flatten_text(t) for t in sample])
    agree, diff = int((js.argmax(1) == py.argmax(1)).sum()), float(np.abs(js - py).max())
    print(f"   same top label: {agree} of 50 | max probability difference {diff:.1e}")
    if agree < 50 or diff > 1e-5:
        sys.exit("STOP: the browser engine disagrees with scikit-learn")
    if args.dry_run:
        print("dry run: every check passed; no MLflow run, no pull request")
        return

    print("4. log the candidate, and propose it as the challenger", flush=True)
    import mlflow

    mlflow.set_experiment("browser-demo")
    with mlflow.start_run(run_name="retrain-challenger") as run:
        mlflow.log_params({"source": meta["source"], "pool_cap": POOL_CAP, "parity_rows": 50})
        mlflow.log_metrics({"cv_balanced_accuracy": candidate.score, "parity_max_diff": diff})
        mlflow.log_metrics({f"vs_champion_{k}": numbers[k] for k in ("diff", "lo", "hi")})
        mlflow.log_artifacts(str(args.out))
    open_pull_request(run.info.run_id, numbers)
    print(f"   run {run.info.run_id} logged, and the pull request is open")


if __name__ == "__main__":
    main()
