"""Log a GPU run's downloaded out-of-fold predictions to W&B, from this PC (P7).

Kaggle hands no secret to a run pushed through its API, so the GPU notebooks can't log to
W&B themselves (P7 Task 5's ruling). Their outputs are downloaded instead (`kaggle kernels
output`), and this logs one OOF file as a CV run, like every CPU run: the summary, per-class
recall, the confusion matrix, and the data and OOF artifacts (tracking.log_cv_run). The
notebook's own numbers (results.json's "scores") go into the summary too.

The score is recomputed from the OOF file on our folds of --seed. If it doesn't match the
notebook's, the rows aren't in the training data's order, and nothing is logged.

Usage (repo root, `wandb login` done):
  python -I -X utf8 scripts/log_gpu_run.py OOF.npy --seed 7 --name llm-first \
      [--results results.json] [--expect LLM]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from name_that_llm.config import LLMConfig, load_config
from name_that_llm.cv import make_folds
from name_that_llm.data import load_train
from name_that_llm.tracking import log_cv_run
from name_that_llm.train import _result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("oof", type=Path, help="an (n, 18) .npy of out-of-fold probabilities")
    parser.add_argument("--seed", type=int, required=True, help="the fold seed of the run")
    parser.add_argument("--name", required=True, help="the W&B run's name")
    parser.add_argument("--config", default="llm", help="the LLM config the run used")
    parser.add_argument("--results", type=Path, help="the notebook's results.json")
    parser.add_argument("--expect", help="results.json's score key that must match the OOF's")
    args = parser.parse_args(argv)

    train = load_train()
    oof = np.load(args.oof).astype(float)
    if oof.shape != (len(train), 18):
        raise SystemExit(f"STOP: the OOF is {oof.shape}, not ({len(train)}, 18)")
    oof /= oof.sum(axis=1, keepdims=True)  # float32 rows sum to 1 only to ~1e-7; log loss checks
    folds = make_folds(train, 5, args.seed).to_numpy()
    result = _result(oof, train["model"].to_numpy(), folds)
    print(f"{args.name}: CV {result.score:.4f} on the seed-{args.seed} folds", flush=True)

    extra = {}
    if args.results:
        notebook = json.loads(args.results.read_text(encoding="utf-8"))
        extra = {f"notebook/{key}": value for key, value in notebook["scores"].items()}
        if args.expect and abs(notebook["scores"][args.expect] - result.score) > 1e-6:
            raise SystemExit(
                f"STOP: the notebook says {args.expect} scored "
                f"{notebook['scores'][args.expect]:.4f}: the rows don't line up"
            )

    cfg = load_config(args.config, model_cls=LLMConfig)
    cfg = cfg.model_copy(update={"name": args.name, "seed": args.seed})
    print("W&B run:", log_cv_run(cfg, result, train, extra))


if __name__ == "__main__":
    main()
