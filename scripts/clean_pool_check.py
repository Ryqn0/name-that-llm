"""The allowlist, or the whole clean pool? (P7, I-032). By hand, on CPU.

The public pool's maintainer published a "clean" release (v9 on 2026-10-29, v10 the next
day): every passage passed their quality, genre and licence checks. Two of our five
allowlisted sources were removed with the rest. Our allowlist came from P7 Task 3's check
on the RAW pool, junk included. On the seed-42 folds, this compares `lr` trained on:
  - our passages only;
  - plus the clean pool's allowlisted sources (the 3 left);
  - plus the same number of passages drawn from EVERY clean source.
The same size, so the last two differ in which passages, not how many. The prompt bootstrap
judges them against each other. Every fold follows the no-leak rule (pool_for_fold). The
clean pool is built once with our own safeguards (pool.build_pool) and cached. Prints
numbers only, never a passage.

Usage (repo root; the v10 file in a folder named after the dataset, under --root):
  python -I -X utf8 scripts/clean_pool_check.py --root data/external-v10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from name_that_llm.config import load_config
from name_that_llm.data import LABELS, load_train
from name_that_llm.metrics import prompt_bootstrap_ci, score
from name_that_llm.pool import build_pool, load_pool_sources
from name_that_llm.train import run_cv_pool

SEED = 42
LABEL_ARRAY = np.array(LABELS)


def lr_oof(cfg, train: pd.DataFrame, pool: pd.DataFrame | None, n: int) -> np.ndarray:
    """`lr`'s out-of-fold probabilities, with at most n passages of `pool` per fold."""
    result = run_cv_pool(cfg, train) if pool is None else run_cv_pool(cfg, train, pool, n)
    return result.oof


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/external-v10"))
    parser.add_argument("--cache", type=Path, default=Path("data/pool/pool-v10-all.parquet"))
    parser.add_argument("--oof-dir", type=Path, default=Path("data/runs"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    train = load_train()
    y, prompts = train["model"].to_numpy(), train["prompt_id"].to_numpy()
    cfg = load_config("lr").model_copy(update={"seed": SEED})
    (source,) = load_pool_sources()  # the open pool, whose allowlist is pool.yaml's include
    include = source.include or []
    if args.cache.exists():
        clean = pd.read_parquet(args.cache)
    else:  # every clean source: the same pipeline as always, without the allowlist
        clean = build_pool([source.model_copy(update={"include": None})], args.root, train)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(args.cache)
    allow = clean[clean["source"].isin(include)].reset_index(drop=True)
    n = len(allow)
    print(
        f"clean pool: {len(clean):,} passages from {clean['source'].nunique()} sources; "
        f"the allowlist keeps {n:,} from {allow['source'].nunique()}",
        flush=True,
    )

    jobs = {"lr alone": None, "+ allowlist": allow, "+ every source": clean}
    oofs = Parallel(n_jobs=3)(delayed(lr_oof)(cfg, train, pool, n) for pool in jobs.values())
    preds = {name: LABEL_ARRAY[oof.argmax(axis=1)] for name, oof in zip(jobs, oofs, strict=True)}
    args.oof_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.oof_dir / f"oof-lr-v10allow{n}-seed{SEED}.npy", oofs[1])
    np.save(args.oof_dir / f"oof-lr-v10all{n}-seed{SEED}.npy", oofs[2])
    for a, b in [("lr alone", "+ allowlist"), ("+ allowlist", "+ every source")]:
        diff, lo, hi = prompt_bootstrap_ci(y, preds[a], preds[b], prompts)
        print(f"{b} vs {a}: {diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    for name, pred in preds.items():
        print(f"  {name}: {score(y, pred):.4f}")


if __name__ == "__main__":
    main()
