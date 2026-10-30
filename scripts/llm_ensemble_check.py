"""Does the LLM lift I-029's ensemble? (P7 Task 7). By hand, on CPU.

Notebook 12 trains the pooled Qwen3-8B LoRA on the seed-42 folds, session by session, and
saves each fold's out-of-fold predictions (runs/<job>/fold<k>.npz, the rows of fold k in
order). On the rows whose LLM fold is finished, this compares:
  - I-029's ensemble (configs/ensemble_pool.yaml): `lr` with the pool, the LM head and the
    encoder, with weights chosen on seed 7, so they're out-of-sample here;
  - the four-member ensemble with the LLM. The LLM has only seed 42, so its weights can't
    be chosen on another seed: they're chosen by NESTED cross-validation. Each finished fold
    is predicted with weights chosen on the OTHER finished folds only, so no row's
    prediction used weights fit on that row.
Then the prompt bootstrap's 95% CI on those rows: the usual A/B gate (lower bound > 0).
With fewer than 5 folds done, it's a preview. It ends with the weights to ship, chosen on
every finished row. `lr` with the pool takes about 50 minutes; its OOF is cached.
Prints numbers only, never a passage.

Usage (repo root):
  python -I -X utf8 scripts/llm_ensemble_check.py data/kaggle-outputs/llm-final-s2/runs [...]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.blend import best_blend_weights, blend
from name_that_llm.config import BlendConfig, EnsembleConfig, load_config
from name_that_llm.cv import make_folds
from name_that_llm.data import LABELS, load_train
from name_that_llm.lm import select_layers
from name_that_llm.metrics import prompt_bootstrap_ci, score
from name_that_llm.pool import load_pool_sources
from name_that_llm.train import run_cv_features, run_cv_pool

SEED = 42  # notebook 12's final seed
LABEL_ARRAY = np.array(LABELS)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="notebook 12's downloaded runs/")
    parser.add_argument("--pool", type=Path, default=Path("data/pool/pool.parquet"))
    parser.add_argument(
        "--lm-features-dir", type=Path, default=Path("data/kaggle-outputs/lm-features")
    )
    parser.add_argument(
        "--encoder-oof-dir", type=Path, default=Path("data/kaggle-outputs/encoder-final")
    )
    parser.add_argument("--cache", type=Path, default=Path("data/runs"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ens = load_config("ensemble_pool", model_cls=EnsembleConfig)
    bcfg = load_config(ens.blend, model_cls=BlendConfig)
    cfg = load_config(bcfg.text_config).model_copy(update={"seed": SEED})
    train = load_train()
    y, prompts = train["model"].to_numpy(), train["prompt_id"].to_numpy()
    folds = make_folds(train, cfg.n_splits, SEED).to_numpy()

    # The LLM's finished folds: each npz holds its fold's rows, in the training data's order.
    llm, done = np.full((len(train), len(LABELS)), np.nan), []
    for path in sorted(p for d in args.runs for p in d.glob("*/fold*.npz")):
        k = int(path.stem.removeprefix("fold"))
        with np.load(path) as npz:
            llm[folds == k] = npz[f"fold{k}"]
        done.append(k)
    rows = np.isin(folds, done)
    print(f"LLM folds done: {sorted(done)} ({rows.sum():,} of {len(train):,} rows)", flush=True)

    # `lr` with the pool, as in I-029's check (the same no-leak folds). Cached under the pool
    # file and its allowlisted size: the public pool changes upstream (its v9 removed 2 of
    # our sources), and one pool's predictions must never stand in for another's.
    include = [name for s in load_pool_sources() if s.keep for name in (s.include or [])]
    pool = pd.read_parquet(args.pool)
    pool = pool[pool["source"].isin(include)].reset_index(drop=True)
    cache = args.cache / f"oof-lr-{args.pool.stem}-{len(pool)}of{ens.text_pool}-seed{SEED}.npy"
    print(f"lr's pool: {len(pool):,} passages from {args.pool}", flush=True)
    if cache.exists():
        lr_pool = np.load(cache)
    else:
        lr_pool = run_cv_pool(cfg, train, pool, max_pool=ens.text_pool).oof
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, lr_pool)
    ids = np.load(args.lm_features_dir / "train_ids.npy", allow_pickle=True)
    if not (ids == train["id"].to_numpy()).all():
        raise SystemExit("STOP: the saved LM features are not in the training data's row order")
    feats = np.load(args.lm_features_dir / f"features-{bcfg.lm_variation}.npy", mmap_mode="r")
    lm = run_cv_features(
        select_layers(feats, bcfg.layer, bcfg.layer_window), train, C=bcfg.C, seed=SEED
    ).oof
    enc = np.load(args.encoder_oof_dir / f"oof-encoder-seed{SEED}.npy").astype(float)
    members = [lr_pool, lm, enc, llm]
    current = blend(members[:3], ens.weights)

    four = np.zeros_like(current)
    for k in sorted(done):
        fit = rows & (folds != k)
        weights, _ = best_blend_weights([m[fit] for m in members], y[fit])
        four[folds == k] = blend([m[folds == k] for m in members], weights)
        print(f"fold {k}: weights {weights}, chosen on folds {sorted(set(done) - {k})}")

    def acc(proba) -> float:
        return score(y[rows], LABEL_ARRAY[proba[rows].argmax(axis=1)])

    print(
        f"\non those {rows.sum():,} rows: the LLM alone {acc(llm):.4f} | I-029's ensemble "
        f"{acc(current):.4f} | with the LLM (nested weights) {acc(four):.4f}"
    )
    old, new = (LABEL_ARRAY[p[rows].argmax(axis=1)] for p in (current, four))
    diff, lo, hi = prompt_bootstrap_ci(y[rows], old, new, prompts[rows])
    print(f"gain {diff:+.4f}, prompt bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    final, _ = best_blend_weights([m[rows] for m in members], y[rows])
    print(f"the weights to ship (chosen on every finished row): {final}")
    preview = "" if len(done) == 5 else f" (a preview: {len(done)} of 5 folds)"
    print(f"\nVERDICT: {'PROMOTE' if lo > 0 else 'KEEP I-029'} (gate: lower bound > 0){preview}")


if __name__ == "__main__":
    main()
