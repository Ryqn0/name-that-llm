"""Does the pool help the ENSEMBLE, through its `lr` member? (I-029, P7). By hand, on CPU.

P7 Task 3 found that `lr` trained on our passages plus 30,000 pool passages scores 0.4205
alone, as high as the whole three-member champion (0.4219). The plan gives the pool only to
the LLM; this checks the cheap half first, with no GPU: the champion's recipe with `lr`
swapped for `lr` + the pool, the LM head and the encoder unchanged.
  1. rebuilds both `lr` variants, the LM head, and the encoder's saved OOF, on four fold
     seeds (every fold follows the no-leak rule: no pool passage about one of its
     validation prompts);
  2. picks the new ensemble's weights on seed 7. The champion keeps its shipped weights;
  3. reports both ensembles on the fresh seeds 42, 43 and 44, with the prompt bootstrap's
     95% CI on seed 42: the usual A/B gate, PROMOTE if the lower bound is above 0.
Expect less than `lr`'s own gain: the other two members already fix some of its mistakes.
Prints numbers only, never a passage.

Usage (repo root; data/pool/pool.parquet from build_pool.py, the P3b features and the
encoder's OOF files downloaded):
  python -I -X utf8 scripts/pool_ensemble_check.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.blend import best_blend_weights, blend
from name_that_llm.config import BlendConfig, EnsembleConfig, load_config
from name_that_llm.data import LABELS, load_train
from name_that_llm.lm import select_layers
from name_that_llm.metrics import prompt_bootstrap_ci, score
from name_that_llm.pool import load_pool_sources
from name_that_llm.train import run_cv, run_cv_features, run_cv_pool

POOL_CAP = 30_000  # as in Task 3's check and the weekly retraining
SEEDS, FRESH = (7, 42, 43, 44), (42, 43, 44)
LABEL_ARRAY = np.array(LABELS)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, default=Path("data/pool/pool.parquet"))
    parser.add_argument(
        "--lm-features-dir", type=Path, default=Path("data/kaggle-outputs/lm-features")
    )
    parser.add_argument(
        "--encoder-oof-dir", type=Path, default=Path("data/kaggle-outputs/encoder-final")
    )
    return parser.parse_args(argv)


def acc(y, proba) -> float:
    return score(y, LABEL_ARRAY[proba.argmax(axis=1)])


def main(argv=None) -> None:
    args = parse_args(argv)
    ens = load_config("ensemble", model_cls=EnsembleConfig)
    bcfg = load_config(ens.blend, model_cls=BlendConfig)
    cfg = load_config(bcfg.text_config)
    train = load_train()
    y = train["model"].to_numpy()

    # The LM head's features, in the training data's row order (as sinkhorn_check.py).
    ids = np.load(args.lm_features_dir / "train_ids.npy", allow_pickle=True)
    if not (ids == train["id"].to_numpy()).all():
        raise SystemExit("STOP: the saved LM features are not in the training data's row order")
    feats = np.load(args.lm_features_dir / f"features-{bcfg.lm_variation}.npy", mmap_mode="r")
    X_lm = select_layers(feats, bcfg.layer, bcfg.layer_window)

    # The allowlisted pool: build_pool dedupes across all 56 sources BEFORE its allowlist,
    # so filtering the full pool file gives exactly the pool it builds.
    include = [name for s in load_pool_sources() if s.keep for name in (s.include or [])]
    pool = pd.read_parquet(args.pool)
    pool = pool[pool["source"].isin(include)].reset_index(drop=True)
    print(f"the pool: {len(pool):,} passages from {pool['source'].nunique()} sources", flush=True)

    members = {}
    for seed in SEEDS:
        seeded = cfg.model_copy(update={"seed": seed})
        members[seed] = {
            "lr": run_cv(seeded, train).oof,
            "lr_pool": run_cv_pool(seeded, train, pool, max_pool=POOL_CAP).oof,
            "lm": run_cv_features(X_lm, train, C=bcfg.C, seed=seed).oof,
            "enc": np.load(args.encoder_oof_dir / f"oof-encoder-seed{seed}.npy").astype(float),
        }
        row = " | ".join(f"{k} {acc(y, v):.4f}" for k, v in members[seed].items())
        print(f"seed {seed}: {row}", flush=True)

    def champion(seed):
        m = members[seed]
        return blend([m["lr"], m["lm"], m["enc"]], ens.weights)

    def candidate(seed, weights):
        m = members[seed]
        return blend([m["lr_pool"], m["lm"], m["enc"]], weights)

    m7 = members[7]
    weights, on7 = best_blend_weights([m7["lr_pool"], m7["lm"], m7["enc"]], y)
    print(f"\n1. weights chosen on seed 7 (lr+pool, LM head, encoder): {weights}")
    print(f"   seed 7: champion {acc(y, champion(7)):.4f}, candidate {on7:.4f} (in-sample)")

    print("\n2. on the fresh seeds")
    gains = []
    for seed in FRESH:
        c, n = acc(y, champion(seed)), acc(y, candidate(seed, weights))
        gains.append(n - c)
        print(f"   seed {seed}: champion {c:.4f}, candidate {n:.4f} ({n - c:+.4f})")
    old = LABEL_ARRAY[champion(42).argmax(axis=1)]
    new = LABEL_ARRAY[candidate(42, weights).argmax(axis=1)]
    diff, lo, hi = prompt_bootstrap_ci(y, old, new, train["prompt_id"].to_numpy())
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    print(f"   mean {np.mean(gains):+.4f} | seed-42 bootstrap {diff:+.4f}, {ci}")
    print(f"\nVERDICT: {'PROMOTE' if lo > 0 else 'KEEP the champion'} (gate: lower bound > 0)")


if __name__ == "__main__":
    main()
