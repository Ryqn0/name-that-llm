"""The gate for a new ensemble member (P8 Task 3; generalizes P7's llm_ensemble_check).

Compares two ensembles on the seed-42 folds, both weighted by NESTED cross-validation
(blend.nested_blend_oof: each fold's weights come from the other folds' rows only, so no
row is scored with weights fitted on it):
  - the baseline, by default `lr` with the pool, the LM head, the encoder and the LLM;
  - the candidate, by default the baseline plus every --extra member.
The members:
  - lr_pool: `lr` trained with the pool's allowlisted sources, on the no-leak folds;
  - lm: the LM head on Qwen 3 4B's layers;
  - enc: the fine-tuned encoder's saved OOF;
  - llm: notebook 12's fold predictions (runs/<job>/fold<k>.npz), on the rows whose fold
    is done;
  - --extra NAME=PATH.npy: a pool member's (n, 18) predictions on all our passages
    (pool_member.py), honest without folds.
Only the rows where every member used has a prediction are scored. The verdict is the usual
gate: PROMOTE if the prompt bootstrap's 95% lower bound is above 0. It ends with the
candidate's weights to ship, chosen on every scored row.

`lr` with the pool takes about 50 minutes; its OOF is cached under the pool file and its
allowlisted size, since the public pool changes upstream. Prints numbers only, never a
passage.

Usage (repo root):
  python -I -X utf8 scripts/ensemble_gate.py --llm-runs DIR [DIR ...] [--pool PARQUET]
      [--extra NAME=PATH.npy ...] [--baseline lr_pool,lm,enc,llm] [--candidate NAMES]
P7's gate (the LLM against I-029's three members):
  ... --baseline lr_pool,lm,enc --candidate lr_pool,lm,enc,llm
P7's choices for the shipped blend (the same members, a different form):
  ... --baseline M --candidate M --candidate-log          (I-015: the log blend)
  ... --baseline M --candidate M --candidate-sinkhorn A   (I-027: Sinkhorn at strength A)
Sinkhorn balances all scored rows at once, as the submission does the whole test.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.balance import sinkhorn_balance
from name_that_llm.blend import best_blend_weights, nested_blend_oof
from name_that_llm.config import BlendConfig, EnsembleConfig, load_config
from name_that_llm.cv import make_folds
from name_that_llm.data import LABELS, load_train
from name_that_llm.lm import select_layers
from name_that_llm.metrics import prompt_bootstrap_ci, score
from name_that_llm.pool import load_pool_sources
from name_that_llm.train import run_cv_features, run_cv_pool

SEED = 42  # the final seed of P7's LLM and P8's members
LABEL_ARRAY = np.array(LABELS)
STANDARD = ["lr_pool", "lm", "enc", "llm"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm-runs", type=Path, nargs="*", default=[], help="runs/ folders")
    parser.add_argument("--extra", action="append", default=[], help="NAME=PATH.npy, repeatable")
    parser.add_argument("--baseline", default=",".join(STANDARD), help="member names")
    parser.add_argument("--candidate", help="member names (default: baseline + extras)")
    parser.add_argument("--candidate-log", action="store_true", help="blend it in log space")
    parser.add_argument(
        "--candidate-sinkhorn", type=float, default=0.0, help="Sinkhorn's strength, in [0, 1]"
    )
    parser.add_argument("--pool", type=Path, default=Path("data/pool/pool-v10-all.parquet"))
    parser.add_argument(
        "--lm-features-dir", type=Path, default=Path("data/kaggle-outputs/lm-features")
    )
    parser.add_argument(
        "--encoder-oof-dir", type=Path, default=Path("data/kaggle-outputs/encoder-final")
    )
    parser.add_argument("--cache", type=Path, default=Path("data/runs"))
    return parser.parse_args(argv)


def _members(args, train: pd.DataFrame, folds: np.ndarray) -> dict[str, np.ndarray]:
    """Every member's (n, 18) predictions, NaN on rows a member hasn't predicted."""
    ens = load_config("ensemble_pool", model_cls=EnsembleConfig)
    bcfg = load_config(ens.blend, model_cls=BlendConfig)
    cfg = load_config(bcfg.text_config).model_copy(update={"seed": SEED})
    members = {}

    llm = np.full((len(train), len(LABELS)), np.nan)
    for path in sorted(p for d in args.llm_runs for p in d.glob("*/fold*.npz")):
        k = int(path.stem.removeprefix("fold"))
        with np.load(path) as npz:
            llm[folds == k] = npz[f"fold{k}"]
    members["llm"] = llm

    include = [name for s in load_pool_sources() if s.keep for name in (s.include or [])]
    pool = pd.read_parquet(args.pool)
    pool = pool[pool["source"].isin(include)].reset_index(drop=True)
    cache = args.cache / f"oof-lr-{args.pool.stem}-{len(pool)}of{ens.text_pool}-seed{SEED}.npy"
    print(f"lr's pool: {len(pool):,} passages from {args.pool}", flush=True)
    if cache.exists():
        members["lr_pool"] = np.load(cache)
    else:
        members["lr_pool"] = run_cv_pool(cfg, train, pool, max_pool=ens.text_pool).oof
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, members["lr_pool"])

    ids = np.load(args.lm_features_dir / "train_ids.npy", allow_pickle=True)
    if not (ids == train["id"].to_numpy()).all():
        raise SystemExit("STOP: the saved LM features are not in the training data's row order")
    feats = np.load(args.lm_features_dir / f"features-{bcfg.lm_variation}.npy", mmap_mode="r")
    X_lm = select_layers(feats, bcfg.layer, bcfg.layer_window)
    members["lm"] = run_cv_features(X_lm, train, C=bcfg.C, seed=SEED).oof
    members["enc"] = np.load(args.encoder_oof_dir / f"oof-encoder-seed{SEED}.npy").astype(float)

    for spec in args.extra:
        name, path = spec.split("=", 1)
        proba = np.load(path).astype(float)
        if proba.shape != (len(train), len(LABELS)):
            raise SystemExit(f"STOP: {name} has shape {proba.shape}, not ({len(train)}, 18)")
        members[name] = proba
    return members


def main(argv=None) -> None:
    args = parse_args(argv)
    train = load_train()
    y, prompts = train["model"].to_numpy(), train["prompt_id"].to_numpy()
    folds = make_folds(train, 5, SEED).to_numpy()
    members = _members(args, train, folds)

    extras = [spec.split("=", 1)[0] for spec in args.extra]
    base = args.baseline.split(",")
    cand = args.candidate.split(",") if args.candidate else base + extras
    used = sorted(set(base) | set(cand))
    rows = np.logical_and.reduce([~np.isnan(members[m]).any(axis=1) for m in used])
    done = sorted(int(k) for k in np.unique(folds[rows]))
    print(f"scored: {rows.sum():,} of {len(train):,} rows (folds {done})", flush=True)

    def acc(proba) -> float:
        return score(y[rows], LABEL_ARRAY[proba[rows].argmax(axis=1)])

    for name in used:
        print(f"  {name} alone: {acc(members[name]):.4f}")
    base_oof, _ = nested_blend_oof([members[m] for m in base], y, folds, rows)
    cand_oof, cand_w = nested_blend_oof(
        [members[m] for m in cand], y, folds, rows, log=args.candidate_log
    )
    if args.candidate_sinkhorn:
        cand_oof[rows] = sinkhorn_balance(cand_oof[rows], args.candidate_sinkhorn)
    for k, w in cand_w.items():
        print(f"fold {k}: candidate weights {w}")
    print(f"\nbaseline {'+'.join(base)}: {acc(base_oof):.4f}")
    form = " (log)" if args.candidate_log else ""
    form += f" (Sinkhorn {args.candidate_sinkhorn})" if args.candidate_sinkhorn else ""
    print(f"candidate {'+'.join(cand)}{form}: {acc(cand_oof):.4f}")
    old, new = (LABEL_ARRAY[p[rows].argmax(axis=1)] for p in (base_oof, cand_oof))
    diff, lo, hi = prompt_bootstrap_ci(y[rows], old, new, prompts[rows])
    print(f"gain {diff:+.4f}, prompt bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    final, _ = best_blend_weights([members[m][rows] for m in cand], y[rows], log=args.candidate_log)
    print(f"the weights to ship ({', '.join(cand)}): {final}")
    preview = "" if len(done) == 5 else f" (a preview: {len(done)} of 5 folds)"
    print(f"\nVERDICT: {'PROMOTE' if lo > 0 else 'KEEP'} (gate: lower bound > 0){preview}")


if __name__ == "__main__":
    main()
