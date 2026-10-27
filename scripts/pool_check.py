"""Does the pool help? Measured with `lr` on our seed-42 folds (P7 Task 3). CPU, by hand.

For every ORIGINAL source of the pool (data/pool/pool.parquet, from build_pool.py):
`lr` trained on our passages plus that source's passages (capped), against `lr` alone, on
the same folds. A source whose Δ is below -0.005 hurts as used here; the others are printed
for configs/pool.yaml's `include`. Caveat: `lr` isn't class-weighted, so a source covering
only some of the 18 models also pulls its predictions toward them, and a low Δ doesn't
prove wrong labels (I-028). Sources with fewer than 1,000 passages are judged together, as
one group. Then `lr` plus every kept source
(capped), against `lr` alone, with the prompt bootstrap: the pool helps `lr` if the 95% CI's
lower bound is above 0.

Every fold follows the no-leak rule (pool_for_fold), and only our passages are scored.
Prints numbers only, never a passage.

Usage:  python -I -X utf8 scripts/pool_check.py [--jobs 3]
"""

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from name_that_llm.config import load_config
from name_that_llm.data import load_train
from name_that_llm.metrics import prompt_bootstrap_ci
from name_that_llm.train import run_cv_pool

PER_SOURCE_CAP, ALL_CAP, SMALL, HURTS = 3_000, 30_000, 1_000, -0.005


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, default=Path("data/pool/pool.parquet"))
    parser.add_argument("--jobs", type=int, default=3, help="parallel checks (memory-bound)")
    return parser.parse_args(argv)


def check(name: str, cfg, train: pd.DataFrame, pool: pd.DataFrame, cap: int):
    """(name, rows, CV score, predictions) of `lr` trained with `pool` (capped)."""
    result = run_cv_pool(cfg, train, pool, max_pool=cap)
    return name, len(pool), result.score, result.pred


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config("lr").model_copy(update={"seed": 42})
    train = load_train()
    y, groups = train["model"].to_numpy(), train["prompt_id"].to_numpy()
    pool = pd.read_parquet(args.pool)

    base = run_cv_pool(cfg, train)
    print(f"lr alone (seed 42): {base.score:.4f}", flush=True)

    sizes = pool["source"].value_counts()
    big = list(sizes.index[sizes >= SMALL])
    groups_of = {name: pool[pool["source"] == name] for name in big}
    groups_of["(sources under 1,000 passages)"] = pool[~pool["source"].isin(big)]
    results = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(check)(name, cfg, train, part, PER_SOURCE_CAP) for name, part in groups_of.items()
    )

    print(f"\nper source (each capped at {PER_SOURCE_CAP:,} passages):")
    print(f"  {'source':58s} {'rows':>7s} {'CV':>7s} {'delta':>8s}  95% CI")
    excluded = []
    for name, rows, cv, pred in sorted(results, key=lambda r: r[2] - base.score):
        diff, lo, hi = prompt_bootstrap_ci(y, base.pred, pred, groups)
        delta = cv - base.score
        if delta < HURTS:
            excluded.append(name)
        flag = "  HURTS" if delta < HURTS else ""
        print(f"  {name:58s} {rows:7,d} {cv:7.4f} {delta:+8.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}")

    small = "(sources under 1,000 passages)"
    drop = [n for n in excluded if n != small]
    if small in excluded:
        drop += list(sizes.index[sizes < SMALL])
    kept = pool[~pool["source"].isin(drop)]
    names = sorted(kept["source"].unique())
    print(f"\ninclude ({len(names)} of {len(sizes)} sources): {names}")
    print(f"lr + all kept sources ({len(kept):,} passages, capped at {ALL_CAP:,})...", flush=True)
    _, _, cv, pred = check("all kept", cfg, train, kept, ALL_CAP)
    diff, lo, hi = prompt_bootstrap_ci(y, base.pred, pred, groups)
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    print(f"  CV {cv:.4f} ({cv - base.score:+.4f}); bootstrap {diff:+.4f}, {ci}")
    helps = lo > 0
    print(f"\nVERDICT: the pool {'HELPS' if helps else 'does NOT clearly help'} lr")
    print("(the LLM's own test of the pool is Task 6, whatever lr says)")


if __name__ == "__main__":
    main()
