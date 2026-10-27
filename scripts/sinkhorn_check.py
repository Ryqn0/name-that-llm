"""Does Sinkhorn balancing help the champion? (P7 Task 1). Run by hand, on CPU.

Rebuilds the champion's out-of-fold predictions on four fold seeds (lr and the LM head
retrained per seed, the encoder's saved OOF files), then:
  1. chooses the balancing strength alpha on seed 7;
  2. reports the gain of that alpha on the fresh seeds 42, 43 and 44, and the prompt
     bootstrap's confidence interval on seed 42 (the usual A/B gate);
  3. stress-tests it: balancing assumes every model wrote the same number of test
     passages, which nobody knows. So it also measures 50 seed-42 subsets where 6 random
     models keep only half of their passages. A trick that only works on a perfectly
     balanced test would show up here as a loss (P3a's lesson about layout tricks).
Keep it only if all three agree. Prints numbers only, never a passage.

Usage (repo root, with the P3b features and the encoder's OOF files downloaded):
  python -I -X utf8 scripts/sinkhorn_check.py
"""

import argparse
from pathlib import Path

import numpy as np

from name_that_llm.balance import sinkhorn_balance
from name_that_llm.blend import blend
from name_that_llm.config import BlendConfig, EnsembleConfig, load_config
from name_that_llm.data import LABELS, load_train
from name_that_llm.lm import select_layers
from name_that_llm.metrics import prompt_bootstrap_ci, score
from name_that_llm.train import run_cv, run_cv_features

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LABEL_ARRAY = np.array(LABELS)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lm-features-dir", type=Path, default=Path("data/kaggle-outputs/lm-features")
    )
    parser.add_argument(
        "--encoder-oof-dir", type=Path, default=Path("data/kaggle-outputs/encoder-final")
    )
    parser.add_argument("--subsets", type=int, default=50)
    return parser.parse_args(argv)


def champion_oof(seed: int, train, X_lm, ens: EnsembleConfig, bcfg: BlendConfig, enc_dir: Path):
    """The champion's (n, 18) out-of-fold probabilities on the folds of `seed`."""
    text = run_cv(load_config(bcfg.text_config).model_copy(update={"seed": seed}), train).oof
    lm = run_cv_features(X_lm, train, C=bcfg.C, seed=seed).oof
    enc = np.load(enc_dir / f"oof-encoder-seed{seed}.npy").astype(float)
    return blend([text, lm, enc], ens.weights)


def acc(y, proba, alpha: float) -> float:
    """Balanced accuracy of the predictions after balancing at strength `alpha`."""
    return score(y, LABEL_ARRAY[sinkhorn_balance(proba, alpha).argmax(axis=1)])


def main(argv=None) -> None:
    args = parse_args(argv)
    ens = load_config("ensemble", model_cls=EnsembleConfig)
    bcfg = load_config(ens.blend, model_cls=BlendConfig)
    train = load_train()
    y = train["model"].to_numpy()
    ids = np.load(args.lm_features_dir / "train_ids.npy", allow_pickle=True)
    if not (ids == train["id"].to_numpy()).all():
        raise SystemExit("STOP: the saved LM features are not in the training data's row order")
    feats = np.load(args.lm_features_dir / f"features-{bcfg.lm_variation}.npy", mmap_mode="r")
    X_lm = select_layers(feats, bcfg.layer, bcfg.layer_window)

    oof = {}
    for seed in (7, 42, 43, 44):
        oof[seed] = champion_oof(seed, train, X_lm, ens, bcfg, args.encoder_oof_dir)
        print(f"seed {seed}: champion OOF {acc(y, oof[seed], 0.0):.4f}", flush=True)

    print("\n1. alpha, chosen on seed 7")
    on_seed7 = {a: acc(y, oof[7], a) for a in ALPHAS}
    for a, s in on_seed7.items():
        print(f"   alpha {a:.2f}: {s:.4f} ({s - on_seed7[0.0]:+.4f})")
    best = max(ALPHAS, key=lambda a: (on_seed7[a], -a))  # ties: the weaker balancing

    print(f"\n2. alpha {best:.2f} on the fresh seeds")
    gains = []
    for seed in (42, 43, 44):
        gains.append(acc(y, oof[seed], best) - acc(y, oof[seed], 0.0))
        print(f"   seed {seed}: {gains[-1]:+.4f}")
    raw = LABEL_ARRAY[oof[42].argmax(axis=1)]
    balanced = LABEL_ARRAY[sinkhorn_balance(oof[42], best).argmax(axis=1)]
    diff, lo, hi = prompt_bootstrap_ci(y, raw, balanced, train["prompt_id"].to_numpy())
    ci = f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    print(f"   mean {np.mean(gains):+.4f} | seed-42 bootstrap {diff:+.4f}, {ci}")

    print(f"\n3. alpha {best:.2f} when 6 models keep only half their passages (seed 42)")
    deltas = []
    for i in range(args.subsets):
        rng = np.random.default_rng(i)
        thinned = set(rng.choice(LABELS, size=6, replace=False))
        keep = np.array([m not in thinned or rng.random() < 0.5 for m in y])
        deltas.append(acc(y[keep], oof[42][keep], best) - acc(y[keep], oof[42][keep], 0.0))
    print(f"   {args.subsets} subsets: mean {np.mean(deltas):+.4f}, min {np.min(deltas):+.4f}")

    keep_it = best > 0 and np.mean(gains) > 0 and lo > 0 and np.mean(deltas) >= 0
    print(
        f"\nVERDICT: {'KEEP' if keep_it else 'DROP'} Sinkhorn"
        + (f" at alpha {best:.2f}" if keep_it else " (alpha = 0)")
    )


if __name__ == "__main__":
    main()
