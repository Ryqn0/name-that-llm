"""Register the leaderboard champion in the MLflow model registry (P4, run in Task 7).

Run by hand, once per new champion (P6 will automate it). In order, it:
  1. fits `lr` on all 2,699 training passages;
  2. fits the LM head on the GPU features the lm-features notebook saved (Qwen 3 4B,
     layers 6-9), after checking that their row order matches the training data;
  3. checks that Qwen from the Hugging Face Hub, cut to 10 blocks and run on CPU, gives
     those same features: the app computes them that way, so a mismatch would quietly
     change the model. It stops here if they differ;
  4. measures the "How sure?" bands on out-of-fold predictions (seed 42);
  5. saves the champion's parts (registry.save_champion_artifacts);
  6. checks PARITY: the champion rebuilt from those files, as the app will run it, against
     the offline computation, on 50 passages. It stops here if they disagree;
  7. unless --dry-run, logs a new registry version and points the alias at it.

The MLflow server comes from the environment: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME
and MLFLOW_TRACKING_PASSWORD (a DagsHub token, which the user sets; never printed).
Nothing here prints a passage: only counts and scores, like everything in this repo.

Usage:
  python scripts/register_champion.py --lm-features-dir DIR --encoder-final-dir DIR [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from name_that_llm import __version__
from name_that_llm.blend import blend
from name_that_llm.champion import fit_confidence_bands
from name_that_llm.config import BlendConfig, EncoderConfig, EnsembleConfig, load_config
from name_that_llm.data import load_train
from name_that_llm.encoder import find_fold_dirs, predict_proba
from name_that_llm.lm import load_lm_layers, pooled_hidden_states, select_layers
from name_that_llm.models import build_dense_model, build_model
from name_that_llm.registry import (
    REGISTERED_NAME,
    load_champion,
    log_champion,
    save_champion_artifacts,
    set_alias,
)
from name_that_llm.submit import _aligned_proba
from name_that_llm.train import run_cv, run_cv_features


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lm-features-dir", type=Path, required=True)
    parser.add_argument("--encoder-final-dir", type=Path, required=True)
    parser.add_argument("--qwen", default="Qwen/Qwen3-4B-Base", help="Hugging Face Hub id")
    parser.add_argument("--revision", help="the Hub commit to pin (default: the current one)")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--out", type=Path, default=Path("data/kaggle-outputs/champion"))
    parser.add_argument("--check-rows", type=int, default=20)
    parser.add_argument("--parity-rows", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="every check, but no registry")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ens = load_config("ensemble", model_cls=EnsembleConfig)
    blend_cfg = load_config(ens.blend, model_cls=BlendConfig)
    text_cfg = load_config(blend_cfg.text_config)
    enc_cfg = load_config(ens.encoder, model_cls=EncoderConfig)
    layer, window = blend_cfg.layer, blend_cfg.layer_window
    train = load_train()
    texts, y = train["text"].tolist(), train["model"].to_numpy()
    rng = np.random.default_rng(0)

    print("1. lr, fit on all of train", flush=True)
    text_model = build_model(text_cfg).fit(texts, y)

    print("2. the LM head, on the GPU features of the lm-features notebook", flush=True)
    ids = np.load(args.lm_features_dir / "train_ids.npy", allow_pickle=True)
    if not (ids == train["id"].to_numpy()).all():
        sys.exit("STOP: the saved features are not in the training data's row order")
    features = np.load(
        args.lm_features_dir / f"features-{blend_cfg.lm_variation}.npy", mmap_mode="r"
    )
    X = select_layers(features, layer, window)
    lm_head = build_dense_model(blend_cfg.C).fit(X, y)

    print("3. Qwen from the Hub (10 blocks, CPU) vs those GPU features", flush=True)
    revision = args.revision
    if revision is None:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(args.qwen).sha
    print(f"   {args.qwen} pinned at revision {revision}")
    lm, tokenizer = load_lm_layers(args.qwen, layer + window, revision)
    rows = rng.choice(len(train), args.check_rows, replace=False)
    hub = pooled_hidden_states(
        [texts[i] for i in rows],
        lm,
        tokenizer,
        layers=range(layer, layer + window),
        batch_size=4,
        max_length=blend_cfg.max_length,
    )
    x_hub = select_layers(hub, 0, window)
    rel = np.linalg.norm(x_hub - X[rows], axis=1) / np.linalg.norm(X[rows], axis=1)
    same = _aligned_proba(lm_head, x_hub).argmax(1) == _aligned_proba(lm_head, X[rows]).argmax(1)
    print(
        f"   relative difference: median {np.median(rel):.4f}, max {rel.max():.4f} | "
        f"same LM-head top label: {same.sum()} of {len(rows)}"
    )
    if np.median(rel) > 0.01 or not same.all():
        sys.exit("STOP: the Hub weights differ from the Kaggle copy the LM head learned from")
    del lm  # free ~5 GB before the parity check loads the champion's own copy

    print("4. the 'How sure?' bands, from out-of-fold predictions (seed 42)", flush=True)
    oof_text = run_cv(text_cfg.model_copy(update={"seed": 42}), train).oof
    oof_lm = run_cv_features(X, train, C=blend_cfg.C, seed=42).oof
    oof_enc = np.load(args.encoder_final_dir / "oof-encoder-seed42.npy", allow_pickle=False)
    bands = fit_confidence_bands(blend([oof_text, oof_lm, oof_enc.astype(float)], ens.weights), y)
    for band in bands:
        print(f"   {band.label:12s} top >= {band.min_top:.3f}: right {band.accuracy:.1%}")
    if not all(a.accuracy >= b.accuracy for a, b in zip(bands, bands[1:], strict=False)):
        print("   WARNING: a more confident band is not more often right")

    print("5. save the champion's parts", flush=True)
    meta = {
        "weights": list(ens.weights),
        "bands": [vars(band) for band in bands],
        "lm": {
            "source": args.qwen,
            "revision": revision,
            "layer": layer,
            "window": window,
            "max_length": blend_cfg.max_length,
        },
        "encoder_max_length": enc_cfg.max_length,
        "package_version": __version__,
        "git_commit": _git_commit(),
    }
    save_champion_artifacts(
        args.out,
        text_model=text_model,
        lm_head=lm_head,
        lm_train_mean=X.mean(axis=0),
        encoder_dirs=find_fold_dirs(args.encoder_final_dir, name="encoder", n_folds=5),
        meta=meta,
    )
    print(f"   saved to {args.out}")

    print("6. parity: the app's computation vs the offline one", flush=True)
    champion, _ = load_champion(args.out)
    rows = rng.choice(len(train), args.parity_rows, replace=False)
    sample = [texts[i] for i in rows]
    app = champion.predict_proba(sample)
    # Offline: the same three parts, but the LM head reads the GPU features. The encoder
    # models are the champion's own, so only the LM features (and flattening) can differ.
    p_enc = np.mean(
        [predict_proba(m, t, sample, max_length=enc_cfg.max_length) for m, t in champion.encoders],
        axis=0,
    )
    offline = blend(
        [_aligned_proba(text_model, sample), _aligned_proba(lm_head, X[rows]), p_enc], ens.weights
    )
    agree = (app.argmax(1) == offline.argmax(1)).sum()
    diff = np.abs(app - offline).max()
    print(f"   same top label: {agree} of {len(rows)} | max probability difference {diff:.4f}")
    if agree < len(rows) or diff >= 0.01:
        sys.exit("STOP: the app's computation disagrees with the offline champion")

    if args.dry_run:
        print("dry run: every check passed; nothing was registered")
        print(json.dumps({k: meta[k] for k in ("weights", "lm", "package_version")}, indent=2))
        return
    print("7. register", flush=True)
    version = log_champion(args.out)
    set_alias(REGISTERED_NAME, args.alias, version)
    print(f"   {REGISTERED_NAME} version {version} is now @{args.alias}")


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


if __name__ == "__main__":
    main()
