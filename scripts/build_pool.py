"""Build the training pool from public Kaggle datasets (P7 Task 2). Run by hand, on CPU.

For each kept source of configs/pool.yaml it:
  1. (with --download) checks the dataset's license through the Kaggle API, refusing any
     license that isn't explicitly open, then downloads the dataset into its own folder
     (downloads are untrusted data: read with pandas only, never executed);
  2. builds the pool (pool.build_pool): cleaned, deduped against our passages, and each
     passage matched to the training prompt it's about;
  3. writes data/pool/pool.parquet (gitignored), and prints counts only: per original
     source, how many passages were kept and how many matched a training prompt.
It also cross-checks our prompt matching against the open pool's own leak flag.

The Kaggle CLI needs the Avast prefix on this PC (Context.md: Environment).

Usage:
  python -I -X utf8 scripts/build_pool.py [--download] [--root DIR] [--out FILE]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from name_that_llm.data import load_train
from name_that_llm.pool import _key, build_pool, license_allowed, load_pool_sources


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="check licenses, download")
    parser.add_argument("--root", type=Path, default=Path("data/external"))
    parser.add_argument("--out", type=Path, default=Path("data/pool/pool.parquet"))
    return parser.parse_args(argv)


def dataset_licenses(slug: str) -> list[str]:
    """The license names Kaggle reports for a dataset (read-only API call)."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["kaggle", "datasets", "metadata", "-p", tmp, slug], check=True)
        info = json.loads((Path(tmp) / "dataset-metadata.json").read_text(encoding="utf-8"))
    return [entry.get("name", "") for entry in info["info"].get("licenses", [])]


def main(argv=None) -> None:
    args = parse_args(argv)
    sources = [s for s in load_pool_sources() if s.keep]
    if args.download:
        for source in sources:
            names = dataset_licenses(source.slug)
            if not names or not all(license_allowed(n) for n in names):
                sys.exit(f"STOP: {source.slug} has license {names}, not an open one")
            folder = args.root / source.slug.replace("/", "__")
            print(f"{source.slug}: license {names}; downloading to {folder}", flush=True)
            subprocess.run(
                ["kaggle", "datasets", "download", source.slug, "-p", str(folder), "--unzip"],
                check=True,
            )

    train = load_train()
    pool = build_pool(sources, args.root, train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out, index=False)

    matched = pool["prompt_id"].notna()
    print(f"\npool: {len(pool):,} passages, {matched.mean():.1%} matched to a training prompt")
    by_source = pool.assign(matched=matched).groupby("source")
    table = by_source.agg(passages=("text", "size"), matched=("matched", "mean"))
    table["models"] = by_source["model"].nunique()
    print(table.sort_values("passages", ascending=False).to_string(float_format="{:.1%}".format))
    per_model = pool["model"].value_counts()
    print(f"\nper model: min {per_model.min():,} ({per_model.idxmin()}), max {per_model.max():,}")

    # Cross-check with the open pool's own leak flag, where the raw file carries one.
    raw_files = list(args.root.rglob("ntl_open_pool.parquet"))
    if raw_files:
        flags = pd.read_parquet(raw_files[0], columns=["text", "flag_near_train_prompt"])
        flagged = set(flags.loc[flags["flag_near_train_prompt"], "text"].map(_key))
        ours = pool["text"].map(_key)
        in_flagged = ours.isin(flagged)
        print(
            f"\ncross-check: {in_flagged.sum():,} pool passages carry the open pool's "
            f"near-train-prompt flag; we matched {matched[in_flagged].mean():.1%} of them, "
            f"and {in_flagged[matched].mean():.1%} of ours carry the flag"
        )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
