"""Compare the public pool's latest release with our pinned copy, source by source (P8).

The public pool (crtproduction/name-that-llm-open-pool) changes upstream without notice:
between two of our runs it lost 2 of our 5 allowlisted sources, and only a total in a log
showed it (P7). We train on a private pinned copy of v10, whose per-source counts are
committed in src/name_that_llm/configs/pool_pinned_counts.json. Every week, the workflow
pool-watch.yml downloads the latest file and runs this check:
  - dropped: sources of our pinned copy that the latest release no longer has;
  - shrunk: sources with fewer passages now;
  - new: sources we don't have.
A dropped source that configs/pool_all.yaml doesn't exclude yet fails the check (exit 1),
and the failure emails the repo's owner. New runs must then exclude it (spec §3).

It reads the source column only, never a passage.

Usage:  python scripts/pool_release_check.py LATEST.parquet [--pinned COUNTS.json]
"""

import argparse
import json
from importlib.resources import files
from pathlib import Path

import pandas as pd

from name_that_llm.pool import load_pool_sources

PINNED = files("name_that_llm") / "configs" / "pool_pinned_counts.json"


def compare_counts(pinned: dict[str, int], latest: dict[str, int]) -> dict[str, list[str]]:
    """The sources dropped, shrunk and new in `latest`, each list sorted."""
    return {
        "dropped": sorted(set(pinned) - set(latest)),
        "shrunk": sorted(s for s in pinned if s in latest and latest[s] < pinned[s]),
        "new": sorted(set(latest) - set(pinned)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("latest", type=Path, help="the latest release's ntl_open_pool.parquet")
    parser.add_argument("--pinned", type=Path, help="the pinned per-source counts (JSON)")
    args = parser.parse_args(argv)

    pinned_text = args.pinned.read_text(encoding="utf-8") if args.pinned else PINNED.read_text()
    pinned = json.loads(pinned_text)
    sources = pd.read_parquet(args.latest, columns=["source_dataset"])["source_dataset"]
    latest = {str(k): int(v) for k, v in sources.value_counts().items()}
    changes = compare_counts(pinned, latest)
    for kind, names in changes.items():
        print(f"{kind}: {names}")
    excluded = {name for s in load_pool_sources("pool_all") for name in (s.exclude or [])}
    unhandled = [name for name in changes["dropped"] if name not in excluded]
    if unhandled:
        print(f"ACTION: add {unhandled} to configs/pool_all.yaml's exclude (spec §3)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
