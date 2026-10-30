"""Tests for the weekly check of the public pool's releases (P8 Task 1).

The public pool changes upstream without notice: between two of our runs it lost 2 of our 5
allowlisted sources (P7). We train on a pinned private copy, and this check compares each
new public release with the pinned per-source counts, so the next change is caught at once.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "pool_release_check", ROOT / "scripts" / "pool_release_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_counts_flags_dropped_shrunk_and_new():
    compare = _script().compare_counts
    assert compare({"a": 10, "b": 5, "c": 3}, {"a": 10, "b": 4, "d": 7}) == {
        "dropped": ["c"],
        "shrunk": ["b"],
        "new": ["d"],
    }


def test_release_check_fails_on_a_dropped_source(tmp_path):
    # A pinned source gone from the latest release fails the check (exit 1), so the weekly
    # workflow emails the owner; new runs must then exclude it (spec §3). The check reads the
    # source column only, never a passage.
    main = _script().main
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps({"a": 2, "b": 1}), encoding="utf-8")
    latest = tmp_path / "latest.parquet"
    pd.DataFrame({"text": ["x", "y"], "source_dataset": ["a", "a"]}).to_parquet(latest)
    assert main([str(latest), "--pinned", str(pinned)]) == 1  # "b" is gone
    pd.DataFrame({"text": ["x", "y", "z"], "source_dataset": ["a", "a", "b"]}).to_parquet(latest)
    assert main([str(latest), "--pinned", str(pinned)]) == 0


def test_pool_watch_workflow_runs_the_check_weekly():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "pool-watch.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads a bare `on:` as True
    assert triggers["schedule"][0]["cron"] and "workflow_dispatch" in triggers
    steps = [step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"]]
    assert any("scripts/pool_release_check.py" in run for run in steps)
