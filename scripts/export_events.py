"""The weekly export of the demo's events to DagsHub (P6): run by weekly-export.yml.

Grafana Cloud's free tier keeps logs for only about two weeks, and visitors' "Do you know
who wrote it?" answers are rare. So every week, this script copies the last 7 days of the
demo's events into the private DagsHub MLflow, where they accumulate:
  1. it reads the demo's events from Loki (GRAFANA_LOGS_URL, with the user
     GRAFANA_LOGS_USER and the token GRAFANA_LOGS_TOKEN), page by page;
  2. it logs one MLflow run in the experiment "live-feedback": the events as a CSV
     artifact; as metrics, the counts, each model run's live accuracy with its Wilson
     interval (name_that_llm.live), and the paired comparison.
The events hold no text, by construction (app/demo/telemetry.js), and this prints counts
only: the repo is public, and so are its Actions logs.

Usage:  python scripts/export_events.py [--days 7]
"""

import argparse
import os
import re
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from name_that_llm.live import event_counts, live_accuracy, paired, parse_loki

QUERY = '{service_name="name-that-llm-demo", kind="event"}'  # as labeled on the real stack
LIMIT = 5000  # Loki's usual maximum of lines per query


def fetch(client, start_ns: int, end_ns: int, limit: int = LIMIT) -> list[dict]:
    """Every Loki response for the demo's events between the two times, oldest first.

    One query returns at most `limit` lines, so the next query starts just after the
    newest line received, until a page comes back short.
    """
    pages = []
    while True:
        params = {"query": QUERY, "start": start_ns, "end": end_ns, "limit": limit}
        response = client.get("/loki/api/v1/query_range", params=params | {"direction": "forward"})
        response.raise_for_status()
        page = response.json()
        pages.append(page)
        stamps = [int(s) for stream in page["data"]["result"] for s, _ in stream["values"]]
        if len(stamps) < limit:
            return pages
        start_ns = max(stamps) + 1


def week_window(now: datetime, end: str | None = None, days: int = 7) -> tuple[datetime, datetime]:
    """The week to export: (start, end), both at 00:00 UTC, end a Monday by default.

    Fixed weeks, not "the last 7 days from now": a late, repeated or failed run would
    otherwise leave a gap, or count events twice, and Loki forgets after about two weeks (the
    final review's finding 5). `end` (YYYY-MM-DD) redoes a given week; by default, the latest
    Monday 00:00 UTC at or before `now`.
    """
    if end is not None:
        stop = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        stop = today - timedelta(days=today.weekday())  # Monday is weekday 0
    return stop - timedelta(days=days), stop


def metric_name(role: str, run: str) -> str:
    """An MLflow metric name for one model run, safe whatever the run id: a fake event on
    the public collector can carry any text, and MLflow refuses some characters."""
    return re.sub(r"[^0-9A-Za-z_.-]", "_", f"{role}_{run}")[:200]


def main(argv=None) -> None:
    import httpx
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7, help="the window's length")
    parser.add_argument("--end", help="redo the week ending on this date (YYYY-MM-DD, 00:00 UTC)")
    args = parser.parse_args(argv)
    keys = ("GRAFANA_LOGS_URL", "GRAFANA_LOGS_USER", "GRAFANA_LOGS_TOKEN", "MLFLOW_TRACKING_URI")
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        sys.exit(f"missing settings: {', '.join(missing)}")

    start, end = week_window(datetime.now(UTC), args.end, args.days)
    start_ns, end_ns = int(start.timestamp()) * 10**9, int(end.timestamp()) * 10**9
    print(f"week: {start:%Y-%m-%d} to {end:%Y-%m-%d} (00:00 UTC)")
    auth = (os.environ["GRAFANA_LOGS_USER"], os.environ["GRAFANA_LOGS_TOKEN"])
    url = os.environ["GRAFANA_LOGS_URL"].rstrip("/")
    # 60 s: a week of events takes Loki longer than httpx's default 5 s (the first run timed out).
    with httpx.Client(base_url=url, auth=auth, timeout=60) as client:
        pages = fetch(client, start_ns, end_ns)
    events = pd.concat([parse_loki(page) for page in pages], ignore_index=True)
    counts = event_counts(events)
    print(", ".join(f"{key} {value}" for key, value in counts.items()))
    accuracy, pairs = live_accuracy(events), paired(events)
    for row in accuracy.itertuples():
        print(
            f"{row.role} {row.run[:7]}: {row.accuracy:.0%} of {row.n} [{row.lo:.0%}, {row.hi:.0%}]"
        )
    print(
        f"paired, on {pairs['n']} texts: only the champion right {pairs['champion_only']}, "
        f"only the challenger {pairs['challenger_only']}"
    )

    mlflow.set_experiment("live-feedback")
    name = f"events-{start:%Y-%m-%d}-to-{end:%Y-%m-%d}"
    with mlflow.start_run(run_name=name), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params({"start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}", "query": QUERY})
        mlflow.log_metrics(counts | {f"paired_{key}": value for key, value in pairs.items()})
        for row in accuracy.itertuples():
            name = metric_name(row.role, row.run)
            mlflow.log_metrics({f"{name}_n": row.n, f"{name}_accuracy": row.accuracy})
            mlflow.log_metrics({f"{name}_lo": row.lo, f"{name}_hi": row.hi})
        path = Path(tmp) / "events.csv"
        events.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
    print("logged to the MLflow experiment live-feedback")


if __name__ == "__main__":
    main()
