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
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from name_that_llm.live import event_counts, live_accuracy, paired, parse_loki

QUERY = '{app_name="name-that-llm-demo", kind="event"}'  # every event of the demo
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


def main(argv=None) -> None:
    import httpx
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)
    keys = ("GRAFANA_LOGS_URL", "GRAFANA_LOGS_USER", "GRAFANA_LOGS_TOKEN", "MLFLOW_TRACKING_URI")
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        sys.exit(f"missing settings: {', '.join(missing)}")

    end_ns = time.time_ns()
    start_ns = end_ns - args.days * 86_400 * 10**9
    auth = (os.environ["GRAFANA_LOGS_USER"], os.environ["GRAFANA_LOGS_TOKEN"])
    with httpx.Client(base_url=os.environ["GRAFANA_LOGS_URL"].rstrip("/"), auth=auth) as client:
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
    day = pd.Timestamp(end_ns, unit="ns").date()
    with mlflow.start_run(run_name=f"events-to-{day}"), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params({"days": args.days, "query": QUERY})
        mlflow.log_metrics(counts | {f"paired_{key}": value for key, value in pairs.items()})
        for row in accuracy.itertuples():
            name = f"{row.role}_{row.run}"
            mlflow.log_metrics({f"{name}_n": row.n, f"{name}_accuracy": row.accuracy})
            mlflow.log_metrics({f"{name}_lo": row.lo, f"{name}_hi": row.hi})
        path = Path(tmp) / "events.csv"
        events.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
    print("logged to the MLflow experiment live-feedback")


if __name__ == "__main__":
    main()
