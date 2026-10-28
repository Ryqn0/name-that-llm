"""The live numbers (P6): how the models do on visitors' texts, from the demo's events.

The public demo asks "Do you know who wrote it?" after each answer, and sends the reply as
an anonymous `feedback` event (app/demo/telemetry.js) to Grafana Cloud, which stores it in
Loki. This module reads those events back, and measures the champion and the challenger
on the texts visitors could vouch for:
  - parse_loki: a Loki query's answer, as one row per event;
  - live_accuracy: per model run, the share of answers it got right, with an honest
    interval, since visitors' answers are few;
  - paired: on the texts both models scored, how often only one of them was right (the
    fair comparison: the same texts, as in our offline A/B tests).

Only real visitors' answers that name one of our 18 models count. The robot's synthetic
events never do, and neither do "another model" or "a person" (no model of ours can be
right there). scripts/export_events.py runs this every week and keeps the numbers.
"""

import math
import shlex

import pandas as pd

from name_that_llm.labels import LABELS

PREFIX = "event_data_"  # how Grafana's Faro pipeline names an event's attributes in Loki


def parse_loki(response: dict) -> pd.DataFrame:
    """One row per event: `time`, `name`, `app_version`, and the event's attributes.

    Loki returns streams of (nanosecond timestamp, log line); Faro writes each event as a
    logfmt line (key=value pairs, values with spaces in quotes).
    """
    rows = []
    for stream in response["data"]["result"]:
        for stamp, line in stream["values"]:
            pairs = dict(token.split("=", 1) for token in shlex.split(line) if "=" in token)
            row = {"time": pd.to_datetime(int(stamp), unit="ns", utc=True)}
            row["name"] = pairs.get("event_name")
            row["app_version"] = pairs.get("app_version")
            row |= {k.removeprefix(PREFIX): v for k, v in pairs.items() if k.startswith(PREFIX)}
            rows.append(row)
    events = pd.DataFrame(rows)
    return events.sort_values("time", ignore_index=True) if len(events) else events


def event_counts(events: pd.DataFrame) -> dict:
    """The week in numbers: every event, then real visits, answers, feedback and errors.

    A visit is the champion's page_load (the challenger loads later, in the same visit).
    Real means not the robot's (synthetic) events.
    """
    out = {"events": len(events)}
    keys = ["real_visits", "real_answers", "real_feedback", "real_errors"]
    if events.empty:
        return out | dict.fromkeys(keys, 0)
    real = events[events["synthetic"] == "false"]
    visits = (real["name"] == "page_load") & (real.get("what") == "champion")
    names = [real["name"] == name for name in ("answer", "feedback", "error")]
    return out | dict(zip(keys, (int(rows.sum()) for rows in [visits, *names]), strict=True))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson 95% interval for k successes out of n.

    Unlike the naive p +- z * sqrt(p (1 - p) / n), it stays inside [0, 1] and stays wide when
    n is small: 1 right out of 1 gives [0.21, 1.0], not [1, 1].
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def _answers(events: pd.DataFrame) -> pd.DataFrame:
    """Real visitors' feedback rows whose answer names one of our 18 models."""
    if events.empty or "actual" not in events:
        return events.iloc[0:0]
    rows = events[(events["name"] == "feedback") & (events["synthetic"] == "false")]
    return rows[rows["actual"].isin(LABELS)]


def _has(column: pd.Series) -> pd.Series:
    return column.notna() & (column != "")


def live_accuracy(events: pd.DataFrame) -> pd.DataFrame:
    """Per model run: role (champion or challenger), run, n, accuracy, lo, hi (Wilson 95%)."""
    answers = _answers(events)
    columns = ["role", "run", "n", "accuracy", "lo", "hi"]
    if answers.empty:
        return pd.DataFrame(columns=columns)
    parts = [("champion", "model_run", "label", answers)]
    if "challenger_label" in answers:
        both = answers[_has(answers["challenger_label"])]
        parts.append(("challenger", "challenger_run", "challenger_label", both))
    rows = []
    for role, run_column, label_column, rows_of in parts:
        for run, group in rows_of.groupby(run_column):
            n, k = len(group), int((group[label_column] == group["actual"]).sum())
            lo, hi = wilson(k, n)
            rows.append({"role": role, "run": run, "n": n, "accuracy": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=columns)


def paired(events: pd.DataFrame) -> dict:
    """On the answered texts both models scored: how often ONLY the champion was right, and
    how often ONLY the challenger was. Texts both got right, or both wrong, can't tell them
    apart, which is why a paired comparison needs far fewer texts than two separate ones."""
    answers = _answers(events)
    if answers.empty or "challenger_label" not in answers:
        return {"n": 0, "champion_only": 0, "challenger_only": 0}
    answers = answers[_has(answers["challenger_label"])]
    champion = answers["label"] == answers["actual"]
    challenger = answers.get("challenger_label") == answers["actual"]
    return {
        "n": len(answers),
        "champion_only": int((champion & ~challenger).sum()),
        "challenger_only": int((challenger & ~champion).sum()),
    }
