"""Tests for the monitoring as code (P6): the Grafana dashboard, the alerts, and their push.

monitoring/ holds the dashboard and alert rules as JSON, and scripts/push_monitoring.py
sends them to Grafana Cloud. What can break silently is tested here: a panel with no
query, robot visits mixed into the real ones, an alert missing, the wrong API, or the
token ending up in a request body.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = json.loads((ROOT / "monitoring" / "dashboard.json").read_text(encoding="utf-8"))
ALERTS = json.loads((ROOT / "monitoring" / "alerts.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def push():
    """scripts/ isn't a package, so load the script as a module from its path."""
    spec = importlib.util.spec_from_file_location("push", ROOT / "scripts" / "push_monitoring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loki_exprs(panels):
    return [t["expr"] for p in panels for t in p.get("targets", []) if "expr" in t]


def test_dashboard_panels_all_have_a_query():
    for panel in DASHBOARD["panels"]:
        if panel["type"] in ("row", "text"):
            continue
        targets = panel["targets"]
        assert targets, panel["title"]
        assert all(t.get("expr") or t.get("url") for t in targets), panel["title"]


def test_live_panels_exclude_synthetic_events():
    # Every query filters `synthetic` one way or the other, so the robot's visits never
    # mix with real visitors' (and real ones never inflate the robot's line).
    exprs = _loki_exprs(DASHBOARD["panels"]) + [r["data"][0]["model"]["expr"] for r in ALERTS]
    assert exprs
    for expr in exprs:
        assert ('event_data_synthetic="false"' in expr) != ('event_data_synthetic="true"' in expr)
    titles = {"Visits (the page loaded)", "Load and answer times (ms)"}
    titles.add("Errors, by where they happened")
    live = [p for p in DASHBOARD["panels"] if p["title"] in titles]
    assert len(live) == 3
    assert all('event_data_synthetic="false"' in e for e in _loki_exprs(live))


def test_alerts_cover_errors_and_robot_silence():
    by_uid = {rule["uid"]: rule for rule in ALERTS}
    errors, robot = by_uid["ntl-errors"], by_uid["ntl-robot-silent"]
    assert 'event_name="error"' in errors["data"][0]["model"]["expr"]
    assert errors["data"][2]["model"]["conditions"][0]["evaluator"] == {"type": "gt", "params": [5]}
    robot_expr = robot["data"][0]["model"]["expr"]
    assert 'event_name="answer"' in robot_expr and "[12h]" in robot_expr
    # No robot answer at all is NO DATA: that must alert too.
    assert robot["noDataState"] == "Alerting"
    for rule in ALERTS:
        assert rule["data"][0]["datasourceUid"] == "${LOKI_UID}"  # filled in at push time


def test_build_requests_targets_the_dashboard_and_alert_apis(push, monkeypatch):
    monkeypatch.setenv("GRAFANA_TOKEN", "glsa_secret-token-for-the-test")
    requests = push.build_requests(DASHBOARD, ALERTS, folder_uid="name-that-llm")
    paths = [path for _, path, _ in requests]
    assert "/api/dashboards/db" in paths
    assert paths.count("/api/v1/provisioning/alert-rules") == len(ALERTS)
    for _, path, body in requests:
        assert "glsa_secret" not in json.dumps(body)  # the token goes in a header only
        if path == "/api/v1/provisioning/alert-rules":
            assert body["folderUID"] == "name-that-llm"


def test_model_history_lists_the_models_by_date():
    lines = (ROOT / "monitoring" / "model_history.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "date,model,cv,lb"
    assert all(re.fullmatch(r"2026-\d\d-\d\d,[^,]+,0\.\d+,(0\.\d+)?", line) for line in lines[1:])


def test_queries_use_the_labels_seen_on_the_real_stack():
    # Checked on the user's stack (2026-10-28) with a test event: Grafana's pipeline puts the
    # app in the stream label service_name (not app_name), next to kind="event".
    exprs = _loki_exprs(DASHBOARD["panels"]) + [r["data"][0]["model"]["expr"] for r in ALERTS]
    assert all('{service_name="name-that-llm-demo", kind="event"}' in e for e in exprs)


def test_both_alerts_are_active():
    # The robot alert started paused (no robot yet). Its events reach Loki now
    # (2026-10-28), so a silent robot must alert.
    assert [rule["isPaused"] for rule in ALERTS] == [False, False]


def test_the_dashboard_opens_on_the_logs_data_source():
    # A Grafana Cloud stack has several Loki data sources (logs, alert state history, usage
    # insights). Without a filter, the picker took the first one alphabetically, the alert
    # history, and every panel came up empty (seen on 2026-10-28).
    (variable,) = [v for v in DASHBOARD["templating"]["list"] if v["name"] == "datasource"]
    pattern = re.compile(variable["regex"].strip("/"))
    names = [
        "grafanacloud-s-alert-state-history",
        "grafanacloud-s-logs",
        "grafanacloud-s-usage-insights",
    ]
    assert [n for n in names if pattern.search(n)] == ["grafanacloud-s-logs"]


LIVE = ("Champion: share right", "Champion vs challenger, on the same texts")


def _panel(title_start: str) -> dict:
    return next(p for p in DASHBOARD["panels"] if p["title"].startswith(title_start))


def test_live_accuracy_counts_only_answers_naming_a_model():
    # The final review's finding 3: "a person" and "another model" answers can't be named by
    # any model, so they must not count as misses (live.py doesn't count them either).
    for title in LIVE:
        for expr in _loki_exprs([_panel(title)]):
            assert 'event_data_actual!="a person"' in expr
            assert 'event_data_actual!="another model"' in expr


def test_live_numbers_are_per_run_on_the_same_texts_and_show_n():
    # Champion and challenger side by side must be on the SAME texts (those the challenger
    # scored), per model run (a promotion mustn't mix two models), with the count shown.
    champion = _loki_exprs([_panel(LIVE[0])])
    assert len(champion) == 2 and all("by (event_data_model_run)" in e for e in champion)
    compared = _loki_exprs([_panel(LIVE[1])])
    assert len(compared) == 3
    assert all('event_data_challenger_label!=""' in e for e in compared)
    assert all("by (event_data_challenger_run)" in e for e in compared)


def test_no_panel_depends_on_an_x_field():
    # The final review's finding 4: instant queries grouped by a label return one frame per
    # value, which a bar chart's x field can't find. Bar gauges draw one bar per series.
    assert not [p["title"] for p in DASHBOARD["panels"] if p["type"] == "barchart"]


def test_unwrapped_values_drop_parse_errors():
    # A non-numeric value (a fake event on the public collector) would make Loki refuse the
    # whole query: such lines are dropped first.
    for expr in _loki_exprs(DASHBOARD["panels"]):
        if "unwrap" in expr:
            assert '__error__=""' in expr


def test_a_model_with_no_right_answer_shows_zero():
    # A count with no matching line returns no series at all, so "0 right" would show as a
    # blank: every "right" count is completed with "or 0 * <all answers>".
    exprs = _loki_exprs([_panel(LIVE[0]), _panel(LIVE[1])])
    assert all(" or 0 * " in e for e in exprs if "right" in e and "label_format" in e)
