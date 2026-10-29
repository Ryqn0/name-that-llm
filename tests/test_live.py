"""Tests for the live numbers (P6): what visitors' "Do you know who wrote it?" answers say.

name_that_llm.live reads the demo's events back from Loki (Grafana's log store) and
measures the models on them. Visitors' answers are rare, so every number comes with an
honest interval, and the robot's synthetic events and answers naming no model of ours
never count.
"""

import pandas as pd
import pytest

from name_that_llm.labels import LABELS
from name_that_llm.live import event_counts, live_accuracy, paired, parse_loki, wilson

A, B, C = LABELS[0], LABELS[1], LABELS[2]


def _line(name: str, **attributes: str) -> str:
    """One event as Grafana's Faro pipeline writes it to Loki: a logfmt line."""
    fields = {"timestamp": "2026-11-02T10:00:00Z", "kind": "event", "event_name": name}
    fields |= {"event_domain": "demo", "app_name": "name-that-llm-demo", "app_version": "abc1234"}
    fields |= {f"event_data_{key}": value for key, value in attributes.items()}
    return " ".join(
        f'{key}="{value}"' if " " in value else f"{key}={value}" for key, value in fields.items()
    )


def _response(*lines: str) -> dict:
    values = [[str(1_793_000_000_000_000_000 + i), line] for i, line in enumerate(lines)]
    stream = {"stream": {"app_name": "name-that-llm-demo", "kind": "event"}, "values": values}
    return {"status": "success", "data": {"resultType": "streams", "result": [stream]}}


def feedback(actual, label, challenger=None, synthetic="false", run="champ1", chal="chal1"):
    attributes = {"actual": actual, "label": label, "band": "Unsure", "synthetic": synthetic}
    attributes |= {"model_run": run, "challenger_run": chal if challenger else ""}
    if challenger:
        attributes["challenger_label"] = challenger
    return _line("feedback", **attributes)


def test_parse_loki_reads_events():
    response = _response(
        _line("answer", answer_ms="12", words="75", label=A, band="Fairly sure", synthetic="false"),
        _line(
            "error", where="answer", kind="TypeError", message="x is undefined", synthetic="false"
        ),
    )
    events = parse_loki(response)
    assert list(events["name"]) == ["answer", "error"]
    assert events.loc[0, "label"] == A and events.loc[0, "answer_ms"] == "12"
    assert events.loc[1, "message"] == "x is undefined"  # a quoted value, spaces kept
    assert events.loc[0, "app_version"] == "abc1234"
    assert pd.api.types.is_datetime64_any_dtype(events["time"])


def test_live_accuracy_ignores_synthetic_and_unknown_authors():
    events = parse_loki(
        _response(
            feedback(A, A),  # right
            feedback(B, A),  # wrong
            feedback(A, A, synthetic="true"),  # the robot: never counted
            feedback("a person", A),  # no model of ours: no right answer exists
            feedback("another model", B),
            feedback(C, C, challenger=B),  # champion right, challenger wrong
        )
    )
    table = live_accuracy(events).set_index(["role", "run"])
    champion = table.loc[("champion", "champ1")]
    assert champion["n"] == 3 and champion["accuracy"] == pytest.approx(2 / 3)
    challenger = table.loc[("challenger", "chal1")]
    assert challenger["n"] == 1 and challenger["accuracy"] == 0
    assert champion["lo"] < champion["accuracy"] < champion["hi"]


def test_wilson_interval_is_honest_for_small_n():
    # 1 right out of 1 is no proof of a perfect model: the 95% interval spans most of the
    # range (exactly [0.207, 1.0]), where a naive one would be [1, 1].
    lo, hi = wilson(1, 1)
    assert lo == pytest.approx(0.2065, abs=1e-3) and hi == pytest.approx(1.0)
    lo, hi = wilson(0, 1)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.7935, abs=1e-3)
    lo, hi = wilson(50, 100)
    assert hi - lo < 0.2  # more answers, a narrower interval


def test_paired_counts_only_rows_with_both_labels():
    events = parse_loki(
        _response(
            feedback(A, A, challenger=B),  # only the champion right
            feedback(B, A, challenger=B),  # only the challenger right
            feedback(C, C, challenger=C),  # both right
            feedback(A, A),  # no challenger label: not paired
            feedback(A, B, challenger=A, synthetic="true"),  # the robot: never counted
        )
    )
    assert paired(events) == {"n": 3, "champion_only": 1, "challenger_only": 1}


def test_export_fetches_every_page_from_loki():
    # Loki returns at most `limit` lines per query: the export must ask again from just
    # after the newest line, until a page comes back short.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "export_events.py"
    spec = importlib.util.spec_from_file_location("export_events", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeLoki:
        def __init__(self):
            self.starts = []

        def get(self, path, params):
            self.starts.append(params["start"])
            first = params["start"] == 0
            stamps = [10, 20] if first else [30]  # a full page of 2, then a short one
            page = _response(*[feedback(A, A) for _ in stamps])
            page["data"]["result"][0]["values"] = [
                [str(s), v]
                for s, (_, v) in zip(stamps, page["data"]["result"][0]["values"], strict=True)
            ]
            return type(
                "R", (), {"json": lambda self: page, "raise_for_status": lambda self: self}
            )()

    loki = FakeLoki()
    pages = module.fetch(loki, 0, 100, limit=2)
    assert len(pages) == 2 and loki.starts == [0, 21]


def test_an_empty_week_gives_zeros():
    # A quiet week is likely (the demo is small): the export must still log a run of zeros.
    events = parse_loki({"status": "success", "data": {"resultType": "streams", "result": []}})
    assert live_accuracy(events).empty
    assert paired(events) == {"n": 0, "champion_only": 0, "challenger_only": 0}
    zeros = dict.fromkeys(["real_visits", "real_answers", "real_feedback", "real_errors"], 0)
    assert event_counts(events) == {"events": 0} | zeros
    busy = parse_loki(
        _response(
            _line("page_load", what="champion", ok="true", synthetic="false"),
            _line("page_load", what="challenger", ok="true", synthetic="false"),
            _line("answer", label=A, synthetic="true"),  # the robot: not a real answer
            feedback(A, A),
        )
    )
    assert event_counts(busy) == {"events": 4} | zeros | {"real_visits": 1, "real_feedback": 1}


def test_parse_loki_survives_junk_values():
    # The final review's finding C: anyone can send events to the public collector, and one
    # junk value must not sink a week's export. An apostrophe broke the old parser (shlex).
    junk = "kind=event event_name=error event_data_kind=can't event_data_synthetic=false "
    junk += r'event_data_message="a \"quoted\" path C:\\x" app_version=abc'  # Go-escaped
    events = parse_loki(_response(junk))
    assert events.loc[0, "kind"] == "can't"
    assert events.loc[0, "message"] == r'a "quoted" path C:\x'


def test_a_window_without_the_synthetic_field_gives_zeros():
    # Junk events may lack fields our page always sends: nothing may crash on them.
    events = parse_loki(_response(_line("feedback", actual=A, label=A)))  # no synthetic
    assert event_counts(events)["real_feedback"] == 0
    assert live_accuracy(events).empty
    assert paired(events) == {"n": 0, "champion_only": 0, "challenger_only": 0}


def _export_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "export_events.py"
    spec = importlib.util.spec_from_file_location("export_events_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_weeks_are_fixed_monday_windows():
    # The final review's finding 5: a window that ends "now" leaves gaps (or overlaps) when
    # a run is late, repeated or missed, and Loki forgets after about two weeks. Weeks are
    # fixed: Monday 00:00 UTC to the next Monday.
    from datetime import UTC, datetime

    module = _export_module()
    wednesday = datetime(2026, 11, 4, 10, 30, tzinfo=UTC)
    start, end = module.week_window(wednesday)
    assert (start, end) == (datetime(2026, 10, 26, tzinfo=UTC), datetime(2026, 11, 2, tzinfo=UTC))
    start, end = module.week_window(wednesday, end="2026-10-26")  # redo a missed week
    assert (start, end) == (datetime(2026, 10, 19, tzinfo=UTC), datetime(2026, 10, 26, tzinfo=UTC))


def test_metric_names_are_safe_whatever_the_run_id():
    # A junk model_run (from a fake event) must not make the MLflow run fail.
    module = _export_module()
    assert module.metric_name("champion", "run/../x y!") == "champion_run_.._x_y_"
