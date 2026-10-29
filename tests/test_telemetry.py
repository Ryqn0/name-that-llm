"""Tests for the demo's telemetry (P6): anonymous events, private by construction.

The page runs app/demo/telemetry.js. These tests drive that same module under Node (the
`event` and `send` commands of app/demo/cli.mjs), as test_browser.py does for the engine.
They skip where Node isn't installed (it is on GitHub's CI machines).
"""

import json
import re
import shutil

import pytest

from name_that_llm.browser import export_browser_model, run_engine
from name_that_llm.champion import Band
from name_that_llm.config import load_config
from name_that_llm.data import LABELS
from name_that_llm.models import build_model

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="these tests need Node")

BANDS = [Band("Fairly sure", 0.2, 0.6), Band("Unsure", 0.1, 0.4), Band("Very unsure", 0.0, 0.2)]
CTX = {"labels": LABELS, "bands": [vars(band) for band in BANDS], "text": ""}
META = {"commit": "abc1234", "model_run": "run1", "challenger_run": "", "synthetic": False}
# A passage written for these tests, never a competition passage.
PASSAGE = (
    "Lighthouses guided sailors through treacherous waters long before satellites existed. "
    "Keepers trimmed wicks nightly, polished enormous lenses, and logged every passing "
    "vessel in weathered notebooks."
)


def make(events: list[dict], text: str = "") -> dict:
    """The events through makeEvent, and their payload: {"events", "payload"} or {"error"}."""
    return run_engine("event", {"events": events, "ctx": {**CTX, "text": text}, "meta": META})


def test_events_keep_only_allowlisted_typed_fields():
    # 10**400 overflows to Infinity in JavaScript: a non-finite number.
    fields = {"answer_ms": 10**400, "label": "not-a-model", "text": "anything", "band": "Unsure"}
    (event,) = make([{"name": "answer", "fields": fields | {"unusual": ["short"]}}])["events"]
    assert event == {"name": "answer", "fields": {"band": "Unsure", "unusual": ["short"]}}


def test_unknown_event_names_are_refused():
    assert "unknown event" in make([{"name": "visit", "fields": {}}])["error"]


def test_string_fields_never_carry_words_of_the_text():
    message = f"could not parse: {PASSAGE[:70]}"
    fields = {"where": "answer", "kind": "TypeError", "message": message}
    (event,) = make([{"name": "error", "fields": fields}], text=PASSAGE)["events"]
    assert event["fields"] == {"where": "answer", "kind": "TypeError", "message": "[redacted]"}


def test_payload_of_a_real_flow_contains_no_word_of_the_passage(fake_train, tmp_path):
    lr = build_model(load_config("lr")).fit(fake_train["text"].tolist(), fake_train["model"])
    export_browser_model(lr, BANDS, tmp_path / "model", meta={"source": "test"})
    request = {"model": str(tmp_path / "model"), "texts": [PASSAGE]}
    proba = run_engine("predict", request)["probabilities"][0]
    label = LABELS[proba.index(max(proba))]
    answer = {"answer_ms": 12.5, "words": len(PASSAGE.split()), "label": label}
    feedback = {"actual": "a person", "label": label, "challenger_label": None}
    events = [
        {"name": "answer", "fields": answer | {"band": "Very unsure", "unusual": []}},
        {"name": "feedback", "fields": feedback | {"band": "Very unsure"}},
        {"name": "error", "fields": {"where": "feedback", "kind": "Error", "message": PASSAGE}},
    ]
    dumped = json.dumps(make(events, text=PASSAGE)["payload"]).lower()
    words = {word.lower() for word in re.findall(r"[^\W\d_]{4,}", PASSAGE)}
    assert words and not [word for word in words if word in dumped]


def test_payload_matches_the_collector_format():
    # Grafana's collector (Faro): events with STRING attributes; meta.app, and meta.sdk,
    # which the collector requires (checked on the real stack: "no SDK version" otherwise).
    # No session, user, page or browser part, so no payload can tell two visitors apart.
    fields = {"load_ms": 812.4, "ok": True, "what": "champion"}
    payload = make([{"name": "page_load", "fields": fields}])["payload"]
    assert set(payload) == {"events", "meta"} and set(payload["meta"]) == {"app", "sdk"}
    assert payload["meta"]["app"]["version"] == "abc1234"
    assert payload["meta"]["sdk"] == {"name": "name-that-llm-telemetry", "version": "1.0.0"}
    (event,) = payload["events"]
    assert event["name"] == "page_load" and event["timestamp"]
    assert event["attributes"] == {
        "load_ms": "812",
        "ok": "true",
        "what": "champion",
        "model_run": "run1",
        "challenger_run": "",
        "synthetic": "false",
    }


def test_word_counts_are_bucketed():
    events = [{"name": "answer", "fields": {"words": n}} for n in (0, 24, 25, 199)]
    assert [event["fields"]["words"] for event in make(events)["events"]] == [0, 0, 25, 175]


def test_send_never_throws():
    # Port 9 refuses connections; an empty URL means no collector is configured.
    for url in ("http://127.0.0.1:9/collect", ""):
        assert run_engine("send", {"url": url})["resolved"] is True


def test_every_request_gets_a_fresh_random_session_id():
    # The collector refuses a request without an X-Faro-Session-Id header (checked on the
    # real stack). A fresh random id per REQUEST satisfies it while linking nothing: not
    # two visits, not even two events of one visit.
    sent = run_engine("send", {"url": "https://collector.test/collect/key", "capture": 2})
    ids = [request["headers"]["X-Faro-Session-Id"] for request in sent["requests"]]
    assert len(ids) == 2 and ids[0] != ids[1] and all(len(i) >= 16 for i in ids)
    assert all(request["keepalive"] for request in sent["requests"])


def test_answer_events_carry_the_challengers_own_time():
    # The final review's finding 6: the challenger now scores after the answer is shown, and
    # its time is a field of its own, so the dashboard can see what shadow mode costs.
    fields = {"answer_ms": 9.6, "challenger_ms": 3.4, "challenger_label": LABELS[2]}
    (event,) = make([{"name": "answer", "fields": fields}])["events"]
    assert event["fields"] == {"answer_ms": 10, "challenger_ms": 3, "challenger_label": LABELS[2]}


def test_redaction_sees_through_unicode_forms():
    # The final review's finding B: a passage pasted in decomposed form (NFD), quoted by an
    # error in composed form (NFC), must still be caught.
    import unicodedata

    text = unicodedata.normalize("NFD", "Café crème brûlée, served at the château.")
    message = unicodedata.normalize("NFC", "failed on: crème brûlée")
    fields = {"where": "answer", "kind": "Error", "message": message}
    (event,) = make([{"name": "error", "fields": fields}], text=text)["events"]
    assert event["fields"]["message"] == "[redacted]"
