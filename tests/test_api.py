"""Tests for the web app (P4): routes, limits, loading states and privacy.

A FAKE predictor stands in for the champion: these tests are about the web layer, so they
run anywhere in milliseconds, without torch or MLflow.
"""

import asyncio
import logging
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from name_that_llm import __version__  # noqa: E402
from name_that_llm.api import LAB_NAMES, Busy, RateLimiter, SingleFlight, create_app  # noqa: E402
from name_that_llm.champion import Band  # noqa: E402
from name_that_llm.data import LAB_OF, LABELS  # noqa: E402

BANDS = [Band("Fairly sure", 0.6, 0.8), Band("Unsure", 0.3, 0.45), Band("Very unsure", 0.0, 0.2)]
# One probability per label, all different, so the expected ranking is unambiguous:
# LABELS[5] gets the most (0.15), then LABELS[0], then the rest in order.
PROBA = np.array([0.12, *np.linspace(0.08, 0.01, 16)[:4], 0.15, *np.linspace(0.08, 0.01, 16)[4:]])
PROBA = PROBA / PROBA.sum()


class FakePredictor:
    def __init__(self):
        self.info = {"model_version": "7", "encoder_folds": 5, "bands": BANDS}

    def predict_proba(self, texts):
        return np.tile(PROBA, (len(texts), 1))


def words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def ready_client(load=FakePredictor, **settings) -> TestClient:
    """A test client whose model has finished loading."""
    client = TestClient(create_app(load, **settings))
    client.__enter__()  # runs the startup, which loads the predictor in a thread
    for _ in range(200):
        if client.get("/health").status_code != 503:
            break
        time.sleep(0.01)
    return client


def test_page_has_the_form_and_the_warning():
    client = ready_client()
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    for required in (
        "Which LLM wrote this?",
        "It is not an AI detector",
        "Your text is not stored",
    ):
        assert required in page.text


def test_predict_returns_all_18_sorted_with_companies():
    body = ready_client().post("/predict", json={"text": words(50)}).json()
    ranking = body["ranking"]
    assert len(ranking) == 18
    assert ranking[0]["model"] == LABELS[5] and ranking[1]["model"] == LABELS[0]
    assert [r["probability"] for r in ranking] == sorted(
        (r["probability"] for r in ranking), reverse=True
    )
    assert sum(r["probability"] for r in ranking) == pytest.approx(1, abs=1e-6)
    assert all(r["lab"] == LAB_NAMES[LAB_OF[r["model"]]] for r in ranking)
    # The top probability is 0.15 / (sum ≈ 1) ≈ 0.15: below 0.3, so the lowest band.
    assert body["confidence"] == {"label": "Very unsure", "right_in_10": 2}
    assert (body["model_version"], body["encoder_folds"]) == ("7", 5)
    assert body["seconds"] >= 0


def test_word_limits_are_inclusive():
    # Review Focus 2: both edges are allowed, one word past either edge is not.
    client = ready_client(per_minute=100)
    assert client.post("/predict", json={"text": words(20)}).status_code == 200
    assert client.post("/predict", json={"text": words(1000)}).status_code == 200
    for n in (19, 1001):
        response = client.post("/predict", json={"text": words(n)})
        assert response.status_code == 422 and str(n) in response.json()["detail"]
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_text_without_words():
    # Review Focus 1: markup alone flattens to nothing, so it's too short, not a crash.
    client = ready_client()
    for text in ("**" * 40, "```\n" * 30):
        assert client.post("/predict", json={"text": text}).status_code == 422


def test_request_size_cap():
    client = ready_client()
    response = client.post(
        "/predict",
        content='{"text": "' + "a" * 32_000 + '"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_rate_limit_per_visitor():
    client = ready_client()
    first = {"X-Forwarded-For": "203.0.113.1"}
    for _ in range(10):
        assert client.post("/predict", json={"text": words(30)}, headers=first).status_code == 200
    assert client.post("/predict", json={"text": words(30)}, headers=first).status_code == 429
    other = {"X-Forwarded-For": "203.0.113.2, 10.0.0.1"}  # the first address is the visitor
    assert client.post("/predict", json={"text": words(30)}, headers=other).status_code == 200


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(per_minute=10, clock=lambda: now[0])
    assert all(limiter.allow("a") for _ in range(10))
    now[0] = 59.0
    assert not limiter.allow("a")
    now[0] = 61.0  # the 10 requests made at second 0 have left the 60-second window
    assert limiter.allow("a")


def test_single_flight_turns_away_beyond_the_line():
    # Review Focus 3: one prediction runs, 4 wait their turn, the 5th is turned away,
    # and every request that got in line is answered.
    async def scenario():
        gate = SingleFlight(max_waiting=4)
        release = asyncio.Event()
        done = []

        async def holder():
            async with gate:
                await release.wait()
                done.append("holder")

        async def waiter(i):
            async with gate:
                done.append(i)

        holding = asyncio.create_task(holder())
        await asyncio.sleep(0)
        waiters = [asyncio.create_task(waiter(i)) for i in range(4)]
        await asyncio.sleep(0)
        with pytest.raises(Busy):
            async with gate:
                pass
        release.set()
        await asyncio.gather(holding, *waiters)
        return done

    assert asyncio.run(scenario()) == ["holder", 0, 1, 2, 3]


def test_health_and_predict_while_loading_and_after_failure():
    # Review Focus 4: before the model is ready, or if it can't load, the app says so
    # plainly instead of failing with a traceback.
    release = threading.Event()

    def slow_load():
        release.wait(5)
        return FakePredictor()

    with TestClient(create_app(slow_load)) as client:
        assert (client.get("/health").status_code, client.get("/health").json()) == (
            503,
            {"status": "loading"},
        )
        loading = client.post("/predict", json={"text": words(30)})
        assert loading.status_code == 503 and "loading" in loading.json()["detail"]
        release.set()
        for _ in range(200):
            if client.get("/health").status_code == 200:
                break
            time.sleep(0.01)
        assert client.post("/predict", json={"text": words(30)}).status_code == 200

    def broken_load():
        raise RuntimeError("the registry is unreachable")

    with TestClient(create_app(broken_load)) as client:
        for _ in range(200):
            if client.get("/health").status_code != 503:
                break
            time.sleep(0.01)
        health = client.get("/health")
        assert health.status_code == 500 and health.json()["status"] == "failed"
        failed = client.post("/predict", json={"text": words(30)})
        assert failed.status_code == 503 and "failed" in failed.json()["detail"]


def test_version_reports_package_commit_and_model(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    monkeypatch.setenv("SPACE_ID", "someone/name-that-llm")
    body = ready_client().get("/version").json()
    assert (body["package"], body["commit"], body["space_id"]) == (
        __version__,
        "abc1234",
        "someone/name-that-llm",
    )
    assert (body["model_version"], body["encoder_folds"]) == ("7", 5)


def test_predict_never_logs_the_text(caplog):
    caplog.set_level(logging.DEBUG)
    secret = "zebracorn"  # a word that appears nowhere else
    client = ready_client()
    assert client.post("/predict", json={"text": f"{secret} " + words(30)}).status_code == 200
    assert not any(secret in record.getMessage() for record in caplog.records)
    assert caplog.records  # the request WAS logged, just without its text


def test_api_never_imports_torch_or_mlflow():
    code = (
        "import sys, name_that_llm.api; "
        "assert not {'torch', 'mlflow', 'transformers'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
