"""The web app (P4): FastAPI serving the champion behind one page.

Routes:
  GET  /         the page (static/index.html)
  POST /predict  {"text": ...} -> all 18 models ranked, with company, probability, and the
                 "How sure?" band; 413 too large, 422 not 20-1,000 words, 429 too many
                 requests, 503 model loading or busy
  GET  /health   200 ready, 503 loading, 500 failed to load (CD's deploy check reads it)
  GET  /version  package, commit and the loaded model's facts

Why the model loads in a background thread: loading the champion takes minutes (Qwen's
weights, 5 encoders). Hugging Face waits for the app's port, so the app answers at once,
says "loading" on /health, and starts predicting when the model is ready.

Why one prediction at a time (SingleFlight): the free Space has 2 CPU cores and the
champion takes about 8 GB. Two predictions in parallel would share the same 2 cores, so
neither would finish sooner, and memory would spike. Requests queue instead, up to a
short line; beyond it, a polite 503 rather than a 2-minute wait.

Privacy: the pasted text is never stored and never logged. Logs keep only its word count,
the time taken and the prediction (tested in test_api.py).

This module needs neither torch nor mlflow (only serve.py wires the real model in), so its
tests run anywhere with a fake predictor.
"""

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from importlib import resources
from typing import Protocol

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool

from name_that_llm import __version__
from name_that_llm.champion import describe_confidence
from name_that_llm.data import LAB_OF, LABELS, flatten_text
from name_that_llm.labels import LAB_NAMES  # re-exported: tests import it from here

log = logging.getLogger("name_that_llm.api")

MIN_WORDS, MAX_WORDS = 20, 1000  # counted after flattening, both ends allowed


class Predictor(Protocol):
    """What the app needs from a model: probabilities, and facts to show about it."""

    info: dict  # model_version, encoder_folds, bands (list[Band]), ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


class PredictRequest(BaseModel):
    text: str


class Busy(Exception):
    """Raised when the line of waiting requests is full."""


class SingleFlight:
    """Let one request through at a time, with at most `max_waiting` waiting their turn."""

    def __init__(self, max_waiting: int):
        self.max_waiting = max_waiting
        self._lock = asyncio.Lock()
        self._waiting = 0

    async def __aenter__(self):
        if self._lock.locked() and self._waiting >= self.max_waiting:
            raise Busy
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        return self

    async def __aexit__(self, *exc_info):
        self._lock.release()


class RateLimiter:
    """At most `per_minute` requests per visitor over any sliding 60-second window."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute = per_minute
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self.clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True


def build_commit() -> str:
    """The git commit this build came from.

    The deploy writes it into the package as BUILD_COMMIT, so it's baked into the Docker
    image. That way /version can't claim a new commit while an old image still runs.
    Locally, GIT_COMMIT can stand in; otherwise "unknown".
    """
    try:
        commit = resources.files("name_that_llm").joinpath("BUILD_COMMIT").read_text().strip()
    except OSError:
        commit = ""
    return commit or os.environ.get("GIT_COMMIT", "unknown")


def _visitor(request: Request) -> str:
    """Who is asking, for the rate limit. Behind Hugging Face's proxy, the visitor's own
    address is the first one in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def create_app(
    load_predictor: Callable[[], Predictor],
    *,
    max_waiting: int = 4,
    per_minute: int = 10,
    max_body_bytes: int = 32_000,
) -> FastAPI:
    """The app, with `load_predictor` run in a background thread at startup."""
    state: dict = {"status": "loading", "predictor": None, "error": None}

    def load() -> None:
        try:
            state["predictor"] = load_predictor()
            state["status"] = "ok"
        except Exception as exc:  # report it on /health instead of crashing the app
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "failed"
            log.exception("the model failed to load")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=load, name="load-model", daemon=True).start()
        yield

    app = FastAPI(title="Name That LLM", version=__version__, lifespan=lifespan)
    gate, limiter = SingleFlight(max_waiting), RateLimiter(per_minute)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return (resources.files("name_that_llm") / "static" / "index.html").read_text("utf-8")

    @app.get("/health")
    def health() -> JSONResponse:
        body = {"status": state["status"]}
        if state["error"]:
            body["error"] = state["error"]
        return JSONResponse(
            body, status_code={"ok": 200, "loading": 503, "failed": 500}[state["status"]]
        )

    @app.get("/version")
    def version() -> dict:
        info = {
            "package": __version__,
            "commit": build_commit(),
            "space_id": os.environ.get("SPACE_ID"),
        }
        if state["predictor"] is not None:  # the bands are for the page, not for /version
            info |= {k: v for k, v in state["predictor"].info.items() if k != "bands"}
        return info

    @app.post("/predict")
    async def predict(request: Request) -> JSONResponse:
        body = await request.body()
        if len(body) > max_body_bytes:
            return _error(413, "That text is too long: paste at most about 1,000 words.")
        if not limiter.allow(_visitor(request)):
            return _error(429, "Too many requests: wait a minute, then try again.")
        try:
            text = PredictRequest.model_validate_json(body).text
        except ValidationError:
            return _error(422, 'Send JSON like {"text": "the passage"}.')
        n_words = len(flatten_text(text).split())
        if not MIN_WORDS <= n_words <= MAX_WORDS:
            return _error(422, f"Paste between 20 and 1,000 words; this text has {n_words}.")
        if state["status"] == "loading":
            return _error(503, "The model is still loading: try again in a minute.")
        if state["status"] == "failed":
            return _error(503, "The model failed to load: the app needs a restart.")
        try:
            async with gate:
                start = time.monotonic()
                proba = await run_in_threadpool(state["predictor"].predict_proba, [text])
                seconds = time.monotonic() - start
        except Busy:
            return _error(503, "Busy: other passages are being read. Try again in a moment.")
        p = proba[0]
        order = np.argsort(-p, kind="stable")
        info = state["predictor"].info
        band = describe_confidence(float(p[order[0]]), info["bands"])
        # Never the text itself: only its length, the time and the outcome.
        log.info("predict: %d words, %.1f s, top %s", n_words, seconds, LABELS[order[0]])
        return JSONResponse(
            {
                "ranking": [
                    {
                        "model": LABELS[i],
                        "lab": LAB_NAMES[LAB_OF[LABELS[i]]],
                        "probability": float(p[i]),
                    }
                    for i in order
                ],
                "confidence": {"label": band.label, "right_in_10": round(band.accuracy * 10)},
                "model_version": info.get("model_version"),
                "encoder_folds": info.get("encoder_folds"),
                "seconds": round(seconds, 1),
            }
        )

    return app
