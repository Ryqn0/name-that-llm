"""Tests for packaging the champion as an MLflow model and loading it from the registry (P4).

A local SQLite store stands in for DagsHub's MLflow server: the same API, no network. The
champion is built from tiny random models, so these run in seconds on CPU.
"""

import shutil

import joblib
import numpy as np
import pytest

from name_that_llm.config import load_config
from name_that_llm.models import build_dense_model, build_model

mlflow = pytest.importorskip("mlflow")

from name_that_llm.registry import (  # noqa: E402  (after the importorskip on purpose)
    load_champion,
    load_registered,
    log_champion,
    save_champion_artifacts,
    set_alias,
)

TEXTS = ["marker3 words about subject 1", "marker5 marker5 — style5"]
BANDS = [
    {"label": "Fairly sure", "min_top": 0.5, "accuracy": 0.8},
    {"label": "Unsure", "min_top": 0.2, "accuracy": 0.5},
    {"label": "Very unsure", "min_top": 0.0, "accuracy": 0.3},
]


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """A throwaway MLflow tracking server and model registry, in a SQLite file."""
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment("tests", artifact_location=(tmp_path / "runs").as_uri())
    mlflow.set_experiment(experiment_id=experiment)
    yield uri
    mlflow.set_tracking_uri(previous)


@pytest.fixture
def saved(fake_train, tiny_qwen, tiny_encoder, tiny_modernbert, tmp_path):
    """A tiny champion saved the way the registration saves the real one.

    Returns (folder, expected): `expected` holds the predictions of an IN-MEMORY twin built
    from the same parts, so the packaging is checked against code that never touched it.
    """
    from name_that_llm.champion import Champion
    from name_that_llm.encoder import load_encoder
    from name_that_llm.lm import load_lm_layers, pooled_hidden_states, select_layers

    texts, y = fake_train["text"].tolist(), fake_train["model"]
    lm, tokenizer = load_lm_layers(tiny_qwen, n_layers=10)
    X = select_layers(pooled_hidden_states(texts, lm, tokenizer, layers=range(6, 10)), 0, 4)
    text_model = build_model(load_config("lr")).fit(texts, y)
    lm_head = build_dense_model(0.1).fit(X, y)
    twin = Champion(
        text_model=text_model,
        lm_head=lm_head,
        lm_train_mean=X.mean(axis=0),
        lm=(lm, tokenizer),
        encoders=[load_encoder(tiny_encoder), load_encoder(tiny_modernbert)],
        weights=(0.7, 0.15, 0.15),
        lm_max_length=48,  # the tiny models have 64 positions
        encoder_max_length=48,
    )
    meta = {
        "weights": [0.7, 0.15, 0.15],
        "bands": BANDS,
        "lm": {
            "source": str(tiny_qwen),
            "revision": None,
            "layer": 6,
            "window": 4,
            "max_length": 48,
        },
        "encoder_max_length": 48,
        "package_version": "test",
        "git_commit": "test",
    }
    folder = save_champion_artifacts(
        tmp_path / "champion",
        text_model=text_model,
        lm_head=lm_head,
        lm_train_mean=X.mean(axis=0),
        encoder_dirs=[tiny_encoder, tiny_modernbert],
        meta=meta,
    )
    return folder, twin.predict_proba(TEXTS)


def test_champion_round_trips_through_the_registry(saved, mlflow_store):
    folder, expected = saved
    version = log_champion(folder)
    set_alias("name-that-llm", "champion", version)
    predictor, info = load_registered()
    assert (version, info["model_version"], info["alias"], info["encoder_folds"]) == (
        "1",
        "1",
        "champion",
        2,
    )
    assert [band.label for band in info["bands"]] == ["Fairly sure", "Unsure", "Very unsure"]
    assert predictor.info == info  # the app reads the same facts from the predictor
    np.testing.assert_allclose(predictor.predict_proba(TEXTS), expected, atol=1e-6)


def test_load_registered_pins_the_alias_version(saved, mlflow_store, fake_train, tmp_path):
    # Review Focus 5: a running app keeps the version it loaded, even after the alias moves;
    # a newly started one picks up the new version.
    folder, _ = saved
    set_alias("name-that-llm", "champion", log_champion(folder))
    running, running_info = load_registered()
    before = running.predict_proba(TEXTS)

    # Version 2 differs in one part: a much more regularized text model.
    v2 = shutil.copytree(folder, tmp_path / "v2")
    other = build_model(load_config("lr").model_copy(update={"C": 0.01}))
    joblib.dump(
        other.fit(fake_train["text"].tolist(), fake_train["model"]), v2 / "text_model.joblib"
    )
    set_alias("name-that-llm", "champion", log_champion(v2))

    assert running_info["model_version"] == "1"
    np.testing.assert_allclose(running.predict_proba(TEXTS), before)
    restarted, restarted_info = load_registered()
    assert restarted_info["model_version"] == "2"
    assert not np.allclose(restarted.predict_proba(TEXTS), before)


def test_load_champion_can_use_fewer_encoder_folds(saved):
    folder, _ = saved
    champion, meta = load_champion(folder, n_encoder_folds=1)
    assert len(champion.encoders) == 1 and meta["encoder_folds"] == 1


def test_registered_model_honors_the_encoder_folds_setting(saved, mlflow_store, monkeypatch):
    # The design's fallback (§10): if the Space is too slow, ENCODER_FOLDS=1 serves one
    # fold model, and the app must report it.
    folder, _ = saved
    set_alias("name-that-llm", "champion", log_champion(folder))
    monkeypatch.setenv("ENCODER_FOLDS", "1")
    _, info = load_registered()
    assert info["encoder_folds"] == 1
