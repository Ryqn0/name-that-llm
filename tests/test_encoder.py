"""Tests for the encoder fine-tuning library (P5), on tiny RANDOM models and fake data.

Tests that need a model skip where torch or transformers isn't installed: the tiny-encoder
fixtures importorskip them. The CI job `torch` installs CPU-only torch and runs them all.
"""

import re

import numpy as np
import pytest

from name_that_llm.config import EncoderConfig
from name_that_llm.cv import make_folds
from name_that_llm.encoder import (
    find_fold_dirs,
    fit_predict_fold,
    load_encoder,
    predict_proba,
    predict_proba_folds,
    run_cv_encoder,
    train_encoder,
)
from name_that_llm.labels import LABELS


def _cfg(path, **overrides):
    # lr 3e-3 suits a tiny random model (the 2026-10-25 spike reached CV 0.86 with it);
    # real encoders train at ~2e-5. max_length 32 fits the fake passages.
    return EncoderConfig(name="t", base_model=str(path), max_length=32, lr=3e-3, **overrides)


def test_predict_proba_columns_follow_labels(tiny_encoder, fake_train):
    model, tokenizer = load_encoder(tiny_encoder)
    assert [model.config.id2label[i] for i in range(len(LABELS))] == LABELS
    proba = predict_proba(model, tokenizer, fake_train["text"].tolist()[:5], max_length=32)
    assert proba.shape == (5, 18) and np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_proba_refuses_another_label_order(tiny_encoder):
    # A model whose columns mean other labels must never be read as LABELS order.
    model, tokenizer = load_encoder(tiny_encoder)
    model.config.id2label = dict(enumerate(reversed(LABELS)))
    with pytest.raises(ValueError):
        predict_proba(model, tokenizer, ["marker1"], max_length=32)


def test_predict_proba_handles_empty_and_long_text(tiny_encoder):
    # The long text has 1,000 words but the model only 64 positions: without truncation at
    # max_length, the position embeddings would overflow.
    model, tokenizer = load_encoder(tiny_encoder)
    proba = predict_proba(model, tokenizer, ["", "marker1 " * 1000], max_length=32)
    assert proba.shape == (2, 18) and np.isfinite(proba).all()


def test_train_encoder_lowers_the_loss(tiny_encoder, fake_train):
    model, tokenizer = load_encoder(tiny_encoder)
    cfg = _cfg(tiny_encoder, epochs=5, warmup_ratio=0.0)
    losses = train_encoder(
        model, tokenizer, fake_train["text"].tolist(), fake_train["model"].to_numpy(), cfg
    )
    assert len(losses) == 5 and losses[-1] < losses[0]


def test_train_encoder_raises_on_non_finite_loss(tiny_encoder, fake_train):
    # Simulates a float16 overflow: the classifier's output becomes NaN.
    model, tokenizer = load_encoder(tiny_encoder)
    model.classifier.register_forward_hook(lambda module, args, out: out * float("nan"))
    with pytest.raises(FloatingPointError):
        train_encoder(
            model,
            tokenizer,
            fake_train["text"].tolist(),
            fake_train["model"].to_numpy(),
            _cfg(tiny_encoder, epochs=1),
        )


def test_fit_predict_fold_starts_fresh_each_time(tiny_encoder, fake_train):
    # If a call reused weights trained by an earlier call, the second result would differ:
    # each fold must start from the pretrained checkpoint, or it leaks earlier folds.
    cfg = _cfg(tiny_encoder, epochs=2, n_splits=3)
    texts, y = fake_train["text"].tolist(), fake_train["model"].to_numpy()
    folds = make_folds(fake_train, 3, cfg.seed).to_numpy()
    first, _ = fit_predict_fold(cfg, texts, y, folds, 0)
    again, _ = fit_predict_fold(cfg, texts, y, folds, 0)
    assert first.shape == ((folds == 0).sum(), 18)
    np.testing.assert_allclose(first, again, atol=1e-6)


def test_run_cv_encoder_learns_the_markers(tiny_encoder, fake_train):
    cfg = _cfg(tiny_encoder, epochs=20, warmup_ratio=0.0, n_splits=3, batch_size=16)
    result = run_cv_encoder(cfg, fake_train)
    assert result.oof.shape == (108, 18)
    assert (result.folds == make_folds(fake_train, 3, cfg.seed).to_numpy()).all()
    assert result.score > 0.5  # chance is 1/18 = 0.056


def test_saved_fold_models_round_trip(tiny_encoder, fake_train, tmp_path):
    # 20 epochs, so the fold models really learn. At 1-3 epochs a tiny random model still
    # predicts ~uniform, swapped fold models differ by only 1e-5, and this test couldn't
    # tell them apart. Measured at 20: a swap differs by >= 0.117, against atol 5e-3.
    cfg = _cfg(tiny_encoder, epochs=20, warmup_ratio=0.0, n_splits=3)
    result = run_cv_encoder(cfg, fake_train, save_dir=tmp_path / "out" / "encoder")
    dirs = find_fold_dirs(tmp_path / "out", "encoder", n_folds=3)
    assert [d.name for d in dirs] == ["fold0", "fold1", "fold2"]

    # Each saved model reproduces the out-of-fold predictions it made during CV. That
    # catches saving the wrong model (untrained, or another fold's) under a fold's name.
    texts = fake_train["text"].tolist()
    for k, d in enumerate(dirs):
        rows = np.flatnonzero(result.folds == k)
        reloaded = predict_proba(*load_encoder(d), [texts[i] for i in rows], max_length=32)
        np.testing.assert_allclose(reloaded, result.oof[rows], atol=5e-3)  # float16 weights

    # The offline average is the plain mean of the three models, built by hand here.
    p0, p1, p2 = (predict_proba(*load_encoder(d), texts[:4], max_length=32) for d in dirs)
    np.testing.assert_allclose(
        predict_proba_folds(dirs, texts[:4], max_length=32), (p0 + p1 + p2) / 3, atol=1e-5
    )


def test_find_fold_dirs_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape(str(tmp_path))):
        find_fold_dirs(tmp_path, "encoder", n_folds=3)
    for k in (0, 1):  # two of the three folds: averaging them silently would be wrong
        (tmp_path / "encoder" / f"fold{k}").mkdir(parents=True)
        (tmp_path / "encoder" / f"fold{k}" / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        find_fold_dirs(tmp_path, "encoder", n_folds=3)


def test_modernbert_trains_and_predicts(tiny_modernbert, fake_train):
    model, tokenizer = load_encoder(tiny_modernbert)
    losses = train_encoder(
        model,
        tokenizer,
        fake_train["text"].tolist(),
        fake_train["model"].to_numpy(),
        _cfg(tiny_modernbert, epochs=1),
    )
    assert len(losses) == 1
    assert predict_proba(model, tokenizer, ["marker1"], max_length=32).shape == (1, 18)
