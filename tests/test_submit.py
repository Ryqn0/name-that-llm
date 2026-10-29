import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from name_that_llm.config import BlendConfig, EnsembleConfig, load_config
from name_that_llm.data import LABELS, load_test
from name_that_llm.submit import (
    predict_submission,
    predict_submission_blend,
    predict_submission_ensemble,
)


def _cfg():
    return load_config("lr").model_copy(update={"min_df": 1})


def test_predict_submission_is_valid_and_keeps_test_order(fake_train, fake_data_dir):
    test = load_test(fake_data_dir)
    sub = predict_submission(_cfg(), fake_train, test)
    assert sub["id"].tolist() == ["000001", "a1b2c3", "999999"]  # leading zeros kept
    assert sub["model"].isin(LABELS).all()


def test_predict_submission_survives_odd_texts(fake_train):
    odd = pd.DataFrame(
        {
            "id": ["empty", "space", "long", "emoji", "ja"],
            "text": ["", " ", "word " * 5000, "\U0001f642" * 3, "日本語のテキストです"],
        }
    )
    sub = predict_submission(_cfg(), fake_train, odd)
    assert len(sub) == 5 and sub["model"].isin(LABELS).all()


def test_submit_and_train_never_import_wandb():
    code = (
        "import sys, name_that_llm.submit, name_that_llm.train; assert 'wandb' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_predict_submission_with_assign_survives_three_rows(fake_train, fake_data_dir):
    test = load_test(fake_data_dir)
    sub = predict_submission(_cfg(), fake_train, test, assign=True)
    assert sub["id"].tolist() == ["000001", "a1b2c3", "999999"]
    assert sub["model"].isin(LABELS).all()


def test_predict_submission_assign_gives_one_of_each_per_topic(fake_train):
    # 18 fresh passages on ONE topic: assignment must use every label exactly once
    test = fake_train[fake_train["prompt_id"] == "prompt0"][["id", "text"]].reset_index(drop=True)
    sub = predict_submission(_cfg(), fake_train, test, assign=True)
    assert sorted(sub["model"]) == sorted(LABELS)


def test_predict_submission_blend_weight_zero_equals_text_only(fake_train, fake_data_dir):
    # weight 0 = the LM side is ignored, so the result must match the text-only submission
    test = load_test(fake_data_dir)
    bcfg = BlendConfig(name="b", lm_variation="x", layer=0, weight=0.0)
    rng = np.random.default_rng(0)
    sub = predict_submission_blend(
        _cfg(),
        bcfg,
        fake_train,
        test,
        rng.normal(size=(len(fake_train), 8)),
        rng.normal(size=(3, 8)),
    )
    assert sub["model"].tolist() == predict_submission(_cfg(), fake_train, test)["model"].tolist()


def test_predict_submission_blend_rejects_misaligned_features(fake_train, fake_data_dir):
    bcfg = BlendConfig(name="b", lm_variation="x", layer=0, weight=0.5)
    with pytest.raises(ValueError):
        predict_submission_blend(
            _cfg(), bcfg, fake_train, load_test(fake_data_dir), np.zeros((5, 8)), np.zeros((3, 8))
        )


def _ensemble(fake_train, fake_data_dir, weights, p_encoder=None):
    """predict_submission_ensemble on the fake data, with random LM features."""
    test = load_test(fake_data_dir)
    rng = np.random.default_rng(0)
    if p_encoder is None:
        p_encoder = rng.dirichlet(np.ones(18), size=len(test))
    sub = predict_submission_ensemble(
        _cfg(),
        BlendConfig(name="b", lm_variation="x", layer=0, weight=0.5),
        EnsembleConfig(name="e", weights=weights),
        fake_train,
        test,
        rng.normal(size=(len(fake_train), 8)),
        rng.normal(size=(len(test), 8)),
        p_encoder,
    )
    return sub, test, p_encoder


def test_predict_submission_ensemble_text_weight_only_equals_text_only(fake_train, fake_data_dir):
    sub, test, _ = _ensemble(fake_train, fake_data_dir, (1.0, 0.0, 0.0))
    assert sub["model"].tolist() == predict_submission(_cfg(), fake_train, test)["model"].tolist()


def test_predict_submission_ensemble_encoder_weight_only_follows_the_encoder(
    fake_train, fake_data_dir
):
    sub, _, p_encoder = _ensemble(fake_train, fake_data_dir, (0.0, 0.0, 1.0))
    assert sub["model"].tolist() == [LABELS[i] for i in p_encoder.argmax(axis=1)]


def test_predict_submission_ensemble_rejects_misaligned_encoder_probabilities(
    fake_train, fake_data_dir
):
    with pytest.raises(ValueError):  # 2 rows of encoder probabilities for 3 test rows
        _ensemble(fake_train, fake_data_dir, (0.4, 0.3, 0.3), p_encoder=np.full((2, 18), 1 / 18))


def _pool_ensemble(fake_train, test, text_pool, pool):
    """predict_submission_ensemble with the text model's weight only, and `pool`."""
    rng = np.random.default_rng(0)
    return predict_submission_ensemble(
        _cfg(),
        BlendConfig(name="b", lm_variation="x", layer=0, weight=0.5),
        EnsembleConfig(name="e", weights=(1.0, 0.0, 0.0), text_pool=text_pool),
        fake_train,
        test,
        rng.normal(size=(len(fake_train), 8)),
        rng.normal(size=(len(test), 8)),
        rng.dirichlet(np.ones(18), size=len(test)),
        pool=pool,
    )


def test_predict_submission_ensemble_trains_lr_on_the_pool(fake_train):
    # I-029: with text_pool, lr also learns from the pool. A word that only the pool uses,
    # always under one label, must pull a test passage made of it to that label.
    text = "zebracorn quillfeather passage " * 6
    pool = pd.DataFrame({"text": [text] * 40, "model": LABELS[3], "source": "s", "prompt_id": None})
    test = pd.DataFrame({"id": ["a", "b"], "text": [text, text]})
    sub = _pool_ensemble(fake_train, test, text_pool=40, pool=pool)
    assert set(sub["model"]) == {LABELS[3]}


def test_predict_submission_ensemble_needs_its_pool(fake_train):
    # A config that trains lr on the pool, called without one, must fail loudly: silently
    # dropping the pool would submit a weaker model than the one that was measured.
    test = pd.DataFrame({"id": ["a"], "text": ["some passage"]})
    with pytest.raises(ValueError, match="pass the pool"):
        _pool_ensemble(fake_train, test, text_pool=40, pool=None)
