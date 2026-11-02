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


def _llm_ensemble(
    fake_train,
    fake_data_dir,
    p_llm,
    alpha=0.0,
    weights=(0.01, 0.01, 0.01, 0.97),
    p_encoder=None,
    log_blend=False,
):
    """predict_submission_ensemble with a 4-member config (P7 Task 8)."""
    test = load_test(fake_data_dir)
    rng = np.random.default_rng(0)
    # Drawn in this order, so every test without its own p_encoder keeps the same inputs.
    lm_train, lm_test = rng.normal(size=(len(fake_train), 8)), rng.normal(size=(len(test), 8))
    if p_encoder is None:
        p_encoder = rng.dirichlet(np.ones(18), size=len(test))
    cfg = EnsembleConfig(
        name="e", weights=weights, llm="llm", sinkhorn_alpha=alpha, log_blend=log_blend
    )
    return predict_submission_ensemble(
        _cfg(),
        BlendConfig(name="b", lm_variation="x", layer=0, weight=0.5),
        cfg,
        fake_train,
        test,
        lm_train,
        lm_test,
        p_encoder,
        p_llm=p_llm,
    )


def test_ensemble_submission_with_an_llm_member(fake_train, fake_data_dir):
    # The LLM joins as a fourth member: a one-hot p_llm holding 0.97 of the weight decides
    # every row.
    test = load_test(fake_data_dir)
    picks = [LABELS[i] for i in (4, 9, 13)]
    p_llm = np.zeros((len(test), 18))
    p_llm[np.arange(len(test)), [LABELS.index(p) for p in picks]] = 1.0
    sub = _llm_ensemble(fake_train, fake_data_dir, p_llm)
    assert list(sub["id"]) == list(test["id"]) and sub["model"].tolist() == picks


def test_ensemble_submission_applies_sinkhorn(fake_train, fake_data_dir):
    # When the LLM favors one label everywhere, full balancing spreads the predictions.
    test = load_test(fake_data_dir)
    p_llm = np.full((len(test), 18), 0.01)
    p_llm[:, 0] = 1.0
    p_llm /= p_llm.sum(axis=1, keepdims=True)
    plain = _llm_ensemble(fake_train, fake_data_dir, p_llm, alpha=0.0)
    balanced = _llm_ensemble(fake_train, fake_data_dir, p_llm, alpha=1.0)
    assert balanced["model"].nunique() > plain["model"].nunique()


def test_ensemble_submission_follows_the_log_blend(fake_train, fake_data_dir):
    # The champion blends in log space (I-015), with weights searched for that form, so the
    # submission must too: the probability blend with those weights is another model, one
    # the gate measured 0.014 lower. Here the two forms disagree. The encoder is sure of
    # label 0; the LLM all but rules it out, and leans to label 1. The plain mean keeps
    # label 0 (0.455 against 0.275); the geometric mean lets the LLM veto it (0.095 against
    # 0.158). P7's final review: nothing caught a submission that dropped the log form.
    test = load_test(fake_data_dir)
    p_encoder = np.full((len(test), 18), 0.05 / 16)
    p_encoder[:, :2] = [0.90, 0.05]
    p_llm = np.full((len(test), 18), 0.49 / 16)
    p_llm[:, :2] = [0.01, 0.50]
    both = dict(weights=(0.0, 0.0, 0.5, 0.5), p_encoder=p_encoder)  # the encoder and the LLM
    plain = _llm_ensemble(fake_train, fake_data_dir, p_llm, **both)
    log = _llm_ensemble(fake_train, fake_data_dir, p_llm, log_blend=True, **both)
    assert set(plain["model"]) == {LABELS[0]}
    assert set(log["model"]) == {LABELS[1]}


def test_llm_probabilities_of_the_wrong_shape_are_refused(fake_train, fake_data_dir):
    test = load_test(fake_data_dir)
    with pytest.raises(ValueError, match="llm"):
        _llm_ensemble(fake_train, fake_data_dir, np.full((len(test), 17), 1 / 17))
    with pytest.raises(ValueError, match="llm"):
        _llm_ensemble(fake_train, fake_data_dir, None)  # the config has an llm member
