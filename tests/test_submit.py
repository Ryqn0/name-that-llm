import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from name_that_llm.config import BlendConfig, load_config
from name_that_llm.data import LABELS, load_test
from name_that_llm.submit import predict_submission, predict_submission_blend


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
