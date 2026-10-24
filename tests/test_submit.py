import subprocess
import sys

import pandas as pd

from name_that_llm.config import load_config
from name_that_llm.data import LABELS, load_test
from name_that_llm.submit import predict_submission


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
