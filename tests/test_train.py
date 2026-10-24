import subprocess
import sys

import numpy as np
import pytest

from name_that_llm.config import load_config
from name_that_llm.data import LABELS
from name_that_llm.metrics import score
from name_that_llm.train import main, run_cv


def _cfg(name):
    return load_config(name).model_copy(update={"min_df": 1, "n_splits": 3})


def test_run_cv_returns_oof_for_every_row(fake_train):
    result = run_cv(_cfg("lr"), fake_train)
    assert result.oof.shape == (len(fake_train), 18)
    assert np.allclose(result.oof.sum(axis=1), 1.0)
    assert len(result.fold_scores) == 3
    assert result.score == pytest.approx(score(fake_train["model"], result.pred))


def test_run_cv_keeps_columns_aligned_when_a_class_is_missing(fake_train):
    # keep kimi-k3 only in prompt0: the fold that validates prompt0 never trains on kimi-k3
    df = fake_train[(fake_train["model"] != "kimi-k3") | (fake_train["prompt_id"] == "prompt0")]
    df = df.reset_index(drop=True)
    result = run_cv(_cfg("mnb"), df)
    in_prompt0 = (df["prompt_id"] == "prompt0").to_numpy()
    assert np.allclose(result.oof[in_prompt0, LABELS.index("kimi-k3")], 0.0)
    assert np.allclose(result.oof.sum(axis=1), 1.0)


def test_main_runs_without_tracking(fake_data_dir, tmp_path, capsys):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text("name: tiny\nmodel: mnb\nmin_df: 1\nn_splits: 3\n")
    assert main(["--config", str(cfg), "--data-dir", str(fake_data_dir)]) == 0
    assert "balanced accuracy" in capsys.readouterr().out


def test_train_never_imports_wandb():
    code = "import sys, name_that_llm.train; assert 'wandb' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
