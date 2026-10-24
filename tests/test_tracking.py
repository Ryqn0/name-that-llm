import pytest

from name_that_llm.config import load_config
from name_that_llm.tracking import log_cv_run
from name_that_llm.train import run_cv


def test_log_cv_run_works_offline(fake_train, tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    # Offline mode: wandb writes the run to a local folder instead of uploading it, so
    # this test needs no account, no key and no network (CI included).
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    cfg = load_config("mnb").model_copy(update={"min_df": 1, "n_splits": 3})
    run_id = log_cv_run(cfg, run_cv(cfg, fake_train), fake_train)
    assert isinstance(run_id, str) and run_id
