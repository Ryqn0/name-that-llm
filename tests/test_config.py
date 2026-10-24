import pytest
from pydantic import ValidationError

from name_that_llm.config import ExperimentConfig, load_config


def test_load_config_by_name_and_by_path(tmp_path):
    cfg = load_config("lr")
    assert cfg.model == "lr" and cfg.n_splits == 5 and cfg.seed == 42
    path = tmp_path / "x.yaml"
    path.write_text("name: x\nmodel: mnb\n")
    assert load_config(path).model == "mnb"


def test_config_rejects_unknown_model_and_typos():
    with pytest.raises(ValidationError):
        ExperimentConfig(name="bad", model="xgboost")
    with pytest.raises(ValidationError):
        ExperimentConfig(name="bad", model="lr", c=1.0)  # typo for C
