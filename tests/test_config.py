import pytest
from pydantic import ValidationError

from name_that_llm.config import BlendConfig, ExperimentConfig, load_config


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


def test_blend_config_round_trips_and_validates_weight(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text("name: b\nlm_variation: 1.7b-base\nlayer: 6\nweight: 0.5\n")
    cfg = load_config(path, model_cls=BlendConfig)
    assert cfg.layer == 6 and cfg.text_config == "lr" and cfg.seed == 42
    with pytest.raises(ValidationError):
        BlendConfig(name="x", lm_variation="1.7b-base", layer=1, weight=1.5)
