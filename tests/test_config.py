import pytest
from pydantic import ValidationError

from name_that_llm.config import (
    BlendConfig,
    EncoderConfig,
    EnsembleConfig,
    ExperimentConfig,
    load_config,
)


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


def test_ensemble_config_validates_weights():
    cfg = EnsembleConfig(name="e", weights=(0.5, 0.2, 0.3))
    assert cfg.blend == "lm_blend" and cfg.encoder == "encoder"
    with pytest.raises(ValidationError):  # a negative weight
        EnsembleConfig(name="e", weights=(0.5, -0.1, 0.6))
    with pytest.raises(ValidationError):  # one weight per model: text, LM head, encoder
        EnsembleConfig(name="e", weights=(0.5, 0.5))


def test_shipped_encoder_configs_load():
    # The GPU notebooks download base_model from the Hugging Face Hub: a typo in the id
    # would only surface on Kaggle, after the GPU session has started.
    for name, base in [
        ("deberta_v3_base", "microsoft/deberta-v3-base"),
        ("modernbert_base", "answerdotai/ModernBERT-base"),
    ]:
        cfg = load_config(name, model_cls=EncoderConfig)
        assert cfg.name == name and cfg.base_model == base and cfg.seed == 42


def test_shipped_encoder_config_loads():
    # The tuned encoder (P5 Task 5). The final notebook trains it by NAME from the
    # installed package, so a missing or malformed file would only fail on Kaggle's GPU.
    cfg = load_config("encoder", model_cls=EncoderConfig)
    assert cfg.name == "encoder" and cfg.base_model == "answerdotai/ModernBERT-base"
    assert cfg.seed == 42 and 2 <= cfg.epochs <= 5 and 5e-6 <= cfg.lr <= 1e-4


def test_shipped_ensemble_config_loads():
    # The ensemble that notebook 09 promoted (P5 Task 6). The offline submission loads it
    # by NAME, then the blend and the encoder it names: a typo in any of them would only
    # surface during the 9-hour scoring run.
    cfg = load_config("ensemble", model_cls=EnsembleConfig)
    assert cfg.name == "ensemble"
    assert load_config(cfg.blend, model_cls=BlendConfig).name == cfg.blend
    assert load_config(cfg.encoder, model_cls=EncoderConfig).name == cfg.encoder


def test_shipped_lm_blend_config_loads_with_its_text_config():
    # The offline submission loads both by NAME from the installed wheel, so a typo in
    # either file would only surface during the 9-hour scoring run.
    cfg = load_config("lm_blend", model_cls=BlendConfig)
    assert cfg.name == "lm_blend" and cfg.lm_variation == "4b-base"
    assert load_config(cfg.text_config).name == cfg.text_config
