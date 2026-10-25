import numpy as np
import pytest
import yaml

from name_that_llm.config import BlendConfig, EncoderConfig, load_config

optuna = pytest.importorskip("optuna")  # optional extra "tune": CI installs it via --all-extras

from name_that_llm.tune import (  # noqa: E402
    best_blend_config,
    best_encoder_config,
    run_encoder_study,
    run_study,
)


def test_run_study_returns_a_valid_round_tripping_config(fake_train, tmp_path):
    rng = np.random.default_rng(0)
    feats = {
        "a": rng.normal(size=(3, len(fake_train), 8)),
        "b": rng.normal(size=(2, len(fake_train), 8)),
    }
    study = run_study(feats, fake_train, n_trials=3, seed=7)
    cfg = best_blend_config(study)
    assert cfg.lm_variation in feats
    assert cfg.layer + cfg.layer_window <= len(feats[cfg.lm_variation])
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    assert load_config(path, model_cls=BlendConfig) == cfg and cfg.seed == 42


def test_run_encoder_study_returns_a_valid_config(tiny_encoder, fake_train, tmp_path):
    # The tiny model needs torch (the fixture skips without it).
    base = EncoderConfig(name="t", base_model=str(tiny_encoder), max_length=32, n_splits=3)
    study = run_encoder_study(base, fake_train, n_trials=2, seed=7, tune_folds=(0,))
    assert len(study.trials) == 2
    # Trial 0 is always the base config (the warm start), so tuning can't end below it.
    assert study.trials[0].params == {
        "lr": base.lr,
        "epochs": base.epochs,
        "warmup_ratio": base.warmup_ratio,
    }
    cfg = best_encoder_config(study)
    assert cfg.name == "encoder" and cfg.seed == 42
    assert 5e-6 <= cfg.lr <= 1e-4 and 2 <= cfg.epochs <= 5
    path = tmp_path / "e.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    assert load_config(path, model_cls=EncoderConfig) == cfg
