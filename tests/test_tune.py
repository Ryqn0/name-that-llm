import numpy as np
import pytest
import yaml

from name_that_llm.config import BlendConfig, load_config

optuna = pytest.importorskip("optuna")  # optional extra "tune": CI installs it via --all-extras

from name_that_llm.tune import best_blend_config, run_study  # noqa: E402


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
