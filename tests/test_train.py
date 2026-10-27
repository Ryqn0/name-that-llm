import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import name_that_llm.train as train_module
from name_that_llm.config import load_config
from name_that_llm.cv import make_folds
from name_that_llm.data import LABELS
from name_that_llm.metrics import score
from name_that_llm.train import main, run_cv, run_cv_features, run_cv_pool


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


def test_evaluate_assignment_reports_all_four_numbers(fake_train):
    from name_that_llm.train import evaluate_assignment

    report = evaluate_assignment(run_cv(_cfg("mnb"), fake_train), fake_train)
    assert set(report) == {"plain", "assigned", "oracle", "grouping_ari"}
    assert (
        report["oracle"] >= report["plain"] - 1e-9
    )  # assignment never hurts with true groups here


def _signal_features(df, noise=0.2, seed=0):
    """Fake dense features: the label as one-hot plus noise, like an informative layer.

    noise 0.2 keeps the signal clear: with 18 classes and noise 0.5, even reading the
    features perfectly is right only ~56% of the time, so a score threshold would test luck.
    """
    rng = np.random.default_rng(seed)
    onehot = (df["model"].to_numpy()[:, None] == np.array(LABELS)).astype(float)
    return onehot + rng.normal(0, noise, onehot.shape)


def test_run_cv_features_scores_informative_features(fake_train):
    result = run_cv_features(_signal_features(fake_train), fake_train, n_splits=3)
    assert result.oof.shape == (len(fake_train), 18) and result.score > 0.8


def test_run_cv_features_keeps_columns_aligned_when_a_class_is_missing(fake_train):
    df = fake_train[(fake_train["model"] != "kimi-k3") | (fake_train["prompt_id"] == "prompt0")]
    df = df.reset_index(drop=True)
    result = run_cv_features(_signal_features(df), df, n_splits=3)
    in_prompt0 = (df["prompt_id"] == "prompt0").to_numpy()
    assert np.allclose(result.oof[in_prompt0, LABELS.index("kimi-k3")], 0)


def test_text_paths_never_import_torch_or_optuna():
    code = (
        "import sys, name_that_llm.submit, name_that_llm.train, name_that_llm.blend, "
        "name_that_llm.encoder, name_that_llm.champion, name_that_llm.balance, "
        "name_that_llm.pool, name_that_llm.llm; "
        "assert not {'torch', 'optuna', 'transformers', 'name_that_llm.tune'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_run_cv_pool_without_pool_matches_run_cv(fake_train):
    cfg = _cfg("lr")
    np.testing.assert_allclose(run_cv_pool(cfg, fake_train).oof, run_cv(cfg, fake_train).oof)


def test_run_cv_pool_trains_only_on_allowed_pool_rows(fake_train, monkeypatch):
    # P7's no-leak rule: a fold never trains on pool passages about its validation prompts.
    seen = []  # the texts each fold's model was fit on

    class Spy:  # a stand-in model: records its training texts, predicts uniform
        def fit(self, X, y):
            seen.append(set(X))
            self.classes_ = np.array(LABELS)
            return self

        def predict_proba(self, X):
            return np.full((len(X), 18), 1 / 18)

    monkeypatch.setattr(train_module, "build_model", lambda cfg: Spy())
    pool = pd.DataFrame(
        {
            "text": [f"pooltag{p}" for p in range(6)],
            "model": LABELS[0],
            "source": "s",
            "prompt_id": [f"prompt{p}" for p in range(6)],
        }
    )
    cfg = _cfg("lr")
    run_cv_pool(cfg, fake_train, pool)
    folds = make_folds(fake_train, cfg.n_splits, cfg.seed).to_numpy()
    assert len(seen) == cfg.n_splits
    for k, texts in enumerate(seen):
        held_out = set(fake_train["prompt_id"][folds == k])
        assert not {f"pooltag{p.removeprefix('prompt')}" for p in held_out} & texts
        # and the other prompts' pool rows DO train it
        assert {f"pooltag{p}" for p in range(6) if f"prompt{p}" not in held_out} <= texts
