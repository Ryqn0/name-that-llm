"""Smoke tests for the GPU notebooks (P7): their own code, run on CPU with a tiny model.

A GPU notebook runs for hours on Kaggle, so a typo found there costs hours. Each test
executes a notebook's code cells (minus the `%pip` lines) with its folders pointed at a
fake /kaggle/input: a tiny random Qwen, fake LM features and fake encoder predictions.
Like tests/test_llm.py, it needs torch, transformers and peft (CI's `torch` job).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def run_notebook(path: Path, env: dict[str, str], monkeypatch) -> None:
    """Execute the code cells of the notebook at `path`, in order, with `env` set."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    lines = [
        line
        for cell in cells
        if cell["cell_type"] == "code"
        for line in "".join(cell["source"]).splitlines()
        if not line.lstrip().startswith(("%", "!"))  # installs: the test env has everything
    ]
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    exec(compile("\n".join(lines), str(path), "exec"), {"__name__": "__main__"})


def fake_kaggle_input(root: Path, tiny_qwen: Path, train) -> None:
    """What notebook 10 finds under /kaggle/input: the base model, and two notebooks' outputs."""
    shutil.copytree(tiny_qwen, root / "qwen-3" / "transformers" / "8b-base" / "1")
    rng = np.random.default_rng(0)
    features = root / "name-that-llm-lm-features"
    features.mkdir(parents=True)
    np.save(features / "train_ids.npy", train["id"].to_numpy())
    # 12 layers, enough for lm_blend's layers 6 to 9.
    np.save(features / "features-4b-base.npy", rng.normal(size=(12, len(train), 8)))
    encoder = root / "name-that-llm-encoder-final"
    encoder.mkdir()
    p = rng.random((len(train), 18))
    np.save(encoder / "oof-encoder-seed7.npy", p / p.sum(axis=1, keepdims=True))


def fake_open_pool(root: Path) -> None:
    """The public pool dataset under /kaggle/input: 36 passages from two of the kept sources."""
    import pandas as pd

    from name_that_llm.labels import LABELS

    pool = root / "name-that-llm-open-pool"
    pool.mkdir()
    kept = ["drjcfmd/ntl-generated-v2", "ayhampar/ntl-pool-ft"]
    rows = [
        {
            "text": f"pooltopic{i} " + f"marker{i % 18} words about things here " * 7,
            "model": LABELS[i % 18],
            "source_dataset": kept[i % 2],
        }
        for i in range(36)
    ]
    pd.DataFrame(rows).to_parquet(pool / "ntl_open_pool.parquet")


def test_llm_first_notebook_runs(tiny_qwen, fake_data_dir, fake_train, tmp_path, monkeypatch):
    pytest.importorskip("peft")
    fake_kaggle_input(tmp_path / "input", tiny_qwen, fake_train)
    work = tmp_path / "working"
    work.mkdir()
    config = tmp_path / "llm.yaml"
    tiny = dict(name="llm", base_model="unused", load_in_4bit=False, n_layers=4, lora_r=4)
    tiny |= dict(lora_alpha=8, max_length=64, batch_size=8, lr=5e-3, epochs=2)
    config.write_text(yaml.safe_dump(tiny), encoding="utf-8")
    env = {
        "NTL_INPUT": str(tmp_path / "input"),
        "NTL_WORK": str(work),
        "NTL_LLM_CONFIG": str(config),
        "NTL_DATA_DIR": str(fake_data_dir),
    }
    run_notebook(NOTEBOOKS / "10-llm-first" / "llm_first.ipynb", env, monkeypatch)

    oof = np.load(work / "oof-llm-seed7.npy")
    assert oof.shape == (len(fake_train), 18)
    np.testing.assert_allclose(oof.sum(axis=1), 1.0, atol=1e-4)
    timing = json.loads((work / "timing.json").read_text())
    keys = {"seconds_per_passage", "model_load_seconds", "peak_memory_gb", "projected_cv_minutes"}
    assert set(timing) == keys  # a tiny model's projection rounds to 0 minutes
    scores = json.loads((work / "results.json").read_text())["scores"]
    assert {"lr", "LM head", "encoder", "ensemble", "LLM", "best 4-member blend"} <= set(scores)


def test_llm_explore_notebook_runs(tiny_qwen, fake_data_dir, fake_train, tmp_path, monkeypatch):
    pytest.importorskip("peft")
    import pandas as pd

    from name_that_llm.config import LLMConfig

    inp = tmp_path / "input"
    fake_kaggle_input(inp, tiny_qwen, fake_train)
    rng = np.random.default_rng(1)
    first = inp / "name-that-llm-llm-first"  # notebook 10's output: its seed-7 OOF
    first.mkdir()
    p = rng.random((len(fake_train), 18))
    np.save(first / "oof-llm-seed7.npy", p / p.sum(axis=1, keepdims=True))
    fake_open_pool(inp)
    work = tmp_path / "working"
    work.mkdir()
    config = tmp_path / "llm.yaml"
    tiny = dict(name="llm", base_model="unused", load_in_4bit=False, n_layers=4, lora_r=4)
    tiny |= dict(lora_alpha=8, max_length=64, batch_size=8, lr=5e-3, epochs=1)
    config.write_text(yaml.safe_dump(tiny), encoding="utf-8")
    env = {
        "NTL_INPUT": str(inp),
        "NTL_WORK": str(work),
        "NTL_LLM_CONFIG": str(config),
        "NTL_DATA_DIR": str(fake_data_dir),
    }
    run_notebook(NOTEBOOKS / "11-llm-explore" / "llm_explore.ipynb", env, monkeypatch)

    results = pd.read_csv(work / "results.csv")
    assert results["name"].iloc[0] == "baseline (notebook 10)"
    assert sum(results["name"].str.startswith("layers")) == 2  # 2/3 and 4/9 of the blocks
    assert {"len320", "pool5000", "pool15000"} <= set(results["name"])
    assert results["mean"].between(0, 1).all()
    chosen = LLMConfig(**yaml.safe_load((work / "llm-chosen.yaml").read_text(encoding="utf-8")))
    assert chosen.seed == 7 and chosen.pool_size in (0, 5000, 15000)


def test_llm_final_notebook_runs(tiny_qwen, fake_data_dir, fake_train, tmp_path, monkeypatch):
    # Notebook 12's first session (P7 Task 7): the pooled LLM at 1 epoch on seed-7 folds 0
    # and 1, against notebook 11's 2-epoch run on the same folds. Within 0.005, the cheaper
    # 1 epoch wins, and every later run costs half.
    pytest.importorskip("peft")
    import pandas as pd

    inp = tmp_path / "input"
    fake_kaggle_input(inp, tiny_qwen, fake_train)
    fake_open_pool(inp)
    explore = inp / "name-that-llm-llm-explore"  # notebook 11's output: its results table
    explore.mkdir()
    pd.DataFrame(
        [{"name": "pool5000", "pool_size": 5000, "fold0": 0.47, "fold1": 0.41, "mean": 0.44}]
    ).to_csv(explore / "results.csv", index=False)
    work = tmp_path / "working"
    work.mkdir()
    config = tmp_path / "llm.yaml"
    tiny = dict(name="llm", base_model="unused", load_in_4bit=False, n_layers=4, lora_r=4)
    tiny |= dict(lora_alpha=8, max_length=64, batch_size=8, lr=5e-3, epochs=2, pool_size=20)
    config.write_text(yaml.safe_dump(tiny), encoding="utf-8")
    env = {
        "NTL_INPUT": str(inp),
        "NTL_WORK": str(work),
        "NTL_LLM_CONFIG": str(config),
        "NTL_DATA_DIR": str(fake_data_dir),
    }
    run_notebook(NOTEBOOKS / "12-llm-final" / "llm_final.ipynb", env, monkeypatch)

    results = pd.read_csv(work / "results.csv")
    assert list(results["name"]) == ["ep1-seed7"]
    assert results[["fold0", "fold1", "mean"]].stack().between(0, 1).all()
    summary = json.loads((work / "results.json").read_text())
    assert summary["reference"]["mean"] == 0.44  # notebook 11's pool5000, the same folds
    assert summary["chosen_epochs"] in (1, 2)
    assert np.load(work / "runs" / "ep1-seed7" / "fold0.npz")["fold0"].shape[1] == 18
