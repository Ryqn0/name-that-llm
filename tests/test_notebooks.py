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
