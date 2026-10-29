"""Tests for the LLM classifier (P7): LoRA on a cut Qwen, on a tiny RANDOM Qwen and fake data.

Every test needs torch, transformers and peft, and this module skips without them. CI's
`torch` job installs all three (CPU-only) and runs it. Loading in 4 bits needs CUDA and
bitsandbytes, so the tests load in float32 (load_in_4bit=False); the rest is the GPU path.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from name_that_llm.config import LLMConfig
from name_that_llm.cv import make_folds
from name_that_llm.encoder import predict_proba, train_encoder
from name_that_llm.labels import LABELS
from name_that_llm.llm import (
    fold_training_set,
    load_llm_classifier,
    main,
    merge_folds,
    predict_proba_adapters,
    run_cv_llm,
    save_adapter,
)

pytest.importorskip("peft")  # torch and transformers: the tiny_qwen fixture skips without them


def _cfg(path, **overrides):
    # A tiny random Qwen learns fastest with big steps over many epochs (Qwen3-8B trains at
    # about 1e-4 for 2 epochs). max_length 64 fits the fake passages.
    base = dict(
        name="t",
        base_model=str(path),
        load_in_4bit=False,
        n_layers=4,
        lora_r=4,
        lora_alpha=8,
        max_length=64,
        batch_size=8,
        lr=5e-3,
        epochs=10,
        seed=0,
    )
    return LLMConfig(**{**base, **overrides})


def test_lora_trains_only_adapters_and_head(tiny_qwen, fake_train):
    # LoRA freezes the base weights and learns small low-rank updates beside them, plus the
    # new 18-way head ("score"): only those may train, and a base weight must not move.
    import torch

    cfg = _cfg(tiny_qwen, epochs=1)
    model, tokenizer = load_llm_classifier(cfg)
    params = dict(model.named_parameters())
    trainable = [n for n, p in params.items() if p.requires_grad]
    assert trainable and all("lora_" in n or "score" in n for n in trainable)
    name = next(n for n in params if n.endswith("layers.0.mlp.up_proj.base_layer.weight"))
    before = params[name].detach().clone()
    texts, y = fake_train["text"].tolist(), fake_train["model"].to_numpy()
    train_encoder(model, tokenizer, texts, y, cfg)
    assert torch.equal(params[name], before)


def test_llm_keeps_only_the_first_layers(tiny_qwen):
    model, _ = load_llm_classifier(_cfg(tiny_qwen))
    assert len(model.get_base_model().model.layers) == 4


def test_llm_learns_the_markers(tiny_qwen, fake_train):
    assert run_cv_llm(_cfg(tiny_qwen), fake_train).score > 0.5


def test_llm_handles_empty_and_long_text(tiny_qwen, tmp_path):
    # Review Focus 4: an empty passage has no token at all (Qwen adds none of its own), and
    # 2,000 words overflow max_length. Both must still give 18 probabilities.
    cfg = _cfg(tiny_qwen)
    model, tokenizer = load_llm_classifier(cfg)
    save_adapter(model, tokenizer, cfg, tmp_path / "fold0")
    for texts in (["", "marker1 " * 2000], [""]):
        proba = predict_proba_adapters(cfg, [tmp_path / "fold0"], texts)
        assert proba.shape == (len(texts), 18) and np.isfinite(proba).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_fold_training_set_follows_the_pool_rule(fake_train):
    # Fold k trains on our passages outside fold k, plus at most pool_size pool rows, none
    # of them about one of fold k's prompts (the pool's no-leak rule).
    folds = make_folds(fake_train, 5, 0).to_numpy()
    prompts = fake_train["prompt_id"].to_numpy()
    topics = sorted(set(prompts))
    pool = pd.DataFrame(
        {
            "text": [f"pooltag {p} {i}" for p in topics for i in range(3)] + ["pooltag none"] * 2,
            "model": [LABELS[i] for _ in topics for i in range(3)] + LABELS[:2],
            "prompt_id": [p for p in topics for _ in range(3)] + [None, None],
        }
    )
    texts, y = fake_train["text"].tolist(), fake_train["model"].to_numpy()
    fit_texts, fit_y = fold_training_set(texts, y, prompts, folds, 0, pool, _cfg("x", pool_size=5))
    ours = [t for t in fit_texts if not t.startswith("pooltag")]
    extra = [t for t in fit_texts if t.startswith("pooltag")]
    assert sorted(ours) == sorted(fake_train["text"][folds != 0]) and len(fit_y) == len(fit_texts)
    assert len(extra) == 5 and not {t.split()[1] for t in extra} & set(prompts[folds == 0])
    # pool_size 0 (the default) or no pool at all: our passages only.
    for args in [(pool, _cfg("x")), (None, _cfg("x", pool_size=5))]:
        assert len(fold_training_set(texts, y, prompts, folds, 0, *args)[0]) == len(ours)


def test_split_folds_merge_to_the_single_run(tiny_qwen, fake_data_dir, fake_train, tmp_path):
    # Two GPUs run different folds in two processes (I-019). Merged, they must equal one run
    # over every fold, or the OOF score would depend on how the work was split.
    cfg = _cfg(tiny_qwen, epochs=2)
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
    parts = []
    for name, folds in [("a", ["0", "1", "2"]), ("b", ["3", "4"])]:
        out = tmp_path / f"{name}.npz"
        argv = ["--config", str(path), "--base-model", str(tiny_qwen), "--seed", "0"]
        argv += ["--data-dir", str(fake_data_dir), "--folds", *folds, "--out", str(out)]
        assert main(argv) == 0
        with np.load(out) as npz:
            parts.append({int(k.removeprefix("fold")): npz[k] for k in npz.files})
    merged = merge_folds(cfg, fake_train, parts)
    np.testing.assert_allclose(merged.oof, run_cv_llm(cfg, fake_train).oof, atol=1e-6)
    with pytest.raises(ValueError, match="missing"):
        merge_folds(cfg, fake_train, parts[:1])
    with pytest.raises(ValueError, match="twice"):
        merge_folds(cfg, fake_train, [parts[0], parts[0], parts[1]])


def test_saved_adapter_predicts_like_the_trained_model(tiny_qwen, fake_train, tmp_path):
    # The submission reloads each fold's adapter onto a fresh base model. It must predict
    # exactly what the trained model did, the new head included.
    cfg = _cfg(tiny_qwen, epochs=2)
    texts = fake_train["text"].tolist()
    model, tokenizer = load_llm_classifier(cfg)
    train_encoder(model, tokenizer, texts, fake_train["model"].to_numpy(), cfg)
    expected = predict_proba(model, tokenizer, texts[:20], max_length=cfg.max_length)
    save_adapter(model, tokenizer, cfg, tmp_path / "fold0")
    got = predict_proba_adapters(cfg, [tmp_path / "fold0"], texts[:20])
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_adapter_on_a_different_cut_is_refused(tiny_qwen, tmp_path):
    # Review Focus 3: an adapter trained on a 4-block cut, loaded onto 6 blocks, would run
    # without an error and return garbage. It must be refused.
    cfg = _cfg(tiny_qwen)
    model, tokenizer = load_llm_classifier(cfg)
    save_adapter(model, tokenizer, cfg, tmp_path / "fold0")
    six = cfg.model_copy(update={"n_layers": 6})
    with pytest.raises(ValueError, match="n_layers"):
        predict_proba_adapters(six, [tmp_path / "fold0"], ["marker1"])


def test_probe_writes_its_timing_from_its_own_process(tiny_qwen, fake_data_dir, tmp_path):
    # The notebooks time one fold's training in a process of its own (`--probe`), so the
    # probe's GPU memory is all returned before the folds start.
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(_cfg(tiny_qwen).model_dump()), encoding="utf-8")
    out = tmp_path / "timing.json"
    argv = ["--config", str(path), "--data-dir", str(fake_data_dir), "--out", str(out)]
    assert main([*argv, "--probe", "16"]) == 0
    timing = json.loads(out.read_text(encoding="utf-8"))
    keys = {"seconds_per_passage", "model_load_seconds", "peak_memory_gb", "projected_cv_minutes"}
    assert set(timing) == keys and timing["seconds_per_passage"] >= 0
    with pytest.raises(SystemExit):  # neither --folds nor --probe
        main(argv)


def test_a_cut_deeper_than_the_model_is_refused(tiny_qwen):
    # tiny_qwen has 12 blocks: a 24-block cut can't be made, and must say so clearly,
    # not fail with an IndexError deep inside transformers.
    with pytest.raises(ValueError, match="12 blocks"):
        load_llm_classifier(_cfg(tiny_qwen, n_layers=24))
