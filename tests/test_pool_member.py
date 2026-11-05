"""Tests for the pool members (P8 Task 2): models trained once, on the pool alone.

P8's new members (DeBERTa-v3-large, then the LLM on far more of the pool) train on about
200,000 public passages. Five folds would cost five such runs, so each trains ONCE, on the
pool only: never on one of our 2,699 passages, and never on a pool passage about one of our
150 prompts. Every one of our passages is then unseen, about an unseen prompt, so the
member's predictions on all of them are honest, as the hidden test's would be (spec §5.2).
Every label weighs the same in the loss, since the pool's sources cover the 18 models very
unevenly (spec §3). Fake data only.
"""

import re

import numpy as np
import pandas as pd
import pytest

from name_that_llm.config import EncoderConfig, LLMConfig
from name_that_llm.labels import LABELS
from name_that_llm.pool_member import (
    find_member_dirs,
    fit_pool_member,
    honest_training_set,
    label_weights,
    probe_pool_member,
    rows_for_budget,
)


def _pool(n_per_label: int = 2) -> pd.DataFrame:
    """A fake pool: marker words the tiny tokenizers know, on topics none of ours share."""
    rows = [
        {
            "text": f"pooltopic{i} marker{m} marker{m} — style{m}",
            "model": label,
            "source": "a/one",
            "prompt_id": None,
        }
        for m, label in enumerate(LABELS)
        for i in range(n_per_label)
    ]
    return pd.DataFrame(rows)


def test_honest_training_set_never_holds_our_prompts_or_passages(fake_train):
    # The two leaks the honest protocol must close, even when the matcher or the dedupe
    # missed them upstream: a passage about one of our prompts, and a copy of our own.
    clean = [f"clean passage {i} marker1" for i in range(10)]
    pool = pd.DataFrame(
        {
            "text": ["a passage about topic0", fake_train["text"].iloc[1] + "  ", *clean],
            "model": [LABELS[0], LABELS[1], *[LABELS[2]] * 10],
            "source": ["a/one"] * 12,
            "prompt_id": [fake_train["prompt_id"].iloc[0], None, *[None] * 10],
        }
    )
    out = honest_training_set(pool, fake_train, max_rows=None, seed=0)
    assert sorted(out["text"]) == sorted(clean)
    assert not set(out["text"]) & set(fake_train["text"])


def test_label_weights_equalize_every_label():
    y = [LABELS[0]] * 3 + [LABELS[1]] + [LABELS[2]] * 2 + list(LABELS[3:])
    w = label_weights(y)
    assert w.shape == (18,)
    totals = np.bincount([LABELS.index(v) for v in y], minlength=18) * w
    np.testing.assert_allclose(totals, totals[0])


def test_label_weights_refuse_a_missing_label():
    # A member that never saw a model can never predict it: stop, and say which.
    y = [label for label in LABELS if label != LABELS[5]]
    with pytest.raises(ValueError, match=re.escape(LABELS[5])):
        label_weights(y)


def test_rows_for_budget_fits_the_session():
    assert rows_for_budget(1.0, 2, 600, 100_000) == 18_000  # 600 minutes, 2 epochs
    assert rows_for_budget(1.0, 2, 600, 5_000) == 5_000  # never more than the pool holds
    assert rows_for_budget(1.0, 2, 0, 5_000) == 0  # no time left: skip, don't start


def _record_probe(monkeypatch, overflow: bool = False) -> list[tuple[list[str], bool]]:
    """Run the probe's training through a recorder of each call's texts and precision (amp).
    With overflow, every float16 call fails, as DeBERTa-v3's can on a T4."""
    pytest.importorskip("torch")  # the probe reads the GPU's memory through torch
    import name_that_llm.encoder as encoder
    import name_that_llm.pool_member as pool_member

    calls = []

    def fake_train_encoder(model, tokenizer, texts, y, cfg, amp=True, **_):
        calls.append((list(texts), amp))
        if overflow and amp:
            raise FloatingPointError("non-finite loss")
        return [1.0]

    monkeypatch.setattr(encoder, "train_encoder", fake_train_encoder)
    monkeypatch.setattr(pool_member, "_load", lambda cfg: (None, None))
    return calls


def test_probe_steps_on_the_longest_batch_first(monkeypatch, fake_train):
    # A batch is padded to its own longest passage, so memory peaks on the rare batch of long
    # ones, which n random passages may never form: notebook 13 v1's probe peaked at 14.1 GB,
    # then its run ran out of memory. So the probe's first step is the worst batch.
    calls = _record_probe(monkeypatch)
    pool = _pool(4)
    pool.loc[[5, 40, 70], "text"] += " long" * 50  # three passages far longer than the rest
    cfg = EncoderConfig(name="t", base_model="unused", batch_size=3)
    out = probe_pool_member(cfg, pool, fake_train, n=16)
    assert sorted(calls[0][0]) == sorted(pool.loc[[5, 40, 70], "text"])
    assert len(calls[1][0]) == 16 and out["finite"]


def test_probe_times_float32_after_a_float16_overflow(monkeypatch, fake_train):
    # The real run then trains in float32, so that's the speed to size it by: the time until
    # an overflow says nothing about a whole epoch's.
    calls = _record_probe(monkeypatch, overflow=True)
    cfg = EncoderConfig(name="t", base_model="unused", batch_size=3)
    out = probe_pool_member(cfg, _pool(4), fake_train, n=16)
    assert not out["finite"]
    assert [amp for _, amp in calls] == [True, False, False]  # overflow, then both in float32


def test_fit_pool_member_predicts_every_passage_and_saves_each_epoch(
    tiny_encoder, fake_train, tmp_path
):
    pytest.importorskip("torch")
    cfg = EncoderConfig(
        name="t",
        base_model=str(tiny_encoder),
        epochs=2,
        pool_size=36,
        batch_size=8,
        max_length=64,
    )
    proba, losses = fit_pool_member(cfg, _pool(), fake_train, save_dir=tmp_path / "m")
    assert proba.shape == (len(fake_train), 18)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert len(losses) == 2
    for e in (1, 2):
        assert (tmp_path / "m" / f"epoch{e}" / "config.json").exists()


def test_fit_pool_member_trains_the_llm_too(tiny_qwen, fake_train, tmp_path):
    pytest.importorskip("peft")
    cfg = LLMConfig(
        name="t",
        base_model=str(tiny_qwen),
        load_in_4bit=False,
        n_layers=4,
        lora_r=4,
        lora_alpha=8,
        max_length=64,
        batch_size=8,
        lr=5e-3,
        epochs=2,
        pool_size=36,
        seed=0,
    )
    proba, losses = fit_pool_member(cfg, _pool(), fake_train, save_dir=tmp_path / "m")
    assert proba.shape == (len(fake_train), 18) and len(losses) == 2
    for e in (1, 2):
        assert (tmp_path / "m" / f"epoch{e}" / "adapter_config.json").exists()


def test_class_weights_change_the_loss(tiny_encoder, fake_train):
    # All-ones weights are the plain mean loss; skewed weights must change the training, or
    # the labels aren't really weighted. A fresh model predicts nearly uniformly, so every
    # passage's loss starts near ln(18) whatever the weights: the training must move first.
    # Hence 3 epochs at the tiny models' big step, comparing the last epoch's loss.
    torch = pytest.importorskip("torch")
    from name_that_llm.encoder import load_encoder, train_encoder

    cfg = EncoderConfig(
        name="t", base_model=str(tiny_encoder), epochs=3, lr=5e-3, batch_size=8, max_length=64
    )
    texts, y = fake_train["text"].tolist(), fake_train["model"].to_numpy()

    def last_loss(weight):
        torch.manual_seed(0)
        model, tokenizer = load_encoder(cfg.base_model)
        return train_encoder(model, tokenizer, texts, y, cfg, class_weight=weight)[-1]

    skewed = np.ones(18)
    skewed[0] = 10.0
    plain = last_loss(None)
    assert last_loss(np.ones(18)) == pytest.approx(plain, rel=1e-4)
    assert last_loss(skewed) != pytest.approx(plain, rel=1e-3)


def _model_dir(root, seed, epoch):
    folder = root / "pool-deberta" / "models" / f"seed{seed}" / f"epoch{epoch}"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    return folder


def test_find_member_dirs_takes_each_seeds_final_epoch(tmp_path):
    wanted = [_model_dir(tmp_path, 42, 2), _model_dir(tmp_path, 43, 2)]
    _model_dir(tmp_path, 42, 1)  # an earlier epoch: the gate never measured these weights
    assert find_member_dirs((42, 43), 2, tmp_path) == wanted


def test_find_member_dirs_refuses_a_missing_seed(tmp_path):
    # One seed in place of two would silently change the submission.
    _model_dir(tmp_path, 42, 2)
    _model_dir(tmp_path, 43, 1)  # stopped part-way: its last finished epoch isn't the final one
    with pytest.raises(FileNotFoundError, match="seed43"):
        find_member_dirs((42, 43), 2, tmp_path)
