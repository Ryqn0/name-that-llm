"""Pool members (P8): models trained ONCE, on the public pool alone.

P8's new members (DeBERTa-v3-large, then the LLM on far more of the pool) learn from about
200,000 public passages. Cross-validating them would cost five such runs. Instead, each
trains once (spec §5.2):
  - never on one of our 2,699 passages;
  - never on a pool passage about one of our 150 prompts (pool.match_prompts).
Every one of our passages is then unseen, and about a prompt the member never trained on:
the hidden test's own situation. So its predictions on ALL of them are honest, from one
run, and the ensemble's nested weights can use them like any member's out-of-fold ones.

Every label weighs the same in the loss (label_weights): the pool's sources cover the 18
models very unevenly, and an unweighted random draw from all of them cost `lr` 0.07 (I-032).

torch and transformers are imported inside the functions (ADR 0002).
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from name_that_llm.config import EncoderConfig, LLMConfig, load_config
from name_that_llm.data import load_train
from name_that_llm.labels import LABELS
from name_that_llm.pool import pool_for_fold


def honest_training_set(
    pool: pd.DataFrame, train: pd.DataFrame, max_rows: int | None, seed: int
) -> pd.DataFrame:
    """The pool rows a pool member may train on: none about our prompts, none of our texts.

    pool_for_fold with EVERY one of our prompts held out drops the rows matched to them.
    Then any row whose text is one of our passages goes too. build_pool's dedupe already
    removes copies, but the honesty of every prediction rests on this, so it's enforced
    again here (Review Focus 2). With max_rows, a seeded sample of that size.
    """
    rows = pool_for_fold(pool, set(train["prompt_id"]), max_rows, seed)
    ours = {str(t).strip() for t in train["text"]}
    return rows[~rows["text"].astype(str).str.strip().isin(ours)].reset_index(drop=True)


def label_weights(y: Sequence[str]) -> np.ndarray:
    """(18,) loss weights in LABELS order, n / (18 * n_j): every label weighs the same.

    A label with no passage stops everything, named: the member could never predict that
    model, and its weight would divide by zero (Review Focus 1).
    """
    counts = pd.Series(list(y), dtype=object).value_counts()
    missing = [label for label in LABELS if counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"the training set has no passage for {missing}")
    n = counts.reindex(LABELS).to_numpy(dtype=float)
    return n.sum() / (len(LABELS) * n)


def rows_for_budget(seconds_per_passage: float, epochs: int, minutes: float, available: int) -> int:
    """How many pool passages one run can train on in `minutes`, at most `available`.

    No time left gives 0: the notebook then skips the run rather than start one that can't
    finish (Review Focus 3).
    """
    if minutes <= 0:
        return 0
    return max(0, min(available, math.floor(minutes * 60 / (seconds_per_passage * epochs))))


def _member_config(name_or_path: str) -> EncoderConfig | LLMConfig:
    """The config, as an encoder's or an LLM's. EncoderConfig forbids the LoRA fields an LLM
    config carries, so an LLM YAML fails it and loads as an LLMConfig."""
    try:
        return load_config(name_or_path, model_cls=EncoderConfig)
    except ValidationError:
        return load_config(name_or_path, model_cls=LLMConfig)


def _load(cfg: EncoderConfig | LLMConfig):
    if isinstance(cfg, LLMConfig):
        from name_that_llm.llm import load_llm_classifier

        return load_llm_classifier(cfg)
    from name_that_llm.encoder import load_encoder

    return load_encoder(cfg.base_model)


def _texts(texts: Sequence[str], tokenizer, cfg: EncoderConfig | LLMConfig) -> list[str]:
    """An LLM can't run on an empty passage (llm._nonempty); an encoder's [CLS] covers it."""
    if isinstance(cfg, LLMConfig):
        from name_that_llm.llm import _nonempty

        return _nonempty(texts, tokenizer)
    return [str(t) for t in texts]


def _save(model, tokenizer, cfg: EncoderConfig | LLMConfig, path: Path) -> None:
    """One epoch's checkpoint: an LLM's adapter, or an encoder's weights in float16."""
    if isinstance(cfg, LLMConfig):
        from name_that_llm.llm import save_adapter

        save_adapter(model, tokenizer, cfg, path)
        return
    # A float16 COPY of the weights: half the files, while the model keeps training in
    # float32 (model.half() would convert it in place, mid-run).
    half = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=half)
    tokenizer.save_pretrained(path)


def fit_pool_member(
    cfg: EncoderConfig | LLMConfig,
    pool: pd.DataFrame,
    train: pd.DataFrame,
    save_dir: Path | None = None,
    amp: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Train a fresh member on the honest pool; predict ALL our passages, in `train`'s order.

    Returns ((len(train), 18) probabilities, the mean loss per epoch). With save_dir, each
    finished epoch is saved to save_dir / f"epoch{e}". A run that stops part-way keeps its
    last finished epoch. If a next session ever continues it, it starts from those weights
    with a fresh optimizer: no optimizer state is saved, since the probe sizes each run to
    fit one session (the plan's call, spec §5.3).
    """
    import torch

    from name_that_llm.encoder import _release_memory, predict_proba, train_encoder

    torch.manual_seed(cfg.seed)  # the new head's random init and the dropout: reproducible
    rows = honest_training_set(pool, train, cfg.pool_size or None, cfg.seed)
    y = rows["model"].to_numpy()
    weights = label_weights(y)
    model, tokenizer = _load(cfg)

    def save(epoch: int) -> None:
        if save_dir is not None:
            _save(model, tokenizer, cfg, Path(save_dir) / f"epoch{epoch}")

    texts = _texts(rows["text"].tolist(), tokenizer, cfg)
    losses = train_encoder(
        model, tokenizer, texts, y, cfg, class_weight=weights, on_epoch_end=save, amp=amp
    )
    proba = predict_proba(
        model,
        tokenizer,
        _texts(train["text"].tolist(), tokenizer, cfg),
        batch_size=2 * cfg.batch_size,  # no gradients to store: bigger batches fit
        max_length=cfg.max_length,
    )
    # Drop the last reference (the epoch-saving closure shares this variable), so the
    # model's GPU memory can be freed before anything else loads.
    model = None
    _release_memory()
    return proba, losses


def probe_pool_member(
    cfg: EncoderConfig | LLMConfig, pool: pd.DataFrame, train: pd.DataFrame, n: int = 256
) -> dict:
    """Time 1 epoch on n honest pool passages, in float16 as the real run would train.

    Returns the seconds per training passage, the model's load time, the peak GPU memory,
    and whether the loss stayed finite: the notebook sizes the real run from it, and falls
    back to float32 if float16 overflowed. Run it in a process of its own (main's --probe):
    a model freed inside a long-lived process can keep its GPU memory (P7 Task 5).
    """
    import torch

    from name_that_llm.encoder import train_encoder

    probe = cfg.model_copy(update={"epochs": 1})
    rows = honest_training_set(pool, train, min(n, len(pool)), cfg.seed)
    torch.manual_seed(cfg.seed)
    start = time.time()
    model, tokenizer = _load(probe)
    load_s = time.time() - start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    texts = _texts(rows["text"].tolist(), tokenizer, cfg)
    start = time.time()
    try:
        train_encoder(model, tokenizer, texts, rows["model"].to_numpy(), probe)
        finite = True
    except FloatingPointError:  # float16 overflowed: the real run must use float32
        finite = False
    per_passage = (time.time() - start) / max(1, len(texts))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {
        "seconds_per_passage": round(per_passage, 3),
        "model_load_seconds": round(load_s, 1),
        "peak_memory_gb": round(peak_gb, 2),
        "finite": finite,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one pool member on the pool alone, then predict all our passages."
    )
    parser.add_argument("--config", required=True, help="a config name or YAML path")
    parser.add_argument("--pool", type=Path, required=True, help="pool.build_pool's parquet")
    parser.add_argument("--seed", type=int, help="the training's seed (overrides)")
    parser.add_argument("--rows", type=int, help="pool passages to train on (overrides)")
    parser.add_argument("--save-dir", type=Path, help="save each epoch to epoch<e>/")
    parser.add_argument("--data-dir", type=Path, help="the folder with train.csv")
    parser.add_argument("--probe", type=int, metavar="N", help="instead: time 1 epoch on N")
    parser.add_argument("--fp32", action="store_true", help="train in float32 (no float16)")
    parser.add_argument("--out", type=Path, required=True, help=".npz (--probe: a .json)")
    args = parser.parse_args(argv)

    cfg = _member_config(args.config)
    updates = {"seed": args.seed, "pool_size": args.rows}
    cfg = cfg.model_copy(update={k: v for k, v in updates.items() if v is not None})
    train = load_train(args.data_dir)
    pool = pd.read_parquet(args.pool)
    if args.probe is not None:
        timing = probe_pool_member(cfg, pool, train, args.probe)
        args.out.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        return 0
    proba, losses = fit_pool_member(cfg, pool, train, args.save_dir, amp=not args.fp32)
    np.savez(args.out, train=proba, losses=np.asarray(losses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
