"""The LLM classifier (P7, model roadmap M5): Qwen 3 fine-tuned with LoRA, on 4-bit weights.

Why a large decoder LLM: the public leaders (0.651) all fine-tune 8-14B LLMs to name the
author. Such a model has read text like every one of our 18 models' writing, and
fine-tuning teaches it which habits tell them apart. Three tricks fit that on one Kaggle
T4 (16 GB):

  - 4-bit weights (QLoRA's NF4): each weight is stored in 4 bits, on a grid of 16 levels
    spaced like a normal distribution (where trained weights actually lie): a quarter of
    float16's memory. The products still run in float16: each block of weights is turned
    back into float16 just before it's used ("dequantized"), then dropped.
  - LoRA: the base weights stay FROZEN. Beside each linear layer W, LoRA learns two thin
    matrices, B (d x r) and A (r x k), and the layer computes W x + (alpha / r) B A x. With
    r = 16 that's well under 1% of the weights, and the optimizer's state (two numbers per
    trained weight) shrinks with it. B starts at zero, so training starts from the base
    model exactly.
  - Gradient checkpointing: the forward pass keeps only each block's input, and the
    backward pass recomputes the rest. About a third more compute, for a fraction of the
    activation memory: that's what lets 512-token passages fit in a batch.

The head reads the hidden state of the LAST token: a decoder reads left to right, so only
the last position has seen the whole passage.

The training loop is the encoder's (encoder.train_encoder, ADR 0003), which optimizes
only the parameters that require gradients. One fresh model per fold, like the encoder.
Kaggle's T4 x2 runs two folds at once, one process per GPU (`main`, I-019), and
merge_folds joins their outputs into one CVResult.

torch, transformers, peft and bitsandbytes are imported INSIDE the functions (ADR 0002):
the text-only paths must never need them.
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.config import LLMConfig, load_config
from name_that_llm.cv import make_folds
from name_that_llm.data import load_train
from name_that_llm.encoder import _release_memory, predict_proba, train_encoder
from name_that_llm.labels import LABELS
from name_that_llm.lm import _dtype_kwargs, _lm_tokenizer
from name_that_llm.metrics import score
from name_that_llm.pool import pool_for_fold
from name_that_llm.train import CVResult, _result

ADAPTER_INFO = "llm_config.json"  # written next to each saved adapter by save_adapter


def _base_model(cfg: LLMConfig, device: str):
    """The base LLM with a new 18-way head, cut to cfg.n_layers blocks: (model, tokenizer).

    In 4 bits on CUDA when cfg.load_in_4bit, else in float32 (CPU: the tests). No LoRA yet.
    """
    import torch
    import transformers

    local = Path(cfg.base_model).exists()  # a local folder never triggers a Hub lookup
    config = transformers.AutoConfig.from_pretrained(
        cfg.base_model,
        local_files_only=local,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    if cfg.n_layers is not None:
        # The cut, as in lm.load_lm_layers: the later blocks are never built, and their
        # weights are skipped while loading.
        config.num_hidden_layers = cfg.n_layers
        if getattr(config, "layer_types", None):
            config.layer_types = config.layer_types[: cfg.n_layers]
    config.use_cache = False  # the generation cache: useless for classifying, and costly
    tokenizer = _lm_tokenizer(cfg.base_model, local_files_only=local)
    # The head reads the last token that isn't padding, so the model must know the pad id.
    config.pad_token_id = tokenizer.pad_token_id
    kwargs = {"config": config, "local_files_only": local, **_dtype_kwargs(torch.float32)}
    if cfg.load_in_4bit and device.startswith("cuda"):
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # the T4 has no bfloat16
            bnb_4bit_use_double_quant=True,  # the blocks' scales in 8 bits too: ~0.4 GB less
        )
        # Quantized weights are placed while loading; a 4-bit model can't be moved after.
        kwargs["device_map"] = {"": torch.device(device).index or 0}
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model, **kwargs
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model, **kwargs
        ).to(device)
    return model, tokenizer


def _base_info(model) -> dict:
    """What an adapter depends on: the base's architecture, width and number of blocks."""
    return {
        "model_type": model.config.model_type,
        "hidden_size": model.config.hidden_size,
        "n_layers": model.config.num_hidden_layers,
    }


def _check_adapter(model, adapter_dir: Path) -> None:
    """Refuse an adapter trained on another base model, or another cut (Review Focus 3).

    An adapter on the wrong cut still loads without an error (its updates land on other
    blocks, or on none) and returns garbage, so what save_adapter recorded is compared.
    """
    saved = json.loads((Path(adapter_dir) / ADAPTER_INFO).read_text(encoding="utf-8"))
    for key, value in _base_info(model).items():
        if saved.get(key) != value:
            raise ValueError(
                f"the adapter in {adapter_dir} was trained with {key}={saved.get(key)!r}, "
                f"but this base model has {key}={value!r}"
            )


def _nonempty(texts: Sequence[str], tokenizer) -> list[str]:
    """The texts, with each empty one replaced by the pad token (Review Focus 4).

    Qwen adds no token of its own (no [CLS], unlike an encoder), so an empty passage would
    be a sequence of length 0, which the model can't run on. The pad token stands in:
    one position, and still no content.
    """
    return [t if t.strip() else tokenizer.pad_token for t in map(str, texts)]


def load_llm_classifier(cfg: LLMConfig, adapter_dir: Path | None = None, device: str | None = None):
    """The LLM classifier, to train (adapter_dir None) or to predict: (model, tokenizer).

    To train: fresh LoRA adapters on every linear layer, plus a trainable copy of the head.
    To predict: the adapters and head that save_adapter wrote to adapter_dir, refused if
    they were trained on another base model or another cut.
    """
    import peft
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _base_model(cfg, device)
    if adapter_dir is not None:
        _check_adapter(model, adapter_dir)
        return peft.PeftModel.from_pretrained(model, adapter_dir), tokenizer
    if getattr(model, "is_loaded_in_4bit", False):
        # Casts the few parts left in 16 bits (norms, embeddings) to float32 for stable
        # training, and turns gradient checkpointing on (see the module docstring).
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            # Passed explicitly, as PyTorch now asks: the non-reentrant variant is the
            # recommended one, and it needs no gradient on the inputs.
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    lora = peft.LoraConfig(
        task_type="SEQ_CLS",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",  # every projection of the attention and the MLP
        modules_to_save=["score"],  # the new head trains in full (it's small: hidden x 18)
    )
    return peft.get_peft_model(model, lora), tokenizer


def save_adapter(model, tokenizer, cfg: LLMConfig, path: Path) -> None:
    """Save a trained fold: its adapters and head (megabytes, not the base's gigabytes),
    the tokenizer, and ADAPTER_INFO (the config, plus what _check_adapter compares)."""
    path = Path(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    info = {**cfg.model_dump(), **_base_info(model.get_base_model())}
    (path / ADAPTER_INFO).write_text(json.dumps(info, indent=2), encoding="utf-8")


def fold_training_set(
    texts: Sequence[str],
    y: np.ndarray,
    prompt_ids: Sequence[str],
    folds: np.ndarray,
    k: int,
    pool: pd.DataFrame | None,
    cfg: LLMConfig,
) -> tuple[list[str], np.ndarray]:
    """What fold k's model trains on: (texts, labels).

    Our passages outside fold k, plus up to cfg.pool_size pool passages, never one about
    a prompt of fold k (pool.pool_for_fold: the pool's no-leak rule).
    """
    folds = np.asarray(folds)
    rows = np.flatnonzero(folds != k)
    fit_texts, fit_y = [str(texts[i]) for i in rows], np.asarray(y)[rows]
    if pool is not None and cfg.pool_size > 0:
        held_out = set(np.asarray(prompt_ids)[folds == k])
        extra = pool_for_fold(pool, held_out, cfg.pool_size, seed=cfg.seed + k)
        fit_texts += extra["text"].astype(str).tolist()
        fit_y = np.concatenate([fit_y, extra["model"].to_numpy()])
    return fit_texts, fit_y


def run_folds_llm(
    cfg: LLMConfig,
    df: pd.DataFrame,
    folds_to_run: Sequence[int],
    pool: pd.DataFrame | None = None,
    save_dir: Path | None = None,
) -> dict[int, np.ndarray]:
    """Train and predict the folds in folds_to_run: {k: probabilities for fold k's rows}.

    Each fold starts again from the base model, so no fold's prompts leak into another.
    torch.manual_seed(cfg.seed + k) comes first: the new head, the adapters' random half
    and the dropout then depend on the fold alone, so folds split across two processes
    give exactly the numbers of one run. One printed line per fold: a GPU fold takes an
    hour, and Kaggle's log keeps only what's printed.
    """
    import torch

    texts, y = df["text"].tolist(), df["model"].to_numpy()
    prompts = df["prompt_id"].to_numpy()
    folds = make_folds(df, cfg.n_splits, cfg.seed).to_numpy()
    labels = np.asarray(LABELS)
    out = {}
    for k in map(int, folds_to_run):
        if k not in folds:
            # Fold 5 of 5 would train on every row, then predict none: an hour wasted.
            raise ValueError(f"fold {k} doesn't exist: the folds are 0 to {cfg.n_splits - 1}")
        start = time.time()
        torch.manual_seed(cfg.seed + k)
        model, tokenizer = load_llm_classifier(cfg)
        fit_texts, fit_y = fold_training_set(texts, y, prompts, folds, k, pool, cfg)
        losses = train_encoder(model, tokenizer, _nonempty(fit_texts, tokenizer), fit_y, cfg)
        valid = np.flatnonzero(folds == k)
        proba = predict_proba(
            model,
            tokenizer,
            _nonempty([texts[i] for i in valid], tokenizer),
            batch_size=2 * cfg.batch_size,  # no gradients to store: bigger batches fit
            max_length=cfg.max_length,
        )
        if save_dir is not None:
            save_adapter(model, tokenizer, cfg, Path(save_dir) / f"fold{k}")
        del model
        _release_memory()
        out[k] = proba
        print(
            f"{cfg.name} seed {cfg.seed} fold {k}: "
            f"CV {score(y[valid], labels[proba.argmax(axis=1)]):.4f} | "
            f"{len(fit_texts)} training passages | epoch losses "
            + ", ".join(f"{loss:.3f}" for loss in losses)
            + f" | {(time.time() - start) / 60:.1f} min",
            flush=True,
        )
    return out


def merge_folds(
    cfg: LLMConfig, df: pd.DataFrame, parts: Sequence[dict[int, np.ndarray]]
) -> CVResult:
    """One CVResult from the fold outputs of several runs (e.g. one process per GPU).

    Every fold must come exactly once. A missing fold would leave its rows at probability
    0, and a fold from two runs was trained twice, and maybe not the same way.
    """
    folds = make_folds(df, cfg.n_splits, cfg.seed).to_numpy()
    oof, seen = np.zeros((len(df), len(LABELS))), set()
    for part in parts:
        for k, proba in part.items():
            if k in seen:
                raise ValueError(f"fold {k} comes twice")
            seen.add(k)
            oof[folds == k] = proba
    missing = sorted(set(np.unique(folds).tolist()) - seen)
    if missing:
        raise ValueError(f"folds missing: {missing}")
    return _result(oof, df["model"].to_numpy(), folds)


def run_cv_llm(
    cfg: LLMConfig,
    df: pd.DataFrame,
    pool: pd.DataFrame | None = None,
    save_dir: Path | None = None,
) -> CVResult:
    """Prompt-grouped CV of the LLM in one process: every fold, then merged."""
    return merge_folds(cfg, df, [run_folds_llm(cfg, df, range(cfg.n_splits), pool, save_dir)])


def predict_proba_adapters(
    cfg: LLMConfig, adapter_dirs: Sequence[Path], texts: Sequence[str], batch_size: int = 16
) -> np.ndarray:
    """The mean probabilities of several fold adapters on ONE base model (the submission).

    The base (gigabytes) loads once, and each fold's adapter (megabytes) is swapped in
    turn. These are the models behind the OOF predictions, so blend weights chosen on the
    OOF apply to this average too.
    """
    if not adapter_dirs:
        raise ValueError("no adapters given")
    model, tokenizer = load_llm_classifier(cfg, adapter_dirs[0])
    texts = _nonempty(texts, tokenizer)
    total = np.zeros((len(texts), len(LABELS)))
    for i, adapter_dir in enumerate(adapter_dirs):
        if i > 0:
            _check_adapter(model.get_base_model(), adapter_dir)
            model.load_adapter(adapter_dir, adapter_name=f"fold{i}")
            model.set_adapter(f"fold{i}")
        total += predict_proba(model, tokenizer, texts, batch_size, cfg.max_length)
    del model
    _release_memory()
    return total / len(adapter_dirs)


def probe_speed(cfg: LLMConfig, df: pd.DataFrame, n: int = 256, gpus: int = 2) -> dict:
    """How long a full CV would take, from one fold's model trained 1 epoch on n passages.

    Returns the seconds per training passage, the model's load time, the peak GPU memory,
    and the projected minutes of a cfg.n_splits-fold CV spread over `gpus` GPUs, where the
    busiest runs ceil(n_splits / gpus) folds. Each fold loads the model, trains on about
    (k - 1) / k of our passages plus cfg.pool_size pool passages for cfg.epochs epochs,
    then predicts the other 1 / k: a forward pass only, about a third of a training step.

    Run it in a process of its own (main's --probe). A model freed inside a long-lived
    process can keep holding GPU memory: in P7 Task 5's first run, 5.8 GB of a freed probe
    model starved the fold that ran next on the same GPU.
    """
    import torch

    probe = cfg.model_copy(update={"epochs": 1})
    folds = make_folds(df, cfg.n_splits, cfg.seed).to_numpy()
    y, prompts = df["model"].to_numpy(), df["prompt_id"].to_numpy()
    texts, fit_y = fold_training_set(df["text"].tolist(), y, prompts, folds, 0, None, probe)
    n = min(n, len(texts))
    torch.manual_seed(cfg.seed)
    start = time.time()
    model, tokenizer = load_llm_classifier(probe)
    load_s = time.time() - start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    train_encoder(model, tokenizer, _nonempty(texts[:n], tokenizer), fit_y[:n], probe)
    per_passage = (time.time() - start) / n
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    k = cfg.n_splits
    trained = len(df) * (k - 1) / k + cfg.pool_size  # passages per fold, per epoch
    fold_s = load_s + per_passage * (trained * cfg.epochs + len(df) / k / 3)
    return {
        "seconds_per_passage": round(per_passage, 3),
        "model_load_seconds": round(load_s, 1),
        "peak_memory_gb": round(peak_gb, 2),
        "projected_cv_minutes": round(math.ceil(k / gpus) * fold_s / 60, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and predict some CV folds of the LLM classifier (one GPU each)."
    )
    parser.add_argument("--config", required=True, help="LLMConfig name (e.g. llm) or YAML path")
    parser.add_argument("--base-model", help="the base model's folder (overrides the config)")
    parser.add_argument("--seed", type=int, help="the folds' and training's seed (overrides)")
    parser.add_argument("--folds", type=int, nargs="+", help="the folds to run, e.g. 0 2 4")
    parser.add_argument(
        "--probe", type=int, metavar="N", help="instead of folds: time 1 epoch on N passages"
    )
    parser.add_argument("--pool", type=Path, help="the pool's parquet (pool.build_pool)")
    parser.add_argument("--save-dir", type=Path, help="save each fold's adapter to fold<k>/")
    parser.add_argument("--data-dir", type=Path, help="the folder with train.csv")
    parser.add_argument(
        "--out", type=Path, required=True, help=".npz: one array per fold (--probe: a .json)"
    )
    args = parser.parse_args(argv)
    if (args.folds is None) == (args.probe is None):
        parser.error("give either --folds or --probe")

    cfg = load_config(args.config, model_cls=LLMConfig)
    overrides = {"base_model": args.base_model, "seed": args.seed}
    cfg = cfg.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    df = load_train(args.data_dir)
    if args.probe is not None:
        timing = probe_speed(cfg, df, args.probe)
        args.out.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        return 0
    pool = pd.read_parquet(args.pool) if args.pool else None
    parts = run_folds_llm(cfg, df, args.folds, pool, args.save_dir)
    np.savez(args.out, **{f"fold{k}": proba for k, proba in parts.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
