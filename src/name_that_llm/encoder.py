"""Fine-tuned transformer encoders (P5, model roadmap M4): DeBERTa-v3 and ModernBERT.

The P3b model kept Qwen FROZEN: it read hidden states, and only a logistic regression on
top learned anything. Fine-tuning instead updates EVERY weight of a pretrained encoder for
our 18 authors, so the network itself learns which habits of style tell them apart.

What this module provides:
  - load_encoder: a pretrained encoder with an 18-way classification head
  - train_encoder: the fine-tuning loop, hand-written (ADR 0003)
  - predict_proba: probabilities, columns in LABELS order
  - fit_predict_fold / run_cv_encoder: the same prompt-grouped CV as every other model, so
    the OOF predictions can be compared and blended with theirs
  - find_fold_dirs / predict_proba_folds: the offline submission averages the 5 fold models

torch and transformers are imported INSIDE the functions (ADR 0002). Kaggle's GPU image
provides them, and the text-only offline path must never need them.
"""

import gc
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.config import EncoderConfig
from name_that_llm.cv import make_folds
from name_that_llm.labels import LABELS
from name_that_llm.lm import _dtype_kwargs
from name_that_llm.metrics import score
from name_that_llm.train import CVResult, _result


def load_encoder(name_or_path: str | Path, device: str | None = None):
    """A sequence classifier with 18 outputs, plus its tokenizer: (model, tokenizer).

    Two kinds of checkpoints come in:
      - a BASE model from the Hub (e.g. "microsoft/deberta-v3-base"). It has no head for
        our task yet, so we ask for a new, randomly initialized 18-way head.
      - one of OUR fine-tuned fold models. Its config already names our 18 labels, so it's
        loaded exactly as saved: if its order ever differed from LABELS, predict_proba
        refuses it rather than silently relabeling the columns.
    """
    import torch
    import transformers

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # A local folder (a saved fold model) must never trigger a Hub lookup: offline there's
    # no internet, and the attempt would only waste time before failing.
    local = Path(name_or_path).exists()
    config = transformers.AutoConfig.from_pretrained(name_or_path, local_files_only=local)
    fine_tuned = set(config.id2label.values()) == set(LABELS)
    head = (
        {}
        if fine_tuned
        else {
            "num_labels": len(LABELS),
            "id2label": dict(enumerate(LABELS)),
            "label2id": {label: i for i, label in enumerate(LABELS)},
        }
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path, local_files_only=local)
    # float32 weights. On CUDA, autocast runs the heavy math in float16 while the weights and
    # the optimizer state stay float32 ("mixed precision"), which keeps training stable.
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        name_or_path, local_files_only=local, **head, **_dtype_kwargs(torch.float32)
    )
    return model.to(device), tokenizer


def train_encoder(
    model, tokenizer, texts: Sequence[str], y: np.ndarray, cfg: EncoderConfig
) -> list[float]:
    """Fine-tune `model` in place on (texts, y); returns the mean training loss per epoch.

    The recipe, step by step:
      - AdamW: Adam with weight decay applied correctly; the standard for transformers.
      - Learning rate: a linear warmup from 0 (big random updates to the new head would
        otherwise wreck the pretrained weights in the first steps), then a linear decay
        to 0 by the end, so the last steps only polish.
      - Mixed precision (CUDA only): float16 math is ~2x faster on a T4. The GradScaler
        multiplies the loss before backward, so tiny gradients don't round to zero in
        float16, then divides them back before the update.
      - Gradient clipping at norm 1.0: one outlier batch can't take a huge step.
      - A non-finite loss stops everything with FloatingPointError. A NaN would otherwise
        spread through the weights and quietly turn every later prediction into garbage.
    The number of epochs is FIXED (cfg.epochs): no early stopping on the validation fold,
    which would leak that fold into its own OOF score.
    """
    import torch

    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    labels = torch.tensor([LABELS.index(v) for v in y])  # an unknown label raises ValueError
    total = cfg.epochs * math.ceil(len(texts) / cfg.batch_size)
    warmup = int(cfg.warmup_ratio * total)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_factor(step: int) -> float:
        """The share of cfg.lr at `step`: ramps 0 -> 1 during warmup, then 1 -> 0."""
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    rng = np.random.default_rng(cfg.seed)  # a seeded shuffle: reruns see the same batches

    model.train()  # dropout on
    losses = []
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(texts))
        batch_losses = []
        for start in range(0, len(texts), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            # Padding to the longest passage IN THE BATCH, not to max_length: shorter
            # batches cost less. Truncation keeps outliers within the position embeddings.
            enc = tokenizer(
                [str(texts[i]) for i in idx],
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**enc, labels=labels[idx].to(device)).loss  # cross-entropy
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}, batch {start // cfg.batch_size}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # back to true gradients, so clipping sees them
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))
    return losses


def predict_proba(
    model, tokenizer, texts: Sequence[str], batch_size: int = 32, max_length: int = 512
) -> np.ndarray:
    """(n, 18) probabilities, columns in LABELS order (refuses any other order)."""
    import torch

    order = [model.config.id2label.get(i) for i in range(len(LABELS))]
    if order != LABELS:
        raise ValueError(f"the model's labels aren't in LABELS order: {order}")
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    model.eval()  # dropout off: predictions must be deterministic
    out = np.zeros((len(texts), len(LABELS)))
    with torch.inference_mode():  # no gradients: faster and much less memory
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start : start + batch_size]]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            # softmax in float32: float16 rounds small probabilities to 0
            out[start : start + len(batch)] = logits.float().softmax(dim=-1).cpu().numpy()
    return out


def _release_memory() -> None:
    """Hand freed GPU memory back before the next model loads (a T4 has 16 GB).

    Callers `del` their model first. Deleting inside this function couldn't free it:
    the caller's own variable would still hold the model.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_predict_fold(
    cfg: EncoderConfig,
    texts: Sequence[str],
    y: np.ndarray,
    folds: np.ndarray,
    k: int,
    save_dir: Path | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Fine-tune a FRESH model on every fold but `k`; predict fold `k`'s rows.

    Returns (probabilities for the rows where folds == k, epoch losses). Each call starts
    again from cfg.base_model. Continuing from an earlier fold's weights would leak that
    fold's validation prompts into this one. With save_dir, the trained model and its
    tokenizer are saved to save_dir / f"fold{k}" for the offline submission.
    """
    import torch

    torch.manual_seed(cfg.seed)  # the new head's random init and the dropout: reproducible
    model, tokenizer = load_encoder(cfg.base_model)
    train_rows, valid_rows = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
    y = np.asarray(y)
    losses = train_encoder(model, tokenizer, [texts[i] for i in train_rows], y[train_rows], cfg)
    proba = predict_proba(
        model,
        tokenizer,
        [texts[i] for i in valid_rows],
        batch_size=2 * cfg.batch_size,  # no gradients to store: bigger batches fit
        max_length=cfg.max_length,
    )
    if save_dir is not None:
        path = Path(save_dir) / f"fold{k}"
        model.half().save_pretrained(path)  # float16 halves the files; plenty for inference
        tokenizer.save_pretrained(path)
    del model
    _release_memory()
    return proba, losses


def run_cv_encoder(cfg: EncoderConfig, df: pd.DataFrame, save_dir: Path | None = None) -> CVResult:
    """Prompt-grouped CV of the encoder: an OOF prediction for every row, like run_cv.

    Prints one line per fold: GPU runs take minutes per fold, and the downloadable Kaggle
    log only keeps what's printed.
    """
    texts, y = df["text"].tolist(), df["model"].to_numpy()
    folds = make_folds(df, cfg.n_splits, cfg.seed).to_numpy()
    labels = np.asarray(LABELS)
    oof = np.zeros((len(df), len(LABELS)))
    for k in np.unique(folds):
        start = time.time()
        proba, losses = fit_predict_fold(cfg, texts, y, folds, int(k), save_dir)
        oof[folds == k] = proba
        fold_score = score(y[folds == k], labels[proba.argmax(axis=1)])
        print(
            f"{cfg.name} seed {cfg.seed} fold {k}: CV {fold_score:.4f} | epoch losses "
            + ", ".join(f"{loss:.3f}" for loss in losses)
            + f" | {(time.time() - start) / 60:.1f} min",
            flush=True,
        )
    return _result(oof, y, folds)


def find_fold_dirs(
    root: Path = Path("/kaggle/input"), name: str = "encoder", n_folds: int = 5
) -> list[Path]:
    """The saved fold models under `root` (folders */<name>/fold<k>/), in fold order.

    Searched, never hard-coded: Kaggle mounts an attached notebook's output under a path
    that has changed before. Anything other than exactly n_folds models is an error.
    Averaging 4 of 5 models would silently change the submission.
    """
    root = Path(root)
    dirs = sorted(
        {
            p.parent
            for p in root.rglob("config.json")
            if re.fullmatch(r"fold\d+", p.parent.name) and p.parent.parent.name == name
        },
        key=lambda d: int(d.name.removeprefix("fold")),
    )
    if len(dirs) != n_folds:
        raise FileNotFoundError(
            f"expected {n_folds} fold models under {root} (folders */{name}/fold<k>/ with a "
            f"config.json), found {len(dirs)}: {[str(d) for d in dirs]}. "
            "Is the encoder-final notebook attached?"
        )
    return dirs


def predict_proba_folds(
    fold_dirs: Sequence[Path], texts: Sequence[str], batch_size: int = 32, max_length: int = 512
) -> np.ndarray:
    """The mean of the fold models' probabilities: the submission's encoder prediction.

    These are the same 5 models that produced the OOF predictions, so blend weights
    chosen on the OOF apply to this average too. One model at a time is in memory.
    """
    if not fold_dirs:
        raise ValueError("no fold models given")
    total = np.zeros((len(texts), len(LABELS)))
    for d in fold_dirs:
        model, tokenizer = load_encoder(d)
        total += predict_proba(model, tokenizer, texts, batch_size, max_length)
        del model
        _release_memory()
    return total / len(fold_dirs)
