"""Language-model features: mean-pooled hidden states from an open LLM (idea I-002, P3b).

A language model reads a passage token by token and builds, at every layer, a vector per
token (a "hidden state"). Averaging those vectors over the passage gives one fixed-size
vector per layer: a numeric summary of the passage as that layer sees it. A linear model on
top of those vectors (train.run_cv_features) can then learn which of the 18 LLMs wrote it.

Why look at several layers: the last layers of a language model specialize in predicting
the NEXT token, so they keep "what comes next" and discard "how this was written". Earlier
layers still carry surface habits (word choice, rhythm, punctuation context), which is
exactly the authorship signal. We extract every layer once on the GPU, then pick the best
one on CPU with cross-validation (notebooks 04 and 05).

torch and transformers are imported INSIDE the functions that need them. They are not
package dependencies: Kaggle's GPU image provides them (ADR 0002), and the text-only
offline path must never need them. `find_model_dir` and `select_layers` need only numpy.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def find_model_dir(variation: str, root: Path = Path("/kaggle/input")) -> Path:
    """The unique model folder under `root` for `variation` (e.g. "1.7b-base").

    A Hugging Face model folder is recognized by its `config.json`. We search instead of
    hard-coding a path because Kaggle's mount layout has already surprised us twice (the
    competition data and our wheel dataset both appeared at unexpected paths).
    """
    root = Path(root)
    matches = sorted({p.parent for p in root.rglob("config.json") if variation in p.parent.parts})
    if not matches:
        raise FileNotFoundError(
            f"no model folder for variation {variation!r} under {root} "
            "(looked for a folder containing config.json); is the model attached?"
        )
    if len(matches) > 1:
        raise ValueError(f"several model folders match {variation!r}: {matches}")
    return matches[0]


def load_lm(model_dir: Path, device: str | None = None, dtype: str = "auto"):
    """Load a model and its tokenizer from a LOCAL folder (the offline submission has no internet).

    dtype "auto" means float16 on a GPU (half the memory, and fast on the T4's tensor cores)
    and float32 on CPU (CPUs gain nothing from float16). Returns (model, tokenizer).
    """
    import torch
    import transformers

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "auto": torch.float16 if device.startswith("cuda") else torch.float32,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    # local_files_only=True: never try to reach the Hugging Face Hub. Offline, a network
    # attempt would only waste time before failing.
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # see pooled_hidden_states for why

    model = transformers.AutoModel.from_pretrained(
        model_dir, local_files_only=True, **_dtype_kwargs(torch_dtype)
    )
    return model.to(device).eval(), tokenizer


def _dtype_kwargs(torch_dtype) -> dict:
    """The keyword for the weights' dtype in `from_pretrained`: `dtype`, or `torch_dtype`.

    transformers 4.56 renamed `torch_dtype` to `dtype`, and Kaggle's image decides which
    version applies. Shared by load_lm and encoder.load_encoder.
    """
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: torch_dtype}


def pooled_hidden_states(
    texts: Sequence[str],
    model,
    tokenizer,
    layers: Sequence[int] | None = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    """Mean-pooled hidden states: float32 array of shape (len(layers), len(texts), hidden).

    `layers` indexes the model's `hidden_states`: 0 is the embedding layer, k is block k.
    None means every layer. Rows come back in the input order.
    """
    import torch

    n_states = model.config.num_hidden_layers + 1  # embeddings + one per block
    layers = list(range(n_states)) if layers is None else list(layers)
    if any(not 0 <= k < n_states for k in layers):
        raise ValueError(f"layers must be in [0, {n_states - 1}], got {layers}")
    device = next(model.parameters()).device
    out = np.zeros((len(layers), len(texts), model.config.hidden_size), dtype=np.float32)

    with torch.inference_mode():  # no gradients: faster and much less memory
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start : start + batch_size]]
            # RIGHT padding: with causal attention a token only sees tokens BEFORE it, so
            # padding placed after a passage can't change the passage's own hidden states.
            # Left padding would shift every token's position and make a passage's vector
            # depend on which other passages share its batch (test_pooling_is_batch_invariant).
            # Truncation at max_length: our passages are <= ~400 tokens, but the hidden test
            # could contain anything.
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                padding_side="right",
            )
            if enc["input_ids"].shape[1] == 0:
                continue  # every text in this batch is empty: no tokens, so zeros stay
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).float()  # 1 = real token, 0 = padding
            # Masked mean, computed in float32: some hidden dimensions of LLMs take very large
            # values, and summing ~400 of them in float16 can overflow to infinity.
            # clamp(min=1): an empty text has 0 tokens; its vector becomes zeros, not NaN.
            n_tokens = mask.sum(dim=1).clamp(min=1)
            for i, k in enumerate(layers):
                pooled = (hidden[k].float() * mask).sum(dim=1) / n_tokens
                out[i, start : start + len(batch)] = pooled.cpu().numpy()
    return out


def select_layers(features: np.ndarray, layer: int, window: int = 1) -> np.ndarray:
    """Mean of `features[layer : layer + window]`: one (n_texts, hidden) matrix.

    A window larger than 1 averages neighboring layers, which smooths layer-to-layer noise.
    Whether that helps is for the tuning to decide (Task 6).
    """
    if layer < 0 or window < 1 or layer + window > len(features):
        raise ValueError(f"layer {layer} with window {window} doesn't fit {len(features)} layers")
    return features[layer : layer + window].astype(np.float32).mean(axis=0)
