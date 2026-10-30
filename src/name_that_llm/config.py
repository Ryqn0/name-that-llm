"""Experiment configuration: every setting that defines a run, in one typed object.

Why a config object instead of loose function arguments? Each experiment then IS its config.
The config gets logged with the results (W&B, from Task 8), so any score can be reproduced
later from exactly the settings that produced it. Pydantic validates the values when they're
loaded, so a typo in a YAML file fails immediately instead of silently using a default.
"""

from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt


class ExperimentConfig(BaseModel):
    # extra="forbid": an unknown key (e.g. "c" instead of "C") is an error, not ignored.
    model_config = ConfigDict(extra="forbid")

    name: str
    model: Literal["mnb", "cnb", "lr", "svm", "nbsvm"]

    # Features: word n-grams (vocabulary habits) + character n-grams (punctuation,
    # spacing, word endings, casing). Both kinds are TF-IDF weighted.
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (1, 5)
    min_df: int = 2  # ignore n-grams seen in fewer documents (mostly noise)
    max_features: int | None = 200_000  # per vectorizer; caps memory and training time

    # Model hyperparameters (each model reads only the ones it uses):
    C: float = 1.0  # inverse regularization strength: bigger = trust the data more
    alpha: float = 0.1  # Naive Bayes smoothing: avoids zero probabilities for unseen n-grams

    # Validation (backtesting): grouped folds on prompt_id, fixed seed (spec §6.6).
    n_splits: int = 5
    seed: int = 42


class BlendConfig(BaseModel):
    """The TF-IDF text model blended with a logistic regression on LM hidden states (P3b).

    Final probabilities = (1 - weight) * text model + weight * LM model.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    text_config: str = "lr"  # name of the ExperimentConfig for the text side
    lm_variation: str  # Kaggle Models variation of Qwen 3, e.g. "1.7b-base"
    layer: int  # hidden_states index: 0 = embeddings, k = transformer block k
    layer_window: int = 1  # average `layer_window` consecutive layers, starting at `layer`
    max_length: int = 512  # tokens per passage (ours are <= ~400)
    batch_size: int = 16
    C: float = 1.0  # regularization of the LM-side logistic regression
    weight: float = Field(ge=0.0, le=1.0)  # the LM's share of the blend
    n_splits: int = 5
    seed: int = 42


class EncoderConfig(BaseModel):
    """A transformer encoder fine-tuned end to end as an 18-way classifier (P5, M4).

    Unlike the LM features of P3b, where Qwen stayed frozen, fine-tuning updates EVERY
    weight of the network for our task. The defaults are common starting points for
    base-size encoders on small datasets. Optuna tunes lr, epochs and warmup_ratio.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    base_model: str  # Hugging Face Hub id, or a folder holding a saved model
    max_length: int = 512  # tokens per passage: ours fit, and longer ones get truncated
    batch_size: int = 16  # bounded by GPU memory: 16 GB on a T4
    lr: float = 2e-5  # peak learning rate: pretrained weights need small steps
    epochs: int = 3  # FIXED, never early-stopped on the validation fold (honest OOF)
    warmup_ratio: float = 0.1  # share of steps where the lr ramps up from 0
    weight_decay: float = 0.01  # mild L2-style shrinkage, the usual for AdamW fine-tuning
    # Pool passages a POOL MEMBER trains on (P8, pool_member.py); 0 means all of the honest
    # pool. Our P5 encoder ignores it: it trains on our passages, fold by fold.
    pool_size: int = 0
    n_splits: int = 5
    seed: int = 42


class LLMConfig(BaseModel):
    """A decoder LLM (Qwen 3) fine-tuned as an 18-way classifier with LoRA (P7, llm.py).

    Like P5's encoder, the network itself learns our 18 authors. But an 8-billion-weight
    model can't be fully fine-tuned on a 16 GB T4, so its weights load in 4 bits and stay
    frozen, and LoRA trains small updates beside them, plus the new head. llm.py explains
    each piece. The defaults are the usual QLoRA starting points.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    base_model: str  # Hugging Face Hub id, or a folder (the notebooks' mounted Kaggle Model)
    n_layers: int | None = None  # keep only the first n blocks (P3b: the middle ones carry style)
    load_in_4bit: bool = True  # NF4 weights on CUDA (needs bitsandbytes); CPU loads float32
    lora_r: int = 16  # the rank of each low-rank update, i.e. its capacity
    lora_alpha: int = 32  # the updates are scaled by lora_alpha / lora_r
    lora_dropout: float = 0.05
    max_length: int = 512  # tokens per passage: ours fit, and longer ones get truncated
    batch_size: int = 8
    lr: float = 1e-4  # LoRA trains far fewer weights than full fine-tuning: bigger steps
    epochs: int = 2  # FIXED, never early-stopped on the validation fold (honest OOF)
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    pool_size: int = 0  # pool passages added to each fold's training (0: pool-free)
    n_splits: int = 5
    seed: int = 42


class EnsembleConfig(BaseModel):
    """Three models mixed by a weighted mean of probabilities (P5).

    The two models of the BlendConfig `blend` (its text model and its LM head) plus the
    fine-tuned EncoderConfig `encoder`. The blend's own weight is ignored: all three
    weights are chosen together, on seed-7 folds (blend.best_blend_weights).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    blend: str = "lm_blend"  # the BlendConfig whose text model and LM head are reused
    encoder: str = "encoder"  # the EncoderConfig of the fine-tuned encoder
    # text model, LM head, encoder, in that order. blend() normalizes them to sum to 1.
    weights: tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat]
    # Pool passages the text model also trains on (I-029: 30,000 lifted the ensemble by
    # +0.049 on seed 42). 0: our passages only, as in P5. The LM head and the encoder never
    # see the pool.
    text_pool: NonNegativeInt = 0


def load_config(name_or_path: str | Path, model_cls: type[BaseModel] = ExperimentConfig):
    """Load a config by NAME (a file shipped inside the package) or by PATH (any YAML file).

    `model_cls` picks the schema: ExperimentConfig (text models, the default) or BlendConfig.

    The named configs live inside the package (`name_that_llm/configs/`), not at the repo
    root. That way the wheel used by the offline Kaggle notebook carries them along
    (ADR 0001).
    """
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} or path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        resource = files("name_that_llm") / "configs" / f"{name_or_path}.yaml"
        if not resource.is_file():
            available = sorted(p.name.removesuffix(".yaml") for p in resource.parent.iterdir())
            raise FileNotFoundError(f"no config named {name_or_path!r}; available: {available}")
        text = resource.read_text(encoding="utf-8")
    return model_cls(**yaml.safe_load(text))
