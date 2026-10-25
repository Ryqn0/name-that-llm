"""Shared pytest fixtures.

Every test runs on FAKE data generated here, never on the real competition data. The
rules forbid redistributing that data, and this repo is public, so the tests must be
able to run anywhere (your PC, GitHub Actions) without it.
"""

import re

import pandas as pd
import pytest

from name_that_llm.data import LABELS, load_train


@pytest.fixture
def fake_data_dir(tmp_path):
    """A tiny FAKE competition folder: 6 prompts x 18 models, plus 3 test rows.

    Each model gets its own marker word, so classifiers have something to learn. The ids
    look numeric on purpose ("000000"), to catch pandas dropping leading zeros.
    """
    rows = [
        {
            "id": f"{p:03d}{m:03d}",
            "prompt_id": f"prompt{p}",
            "text": f"topic{p} words about subject {p}. marker{m} marker{m} — style{m}",
            "model": label,
        }
        for p in range(6)
        for m, label in enumerate(LABELS)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        {"id": ["000001", "a1b2c3", "999999"], "text": ["marker3 text", None, "marker5 text"]}
    ).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


@pytest.fixture
def fake_train(fake_data_dir):
    return load_train(fake_data_dir)


def make_tiny_encoder(texts, path, arch="deberta"):
    """A tiny RANDOM classifier (2 blocks, hidden 32) plus a word-level tokenizer built from
    `texts`, saved to `path`. Like real encoders, the tokenizer wraps each text in
    [CLS] ... [SEP], so even an empty text has 2 tokens.

    Built offline from a config, so no download is needed. torch and transformers come
    through importorskip: only the tests that use this skip where they aren't installed.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    # One token per word or punctuation mark of the fake data, so marker{m} gets its own id
    # and a model can learn "marker7 -> the 8th label".
    words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t)})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", *words]
    word_level = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    word_level.pre_tokenizer = pre_tokenizers.Whitespace()
    word_level.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=word_level,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    common = dict(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    if arch == "deberta":
        config = transformers.DebertaV2Config(**common)
    else:
        config = transformers.ModernBertConfig(
            **common, bos_token_id=2, eos_token_id=3, cls_token_id=2, sep_token_id=3
        )
    torch.manual_seed(0)
    transformers.AutoModelForSequenceClassification.from_config(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def tiny_encoder(fake_train, tmp_path_factory):
    """The folder of a tiny DeBERTa-style classifier whose vocabulary covers fake_train."""
    return make_tiny_encoder(fake_train["text"].tolist(), tmp_path_factory.mktemp("deberta"))


@pytest.fixture
def tiny_modernbert(fake_train, tmp_path_factory):
    """The same, with ModernBERT's architecture (its special tokens differ)."""
    return make_tiny_encoder(
        fake_train["text"].tolist(), tmp_path_factory.mktemp("modernbert"), arch="modernbert"
    )
