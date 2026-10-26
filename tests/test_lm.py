import numpy as np
import pytest

from name_that_llm.lm import (
    find_model_dir,
    load_lm,
    load_lm_layers,
    pooled_hidden_states,
    select_layers,
)

# A 5-word vocabulary is enough to exercise tokenization, padding and pooling.
VOCAB = ["[PAD]", "[UNK]", "a", "b", "c"]


@pytest.fixture(scope="module")
def tiny_lm():
    """A tiny RANDOM Qwen 3 model (2 blocks, hidden size 16) plus a word-level tokenizer.

    Built entirely offline from a config, so no download is needed. The model tests are
    skipped where torch/transformers aren't installed (CI): Kaggle's GPU image provides them
    in production (ADR 0002).
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    word_level = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    word_level.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=word_level, pad_token="[PAD]", unk_token="[UNK]"
    )
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=len(VOCAB),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    return transformers.AutoModel.from_config(config).eval(), tokenizer


def test_pooled_hidden_states_shape_and_order(tiny_lm):
    model, tok = tiny_lm
    texts = ["a b c", "b c", "c a b a"]
    feats = pooled_hidden_states(texts, model, tok, batch_size=2)
    assert feats.shape == (3, 3, 16)  # embeddings + 2 blocks
    alone = pooled_hidden_states(texts[2:], model, tok)
    np.testing.assert_allclose(feats[:, 2], alone[:, 0], atol=1e-5)


def test_pooling_is_batch_invariant(tiny_lm):
    model, tok = tiny_lm
    short = pooled_hidden_states(["a b"], model, tok)
    padded = pooled_hidden_states(["a b", "c a b c a b c a"], model, tok, batch_size=2)
    np.testing.assert_allclose(short[:, 0], padded[:, 0], atol=1e-5)


def test_pooled_hidden_states_handles_empty_text(tiny_lm):
    model, tok = tiny_lm
    feats = pooled_hidden_states(["", "a b"], model, tok)
    assert np.isfinite(feats).all() and np.allclose(feats[:, 0], 0.0)
    # a batch made ONLY of empty texts has zero tokens: no forward pass, just zeros
    alone = pooled_hidden_states([""], model, tok)
    assert alone.shape == (3, 1, 16) and np.allclose(alone, 0.0)


def test_pooled_hidden_states_truncates_long_text(tiny_lm):
    model, tok = tiny_lm
    feats = pooled_hidden_states(["a " * 5000], model, tok, max_length=32)
    assert feats.shape == (3, 1, 16) and np.isfinite(feats).all()


def test_pooled_hidden_states_selects_layers(tiny_lm):
    model, tok = tiny_lm
    every = pooled_hidden_states(["a b c"], model, tok)
    some = pooled_hidden_states(["a b c"], model, tok, layers=[2, 0])
    np.testing.assert_allclose(some, every[[2, 0]], atol=1e-6)


def test_load_lm_round_trips_a_saved_model(tiny_lm, tmp_path):
    from name_that_llm.lm import load_lm

    model, tok = tiny_lm
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    loaded, loaded_tok = load_lm(tmp_path, device="cpu")
    assert loaded_tok.padding_side == "right" and not loaded.training
    np.testing.assert_allclose(
        pooled_hidden_states(["a b c"], loaded, loaded_tok),
        pooled_hidden_states(["a b c"], model, tok),
        atol=1e-5,
    )


def test_find_model_dir_finds_the_unique_variation(tmp_path):
    for variation in ("1.7b-base", "0.6b-base"):
        folder = tmp_path / "models/qwen-3/transformers" / variation / "1"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
    found = find_model_dir("1.7b-base", tmp_path)
    assert found.name == "1" and "1.7b-base" in found.parts


def test_find_model_dir_error_names_variation_and_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="4b-base"):
        find_model_dir("4b-base", tmp_path)


def test_select_layers_averages_a_window():
    feats = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    np.testing.assert_allclose(select_layers(feats, 1, 2), (feats[1] + feats[2]) / 2)
    with pytest.raises(ValueError):
        select_layers(feats, 3, 2)


def test_load_lm_layers_matches_the_full_model_on_kept_layers(tiny_qwen, fake_train):
    # The app loads only Qwen's first 10 blocks: the LM head reads layers 6-9, so the other
    # 26 blocks would cost memory and time for nothing. Layers 6-9 must be EXACTLY what the
    # full model gives, or the LM head (trained on full-model features) would be fed
    # different inputs than it learned from.
    texts = fake_train["text"].tolist()[:5]
    full, tokenizer = load_lm(tiny_qwen, device="cpu")
    cut, cut_tokenizer = load_lm_layers(tiny_qwen, n_layers=10)
    assert len(cut.layers) == 10
    a = pooled_hidden_states(texts, full, tokenizer, layers=range(6, 10), batch_size=2)
    b = pooled_hidden_states(texts, cut, cut_tokenizer, layers=range(6, 10), batch_size=2)
    np.testing.assert_array_equal(a, b)


def test_load_lm_layers_with_nine_blocks_changes_layer_nine(tiny_qwen, fake_train):
    # The trap: the LAST hidden state goes through the model's final normalization, so a
    # 9-block cut no longer returns the plain output of block 9. Keep one block more than
    # the last layer you read.
    texts = fake_train["text"].tolist()[:5]
    full, tokenizer = load_lm(tiny_qwen, device="cpu")
    cut9, _ = load_lm_layers(tiny_qwen, n_layers=9)
    a = pooled_hidden_states(texts, full, tokenizer, layers=[9], batch_size=2)
    b = pooled_hidden_states(texts, cut9, tokenizer, layers=[9], batch_size=2)
    assert np.abs(a - b).max() > 1e-3
