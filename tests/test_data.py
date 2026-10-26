import pandas as pd
import pytest
from pandera.errors import SchemaError

from name_that_llm.data import (
    LAB_OF,
    LABELS,
    find_data_dir,
    flatten_text,
    load_test,
    load_train,
    normalize_text,
)

EXPECTED = {
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
}


def test_labels_are_the_18_competition_labels_sorted():
    assert sorted(EXPECTED) == LABELS and len(LABELS) == 18


def test_every_label_has_one_of_six_labs():
    assert set(LAB_OF) == EXPECTED
    assert set(LAB_OF.values()) == {"anthropic", "google", "moonshot", "openai", "qwen", "zai"}


def test_normalize_text_keeps_style_marks():
    raw = "  He said “hi” — it’s fine… ok then  "
    assert normalize_text(raw) == "He said “hi” — it’s fine… ok then"


def test_normalize_text_composes_accents():
    assert normalize_text("café") == "café"


# A chatbot answer as a visitor would paste it: every kind of markdown the app must undo.
MARKDOWN = (
    "# Title\n\nSome **bold** and *italic* text.\n\n- first point\n- second point\n\n"
    "1. step one\n2) step two\n\n> a quote\n\n```python\nx = 1\n```\n\n"
    "A [link](https://example.com) and `code`.\n\n| a | b |\n|---|---|\n| 1 | 2 |"
)


def test_flatten_text_turns_markdown_into_one_paragraph():
    # Hand-derived: the markup goes, the words stay, single *emphasis* survives (the host
    # kept it in 14 training passages), and everything joins into one paragraph.
    assert flatten_text(MARKDOWN) == (
        "Title Some bold and *italic* text. first point second point step one step two "
        "a quote x = 1 A link and code. a b 1 2"
    )


def test_flatten_text_leaves_training_style_text_alone():
    # One paragraph, single *emphasis*, a number ending a sentence, an em dash, an inline
    # "1." : all of these occur in the real training passages and must survive unchanged.
    text = "It costs $5. Then *this* matters — as it did in 2023. Done (see 1. above)."
    assert flatten_text(text) == text


def test_flatten_text_collapses_whitespace_and_normalizes():
    assert flatten_text("  a\n\nb\t c   d  ") == "a b c d"
    assert flatten_text("caf" + "é") == "café"  # NFC, as normalize_text does


def test_load_train_keeps_ids_as_strings(fake_data_dir):
    train = load_train(fake_data_dir)
    assert train.shape == (108, 4) and train["id"].iloc[0] == "000000"


def test_load_train_rejects_unknown_label(fake_data_dir):
    df = pd.read_csv(fake_data_dir / "train.csv", dtype=str)
    df.loc[0, "model"] = "gpt-9"
    df.to_csv(fake_data_dir / "train.csv", index=False)
    with pytest.raises(SchemaError):
        load_train(fake_data_dir)


def test_load_train_rejects_duplicate_prompt_model_pair(fake_data_dir):
    df = pd.read_csv(fake_data_dir / "train.csv", dtype=str)
    df.loc[1, "model"] = df.loc[0, "model"]  # same prompt answered twice by one model
    df.to_csv(fake_data_dir / "train.csv", index=False)
    with pytest.raises(SchemaError):
        load_train(fake_data_dir)


def test_load_test_survives_missing_text(fake_data_dir):
    test = load_test(fake_data_dir)
    assert test["id"].tolist() == ["000001", "a1b2c3", "999999"]
    assert test["text"].iloc[1] == ""


def test_find_data_dir_prefers_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NTL_DATA_DIR", str(tmp_path))
    assert find_data_dir() == tmp_path


def test_find_data_dir_error_lists_what_it_tried(tmp_path, monkeypatch):
    monkeypatch.delenv("NTL_DATA_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="nope1"):
        find_data_dir(candidates=[tmp_path / "nope1", tmp_path / "nope2"])
