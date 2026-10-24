import pandas as pd
import pytest
from pandera.errors import SchemaError

from name_that_llm.data import LAB_OF, LABELS, find_data_dir, load_test, load_train, normalize_text

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
