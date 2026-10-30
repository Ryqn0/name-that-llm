"""Tests for the pool of generated passages (P7 Task 2).

The pool only ever TRAINS (design §3): every score is measured on our own passages, held
out by prompt. These tests pin the safeguards that make that true: only open licenses,
labels mapped to our 18 or dropped, the competition's shape (flattened, 30 to 320 words),
copies of our passages and conflicting labels removed, and each pool passage matched to the
training prompt it's about, so a fold never trains on its own validation topics.
"""

import pandas as pd

from name_that_llm.pool import (
    PoolSource,
    build_pool,
    calibrate_match_threshold,
    clean_pool,
    dedupe,
    license_allowed,
    load_pool_sources,
    match_prompts,
    normalize_label,
    pool_for_fold,
    source_counts,
)


def test_license_filter_keeps_only_open_licenses():
    allowed = [
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "apache-2.0",
        "MIT",
        # Kaggle's long display names, as its API reports most licenses:
        "Attribution 4.0 International (CC BY 4.0)",
        "Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
        "CC0: Public Domain",
        "Apache 2.0",
    ]
    refused = [
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-ND-4.0",
        "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
        "ODbL-1.0",
        "unknown",
        "other",
        "",
    ]
    assert all(license_allowed(name) for name in allowed)
    assert not any(license_allowed(name) for name in refused)


def test_normalize_label_maps_vendor_prefixes_and_drops_unknown():
    assert normalize_label("openai/gpt-5.5") == "gpt-5.5"
    assert normalize_label(" Claude-Opus-5 ") == "claude-opus-5"
    assert normalize_label("z-ai/glm-5.3-flash") == "glm-5.3-flash"
    assert normalize_label("gpt-4o") is None


def test_clean_pool_flattens_and_filters_length():
    def words(n):
        return " ".join(["word"] * n)

    raw = pd.DataFrame(
        {"text": [f"# Title\n\n- {words(40)}", words(10), words(400)], "model": ["gpt-5.5"] * 3}
    )
    out = clean_pool(raw, "src")
    assert len(out) == 1 and not out.text[0].startswith("#") and out.source[0] == "src"


def test_dedupe_removes_train_copies_and_conflicting_labels(fake_train):
    t0 = fake_train.text[0]
    pool = pd.DataFrame(
        {
            "text": [t0, "x y z " * 15, "x y z " * 15, "a b c " * 15, "a b c " * 15],
            "model": ["gpt-5.5", "kimi-k3", "kimi-k3", "kimi-k3", "gpt-5.5"],
            "source": "s",
        }
    )
    out = dedupe(pool, fake_train.text)
    assert list(out.text) == ["x y z " * 15]  # a train copy, and a label conflict, are gone


def test_match_prompts_finds_the_same_topic(fake_train):
    thr = calibrate_match_threshold(fake_train.text, fake_train.prompt_id)
    same = "topic2 words about subject 2. marker5 again, rephrased — style5"
    other = "bananas oranges apples pears plums cherries"
    got = match_prompts([same, other], fake_train.text, fake_train.prompt_id, thr)
    assert got[0] == "prompt2" and got[1] is None


def test_calibrated_threshold_matches_most_training_passages(fake_train):
    thr = calibrate_match_threshold(fake_train.text, fake_train.prompt_id)
    got = match_prompts(fake_train.text, fake_train.text, fake_train.prompt_id, thr)
    assert (got == fake_train.prompt_id.to_numpy()).mean() >= 0.95


def test_pool_for_fold_excludes_validation_prompts():
    pool = pd.DataFrame(
        {
            "text": list("abcdef"),
            "model": "gpt-5.5",
            "source": "s",
            "prompt_id": ["p1", "p2", None, "p1", None, "p3"],
        }
    )
    out = pool_for_fold(pool, {"p1"}, max_rows=None, seed=0)
    assert set(out.text) == {"b", "c", "e", "f"}
    assert len(pool_for_fold(pool, {"p1"}, max_rows=2, seed=0)) == 2


def test_shipped_pool_sources_load():
    sources = load_pool_sources()
    assert any(s.keep for s in sources)
    assert all(s.slug.count("/") == 1 for s in sources)


def test_build_pool_credits_each_row_to_its_original_source(fake_train, tmp_path):
    # A merged pool names each row's original dataset in a column (source_column).
    folder = tmp_path / "owner__fake-pool"
    folder.mkdir()
    sentence = "Sleep matters because the brain sorts the day's memories at night. "
    pd.DataFrame(
        {
            "text": [sentence * 4, sentence * 5, sentence * 6],
            "model": ["kimi-k3", "gpt-5.5", "qwen3.8-max"],
            "source_dataset": ["a/one", "b/two", "a/one"],
        }
    ).to_parquet(folder / "pool.parquet")
    sources = [
        PoolSource(
            slug="owner/fake-pool",
            files="*.parquet",
            text="text",
            model="model",
            source_column="source_dataset",
        ),
        PoolSource(slug="owner/absent", files="*.csv", text="t", model="m", keep=False),
    ]
    out = build_pool(sources, tmp_path, fake_train)
    assert list(out.columns) == ["text", "model", "source", "prompt_id"]
    assert sorted(out.source) == ["a/one", "a/one", "b/two"]


def test_build_pool_keeps_only_included_original_sources(fake_train, tmp_path):
    # P7 Task 3 keeps a LIST of the merged pool's original datasets (an allowlist: a source
    # added in a later version stays out until it's checked). Copies and label conflicts
    # are still judged across EVERY source: "b/two" labels a's second text differently, so
    # that text goes too, although "b/two" itself isn't kept.
    folder = tmp_path / "owner__fake-pool"
    folder.mkdir()
    sentence = "Sleep matters because the brain sorts the day's memories at night. "
    pd.DataFrame(
        {
            "text": [sentence * 4, sentence * 5, sentence * 6, sentence * 5],
            "model": ["kimi-k3", "gpt-5.5", "qwen3.8-max", "gemini-3.7-flash"],
            "source_dataset": ["a/one", "a/one", "b/two", "b/two"],
        }
    ).to_parquet(folder / "pool.parquet")
    source = PoolSource(
        slug="owner/fake-pool",
        files="*.parquet",
        text="text",
        model="model",
        source_column="source_dataset",
        include=["a/one"],
    )
    pool = build_pool([source], tmp_path, fake_train)
    assert list(pool.source) == ["a/one"]
    assert list(pool.model) == ["kimi-k3"]


def test_exclude_drops_a_source_even_with_every_source_included(fake_train, tmp_path):
    # P8 (spec §3): the whole clean pool trains (include=None), minus the sources its
    # maintainer dropped since our pinned copy. A dropped source must stay out even though
    # our pinned copy still holds it, and even if a release re-adds it under the same name.
    folder = tmp_path / "owner__fake-pool"
    folder.mkdir()
    a = "Sleep matters because the brain sorts the day's memories at night. "
    b = "Tides follow the moon, whose pull raises the oceans on either side. "
    pd.DataFrame(
        {
            "text": [a * 4, a * 5, b * 4, b * 5],
            "model": ["kimi-k3", "gpt-5.5", "qwen3.8-max", "gemini-3.7-flash"],
            "source_dataset": ["a/one", "a/one", "b/two", "b/two"],
        }
    ).to_parquet(folder / "pool.parquet")
    source = PoolSource(
        slug="owner/fake-pool",
        files="*.parquet",
        text="text",
        model="model",
        source_column="source_dataset",
        include=None,
        exclude=["b/two"],
    )
    pool = build_pool([source], tmp_path, fake_train)
    assert set(pool["source"]) == {"a/one"}


def test_source_counts():
    # Logged in every run (spec §5.2): a source that silently shrinks or vanishes upstream
    # shows up in the counts at once, as it didn't in P7.
    assert source_counts(pd.DataFrame({"source": ["b", "a", "b"]})) == {"a": 1, "b": 2}
