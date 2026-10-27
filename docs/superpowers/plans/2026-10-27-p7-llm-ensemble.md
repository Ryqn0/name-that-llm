# P7: A Fine-Tuned LLM and a Bigger Ensemble — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the model clearly more accurate. Fine-tune Qwen3-8B-Base with LoRA as an
18-way classifier, train it on our data plus a leak-free public pool of generated passages,
and blend it with the current members. Promote it through the usual gate, and submit it
offline.

**Architecture:**
- **New, torch-free modules:**
  - `balance.py`: Sinkhorn class balancing;
  - `pool.py`: licenses, cleaning, dedupe, prompt matching, and the per-fold pool rule.
- **`train.py`** gains `run_cv_pool`, so `lr` can measure the pool on CPU.
- **New `llm.py`** mirrors `encoder.py`: loading (4-bit, LoRA), per-fold training, saving
  and reloading adapters, and fold averaging. It reuses `encoder.train_encoder` and
  `encoder.predict_proba`. It also has a CLI, so a notebook can train two folds at once, one
  per T4.
- **Three GPU notebooks** (first run, exploration, final), then the offline submission,
  extended.

**Tech Stack:**
- the existing stack;
- peft (the LoRA layers) and bitsandbytes (4-bit NF4 weights): pip-installed in the
  internet-on notebooks, and shipped as wheels in the lib dataset for the offline
  submission;
- Qwen3-8B-Base (Apache 2.0) from Kaggle Models;
- public Kaggle datasets of generated passages.

**Spec:** `docs/superpowers/specs/2026-10-27-p7-llm-ensemble-design.md`. It builds on the
project design (`2026-10-23-name-that-llm-design.md`): §6.6 folds and OOF, §6.9 the A/B gate,
§11 risks. Out of this plan: P6, and the public demo.

## Global Constraints

The P5 constraints still apply:
- the labels, the data rules, the folds, privacy and account-visible OKs;
- the Ship procedure, with merges only inside `if gh pr checks --watch; then gh pr merge ...; fi`;
- the Avast prefix for local Kaggle CLI calls: `unset SSLKEYLOGFILE;
  export REQUESTS_CA_BUNDLE="$HOME/.certs/combined-ca.pem"`.

Also:

- **The pool's three safeguards** (spec §3), verbatim:
  1. "only free, public Kaggle datasets with an explicit open license on their page (such
     as CC0, CC BY, CC BY-SA, Apache 2.0 or MIT; "unknown" or "other" is excluded); we never
     pay to generate data";
  2. "the pool only ever trains: every score is measured on our own passages, held out by
     prompt, after removing pool passages that duplicate them";
  3. "one of the two final submissions stays pool-free".
- **Our own models only** (spec §3):
  - no fork of the public notebooks, and no other participant's trained weights or adapters;
  - no paid compute, and no API generation;
  - Llama only after the user accepts its license on Kaggle, and only if it's ever chosen.
- **Never print a passage**, ours or the pool's: only counts, scores and column names.
- **Pool files stay out of git.** They live in `data/external/` and `data/pool/`, under the
  gitignored `data/`. They are never published by us: notebooks attach the public source
  datasets and rebuild the pool from them.
- **Base model:** Kaggle Models `qwen-lm/qwen-3/transformers/8b-base/1`, found with
  `lm.find_model_dir("8b-base")`, never a hard-coded path.
- **GPU budget:** ≤ 90 GPU-hours for the plan, aiming to finish by 2026-11-20 (spec §6).
  - Push with `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4` and
    `"enable_gpu": true`.
  - The ledger records each run's minutes.
  - If the next run would cross the budget, stop and ask the user.
- **Two folds at once:** every GPU CV runs two `python -m name_that_llm.llm` processes, one
  per T4 (`CUDA_VISIBLE_DEVICES=0` and `=1`), then merges their outputs.
- **Approvals, asked in the session that uses them:** each Kaggle push, each dataset creation
  or version, each submission, and the tag that publishes the lib dataset.
- **torch, transformers, peft and bitsandbytes are never package dependencies** (ADR 0002).
  They're imported inside functions; `balance.py`, `pool.py` and `train.py` import none of
  them.
- **Seeds:** choices on seed 7; reports on 42, 43 and 44; the gate on seed 42.

## Review Focus

1. **A pool passage about a validation prompt's topic**, but not a copy of any passage,
   must not train that fold's model. Test: `test_match_prompts_finds_the_same_topic`
   (Task 2) and `test_run_cv_pool_trains_only_on_allowed_pool_rows` (Task 3).
2. **A source that names the models differently** (`openai/gpt-5.5`, `Claude-Opus-5`) must
   map to our labels, and unknown models must be dropped, never guessed. Test:
   `test_normalize_label_maps_vendor_prefixes_and_drops_unknown` (Task 2).
3. **An adapter loaded onto a different cut of the base model** must be refused, not return
   garbage. Test: `test_adapter_on_a_different_cut_is_refused` (Task 4).
4. **Empty or very long text** in the LLM must still give 18 probabilities. Test:
   `test_llm_handles_empty_and_long_text` (Task 4).
5. **The offline submission without its 4-bit library** must fail at the first cell, with a
   clear message, not after an hour. Step: the import check in Task 8's notebook.

---

### Task 1: Sinkhorn balancing, and its verdict on our folds

**Files:**
- Create: `src/name_that_llm/balance.py`, `tests/test_balance.py`,
  `scripts/sinkhorn_check.py`
- Modify: `tests/test_train.py` (the import guard: add `name_that_llm.balance`)

**Interfaces:**
- Produces: `sinkhorn_balance(proba: np.ndarray, alpha: float = 1.0, n_iter: int = 200,
  tol: float = 1e-10) -> np.ndarray`.
  - The output is (n, K), with rows summing to 1.
  - `alpha` must be in [0, 1], else ValueError.
  - At `alpha=1`, every column sums to n/K.

- [ ] **Step 1: Write the failing tests.**

```python
def test_full_balancing_equalizes_class_mass():
    p = np.random.default_rng(0).dirichlet(np.r_[8.0, np.ones(17)], size=180)
    b = sinkhorn_balance(p, alpha=1.0)
    np.testing.assert_allclose(b.sum(axis=0), 10.0, atol=1e-3)
    np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-9)


def test_alpha_zero_changes_nothing():
    p = np.random.default_rng(1).dirichlet(np.ones(18), size=50)
    np.testing.assert_allclose(sinkhorn_balance(p, alpha=0.0), p, atol=1e-12)


def test_partial_balancing_moves_part_way():
    p = np.random.default_rng(2).dirichlet(np.r_[8.0, np.ones(17)], size=180)
    spread = [np.ptp(sinkhorn_balance(p, a).sum(axis=0)) for a in (0.0, 0.5, 1.0)]
    assert spread[0] > spread[1] > spread[2]


def test_balancing_moves_picks_away_from_an_over_predicted_class():
    # 18 true classes, 10 rows each; class 0 gets an extra 0.3 everywhere.
    y = np.repeat(np.arange(18), 10)
    p = np.full((180, 18), 0.02)
    p[np.arange(180), y] += 0.3
    p[:, 0] += 0.35
    p /= p.sum(axis=1, keepdims=True)
    before = (p.argmax(1) == y).mean()
    after = (sinkhorn_balance(p).argmax(1) == y).mean()
    assert after > before


def test_zero_columns_stay_finite():
    p = np.random.default_rng(3).dirichlet(np.ones(18), size=40)
    p[:, 5] = 0
    p /= p.sum(axis=1, keepdims=True)
    assert np.isfinite(sinkhorn_balance(p)).all()
```

- [ ] **Step 2: Run** `uv run --all-extras pytest -q tests/test_balance.py`. Expected: FAIL
  (ModuleNotFoundError).
- [ ] **Step 3: Implement `sinkhorn_balance`.**
  - Clip `proba` at 1e-12 into `P`, and start with column scales `c = 1`.
  - Repeat: `M = rownorm(P * c)`; then `c *= (n/K) / M.sum(0)`. Stop when
    `max|M.sum(0) - n/K| < tol * n`, or after `n_iter` rounds.
  - Return `rownorm(P * c**alpha)`.

  It's comment-heavy: why the column scales are the balancing, and what `alpha` trades off.
- [ ] **Step 4: Run the tests.** Expected: 5 passed. Add `name_that_llm.balance` to the
  import guard and run `tests/test_train.py`. Expected: PASS.
- [ ] **Step 5: Get the encoder's out-of-fold files** (read-only, small):
  `kaggle kernels output ryqn00/name-that-llm-encoder-final -p data/kaggle-outputs/encoder-final --file-pattern "oof-encoder-seed"`.
  Expected: `oof-encoder-seed{7,42,43,44}.npy`.
- [ ] **Step 6: Write `scripts/sinkhorn_check.py`.**
  1. Rebuild the champion's OOF per seed, as the recon did: `lr` with `run_cv`, the LM head
     with `run_cv_features` on the saved 4B features, the encoder from its file, blended
     with `ensemble.yaml`'s weights.
  2. Choose `alpha` from {0, 0.25, 0.5, 0.75, 1} on seed 7.
  3. Report Δ (`alpha*` minus 0) on seeds 42, 43 and 44, and the seed-42
     `prompt_bootstrap_ci`.
  4. **The unbalanced check:** 50 seed-42 subsets. Each keeps half the rows of 6 random
     models (seeds 0..49). Print the mean and min Δ.
  5. **The verdict:** KEEP if the 42–44 mean Δ > 0, the CI lower bound > 0, *and* the mean
     unbalanced Δ ≥ 0. Otherwise `alpha = 0`.
- [ ] **Step 7: Run it** (`uv run --all-extras python -I -X utf8 scripts/sinkhorn_check.py`).
  Expected: a table and one verdict line. Ledger the numbers, and add IDEAS I-027
  (Sinkhorn) with the result.
- [ ] **Step 8: Commit and ship** (branch `p7-01-sinkhorn`). The commit also carries the
  approved spec and this plan, as in P4 and P5.

---

### Task 2: The pool: licenses, cleaning, dedupe, prompt matching

**Files:**
- Create: `src/name_that_llm/pool.py`, `src/name_that_llm/configs/pool.yaml`,
  `scripts/build_pool.py`, `tests/test_pool.py`
- Modify: `tests/test_train.py` (the import guard: add `name_that_llm.pool`)

**Interfaces:**
- Produces:
  - `PoolSource`: pydantic, `extra="forbid"`. Fields: `slug: str`, `files: str` (a glob
    inside the dataset), `text: str` and `model: str` (column names), `keep: bool = True`.
  - `load_pool_sources(name: str = "pool") -> list[PoolSource]`: reads the `sources` key of
    the packaged YAML.
  - `license_allowed(name: str) -> bool`.
  - `normalize_label(raw: str) -> str | None`.
  - `clean_pool(df: pd.DataFrame, source: str) -> pd.DataFrame`: in, the columns `text` and
    `model`; out, `text`, `model` and `source`.
  - `dedupe(pool: pd.DataFrame, train_texts: Sequence[str]) -> pd.DataFrame`.
  - `calibrate_match_threshold(train_texts, train_prompt_ids, quantile: float = 0.01) -> float`.
  - `match_prompts(pool_texts, train_texts, train_prompt_ids, threshold: float) -> np.ndarray`:
    an object array holding a train `prompt_id`, or None.
  - `pool_for_fold(pool: pd.DataFrame, valid_prompt_ids: Collection[str],
    max_rows: int | None, seed: int) -> pd.DataFrame`.
  - `build_pool(sources: Sequence[PoolSource], root: Path, train: pd.DataFrame) -> pd.DataFrame`:
    columns `text`, `model`, `source` and `prompt_id` (null when unmatched).

- [ ] **Step 1: Inspect the candidates (read-only).** The candidates:
  - `seyitkaangunes/ntl-extra-api-passages`
  - `abdullah1afify/ntl-generations`
  - `dophuchung/ntl-proxy-generations`
  - `lucasboesen/ntl-free-proxy-generations`
  - `danielandresmiranda/ntl-openrouter-glm-kimi`
  - `prvsiyan/prvsiyans-extended-name-that-llm-dataset`
  - the merged pool of forum thread 752766 (find its slug with
    `kaggle datasets list -s "name that llm"`).

  For each: `kaggle datasets metadata -d <slug> -p <tmp>` (the license) and
  `kaggle datasets files <slug>`. Read only the column names. Write `configs/pool.yaml`: the
  allowed-license sources, with their file glob and columns. Ledger every candidate, its
  license and its verdict.
- [ ] **Step 2: Write the failing tests** (fake data: 6 prompts × 18 models, texts
  `topic{p} words about subject {p}. marker{m} …`).

```python
def test_license_filter_keeps_only_open_licenses():
    for ok in ["CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "apache-2.0", "MIT"]:
        assert license_allowed(ok)
    for bad in [
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "CC-BY-ND-4.0",
        "ODbL-1.0",
        "unknown",
        "other",
        "",
    ]:
        assert not license_allowed(bad)


def test_normalize_label_maps_vendor_prefixes_and_drops_unknown():
    assert normalize_label("openai/gpt-5.5") == "gpt-5.5"
    assert normalize_label(" Claude-Opus-5 ") == "claude-opus-5"
    assert normalize_label("z-ai/glm-5.3-flash") == "glm-5.3-flash"
    assert normalize_label("gpt-4o") is None


def test_clean_pool_flattens_and_filters_length():
    words = lambda n: " ".join(["word"] * n)
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
```

- [ ] **Step 3: Run** `uv run --all-extras pytest -q tests/test_pool.py`. Expected: FAIL.
- [ ] **Step 4: Implement `pool.py`.**
  - **Licenses:** lowercase the name. Accept `cc0*`, `cc-by-<digit>*`, `cc-by-sa-*`,
    `apache-2.0` and `mit`.
  - **Labels:** strip, lowercase, drop a `vendor/` prefix, turn spaces and underscores into
    `-`, then require membership in LABELS.
  - **Cleaning:** `flatten_text`, keep 30 to 320 words. That's a margin around the training
    passages' 36 to 274.
  - **Dedupe:** the key is `normalize_text(flatten_text(t))`. Drop pool rows whose key is a
    training key, and drop every row of a key that carries two labels. Keep one row per
    remaining key.
  - **Matching:** a content-word TF-IDF (`stop_words="english"`, as `grouping.py`) fit on
    the train texts, and an L2-normalized centroid per prompt. Each pool passage takes the
    argmax prompt if its cosine is ≥ the threshold.
  - **The threshold:** each training passage's cosine to its own prompt's centroid computed
    *without itself* (leave-one-out); the threshold is that distribution's `quantile`.
  - **`pool_for_fold`:** keep the rows whose `prompt_id` is null or outside the set, then
    sample `max_rows` without replacement (`seed`). Keep the order stable.
  - **`build_pool`:** read each kept source's files under `root` (parquet, csv or jsonl),
    then `clean_pool`, concatenate, `dedupe`, and `match_prompts` at the calibrated
    threshold.
- [ ] **Step 5: Run the tests.** Expected: 7 passed. Add `name_that_llm.pool` to the import
  guard. Expected: PASS.
- [ ] **Step 6: Write and run `scripts/build_pool.py --download --root data/external --out data/pool/pool.parquet`.**
  - With `--download`, it re-checks each source's license (refusing a disallowed one) and
    downloads it into its own folder: `kaggle datasets download <slug> -p <root>/<owner>__<name> --unzip`.
  - It reads the files with pandas only (untrusted data), then calls `build_pool`.

  Expected per source: rows read, after cleaning, after dedupe, the matched share, and the
  label counts. Ledger them.
- [ ] **Step 7: Commit and ship** (branch `p7-02-pool`).

---

### Task 3: Does the pool help? Measured with `lr` on CPU

**Files:**
- Modify: `src/name_that_llm/train.py`, `tests/test_train.py`,
  `src/name_that_llm/configs/pool.yaml` (the `keep` flags)
- Create: `scripts/pool_check.py`

**Interfaces:**
- Consumes: `pool_for_fold`, `load_pool_sources` (Task 2).
- Produces: `run_cv_pool(cfg: ExperimentConfig, df: pd.DataFrame, pool: pd.DataFrame | None = None,
  max_pool: int | None = None) -> CVResult`. Fold k trains on its training rows plus
  `pool_for_fold(pool, prompts of fold k, max_pool, seed=cfg.seed + k)`, and predicts only
  the real rows of fold k.

- [ ] **Step 1: Write the failing tests.**

```python
def test_run_cv_pool_without_pool_matches_run_cv(fake_train):
    cfg = load_config("lr")
    np.testing.assert_allclose(run_cv_pool(cfg, fake_train).oof, run_cv(cfg, fake_train).oof)


def test_run_cv_pool_trains_only_on_allowed_pool_rows(fake_train, monkeypatch):
    seen = []  # the texts each fold's model was fit on

    class Spy:  # a stand-in model: records its training texts, predicts uniform
        def fit(self, X, y):
            seen.append(set(X))
            self.classes_ = np.array(LABELS)
            return self

        def predict_proba(self, X):
            return np.full((len(X), 18), 1 / 18)

    monkeypatch.setattr(train, "build_model", lambda cfg: Spy())
    pool = pd.DataFrame(
        {
            "text": [f"pooltag{p}" for p in range(6)],
            "model": LABELS[0],
            "source": "s",
            "prompt_id": [f"prompt{p}" for p in range(6)],
        }
    )
    run_cv_pool(load_config("lr"), fake_train, pool)
    folds = make_folds(fake_train, 5, 42)
    for k, texts in enumerate(seen):
        held_out = set(fake_train.prompt_id[folds == k])
        assert not {f"pooltag{p[6:]}" for p in held_out} & texts
```

- [ ] **Step 2: Run them.** Expected: FAIL (`run_cv_pool` not defined).
- [ ] **Step 3: Implement `run_cv_pool`.** Generalize `_oof` with an optional
  `extra_for_fold(k) -> (texts, labels)`, so `run_cv` stays unchanged. Columns stay aligned
  with LABELS, as `_oof` already does.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Write and run `scripts/pool_check.py`** (seed 42):
  - `lr` alone;
  - `lr` plus each source alone, capped at 6,000 rows;
  - `lr` plus all kept sources, capped at 30,000.

  For each: CV, Δ against `lr` alone, and the bootstrap CI. Set `keep: false` for a source
  with Δ < −0.005. Then print the all-kept verdict: the pool helps `lr` if the CI lower
  bound > 0. Ledger the table. The LLM gets its own test of the pool in Task 6, whatever
  `lr` says.
- [ ] **Step 6: Commit and ship** (branch `p7-03-pool-check`).

---

### Task 4: The LLM classifier: LoRA on 4-bit Qwen, CPU-tested

**Files:**
- Modify:
  - `src/name_that_llm/config.py`: `LLMConfig`;
  - `src/name_that_llm/encoder.py`: `train_encoder` optimizes and clips only the parameters
    with `requires_grad`, and `find_fold_dirs` gains `marker: str = "config.json"`;
  - `tests/test_config.py`, and the import guard in `tests/test_train.py` (add
    `name_that_llm.llm`);
  - `.github/workflows/ci.yml`: the torch job adds `uv pip install peft` and runs
    `tests/test_llm.py`.
- Create: `src/name_that_llm/llm.py`, `src/name_that_llm/configs/llm.yaml`,
  `tests/test_llm.py`

**Interfaces:**
- Consumes:
  - `train_encoder`, `predict_proba`, `_release_memory` and `find_fold_dirs` from `encoder.py`;
  - `make_folds`, `CVResult` and `_result`;
  - `pool_for_fold` (Task 2).
- Produces:
  - `LLMConfig` (pydantic, `extra="forbid"`). Fields, with defaults: `name: str`,
    `base_model: str`, `n_layers: int | None = None`, `load_in_4bit: bool = True`,
    `lora_r: int = 16`, `lora_alpha: int = 32`, `lora_dropout: float = 0.05`,
    `max_length: int = 512`, `batch_size: int = 8`, `lr: float = 1e-4`, `epochs: int = 2`,
    `warmup_ratio: float = 0.05`, `weight_decay: float = 0.0`, `pool_size: int = 0`,
    `n_splits: int = 5`, `seed: int = 42`.
  - `load_llm_classifier(cfg: LLMConfig, adapter_dir: Path | None = None, device: str | None = None) -> tuple[model, tokenizer]`.
  - `fold_training_set(texts, y, prompt_ids, folds, k, pool: pd.DataFrame | None, cfg) -> tuple[list[str], np.ndarray]`.
  - `run_folds_llm(cfg, df, folds_to_run: Sequence[int], pool=None, save_dir: Path | None = None) -> dict[int, np.ndarray]`.
  - `merge_folds(cfg, df, parts: Sequence[dict[int, np.ndarray]]) -> CVResult`: raises
    ValueError on a missing or duplicate fold.
  - `run_cv_llm(cfg, df, pool=None, save_dir=None) -> CVResult`.
  - `save_adapter(model, tokenizer, cfg, path: Path) -> None`.
  - `predict_proba_adapters(cfg, adapter_dirs: Sequence[Path], texts, batch_size: int = 16) -> np.ndarray`.
  - `main(argv) -> int`, the CLI: `python -m name_that_llm.llm --config PATH --base-model DIR
    --seed S --folds 0 2 4 [--pool PATH] [--save-dir DIR] --out PATH.npz`. It writes one
    array per fold, `fold{k}`.

- [ ] **Step 1: Write the failing tests.** All use `tiny_qwen` (12 random layers), with
  `importorskip` on torch, transformers and peft. The base config is
  `CFG = LLMConfig(name="t", base_model=str(tiny_qwen), load_in_4bit=False, n_layers=4,
  lora_r=4, lora_alpha=8, max_length=64, batch_size=8, lr=5e-3, epochs=10, seed=0)`.
  - `test_lora_trains_only_adapters_and_head`: the trainable names are all `lora_` or
    `score`; one base weight is bit-identical after `train_encoder`.
  - `test_llm_keeps_only_the_first_layers`: the model has 4 decoder layers.
  - `test_llm_learns_the_markers`: `run_cv_llm(CFG, fake_train).score > 0.5`.
  - `test_llm_handles_empty_and_long_text`: `""` and 2,000 words give 2 × 18 rows summing
    to 1.
  - `test_fold_training_set_follows_the_pool_rule`: no pool row of fold k's prompts, and at
    most `pool_size` pool rows.
  - `test_split_folds_merge_to_the_single_run`: `main` on folds {0,1,2} and on {3,4}, then
    `merge_folds`, equals `run_cv_llm(...).oof` (atol 1e-6).
  - `test_saved_adapter_predicts_like_the_trained_model`: `predict_proba_adapters([dir])`
    matches the in-memory model (atol 1e-5).
  - `test_adapter_on_a_different_cut_is_refused`: an adapter saved with `n_layers=4`, loaded
    with `n_layers=6`, raises ValueError.
  - In `tests/test_config.py`: `test_shipped_llm_config_loads`.
- [ ] **Step 2: Run** `uv run --all-extras --with torch --with transformers --with peft pytest -q tests/test_llm.py`.
  Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **The cut:** `load_llm_classifier` cuts depth like `load_lm_layers` (set
    `num_hidden_layers`, slice `layer_types`).
  - **The head:** 18 outputs with LABELS' `id2label`; if the tokenizer has no pad token,
    use its eos token, and set `model.config.pad_token_id`.
  - **4-bit:** on CUDA with `load_in_4bit`: `BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True)`, then
    `prepare_model_for_kbit_training(use_gradient_checkpointing=True)`.
  - **LoRA:** `LoraConfig(task_type="SEQ_CLS", r, lora_alpha, lora_dropout,
    target_modules="all-linear", modules_to_save=["score"])`.
  - **Saving:** `save_adapter` writes the peft adapter, the tokenizer, and `llm_config.json`
    (the config plus the base's `model_type` and `hidden_size`). Loading compares those,
    and `n_layers`.
  - **Training per fold:** `torch.manual_seed(cfg.seed + k)` before building each fold's
    model, so split runs equal one run.
  - **Prediction:** `predict_proba_adapters` loads the base once and swaps adapters
    (`load_adapter` / `set_adapter`).
  - **Comments** explain LoRA, NF4 and gradient checkpointing.
- [ ] **Step 4: Run the tests**, then the whole suite with torch, transformers and peft.
  Expected: all pass, `test_encoder` included.
- [ ] **Step 5: Write `configs/llm.yaml`**: `name: llm`, `base_model: Qwen/Qwen3-8B-Base`
  (the notebooks override it with the mounted path), and the defaults above.
- [ ] **Step 6: Commit and ship** (branch `p7-04-llm`). CI's torch job must show
  `test_llm.py` passed, not skipped.

---

### Task 5: The first GPU run: pool-free Qwen3-8B, seed 7 (`notebooks/10-llm-first`)

**Files:**
- Create: `notebooks/10-llm-first/llm_first.ipynb` and its `kernel-metadata.json`:
  - id `ryqn00/name-that-llm-llm-first`, GPU, internet on;
  - `model_sources: ["qwen-lm/qwen-3/transformers/8b-base/1"]`;
  - `kernel_sources: ["ryqn00/name-that-llm-lm-features", "ryqn00/name-that-llm-encoder-final"]`.
- Create: `scripts/smoke_llm_notebook.py`

**Interfaces:**
- Consumes: Task 4's CLI and `llm.yaml`; the 4B features; `oof-encoder-seed7.npy`.
- Produces: `oof-llm-seed7.npy`, `timing.json`, and the printed seed-7 comparison.

- [ ] **Step 0: Fix W&B for all three P7 notebooks at once.** A Kaggle Secret can only be
  attached to a notebook that already exists. So:
  1. With one OK from the user, push stubs of `10-llm-first`, `11-llm-explore` and
     `12-llm-final`: CPU only, one cell that prints the W&B status.
  2. The user attaches `WANDB_API_KEY` to all three in the Kaggle editor (Add-ons → Secrets).
  3. A stub re-run prints `W&B tracking is ON`. That is the design's goal 6, met.
- [ ] **Step 1: Write the notebook.**
  1. **Install:** `name-that-llm[track] @ git+…@<main SHA>`, plus `peft bitsandbytes`. Print
     the torch, transformers, peft, bitsandbytes and CUDA versions, and both GPUs.
  2. **W&B:** print `W&B tracking is ON` or `OFF (<reason>)`. Training goes on either way.
  3. **Timing probe:** one fold's training on 256 passages, 1 epoch. Print seconds per
     passage, the peak memory, and the projected minutes for a 5-fold CV on two GPUs. If
     that's over 6 hours, skip step 4 and print why.
  4. **Seed-7 CV, pool-free:** two CLI processes (folds 0, 2, 4 on GPU 0; folds 1, 3 on
     GPU 1), then `merge_folds`. Save `oof-llm-seed7.npy`.
  5. **The comparison on seed 7** (printed):
     - `lr`, the LM head, the encoder and the current ensemble;
     - the LLM alone;
     - `best_blend_weights` over all four (an in-sample guide, not a promotion).
  6. Findings, filled in after the run.
- [ ] **Step 2: Smoke-test locally:** `scripts/smoke_llm_notebook.py` runs the notebook's
  code with `tiny_qwen`, fake data and fake features, `load_in_4bit=False`. Expected: it
  runs end to end and writes the outputs.
- [ ] **Step 3: Ask the user's OK, then push.**
  - Poll, and download only the small outputs.
  - Ledger the minutes. The run's budget is ≤ 8 GPU-hours.
  - Expected: `W&B tracking is ON`, and the run appears in the W&B project.
- [ ] **Step 4: Decide.** Continue to Task 6 if either holds:
  - the LLM alone ≥ the encoder alone on seed 7 (0.344 there in P5);
  - the four-member blend beats the three-member one by ≥ 0.01.

  Otherwise, stop and ask: that's a finding, as P3a's was.
- [ ] **Step 5: Fill in the Findings and ship** (branch `p7-05-llm-first`).

---

### Task 6: Explore depth, the pool and the step size (`notebooks/11-llm-explore`)

**Files:**
- Create: `notebooks/11-llm-explore/llm_explore.ipynb` and its `kernel-metadata.json`:
  - id `ryqn00/name-that-llm-llm-explore`, GPU, internet on;
  - `dataset_sources`: the `keep: true` slugs of `pool.yaml`;
  - `model_sources`: 8b-base.
- Modify: `src/name_that_llm/configs/llm.yaml` (the chosen settings, with a comment block,
  as in `encoder.yaml`)

**Protocol:** proxy folds 0 and 1 of seed 7, one per GPU. Stages, one change at a time:
1. `n_layers` ∈ {None, 24, 16}, pool-free;
2. at the best depth, `max_length` ∈ {512, 320}, pool-free (shorter is faster, and cuts
   only the longest passages);
3. at the best of those, `pool_size` ∈ {0, 5,000, 15,000}, with the pool built in the
   notebook by `build_pool` from the attached sources;
4. only if budget remains: `lr` ∈ {1e-4, 2e-4} × `epochs` ∈ {1, 2}.

A second LLM member (Qwen3-14B-Base) is tried on the same folds only if 8B clearly paid off
and the budget allows. Otherwise it's recorded as IDEAS I-028.

The best mean of folds 0 and 1 wins. Within 0.005, the cheaper setting wins. Before each
stage, the notebook prints its projected GPU time, and skips the stage if it would cross
**20 GPU-hours** for this notebook.

- [ ] **Step 1: Write the notebook:** install, W&B, build the pool (print its counts), the
  three stages, a results table (csv), the chosen config as YAML, and Findings.
- [ ] **Step 2: Smoke-test locally** (tiny model, a fake pool).
- [ ] **Step 3: Ask the user's OK, then push.** Poll, and ledger the minutes.
- [ ] **Step 4: Copy the chosen settings into `configs/llm.yaml`** with their evidence,
  including whether the pool helped the LLM. Ship (branch `p7-06-llm-explore`).

---

### Task 7: The final run, the four-member ensemble, and the gate (`notebooks/12-llm-final`)

**Files:**
- Modify:
  - `src/name_that_llm/config.py`: `EnsembleConfig` gains `llm: str | None = None` (the LLM
    member) and `sinkhorn_alpha: float = 0.0`, in [0, 1]. `weights` takes 3 or 4 values,
    and the count must match whether `llm` is set. It also gains `log_blend: bool = False`.
  - `src/name_that_llm/blend.py`: `blend(probas, weights, log: bool = False)`. The log form
    is a weighted geometric mean, renormalized (I-015).
  - `tests/test_blend.py`, `tests/test_config.py`.
- Create: `notebooks/12-llm-final/llm_final.ipynb` and its `kernel-metadata.json`
  (id `ryqn00/name-that-llm-llm-final`; the pool sources, 8b-base, lm-features and
  encoder-final).
- Create, if promoted: `src/name_that_llm/configs/ensemble.yaml` (the new champion) and
  `configs/ensemble_poolfree.yaml` (the hedge).

**Interfaces:**
- Produces:
  - `/kaggle/working/llm/fold{0..4}` (seed 42, with the pool);
  - `/kaggle/working/llm_poolfree/fold{0..4}` (seed 42, pool-free);
  - the OOF files, `ensemble_report.csv`, and the YAML files.

- [ ] **Step 1: Write the failing tests:**
  - `test_log_blend_is_a_normalized_geometric_mean`: for two members with weights 0.5 and
    0.5, each row equals `sqrt(p1 * p2)` renormalized; rows sum to 1.
  - `test_ensemble_config_accepts_three_or_four_members`: 3 weights without `llm`, and 4
    with `llm="llm"`, both load.
  - `test_ensemble_config_rejects_a_weight_count_mismatch`: 4 weights without `llm` raise a
    ValidationError, and so do 3 weights with `llm`.
  - `test_ensemble_config_rejects_alpha_outside_0_1`: `sinkhorn_alpha=1.5` raises a
    ValidationError.
  - `test_shipped_ensemble_config_still_loads`: today's `ensemble.yaml` loads, with
    `llm is None`, `sinkhorn_alpha == 0` and `log_blend is False`.
- [ ] **Step 2: Run them** (expected: FAIL), **implement**, **run them** (expected: PASS),
  then **ship** (branch `p7-07a-ensemble-config`).
- [ ] **Step 3: Write the notebook.**
  1. **The LLM with the pool:** seeds 7, 42, 43 and 44, saving seed 42's adapters.
  2. **The LLM pool-free:** seeds 7 and 42, saving seed 42's adapters.
  3. **The other members per seed:** `lr`, the LM head, and the encoder's OOF files.
  4. **Choices on seed 7:** the weights (`best_blend_weights`), the blend form (probability
     or log), and `sinkhorn_alpha` (Task 1's verdict, rechecked).
  5. **The report on 42, 43 and 44:** each member, the current champion, the new ensemble,
     and the pool-free ensemble, plus a mean row.
  6. **The gate on seed 42:** new minus current champion, with `PROMOTE = lower bound > 0`.
     The same for the pool-free ensemble.
  7. **Outputs and Findings.**

  The run's budget is ≤ 30 GPU-hours. It projects first, and stops if it would cross that.
- [ ] **Step 4: Smoke-test locally, then ask the user's OK and push.** Poll, and ledger the
  minutes.
- [ ] **Step 5: Decide.**
  - **Promoted:** ship the YAML files, with
    `test_shipped_ensemble_config_loads` updated.
  - **Not promoted:** record it, and skip Task 8's submission (its docs still run).

  Ship (branch `p7-07-llm-final`).

---

### Task 8: Submit the new ensemble, release v0.9.0

**Files:**
- Modify:
  - `src/name_that_llm/submit.py`: `predict_submission_ensemble(..., p_llm: np.ndarray | None = None)`.
    It applies `log_blend`, then `sinkhorn_balance(alpha=ens_cfg.sinkhorn_alpha)`.
  - `tests/test_submit.py`.
  - `notebooks/90-submit/submit.ipynb` and its `kernel-metadata.json` (add 8b-base and
    `ryqn00/name-that-llm-llm-final`).
  - `.github/workflows/release-lib.yml`: download the `peft` and `bitsandbytes` wheels
    (`--only-binary=:all: --platform manylinux_2_24_x86_64 --python-version 3.13`), plus any
    dependency the Kaggle image lacks.
  - `pyproject.toml`: version 0.9.0.

- [ ] **Step 1: Write the failing tests** (fake data, fake LM features, as the P5
  submission tests do):
  - `test_ensemble_submission_with_an_llm_member`: with a 4-member config, the submission
    has one row per test id and only LABELS values. A one-hot `p_llm` with weight 0.97
    decides every row.
  - `test_ensemble_submission_applies_sinkhorn`: when `p_llm` favors one label everywhere,
    `sinkhorn_alpha=1` gives more distinct predicted labels than `sinkhorn_alpha=0`.
  - `test_llm_probabilities_of_the_wrong_shape_are_refused`: a (n_test, 17) `p_llm` raises
    ValueError.
- [ ] **Step 2: Run them** (expected: FAIL), **implement**, **run them** (expected: PASS),
  and **update the release workflow**. Ship (branch `p7-08-submit`).
- [ ] **Step 3: Ask the user's OK, then tag `v0.9.0`.** The `release-lib` workflow
  publishes the new lib dataset version.
- [ ] **Step 4: Extend the submission notebook.**
  - **Its first cell imports peft and bitsandbytes offline**, and stops with "the lib
    dataset lacks <package>" if one fails (Review Focus 5).
  - **The LLM step:** `find_fold_dirs(name="llm", marker="adapter_config.json")`, then
    `predict_proba_adapters`, with the base from `find_model_dir("8b-base")`.
  - **The runtime check** on the 4,000-row stand-in must project under 7.5 hours (9 is the
    limit).
- [ ] **Step 5: Ask the user's OK, then push and submit** (in the foreground, right after
  the OK). It's an A/B against 0.410.
  - If Sinkhorn was kept, a second submission without it confirms it on the leaderboard
    (still within 5 a day).
  - Ledger the scores. Name the two final-pick candidates: the best overall, and the best
    pool-free (`ensemble_poolfree`, or else the current champion).
- [ ] **Step 6: Ship the results** (branch `p7-08b-results`): Context's table and the
  notebook's Findings.

---

### Task 9: Record and ship

**Files:**
- Create:
  - `docs/adr/0006-generated-data-and-lora.md`: the pool policy (my call, with its three
    safeguards), and LoRA on 4-bit weights inside our own loop;
  - `docs/journal/p7-llm-ensemble.md`, written from the ledger before the workspace is
    deleted.
- Modify: `DIARY.md` (session 8), `Context.md` (status, champion, table, next steps),
  `IDEAS.md`, and `docs/journal/README.md`.
  - IDEAS: I-003 used, I-015 and I-019 with their results, I-027 Sinkhorn.

- [ ] **Step 1: Write them.** Then run `uv run --all-extras pytest -q` and
  `uv run pre-commit run --all-files`. Expected: PASS.
- [ ] **Step 2: Commit and ship** (branch `p7-09-results`). Then the final review, the
  journal, deleting the workspace, and finishing the branch, as in P4.
