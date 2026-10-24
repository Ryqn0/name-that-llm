# P3b: Language-Model Features, then Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, stronger signal to the champion: mean-pooled hidden states from an
open Qwen 3 base model, extracted on Kaggle's GPU, with logistic regression on top. Blend
it with the TF-IDF `lr` model, submit it as an A/B test, then tune the blend with Optuna.

**Architecture:**
- **`lm.py`** extracts per-layer, mean-pooled hidden states. torch and transformers are
  imported lazily, so the text-only path never needs them.
- **`train.run_cv_features`** cross-validates a scaled logistic-regression head on any dense
  feature matrix, using the same prompt-grouped folds as `run_cv`.
- **`blend.py`** combines the two models' probabilities.
- **Kaggle notebooks are chained.** A GPU notebook writes the features. CPU notebooks read
  them as input to pick a layer (sweep) and to tune (Optuna). The offline submission
  notebook recomputes the features for train and test on GPU.

**Tech Stack:** existing stack, plus:
- torch and transformers, both preinstalled in Kaggle's GPU image. They are **not** package
  dependencies.
- Qwen 3 base checkpoints from Kaggle Models.
- Optuna, as a new optional extra `tune`.

**Spec:** `docs/superpowers/specs/2026-10-23-name-that-llm-design.md`. Relevant sections:
- §6.5: dense features, including LLM hidden states and early layers.
- §6.7: M5 (LLM features), pulled forward from P7 because of the evidence below.
- §6.9: the A/B promotion gate.
- §8: Optuna.

**Evidence, from 2026-10-24 spikes on real data with prompt-grouped CV (seed 42):**
- **The TF-IDF family is saturated.** Nine variants scored 0.307–0.323 against the
  champion's 0.3212: no cap, C = 16, longer n-grams, start/end markers, linear SVM with
  C = 0.5 or 0.1, `char_wb`, characters only.
- **Tricks based on prompt structure are dead** for this test layout (P3a diagnosis).
- **External data isn't usable:** the host hasn't ruled on API-generated data, the public
  corpus reportedly hurts (0.326 → 0.279 in a forum report), and paid generation is outside
  the design.
- **The forum claims about 0.39 CV for early Qwen layers.** That's unverified; this plan
  measures it ourselves.
- **Qwen 3 base checkpoints are on Kaggle Models** under the Apache 2.0 license:
  `qwen-lm/qwen-3/transformers/{0.6b-base,1.7b-base,4b-base,8b-base}/1`.

**Scope:** the user chose *"LM features, then tune"* on 2026-10-24.
- Out of this plan: the lab-then-model hierarchy (M3) and transformer fine-tuning (P5).
- Later idea I-016: `8b-base` split across 2×T4.

## Global Constraints

Plan 1's Global Constraints still apply: the labels, the data rules, the folds (seed 42),
privacy, account-visible OKs, the Ship procedure, tagging and the Avast prefix. Also:

- **Merges:** only inside `if gh pr checks --watch; then gh pr merge ...; fi`. This is the
  P3a lesson.
- **Models:** `qwen-lm/qwen-3/transformers/0.6b-base/1`, `1.7b-base/1` and `4b-base/1`,
  all Apache 2.0.
- **Locating models:** always find a model folder with `lm.find_model_dir`, never with a
  hard-coded `/kaggle/input/...` path. Kaggle's mount layout has surprised us twice.
- **GPU:** push GPU notebooks with `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`
  and `"enable_gpu": true`. The budget for this plan is **≤ 4 GPU-hours** out of about 30
  per week.
- **Runtime libraries:** torch and transformers come from Kaggle's image. The extraction
  notebook asserts `transformers >= 4.51`, the first release with Qwen 3.
  - Locally, the LM tests run with
    `uv run --all-extras --with torch --with transformers pytest tests/test_lm.py`.
  - In CI they're skipped, via `pytest.importorskip`.
- **Feature files:** `features-<variation>.npy`, float16, shape
  (layers + 1, n_texts, hidden). Index 0 is the embedding layer. Every features file is
  saved with `train_ids.npy`, and each reader asserts that the ids order matches
  `load_train()`.
- **Selection bias:** anything chosen by CV (layer, C, blend weight) is chosen on folds with
  **seed 7**. Final numbers are reported on seeds **42, 43 and 44**, plus the
  prompt-bootstrap A/B test (promote only if the CI lower bound is > 0).
- **Versioning:**
  - Task 2 bumps the version to 0.5.0 and tags `v0.5.0` (the notebooks pin it).
  - Task 5 tags `v0.6.0`.
  - Task 6, if promoted, tags `v0.7.0`.

## Review Focus

1. **Features misaligned with labels across chained notebooks:** rows silently shuffled
   between the GPU notebook and the CPU notebook. Guard: `train_ids.npy` plus an assert in
   every reader (Tasks 3, 4, 6), and `test_select_layers_*` (Task 1).
2. **Padding changes the features:** a text's vector depends on the other texts in its batch
   when padding is on the left or the mask is ignored. Test:
   `test_pooling_is_batch_invariant` (Task 1).
3. **Empty or very long passages:** an empty text must give zeros, not NaN, and a long text
   must be truncated at `max_length`. Tests:
   `test_pooled_hidden_states_handles_empty_text` and
   `test_pooled_hidden_states_truncates_long_text` (Task 1).
4. **The model folder is missing or ambiguous offline:** a clear error that names the
   variation and the search root. Tests: `test_find_model_dir_*` (Task 1). Loading always
   uses `local_files_only=True`.
5. **The offline text-only path must never need torch or optuna:** test
   `test_text_paths_never_import_torch_or_optuna` (Task 2, extended in Task 6).

---

### Task 1: Hidden-state extraction (`lm.py`)

**Files:**
- Create: `src/name_that_llm/lm.py`, `tests/test_lm.py`,
  `docs/adr/0002-lm-features-without-torch-dependency.md`

**Interfaces:**
- Produces:
  - `find_model_dir(variation: str, root: Path = Path("/kaggle/input")) -> Path`: the
    unique folder under `root` that contains `config.json` and has `variation` as one of
    its path parts. Raises `FileNotFoundError` naming the variation and root if there's
    none, and `ValueError` listing the matches if there are several.
  - `load_lm(model_dir: Path, device: str | None = None, dtype: str = "auto") -> tuple[model, tokenizer]`
    - Uses `AutoModel` and `AutoTokenizer` with `local_files_only=True`.
    - `"auto"` means float16 on CUDA, float32 on CPU.
    - Sets `tokenizer.padding_side = "right"`, and the pad token falls back to the EOS
      token.
    - Returns the model in eval mode.
  - `pooled_hidden_states(texts: Sequence[str], model, tokenizer, layers: Sequence[int] | None = None, batch_size: int = 16, max_length: int = 512) -> np.ndarray`
    - Returns float32 with shape (len(layers), len(texts), hidden). `None` means all
      `hidden_states` (embeddings plus each block).
    - Pooling is a masked mean over tokens. An empty text gives zeros.
    - Rows come back in input order, computed under `torch.inference_mode()`.
  - `select_layers(features: np.ndarray, layer: int, window: int = 1) -> np.ndarray`: the
    mean of `features[layer : layer + window]`, shape (n, hidden). It raises `ValueError`
    if `layer + window > len(features)`.
- torch and transformers are imported **inside** `load_lm` and `pooled_hidden_states`, so
  `find_model_dir` and `select_layers` work without them.

- [ ] **Step 1: Write the failing tests** (`tests/test_lm.py`). The model tests share a
  fixture that builds a tiny random model fully offline:
  - `Qwen3Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8)`,
    passed to `AutoModel.from_config`, with `torch.manual_seed(0)`.
  - A `PreTrainedTokenizerFast` built on a `tokenizers` WordLevel vocabulary with
    whitespace pre-tokenization and `pad_token="[PAD]"`.
  - It starts with `pytest.importorskip("torch")` and `pytest.importorskip("transformers")`.

```python
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


def test_find_model_dir_finds_the_unique_variation(tmp_path):
    (tmp_path / "models/qwen-3/transformers/1.7b-base/1").mkdir(parents=True)
    (tmp_path / "models/qwen-3/transformers/1.7b-base/1/config.json").write_text("{}")
    (tmp_path / "models/qwen-3/transformers/0.6b-base/1").mkdir(parents=True)
    (tmp_path / "models/qwen-3/transformers/0.6b-base/1/config.json").write_text("{}")
    assert find_model_dir("1.7b-base", tmp_path).name == "1"
    assert "1.7b-base" in find_model_dir("1.7b-base", tmp_path).parts


def test_find_model_dir_error_names_variation_and_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="4b-base"):
        find_model_dir("4b-base", tmp_path)


def test_select_layers_averages_a_window():
    feats = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    np.testing.assert_allclose(select_layers(feats, 1, 2), (feats[1] + feats[2]) / 2)
    with pytest.raises(ValueError):
        select_layers(feats, 3, 2)
```

- [ ] **Step 2: Run the tests locally with torch.**
  `uv run --all-extras --with torch --with transformers pytest tests/test_lm.py -q`.
  Expected: FAIL with an import error.

- [ ] **Step 3: Implement `lm.py`** with heavy "why" comments. Explain:
  - why right-padding: causal attention means padding after the text can't change the
    text's own hidden states.
  - why a masked mean: so padding never enters the average.
  - why `clamp(min=1)` on the token count: an empty text gives zeros instead of NaN.
  - why early layers might carry style: later layers specialize toward predicting the next
    token.

- [ ] **Step 4: Run the tests.** The local command from Step 2. Expected: all 7 pass. Also
  run `uv run --all-extras pytest -q` (no torch). Expected: the 4 model tests are
  **skipped**, everything else passes.

- [ ] **Step 5: Write ADR 0002.** Covers: torch and transformers aren't package
  dependencies (Kaggle's image provides them, and CI stays light); the cost is that the
  model tests run locally, not in CI; and the integration check is the Kaggle notebook
  itself.

- [ ] **Step 6: Ship.** Branch `p3b-01-lm`. The P3b plan file is committed in this PR.

---

### Task 2: Dense-feature CV, blending, BlendConfig

**Files:**
- Modify: `src/name_that_llm/config.py`, `src/name_that_llm/models.py`,
  `src/name_that_llm/train.py`
- Create: `src/name_that_llm/blend.py`, `tests/test_blend.py`
- Test: append to `tests/test_config.py`, `tests/test_train.py`

**Interfaces:**
- Produces:
  - `BlendConfig` (pydantic, `extra="forbid"`), with these fields:
    - `name: str`
    - `text_config: str = "lr"`
    - `lm_variation: str`
    - `layer: int`
    - `layer_window: int = 1`
    - `max_length: int = 512`
    - `batch_size: int = 16`
    - `C: float = 1.0`
    - `weight: float = Field(ge=0.0, le=1.0)` (the LM's share of the blend)
    - `n_splits: int = 5`
    - `seed: int = 42`
  - `load_config(name_or_path, model_cls: type[BaseModel] = ExperimentConfig)`: existing
    callers are unchanged.
  - `build_dense_model(C: float) -> Pipeline`: `StandardScaler()` followed by
    `LogisticRegression(C=C, max_iter=2000)`.
  - `run_cv_features(X: np.ndarray, df: pd.DataFrame, C: float = 1.0, n_splits: int = 5, seed: int = 42) -> CVResult`:
    the same folds, column alignment and CVResult as `run_cv`. Both share one private fold
    loop, `_oof(make_model, X, y, folds) -> np.ndarray`.
  - `blend(probas: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray`: a
    weighted arithmetic mean, with the weights normalized to sum to 1.
  - `best_blend_weight(p_text, p_lm, y, grid=np.round(np.linspace(0, 1, 21), 2)) -> tuple[float, float]`:
    returns (the LM weight, its balanced accuracy). Ties go to the **smaller** weight, the
    simpler model.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_train.py (append)
def _signal_features(df, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    onehot = (df["model"].to_numpy()[:, None] == np.array(LABELS)).astype(float)
    return onehot + rng.normal(0, noise, onehot.shape)


def test_run_cv_features_scores_informative_features(fake_train):
    result = run_cv_features(_signal_features(fake_train), fake_train, n_splits=3)
    assert result.oof.shape == (len(fake_train), 18) and result.score > 0.8


def test_run_cv_features_keeps_columns_aligned_when_a_class_is_missing(fake_train):
    df = fake_train[(fake_train["model"] != "kimi-k3") | (fake_train["prompt_id"] == "prompt0")]
    df = df.reset_index(drop=True)
    result = run_cv_features(_signal_features(df), df, n_splits=3)
    assert np.allclose(
        result.oof[(df["prompt_id"] == "prompt0").to_numpy(), LABELS.index("kimi-k3")], 0
    )


def test_text_paths_never_import_torch_or_optuna():
    code = (
        "import sys, name_that_llm.submit, name_that_llm.train, name_that_llm.blend; "
        "assert not {'torch', 'optuna', 'transformers'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

```python
# tests/test_blend.py
def test_blend_normalizes_weights_and_rows_sum_to_one():
    a, b = np.full((2, 18), 1 / 18), np.eye(18)[:2]
    out = blend([a, b], [2, 2])
    assert np.allclose(out.sum(axis=1), 1.0) and np.allclose(out, (a + b) / 2)


def test_best_blend_weight_prefers_the_informative_source():
    y = np.array(LABELS * 3)
    good = np.eye(18)[np.tile(np.arange(18), 3)] * 0.9 + 0.1 / 18
    bad = np.full_like(good, 1 / 18)
    weight, score = best_blend_weight(bad, good, y)
    assert weight > 0 and score == 1.0
    assert best_blend_weight(good, bad, y)[0] == 0.0  # tie → smaller weight
```

```python
# tests/test_config.py (append)
def test_blend_config_round_trips_and_validates_weight(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text("name: b\nlm_variation: 1.7b-base\nlayer: 6\nweight: 0.5\n")
    cfg = load_config(path, model_cls=BlendConfig)
    assert cfg.layer == 6 and cfg.text_config == "lr" and cfg.seed == 42
    with pytest.raises(ValidationError):
        BlendConfig(name="x", lm_variation="1.7b-base", layer=1, weight=1.5)
```

- [ ] **Step 2: Run the tests.** Expected: FAIL with import errors.

- [ ] **Step 3: Implement.**
  - Refactor `run_cv` onto `_oof` first, then run the existing train tests to confirm
    they're unchanged.
  - Then add `run_cv_features`, `build_dense_model`, `blend.py` and `BlendConfig`.
  - Comment why scaling matters: hidden-state dimensions have wildly different scales, and
    logistic regression's penalty assumes comparable ones.

- [ ] **Step 4: Run the tests.** `uv run --all-extras pytest -q`. Expected: all pass, and
  the torch tests are skipped.

- [ ] **Step 5: Ship and tag.** Branch `p3b-02-dense-blend`. Bump the version to **0.5.0**.
  After the merge, tag `v0.5.0` and watch `release-lib` until it's green.

---

### Task 3: GPU extraction notebook (`notebooks/04-lm-features`)

**Files:**
- Create: `notebooks/04-lm-features/lm_features.ipynb`, `notebooks/04-lm-features/kernel-metadata.json`

**Interfaces:**
- Consumes: `load_train`, plus `find_model_dir`, `load_lm` and `pooled_hidden_states`
  (Task 1), from `v0.5.0`.
- Produces: Kaggle outputs of `ryqn00/name-that-llm-lm-features`:
  `features-0.6b-base.npy`, `features-1.7b-base.npy`, `features-4b-base.npy`,
  `train_ids.npy`. It also records the Kaggle GPU image's torch and transformers versions
  in Context.md.

- [ ] **Step 1: Write `kernel-metadata.json`.**
  - `"id": "ryqn00/name-that-llm-lm-features"`, `"title": "Name That LLM - LM features"`
  - `"enable_gpu": true`, `"enable_internet": true`, `"is_private": false`
  - `"competition_sources": ["name-that-llm"]`
  - `"model_sources"`: `qwen-lm/qwen-3/transformers/0.6b-base/1`, `.../1.7b-base/1` and
    `.../4b-base/1`

- [ ] **Step 2: Write the notebook** (generated by a scratchpad builder, as before). Cells:
  1. Install `@v0.5.0`. Print the torch, transformers and CUDA versions and the GPU name,
     then `assert` that transformers is at least 4.51.
  2. Load train and save `train_ids.npy`.
  3. Loop over the three variations:
     - Call `find_model_dir`, then `load_lm`.
     - Extract all layers with `pooled_hidden_states(train.text, ...)`.
     - `assert np.isfinite(...).all()`. On failure, re-extract that variation with
       `dtype="float32"` if it's ≤ 1.7B; otherwise drop it and print why.
     - Save as float16. Print the shape and minutes taken.
     - `del` the model, then `torch.cuda.empty_cache()`.
  4. Add a markdown cell explaining the pooling and why we store every layer: the layer
     choice happens later, on CPU, so it doesn't spend GPU quota.

- [ ] **Step 3: Smoke-test locally on CPU.** Run the extraction cells against a local Qwen
  3 `0.6b-base` snapshot on 20 rows, if one is cached. Otherwise skip, because Task 1's
  tiny-model tests cover the logic. Record which in the ledger.

- [ ] **Step 4: Optional user step:** attach the Kaggle Secret `WANDB_API_KEY` to notebooks
  04 to 06 for tracking. Without it, everything still runs.

- [ ] **Step 5: Ask the user's OK, then push.**
  `kaggle kernels push -p notebooks/04-lm-features --accelerator NvidiaTeslaT4`. Poll until
  it reports `complete`, then read the log for the shapes, timings and versions. Expected:
  three feature files with 29, 29 and 37 layers, in under 45 GPU-minutes.

- [ ] **Step 6: Ship.** Branch `p3b-03-lm-features`. Add the GPU image versions to
  Context.md.

---

### Task 4: Layer sweep and blend A/B (`notebooks/05-lm-sweep`, CPU)

**Files:**
- Create: `notebooks/05-lm-sweep/lm_sweep.ipynb`, `notebooks/05-lm-sweep/kernel-metadata.json`,
  `src/name_that_llm/configs/lm_blend.yaml` (from the notebook output)

**Interfaces:**
- Consumes: Task 3's outputs (through `"kernel_sources": ["ryqn00/name-that-llm-lm-features"]`),
  plus `select_layers`, `run_cv_features`, `run_cv`, `best_blend_weight`,
  `prompt_bootstrap_ci` and `BlendConfig`.
- Produces: `lm_blend.yaml` (a `BlendConfig`), `lm_sweep.csv`, and a promote-or-not
  decision recorded in Context.md and IDEAS.md (I-002).

- [ ] **Step 1: Write the notebook.**
  1. **Load and check alignment.** Find the inputs with
     `glob("/kaggle/input/**/features-*.npy", recursive=True)`, and assert that
     `train_ids.npy` equals `load_train().id`.
  2. **Sweep layers** on the **seed-7** folds with `run_cv_features(..., C=1.0, seed=7)`:
     every 2nd layer of each model, then both neighbors of the best one. Show a chart of CV
     against relative depth (layer divided by the number of layers), one line per model.
     Load the `dataviz` skill before writing it.
  3. **Refine around the best (variation, layer):** windows {1, 2, 4} × C {0.1, 0.3, 1, 3}.
  4. **Choose the blend weight** with `best_blend_weight` against `run_cv(lr, seed=7)`.
  5. **Report on seeds 42, 43 and 44:** the LM model alone, `lr`, and the blend, plus
     `prompt_bootstrap_ci(y, lr.pred, blend.pred, groups)` on seed 42.
  6. **Write the outputs:** `lm_blend.yaml`, with name `lm_blend`, the chosen values and
     seed 42, and `lm_sweep.csv`.
  7. **Findings:** a markdown cell, filled in after the run.

- [ ] **Step 2: Ask the user's OK, then push** (CPU, no accelerator). The expected runtime
  is ≤ 3 h (about 60 dense CVs), so poll in the background until it reports `complete`.
  Then download only the small outputs: `lm_blend.yaml` and `lm_sweep.csv`.

- [ ] **Step 3: Decide.** Promote the blend only if the bootstrap CI lower bound is > 0.
  Copy `lm_blend.yaml` into `src/name_that_llm/configs/`. Then update:
  - DIARY.md: the sweep results.
  - Context.md: the score table.
  - IDEAS.md: I-002 gets its result, and I-012 is re-scoped to "tune the blend (Task 6)".

- [ ] **Step 4: Ship.** Branch `p3b-04-lm-sweep`.

---

### Task 5: Offline submission with the LM blend

**Files:**
- Modify: `src/name_that_llm/submit.py`, `tests/test_submit.py`,
  `notebooks/90-submit/submit.ipynb` (regenerated by the builder),
  `notebooks/90-submit/kernel-metadata.json`

**Interfaces:**
- Consumes: `BlendConfig` (`lm_blend.yaml`), `build_dense_model`, `blend`, and
  `lm.select_layers` (in the notebook).
- Produces:
  - `predict_proba_test(cfg: ExperimentConfig, train, test) -> np.ndarray` (n_test, 18),
    with columns aligned to LABELS. `predict_submission` now uses it, with unchanged
    behavior.
  - `predict_submission_blend(text_cfg: ExperimentConfig, blend_cfg: BlendConfig, train, test, lm_train: np.ndarray, lm_test: np.ndarray) -> pd.DataFrame`.
    It raises `ValueError` if the `lm_*` row counts don't match.

- [ ] **Step 1: Write the failing tests.**

```python
def test_predict_submission_blend_weight_zero_equals_text_only(fake_train, fake_data_dir):
    test = load_test(fake_data_dir)
    bcfg = BlendConfig(name="b", lm_variation="x", layer=0, weight=0.0)
    rng = np.random.default_rng(0)
    sub = predict_submission_blend(
        _cfg(),
        bcfg,
        fake_train,
        test,
        rng.normal(size=(len(fake_train), 8)),
        rng.normal(size=(3, 8)),
    )
    assert sub["model"].tolist() == predict_submission(_cfg(), fake_train, test)["model"].tolist()


def test_predict_submission_blend_rejects_misaligned_features(fake_train, fake_data_dir):
    bcfg = BlendConfig(name="b", lm_variation="x", layer=0, weight=0.5)
    with pytest.raises(ValueError):
        predict_submission_blend(
            _cfg(), bcfg, fake_train, load_test(fake_data_dir), np.zeros((5, 8)), np.zeros((3, 8))
        )
```

- [ ] **Step 2: Run them.** Expected: FAIL with an import error.

- [ ] **Step 3: Implement**, then run `uv run --all-extras pytest -q`. Expected: all pass.

- [ ] **Step 4: Update the submit notebook.**
  - Add `USE_LM = True` and `BLEND = "lm_blend"`.
  - If `USE_LM`, it does the following, all offline:
    1. `find_model_dir(blend_cfg.lm_variation)`, then `load_lm`.
    2. Extract **only** layers `layer .. layer + window - 1` for train and test.
    3. `select_layers(..., 0, window)`, then `predict_submission_blend`.
  - The timing cell also covers extraction on 4,000 stand-in rows.
  - Metadata: `"enable_gpu": true` and `model_sources` = the chosen variation, internet
    stays **off**.
  - Explain in markdown why the train features are recomputed instead of attached:
    one artifact less to keep in sync.

- [ ] **Step 5: Release.** Ship as branch `p3b-05-submit-lm`, with version 0.6.0. After the
  merge, tag `v0.6.0` and check that the dataset lists `name_that_llm-0.6.0`.

- [ ] **Step 6: Ask the user's OK, then push and submit.**
  - Push with `--accelerator NvidiaTeslaT4`, poll until `complete`, and check the 3-row
    output.
  - Submit with the message `"lr + LM blend (<variation> L<layer>), v0.6.0"`, then poll
    for the score.
  - Record CV against LB. Promote the blend only if the LB is above 0.320. Otherwise go
    back to `USE_LM = False`, and diagnose before tuning.
  - Ship the records as branch `p3b-06-results`.

---

### Task 6: Tune the blend with Optuna (`tune.py`, `notebooks/06-tuning`, CPU)

**Files:**
- Create: `src/name_that_llm/tune.py`, `tests/test_tune.py`,
  `notebooks/06-tuning/tuning.ipynb`, `notebooks/06-tuning/kernel-metadata.json`
- Modify: `pyproject.toml` (`uv add --optional tune "optuna>=3.6"`), and the import test
  in `tests/test_train.py`. After the text paths are imported, it also asserts that
  `name_that_llm.tune` is **not** in `sys.modules`.

**Interfaces:**
- Consumes: Task 3's feature files, plus `select_layers`, `run_cv_features`, `run_cv`,
  `blend`, `score` and `BlendConfig`.
- Produces:
  - `suggest_blend(trial, n_layers: dict[str, int]) -> BlendConfig`. Search space:
    - `lm_variation` ∈ the keys of `n_layers`
    - `layer` ∈ [0, L − 1]
    - `layer_window` ∈ {1, 2, 4}, clipped so that `layer + window ≤ L`
    - `C` log-uniform on [0.01, 10]
    - `weight` uniform on [0, 1]
  - `run_study(features: dict[str, np.ndarray], df: pd.DataFrame, n_trials: int, seed: int = 7) -> optuna.Study`:
    TPE sampler with `seed`. The objective is the blend's balanced accuracy on seed-`seed`
    folds. `lr` OOF is computed once on the same folds.
  - `best_blend_config(study, name: str = "lm_blend_tuned") -> BlendConfig`: seed reset to
    42.

- [ ] **Step 1: Write the failing tests** (`tests/test_tune.py`, with
  `pytest.importorskip("optuna")`). The features are tiny random arrays shaped
  (L, n, 8) for two fake variations on `fake_train`, so 3 trials take seconds.

```python
def test_run_study_returns_a_valid_round_tripping_config(fake_train, tmp_path):
    rng = np.random.default_rng(0)
    feats = {
        "a": rng.normal(size=(3, len(fake_train), 8)),
        "b": rng.normal(size=(2, len(fake_train), 8)),
    }
    study = run_study(feats, fake_train, n_trials=3, seed=7)
    cfg = best_blend_config(study)
    assert cfg.lm_variation in feats and cfg.layer + cfg.layer_window <= len(
        feats[cfg.lm_variation]
    )
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    assert load_config(path, model_cls=BlendConfig) == cfg and cfg.seed == 42
```

- [ ] **Step 2: Run it.** Expected: FAIL with an import error.

- [ ] **Step 3: Implement `tune.py`.** Comment: why TPE beats a grid here; why the seed-7
  folds (selection bias); and why there's no pruning (each trial is only a few dense fits,
  and pruning can be added if trials get expensive).

- [ ] **Step 4: Run the tests.** `uv run --all-extras pytest -q`. Expected: all pass.

- [ ] **Step 5: Write the tuning notebook.**
  - Metadata: CPU, `kernel_sources` = Task 3's notebook, internet on.
  - Steps:
    1. Load the features and assert id alignment.
    2. `run_study(..., n_trials=60, seed=7)`, logged to W&B if the secret is attached.
    3. Show the optimization history and the most important parameters.
    4. Evaluate the tuned config against `lm_blend` on seeds 42, 43 and 44, plus the
       bootstrap on seed 42.
    5. Write `lm_blend_tuned.yaml`.

- [ ] **Step 6: Ask the user's OK, then push.** Poll until it reports `complete`, then
  download the YAML.
  - If the CI lower bound is > 0 against `lm_blend`, add the YAML to `configs/` and set
    `BLEND = "lm_blend_tuned"`.
  - Release `v0.7.0`, then ask the user's OK and submit.
  - Otherwise, record that tuning didn't help.
  - Either way, update DIARY.md, IDEAS.md (I-012) and Context.md. Ship as branch
    `p3b-07-tuning`.
