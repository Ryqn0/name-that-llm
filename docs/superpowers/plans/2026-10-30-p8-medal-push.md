# P8: Our own strong models on the clean pool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a top-10% finish by 2027-01-13, by training our own strong members on the whole
clean public pool and blending them honestly.

**Architecture:**
- **A pinned, watched pool.** A private copy of the clean pool v10. Per-source counts are
  logged in every run, and a weekly check compares new public releases with it.
- **"Pool members" in a new module, `pool_member.py`.** Each trains once on the pool alone:
  no passage about our 150 prompts, and none of our passages. Every label is weighted
  equally, and the member predicts all 2,699 of our passages honestly.
- **Two members:** DeBERTa-v3-large, then the Qwen3-8B LoRA, each trained on Kaggle's T4s
  with one process and one seed per GPU.
- **A generalized gate:** nested blend weights on the seed-42 folds decide what joins the
  ensemble. The offline submission then runs every member, from private weights.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, torch, transformers and peft (not
package dependencies: ADR 0002), the Kaggle CLI, GitHub Actions, and uv.

**Spec:** `docs/superpowers/specs/2026-10-30-p8-medal-push-design.md`

**Prerequisite (P7's own plan and ledger, `.superpowers/sdd/2026-10-27-p7-llm-ensemble/`):**
1. session 3 (seed 42's fold 4);
2. the four-member gate on all 5 folds (`scripts/llm_ensemble_check.py`);
3. P7 Task 7's `EnsembleConfig` (`llm`, `sinkhorn_alpha`, `log_blend`);
4. P7 Task 8 (`predict_submission_ensemble(..., p_llm=None)`, the LLM in the submit
   notebook, lib v0.10.0);
5. P7 Task 9 (its journal).

P8's Tasks 1–3 are CPU work, and may run during P7's GPU waits. P8's Task 4 starts after
P7 Task 8 ships.

## Global Constraints

- **Free compute only:** Kaggle's T4 × 2, 12-hour sessions, about 30 GPU-hours a week.
  `kaggle quota` gives the exact amount left and the refresh time. Every GPU push gets a
  session budget at least 30 minutes under the quota left.
- **The user's explicit OK** for every Kaggle push, dataset creation, submission and release
  tag. A scheduled workflow is switched on only with the user's OK.
- **Private until 2027-01-13:**
  - the trained weights, and every Kaggle notebook that uses them, the submit notebook
    included (`"is_private": true`);
  - the code, the configs and the diary stay public.
- **Data:** the pinned pool copy only, minus `exclude`. Openly licensed, published data.
  Never commit competition data or pool passages. The tests use fake data only.
- **The honest protocol:** no pool row whose `prompt_id` is one of our prompts trains, and
  no row whose text equals one of our passages trains. None of our 2,699 passages trains a
  pool member.
- **Every label is weighted equally** in a pool member's loss: `w_j = n / (18 · n_j)`.
- **CV decides:** nested blend weights on the seed-42 folds. PROMOTE if and only if the
  prompt bootstrap's 95% lower bound is above 0. The public LB alone never decides.
- **One final pick stays pool-free:** `ensemble`, LB 0.410.
- **Committed notebooks:** outputs stripped, aggregate numbers only. W&B is logged from the
  PC (`scripts/log_gpu_run.py`).
- **Comments explain the why,** generously: the user's standing request.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. **PR
  bodies** end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## Review Focus

1. **A training pool missing one of the 18 labels** (after `exclude`, matching and sampling)
   must stop the run with that label named. A member that never saw a model can never
   predict it. Test: Task 2, `test_label_weights_refuse_a_missing_label`.
2. **A pool passage identical to one of ours, or matched to one of our prompts,** must never
   reach training, even if `dedupe` or the matcher missed it upstream. Test: Task 2,
   `test_honest_training_set_never_holds_our_prompts_or_passages`.
3. **A speed probe that projects more passages than the session allows** must shrink the run
   to fit, and a zero budget must skip training rather than start it. Test: Task 2,
   `test_rows_for_budget_fits_the_session`.
4. **A source listed in `exclude` must be dropped even when `include` is None** (all
   sources), and even if a new release re-adds it under the same name. Test: Task 1,
   `test_exclude_drops_a_source_even_with_every_source_included`.
5. **A submission whose extra member's probabilities are missing, or of the wrong shape,**
   must refuse to write a file rather than blend the wrong rows. Test: Task 5,
   `test_extra_member_probabilities_are_required_and_aligned`.

---

### Task 1: The pinned, watched pool

**Files:**
- Modify: `src/name_that_llm/pool.py`:
  - `PoolSource` gains `exclude`;
  - `build_pool` applies it after `include`;
  - new `source_counts`.
- Create:
  - `src/name_that_llm/configs/pool_all.yaml`;
  - `src/name_that_llm/configs/pool_pinned_counts.json`;
  - `scripts/pool_release_check.py`;
  - `.github/workflows/pool-watch.yml`.
- Test: `tests/test_pool.py`, `tests/test_pool_release_check.py`.

**Interfaces:**
- Produces:
  - `PoolSource.exclude: list[str] | None = None`;
  - `source_counts(pool: pd.DataFrame) -> dict[str, int]`: passages per `source`, keys
    sorted;
  - the config name `"pool_all"`: one source, slug `ryqn00/name-that-llm-pool-v10`,
    `source_column: source_dataset`, `include: null`, `exclude: []`;
  - `scripts/pool_release_check.py`:
    `compare_counts(pinned: dict[str, int], latest: dict[str, int]) -> dict[str, list[str]]`
    with keys `"dropped"`, `"shrunk"` and `"new"`.

- [ ] **Step 1: Write the failing tests.**
  - `test_exclude_drops_a_source_even_with_every_source_included`: a fake pool parquet with
    sources `a` and `b`. Run `build_pool([PoolSource(..., source_column="source_dataset",
    include=None, exclude=["b"])], root, fake_train)` and assert
    `set(pool["source"]) == {"a"}`.
  - `test_source_counts`: `source_counts(pd.DataFrame({"source": ["b", "a", "b"]})) ==
    {"a": 1, "b": 2}`.
  - `test_compare_counts_flags_dropped_shrunk_and_new`:
    `compare_counts({"a": 10, "b": 5, "c": 3}, {"a": 10, "b": 4, "d": 7}) ==
    {"dropped": ["c"], "shrunk": ["b"], "new": ["d"]}`.
  - `test_release_check_fails_on_a_dropped_source`: `main([latest_parquet, "--pinned",
    pinned_json])` returns 1 when a pinned source is gone from the latest file, and 0 when
    none is.
  - `test_pool_watch_workflow_runs_the_check_weekly`: the workflow YAML has a `schedule`
    cron and `workflow_dispatch`, and runs `scripts/pool_release_check.py`.
- [ ] **Step 2: Run them.**
  Run: `uv run --all-extras pytest -q tests/test_pool.py tests/test_pool_release_check.py`.
  Expected: FAIL (`exclude` and `source_counts` are undefined, and the script is missing).
- [ ] **Step 3: Implement.**
  - **`exclude`:** filter `kept` after `include` in `build_pool`; the dedupe stays across
    every source.
  - **`source_counts`:** `value_counts`, as a sorted dict.
  - **`compare_counts`:**
    - dropped: in pinned, not in latest;
    - shrunk: fewer passages in latest;
    - new: in latest, not in pinned;
    - each list sorted.
  - **The script's `main`:**
    - read the latest file's `source_dataset` column only (never the text);
    - print the three lists;
    - return 1 if `dropped` holds a source that isn't already in `pool_all.yaml`'s
      `exclude`.
  - **`pool_pinned_counts.json`:** written once, by a one-off command, from
    `data/external-v10/.../ntl_open_pool.parquet` (v10). Counts only.
  - **`pool-watch.yml`:**
    - schedule `"0 6 * * 3"` (Wednesdays), plus `workflow_dispatch`;
    - `uv`, and `pip install kaggle==2.2.5`;
    - download only `ntl_open_pool.parquet` with the secret `KAGGLE_API_TOKEN`;
    - run the check.

    A failure emails the owner, which is the "watch".
- [ ] **Step 4: Run them.** Same command. Expected: PASS.
- [ ] **Step 5: Ship** (branch `p8-01-pool-pin`): PR, CI, merge.
- [ ] **Step 6: Pin the pool. Ask the user's OK first.**
  - Create the **private** dataset `ryqn00/name-that-llm-pool-v10` from the v10 file, licence
    `CC-BY-SA-4.0` as upstream. Its description credits the maintainer and the v10 source
    list.
  - Run: `kaggle datasets create -p <dir>` with `dataset-metadata.json`. Expected: a private
    dataset, listed by `kaggle datasets list --mine`.
  - Ledger the version.
- [ ] **Step 7: Switch on the weekly workflow (the user's OK).** Run it once by hand
  (`gh workflow run pool-watch.yml`). Expected: green, with "dropped: []".

---

### Task 2: The honest pool member (CPU-tested)

**Files:**
- Modify:
  - `src/name_that_llm/encoder.py`: `train_encoder` gains `class_weight` and `on_epoch_end`;
  - `src/name_that_llm/config.py`: `EncoderConfig.pool_size: int = 0`.
- Create:
  - `src/name_that_llm/pool_member.py`;
  - `src/name_that_llm/configs/deberta_pool.yaml`;
  - `src/name_that_llm/configs/llm_pool.yaml`.
- Test: `tests/test_pool_member.py`, `tests/test_config.py`.

**Interfaces:**
- Consumes:
  - `pool.pool_for_fold(pool, valid_prompt_ids, max_rows, seed)`;
  - `encoder.load_encoder`, `encoder.predict_proba` and `encoder._release_memory`;
  - `llm.load_llm_classifier`, `llm.save_adapter` and `llm._nonempty`.
- Produces:
  - `train_encoder(model, tokenizer, texts, y, cfg, class_weight: np.ndarray | None = None,
    on_epoch_end: Callable[[int], None] | None = None, amp: bool = True) -> list[float]`.
    `amp=False` trains in float32 on CUDA too: the fallback if float16 overflows (spec §8).
  - `honest_training_set(pool: pd.DataFrame, train: pd.DataFrame, max_rows: int | None,
    seed: int) -> pd.DataFrame`;
  - `label_weights(y: Sequence[str]) -> np.ndarray`: shape (18,), in LABELS order;
  - `rows_for_budget(seconds_per_passage: float, epochs: int, minutes: float, available: int)
    -> int`;
  - `fit_pool_member(cfg: EncoderConfig | LLMConfig, pool: pd.DataFrame, train:
    pd.DataFrame, save_dir: Path | None = None) -> tuple[np.ndarray, list[float]]`;
  - `probe_pool_member(cfg, pool, train, n: int = 256) -> dict`, with keys
    `seconds_per_passage`, `model_load_seconds`, `peak_memory_gb` and `finite`;
  - the CLI `python -m name_that_llm.pool_member --config NAME --pool PARQUET --seed S
    --rows R --out NPZ [--save-dir DIR] [--data-dir DIR] [--probe N] [--fp32]`. The npz
    holds `train` (n, 18), in `load_train()` order, and `losses`.

- [ ] **Step 1: Write the failing tests.**
  - `test_honest_training_set_never_holds_our_prompts_or_passages`:
    - a fake pool with one row matched to a train prompt (`prompt_id` set), one row whose
      text equals a `fake_train` text but whose `prompt_id` is None, and ten clean rows;
    - assert the result has exactly the ten clean rows, and that no text in it is in
      `set(fake_train["text"])`.
  - `test_label_weights_equalize_every_label`: for `y` with counts 3, 1 and 2 over three
    labels of the 18, plus every other label once, `np.bincount` of the label indices,
    weighted by `w`, is constant across labels.
  - `test_label_weights_refuse_a_missing_label`: drop `LABELS[5]` from `y`. A `ValueError`
    whose message names `LABELS[5]`.
  - `test_rows_for_budget_fits_the_session`:
    - `rows_for_budget(1.0, 2, 600, 100_000) == 18_000`;
    - `rows_for_budget(1.0, 2, 600, 5_000) == 5_000`;
    - `rows_for_budget(1.0, 2, 0, 5_000) == 0`.
  - `test_fit_pool_member_predicts_every_passage_and_saves_each_epoch` (torch,
    `tiny_encoder`):
    - with `EncoderConfig(base_model=str(tiny_encoder), epochs=2, pool_size=36,
      batch_size=8, max_length=64)`;
    - a fake pool of 36 rows, two per label, `prompt_id` None;
    - assert the shape is `(len(fake_train), 18)`, the rows sum to 1, there are 2 losses,
      and `save_dir/"epoch1"` and `save_dir/"epoch2"` hold a `config.json`.
  - `test_fit_pool_member_trains_the_llm_too` (torch and peft, `tiny_qwen`): the same with a
    tiny `LLMConfig` (`load_in_4bit=False`). Each epoch folder holds an
    `adapter_config.json`.
  - `test_class_weights_change_the_loss` (torch, `tiny_encoder`): one epoch with
    `class_weight` all ones gives the same loss as `None`, and a skewed weight gives a
    different one.
  - `test_shipped_pool_member_configs_load`:
    - `deberta_pool` is an `EncoderConfig` with base `microsoft/deberta-v3-large`;
    - `llm_pool` is an `LLMConfig` with `pool_size == 18_000`.
- [ ] **Step 2: Run them.**
  Run: `env MLFLOW_DISABLE_AGENT_HINT=1 uv run --all-extras --with torch --with transformers
  --with peft pytest -q tests/test_pool_member.py tests/test_config.py`.
  Expected: FAIL (`pool_member` is missing).
- [ ] **Step 3: Implement.**
  - **`train_encoder`:**
    - with `class_weight`, the loss is `torch.nn.functional.cross_entropy(logits, labels,
      weight=w)`, with `w` on the logits' device and in float32;
    - without it, unchanged (the model's own loss);
    - `on_epoch_end(epoch)` is called after each epoch.
  - **`honest_training_set`:** `pool_for_fold(pool, set(train["prompt_id"]), max_rows,
    seed)`, then drop the rows whose stripped text is in our stripped texts.
  - **`fit_pool_member`:**
    - seed torch with `cfg.seed`;
    - load: `load_encoder(cfg.base_model)` for an `EncoderConfig`,
      `load_llm_classifier(cfg)` for an `LLMConfig`;
    - train on `honest_training_set(pool, train, cfg.pool_size or None, cfg.seed)` with
      `class_weight=label_weights(...)`;
    - save after each epoch to `save_dir/f"epoch{e}"`: an encoder in float16 with
      `save_pretrained`, an LLM with `save_adapter`;
    - **resuming (spec §5.3), my call:** a run that stops part-way keeps its last finished
      epoch. If a next session ever continues it, it starts from that epoch's weights with a
      fresh optimizer; no optimizer state is saved. The probe sizes each run to fit one
      session, so this is only a fallback;
    - predict all of `train["text"]` with `encoder.predict_proba`;
    - release memory.
  - **`probe_pool_member`:** train `n` rows for 1 epoch, timed, as `llm.probe_speed` does.
    `finite` is False if the loss turned non-finite (catch `FloatingPointError`).
  - **`rows_for_budget`:** `min(available, floor(minutes * 60 / (seconds_per_passage *
    epochs)))`, never below 0.
  - **`deberta_pool.yaml`:**
    - `max_length 512`, `batch_size 8`, `lr 1e-5`, `epochs 2`, `warmup_ratio 0.1`,
      `weight_decay 0.01`, `pool_size 0` (the notebook sets it), `seed 42`;
    - the comment: "settings left to the builder by the spec (§7), starting points for a
      large encoder".
  - **`llm_pool.yaml`:** `llm.yaml`'s values, with `pool_size: 18000` and the comment
    "about 10 hours for 2 epochs on a T4 at about 1 s per passage per epoch (P7)".
- [ ] **Step 4: Run them.** Same command. Expected: PASS.
- [ ] **Step 5: Ship** (branch `p8-02-pool-member`). CI's torch job runs the tiny tests.

---

### Task 3: The generalized gate

**Files:**
- Modify: `src/name_that_llm/blend.py`, which gains `nested_blend_oof`.
- Create: `scripts/ensemble_gate.py`, which supersedes `scripts/llm_ensemble_check.py`.
  Delete the latter here, and change Context's references.
- Test: `tests/test_blend.py`.

**Interfaces:**
- Produces:
  - `nested_blend_oof(probas: Sequence[np.ndarray], y: np.ndarray, folds: np.ndarray, rows:
    np.ndarray | None = None, step: float = 0.05) -> tuple[np.ndarray, dict[int,
    tuple[float, ...]]]`. It returns the blended probabilities (zeros outside `rows`) and
    each fold's weights, chosen on the other folds' rows only.
  - The CLI `python -I -X utf8 scripts/ensemble_gate.py --llm-runs DIR [DIR ...]
    [--extra NAME=PATH.npy ...] [--pool PARQUET]`:
    - the baseline is `lr` + pool, the LM head, the encoder and the LLM folds;
    - the candidate is the baseline plus every `--extra`;
    - it prints the nested scores, the bootstrap CI, the weights to ship, and `VERDICT:
      PROMOTE|KEEP`.

- [ ] **Step 1: Write the failing tests.**
  - `test_nested_blend_oof_chooses_each_folds_weights_elsewhere`: three folds and two
    members. Assert `weights[k] == best_blend_weights([p[rows_not_k] ...], y[rows_not_k])[0]`
    for every k.
  - `test_nested_blend_oof_leaves_rows_outside_untouched`: with `rows` covering folds 0–1,
    the output is 0 on fold 2's rows.
- [ ] **Step 2: Run them.** Run: `uv run --all-extras pytest -q tests/test_blend.py`.
  Expected: FAIL.
- [ ] **Step 3: Implement** `nested_blend_oof`, then `ensemble_gate.py`. Keep
  `llm_ensemble_check.py`'s loading and caching (the cache key names the pool file and its
  size), and call `nested_blend_oof`.
- [ ] **Step 4: Run them.** Expected: PASS. Then run the script on P7's five LLM folds with
  no `--extra`. Expected: `llm_ensemble_check.py`'s P7 numbers for the four-member
  ensemble, to 4 decimals.
- [ ] **Step 5: Ship** (branch `p8-03-gate`).

---

### Task 4: DeBERTa-v3-large on the pool (notebook 13, GPU)

**Files:**
- Create:
  - `notebooks/13-pool-deberta/pool_deberta.ipynb` and its `kernel-metadata.json`:
    - id `ryqn00/name-that-llm-pool-deberta`, **`is_private: true`**;
    - GPU on, internet on;
    - `dataset_sources: ["ryqn00/name-that-llm-pool-v10"]`, competition `name-that-llm`.
- Test: `tests/test_notebooks.py`.

**Interfaces:**
- Consumes: Task 1 (`pool_all`, `source_counts`) and Task 2 (`probe_pool_member`,
  `rows_for_budget`, and the `pool_member` CLI).
- Produces, as notebook outputs:
  - `pool_counts.json` and `probe.json`;
  - `runs/seed{42,43}.npz`;
  - `models/seed{s}/epoch{e}/`;
  - `results.json`, with keys `rows`, `epochs`, `scores` (per seed, and `mean` of the two
    seeds' probabilities, on our 2,699 passages) and `minutes`.

- [ ] **Step 1: Write the failing test.** `test_pool_deberta_notebook_runs`:
  - `tiny_encoder` as `NTL_ENCODER_CONFIG`'s base;
  - a fake pinned pool under `NTL_INPUT/name-that-llm-pool-v10/`;
  - assert:
    - `results.json`'s `scores` has keys `42`, `43` and `mean`, all in [0, 1];
    - both `runs/seed*.npz` hold `train` of shape `(len(fake_train), 18)`;
    - `pool_counts.json` sums to the built pool's size.
- [ ] **Step 2: Run it.** Expected: FAIL (no notebook).
- [ ] **Step 3: Write the notebook.**
  1. Install, pinned to a commit holding Tasks 1–2. The config is the name or YAML path in
     `NTL_ENCODER_CONFIG` (default `deberta_pool`); the folders come from `NTL_INPUT`,
     `NTL_WORK` and `NTL_DATA_DIR`, as in notebooks 10–12.
  2. Build the pool from `pool_all`, and save it with its counts.
  3. The probe on GPU 0, then `rows_for_budget` with `SESSION_MIN` minus the time spent, a
     20-minute margin for predicting, and `epochs` from the config. If the probe's `finite`
     is False, the two runs get `--fp32`, at about half the speed (the probe's speed is
     halved before sizing).
  4. Two processes, one per GPU (seeds 42 and 43), each on its own sample (`seed`)
     of `rows` passages.
  5. Average the two seeds' probabilities. Save the scores.
  6. Findings (filled in after the run).

  `SESSION_MIN` is set from `kaggle quota` before each push (the global constraint).
- [ ] **Step 4: Run the smoke test.** Expected: PASS.
- [ ] **Step 5: Ship the notebook** (branch `p8-04-pool-deberta`), **then push it with the
  user's OK.** Poll it, download the outputs to `data/kaggle-outputs/pool-deberta`, and log
  to W&B from the PC.
- [ ] **Step 6: The gate.**
  Run: `python -I -X utf8 scripts/ensemble_gate.py --llm-runs <P7's runs> --extra
  deberta=data/kaggle-outputs/pool-deberta/mean.npy --pool data/pool/pool-v10-all.parquet`.
  Here `mean.npy` is the two seeds' average, saved on the PC.
  - Ledger the verdict.
  - **PROMOTE:** go to Task 5.
  - **KEEP:** stop this member, and ledger why: the speed, the score alone, the weights.
  Fill in the notebook's Findings.

---

### Task 5: The submission with pool members (private)

**Files:**
- Modify:
  - `src/name_that_llm/config.py`: `EnsembleConfig.extra_members: list[str] = []`. The
    weights count = 3 + (1 if `llm` else 0) + len(`extra_members`).
  - `src/name_that_llm/submit.py`: `predict_submission_ensemble(...)` gains `p_extra`.
  - `notebooks/90-submit/` (`is_private: true` from now on): attach the pool member's
    private notebook output, predict the test with its seeds, and pass `p_extra`.
  - `pyproject.toml`: version 0.11.0.
- Create: `src/name_that_llm/configs/ensemble_p8.yaml`, with the gate's weights and
  `extra_members: [deberta_pool]`.
- Test: `tests/test_submit.py`, `tests/test_config.py`.

**Interfaces:**
- Consumes: P7 Task 8's `predict_submission_ensemble(..., p_llm=None)`, and Task 4's
  `models/seed{s}/epoch{last}/`.
- Produces: `predict_submission_ensemble(..., p_llm: np.ndarray | None = None, p_extra:
  Mapping[str, np.ndarray] | None = None) -> pd.DataFrame`. Members in config order: text,
  LM head, encoder, [llm], then `*extra_members`.

- [ ] **Step 1: Write the failing tests.**
  - `test_extra_member_probabilities_are_required_and_aligned`: with `extra_members=["x"]`:
    - `p_extra=None` raises a `ValueError` naming "x";
    - a `(n_test, 17)` array raises a `ValueError`;
    - a `(n_test - 1, 18)` array raises a `ValueError`.
  - `test_an_extra_member_with_all_the_weight_decides`: weights `(0, 0, 0, 1)` and a one-hot
    `p_extra["x"]`. Every row's label is the one-hot's.
  - `test_ensemble_config_counts_its_weights`: `extra_members=["x"]` with 3 weights raises a
    `ValidationError`; `ensemble_p8` loads.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Then the notebook:
  - find each seed's model folder, searched and never hard-coded (as `find_fold_dirs`);
  - average the seeds' `predict_proba` on the test, with `max_length` from the config;
  - include it in the runtime check;
  - `is_private: true`.
- [ ] **Step 4: Run them.** Expected: PASS. Then the whole suite.
- [ ] **Step 5: Ship**, then **tag `v0.11.0` (the user's OK)**, then **push and submit the
  notebook (the user's OK, in the foreground)**. Ledger the LB, and the CV–LB gap against
  the gate's CV.

---

### Task 6: The LLM on far more of the pool (notebook 14, GPU)

**Files:**
- Create: `notebooks/14-pool-llm/pool_llm.ipynb` and its `kernel-metadata.json`:
  - `is_private: true`;
  - the 8b-base model source;
  - the pinned pool.
- Test: `tests/test_notebooks.py`.

**Interfaces:**
- Consumes: Task 2's `fit_pool_member` with `llm_pool`, Task 3's gate, and Task 5's
  `extra_members`.
- Produces: the same outputs as notebook 13, with each epoch's adapters.

- [ ] **Step 1: Write the failing test.** `test_pool_llm_notebook_runs`: `tiny_qwen`, with
  the same assertions as Task 4's.
- [ ] **Step 2: Run it.** Expected: FAIL.
- [ ] **Step 3: Write the notebook:**
  - the config from `NTL_LLM_CONFIG` (default `llm_pool`);
  - two processes (seeds 42 and 43), each on its own pool sample of `rows` passages;
  - `rows` comes from `rows_for_budget`, with the probe's speed and `llm_pool`'s epochs;
  - the rest as notebook 13.
- [ ] **Step 4: Run the smoke test.** Expected: PASS.
- [ ] **Step 5: Ship, then push (the user's OK). Then the gate,** with
  `--extra deberta=... --extra llm_pool=...` (the candidate) against the current champion.
  - **PROMOTE:** add `llm_pool` to `extra_members` with the gate's weights, then release and
    submit, as Task 5's Step 5.
  - **KEEP:** ledger why.

---

### Task 7: Centering, under P3a's layout check

**Files:**
- Modify: `src/name_that_llm/blend.py`, which gains `center`.
- Create: `scripts/centering_check.py`.
- Test: `tests/test_blend.py`.

**Interfaces:**
- Produces:
  - `center(proba: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray`:
    log-probabilities minus `alpha` times their group mean, re-normalized by softmax;
    `0 <= alpha < 1`;
  - `scripts/centering_check.py`: the champion blend's OOF, scored with and without
    centering, for alpha in {0.25, 0.5, 0.75}, on subsampled layouts m in {1, 2, 3, 6, 18}
    answers per prompt. The groups come from `grouping.topic_groups` on each layout's texts.

- [ ] **Step 1: Write the failing tests.**
  - `test_center_with_alpha_zero_changes_nothing`;
  - `test_center_removes_a_shared_group_bias`: two groups, where every row of group 0
    carries +2 on label 0's log-probability. After `center(..., alpha=0.9)`, label 0 no
    longer wins group 0's rows.
  - `test_center_refuses_alpha_one_or_more`: a `ValueError`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** `center`, then the script.
- [ ] **Step 4: Run them.** Expected: PASS. Then run the script.
  - **Keep centering only if it gains at every m**, with the m = 1, 2 and 3 layouts weighing
    most (P3a's lesson).
  - If kept: `EnsembleConfig` gains `center_alpha: float = 0.0`, and the submission applies
    it. Release and submit, each with the user's OK.
  - Ledger the per-m table either way.
- [ ] **Step 5: Ship** (branch `p8-07-centering`).

---

### Task 8: The checkpoint (2026-12-15)

- [ ] **Step 1:** `kaggle competitions leaderboard name-that-llm -d`: our best public LB, the
  top-10% cutoff, and our rank.
- [ ] **Step 2: If we're below the line:** ask the user whether to add published models
  (§3 of the spec), with the gap and the remaining weeks. Ledger the decision. **If we're
  above it:** ledger that, and go on.

---

### Task 9: Final picks, publishing, and the record

- [ ] **Step 1 (by 2027-01-13): the two final picks** (the user selects them on Kaggle):
  - the best CV-backed pool ensemble;
  - **the pool-free `ensemble` (LB 0.410)**.
  Ledger both.
- [ ] **Step 2 (after 2027-01-13 23:59 UTC, the user's OK):** make the P8 notebooks
  (13, 14, the submit notebook, `llm-final`) and the pool-member weights public.
- [ ] **Step 3:**
  - `docs/journal/p8-medal-push.md` from the ledger;
  - an ADR on public code with private weights;
  - Context and the diary.

  Delete the workspace only after the journal.
