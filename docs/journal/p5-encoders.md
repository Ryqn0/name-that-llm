# Journal: P5, fine-tuned encoders (2026-10-25 → 26)

**Goal:** fine-tune a transformer encoder to name the model behind a passage, then blend it
with the champion `lm_blend` (TF-IDF `lr` plus a logistic regression on Qwen 3 4B hidden
states).

**Result:** ModernBERT-base beat DeBERTa-v3-base. The three-model ensemble scored **public
LB 0.410**, up from 0.389, and became the champion.

**In numbers:** 7 tasks, 8 pull requests (#22 to #29), release v0.7.0, and 4 GPU runs on
Kaggle (about 4.1 GPU-hours). The plan:
[2026-10-25-p5-encoders.md](../superpowers/plans/2026-10-25-p5-encoders.md).

## How the work was run

- **Native execution:** Claude implemented every task in the session, following the plan,
  with one review of the whole phase at the end. The alternative, a fresh sub-agent per
  task plus a reviewer per task, costs more.
- **Test first:** every behavior got a failing test first. Watching it fail proves the test
  can catch the bug; then the code makes it pass.
- **Mutation checks:** for the important tests, Claude broke the code on purpose (for
  example, by swapping two fold models) to confirm that the test fails. Two tests turned
  out too weak this way, and were fixed.
- **A ledger:** a running log of every step, result and judgment call, kept in a
  git-ignored folder while the phase ran. Its content is folded into this journal.
- **Shipping:** one branch per task, then a pull request. CI runs three jobs (`test`,
  `kaggle-env`, `torch`), and the PR is squash-merged only when all three pass and the
  `torch` job shows that the model tests ran rather than skipped.
- **Approvals:** every Kaggle push and the submission waited for your explicit OK, given in
  the same session.
- **The final review** was a self-review (sub-agents run only when you ask for them). It
  found no critical or important issue; three minor ones are listed at the end.

## Step by step

### Task 1: the fine-tuning library (PR #22)

**Built:**
- `src/name_that_llm/encoder.py`:
  - `load_encoder`: a pretrained encoder with a new 18-way head, or one of our saved fold
    models, loaded as it is.
  - `train_encoder`: the training loop, written by hand instead of using Hugging Face's
    `Trainer` (why: [ADR 0003](../adr/0003-hand-written-fine-tuning-loop.md)). AdamW, a
    linear warmup then decay, mixed precision on the GPU (float16 math, float32 weights),
    gradient clipping at 1.0, and a hard stop on a non-finite loss.
  - `predict_proba`: probabilities whose columns follow `LABELS`. A model with another
    label order is refused, never silently relabeled.
  - `fit_predict_fold` and `run_cv_encoder`: the same prompt-grouped 5-fold CV as every
    other model, with a fresh pretrained model for each fold.
  - `find_fold_dirs` and `predict_proba_folds`: for the offline submission, find the 5
    saved fold models and average their predictions.
- `EncoderConfig`, and two configs: `deberta_v3_base.yaml` and `modernbert_base.yaml`.
  They write `lr: 2.0e-5`, not `2e-5`, because PyYAML reads `2e-5` as a string.
- Tests on tiny random models built offline (`make_tiny_encoder` in `tests/conftest.py`):
  a word-level tokenizer and a 2-layer DeBERTa or ModernBERT, so no test downloads
  anything.
- A new CI job, `torch`, which installs CPU-only torch so that the model tests run on every
  PR.

**Checked:** 10 encoder tests; the whole suite passed with torch installed (83 tests).

**What went wrong, and the fix:**
- A helper `_free(model)` couldn't free the GPU memory: deleting a name inside a function
  doesn't delete the caller's variable. Fix: callers `del model`, then call
  `_release_memory()`.
- The test that saved fold models reload correctly couldn't tell two fold models apart. At
  1 to 3 epochs the tiny models still predicted the uniform guess, so a swap changed the
  output by only 1e-5. Fix: 20 epochs, after which a swap changes it by at least 0.117.
- A commit was blocked by the line-length check, yet the ledger had already recorded the
  task as complete. Cause: `pre-commit run --all-files` skips files that git doesn't track
  yet. Fix: stage the files before running the hooks, and record completion only after
  the commit succeeds.

### Task 2: Optuna for the encoder (PR #23)

**Built**, in `tune.py`:
- `suggest_encoder`: samples the three knobs that matter most when fine-tuning: the
  learning rate (log scale, 5e-6 to 1e-4), the number of epochs (2 to 5) and the warmup
  (0 to 0.2).
- `run_encoder_study`, built to be cheap because each trial is a GPU run:
  - proxy folds: a trial trains on 2 of the 5 folds (0 and 1);
  - a warm start: trial 0 is always the default config, so tuning can't end below it;
  - pruning: after its first fold, a trial below the median of earlier trials stops.
- `best_encoder_config`: the winner as a config, with the standard seed 42.

**Checked:** `test_run_encoder_study_returns_a_valid_config`: trial 0 is the default, and
the result survives a round trip through YAML.

### Task 3: blending three models (PR #24)

**Built:**
- `best_blend_weights` in `blend.py`: it tries every combination of weights in steps of
  0.05 that sums to 1 (231 combinations for 3 models) and keeps the best. On a tie, the
  newcomer gets the smaller weight, so a model that adds nothing gets 0.
- `EnsembleConfig`: the three weights (text model, LM head, encoder) and the names of the
  configs they reuse.
- `predict_submission_ensemble` in `submit.py`, plus a refactor: the steps shared by the
  three submission functions became small helpers, with the same behavior.

**What went wrong, and the fix:** a mutation check showed that no test covered the tie
rule: removing the `reversed()` that implements it still passed every test. Fix: a new
test, `test_best_blend_weights_gives_a_redundant_newcomer_no_weight`, which fails under
that mutation.

### Task 4: the head-to-head, notebook 07 (PR #25, 38 GPU-minutes)

**Did:** [encoder-select](https://www.kaggle.com/code/ryqn00/name-that-llm-encoder-select)
fine-tuned DeBERTa-v3-base and ModernBERT-base on the same seed-7 folds, each at its usual
settings. A fail-fast step came first: parameter counts, peak GPU memory, and a check that
both tokenizers keep curly quotes and the spaced em dash, which are strong style clues.

**Before pushing:** a local run of the whole notebook with tiny random stand-ins.

| model | seed-7 CV | note |
|---|---|---|
| **ModernBERT-base** | **0.344** | folds 0.326 to 0.369, 15 min |
| DeBERTa-v3-base | 0.206 | underfit: its loss barely left the uniform guess |
| TF-IDF `lr`, the bar to beat | 0.330 | |

**What went wrong, and the fix:**
- The local run crashed on `torch.cuda.reset_peak_memory_stats()`, which needs a GPU. Fix:
  measure GPU memory only when a GPU is present.
- Since DeBERTa underfit, the comparison could be an artifact of its learning rate.
  Decision: give it a rematch at ModernBERT's settings before tuning (judgment call 6).

**Found along the way:** Kaggle's T4 accelerator has 2 GPUs, and we use one (idea I-019).

### Task 5: tuning, notebook 08 (PR #26, 59 GPU-minutes)

**Did:** [encoder-tune](https://www.kaggle.com/code/ryqn00/name-that-llm-encoder-tune):
1. The rematch: DeBERTa at ModernBERT's settings scored 0.171, and one of its folds never
   learned. So ModernBERT was the one tuned.
2. 10 Optuna trials on folds 0 and 1, in 41 minutes; 7 were pruned after one fold.

| trial | lr | epochs | warmup | folds 0 and 1 |
|---|---|---|---|---|
| 0, the default | 5.0e-5 | 3 | 0.10 | 0.344 |
| **2, the best** | 4.4e-5 | **5** | 0.11 | **0.374** |

The winner was saved as `configs/encoder.yaml`, with a test that loads it.

**Lessons:**
- Optuna's TPE sampler draws its first 10 trials at random by default. With exactly 10
  trials, it never learned anything (idea I-020).
- The best trial used 5 epochs, the top of the range, so more epochs might help (I-020).
- A text search over the trials table missed the pruned rows, whose spacing differs.
  Reading the CSV file with pandas is safer.

### Task 6: the final run and the A/B gate, notebook 09 (PR #27, 122 GPU-minutes)

**Did:** [encoder-final](https://www.kaggle.com/code/ryqn00/name-that-llm-encoder-final)
trained the tuned encoder on 4 fold seeds (20 runs of about 5 minutes each), next to the
text model and the LM head:
- **Seed 7 chooses** the weights: text 0.70, LM head 0.15, encoder 0.15.
- **Seeds 42, 43 and 44 report**, on folds that no choice ever saw.
- **The gate:** a bootstrap over prompts on seed 42. Promote only if the whole 95%
  interval of "ensemble minus `lm_blend`" is above 0.
- The 5 seed-42 fold models were saved for the submission (289 MB each).

**Before pushing:** a local run with fake LM features. On that noise, the weight search
still gave the encoder a weight of 0.3, and the gate refused it (95% CI [−0.007, +0.003]).
That's the gate doing its job: a lucky weight can't ship.

| seed | `lr` | `lm_blend` | encoder | **ensemble** |
|---|---|---|---|---|
| 42 | 0.321 | 0.405 | 0.350 | **0.422** |
| 43 | 0.330 | 0.400 | 0.354 | **0.425** |
| 44 | 0.325 | 0.402 | 0.356 | **0.423** |

The gate on seed 42: +0.017, 95% CI [+0.004, +0.030], so the ensemble was **promoted** and
saved as `configs/ensemble.yaml`, with a test that loads it.

**What went wrong, and the fix:**
- `yaml.safe_dump` can't write Python tuples, like the weights. Fix:
  `model_dump(mode="json")` turns them into lists first.
- W&B was off in this run, as in 07 and 08: reading the Kaggle Secret raised
  `ConnectionError`. Tracking is optional, so the runs went on. The fix is still open.

**Lesson:** with the same config, folds and seed, the encoder scored 0.374 on folds 0 and
1 in the tuning run, and 0.360 here. GPU training isn't bit-for-bit reproducible, so part
of the tuning gain was luck (idea I-022).

### Task 7: the submission (PRs #28 and #29)

1. **The notebook:** the 90-submit notebook got a `USE_ENCODER` switch. It averages the 5
   fold models on the test set, cutting passages at the training length (512 tokens), and
   blends the three models with the weights in `ensemble.yaml`. If the encoder's output
   for a passage isn't a number, that row becomes uniform, so the other two models decide.
2. **A local run:** a tiny Qwen, plus one tiny encoder copied into 5 fold folders (the
   layout notebook 09 writes). All cells ran, and a valid 3-row submission came out.
3. **Release v0.7.0:** the version bump, the PR, then the tag. GitHub Actions published the
   wheel to the Kaggle dataset `name-that-llm-lib`. Two checks: the dataset lists
   `name_that_llm-0.7.0`, and the wheel contains `configs/ensemble.yaml`.
4. **The push, with your OK:** 28 minutes on the T4. The 5 fold models were found, the
   3-row output passed the check, and the hidden rerun was estimated at 19 minutes (the
   limit is 9 hours).
5. **The submission, with your go-ahead:** **public LB 0.410**, +0.021 over 0.389, the same
   gain as on fresh folds.
6. **The record:** Context, DIARY (Session 6) and IDEAS (I-021 to I-023), in PR #29.

## Every judgment call

The plan didn't settle these, so Claude decided, and logged each with its cost if wrong.

1. No git worktree: one branch per task off `main`, shipped by PR (the project's usual
   procedure). Cost if wrong: none.
2. The fold-model round-trip test compares each saved model with its own out-of-fold
   predictions, and builds the expected average by hand, so it doesn't just repeat the
   code's logic. Cost if wrong: none.
3. That test trains 20 epochs instead of the plan's 1 (see Task 1). Cost if wrong: about
   10 seconds more test time.
4. The encoder tests skip one by one rather than per file, so the torch-free fold-folder
   test runs in every CI job. Cost if wrong: none.
5. Two blend tests beyond the plan: an informative third model gets found, and a redundant
   one gets weight 0 (see Task 3). Cost if wrong: none.
6. DeBERTa got a rematch at ModernBERT's settings before tuning (see Task 4). Cost if
   wrong: about 18 GPU-minutes.
7. Idea I-019 (2 GPUs) was logged as soon as it was found. Cost if wrong: none.
8. Notebook 08 starts with the rematch and tunes whichever model wins it. Cost if wrong:
   none.
9. The test prediction averages the 5 fold models, while the weights were chosen on
   single-model predictions, whose probabilities are sharper. It stands, since the
   leaderboard gain matched CV's. Cost if wrong: the encoder is slightly under-weighted on
   test.

## Open issues

- `EncoderConfig` doesn't check its ranges: `epochs: 0` would silently skip training.
- The submission attaches the latest output of notebook 09. Re-running 09 would swap the
  fold models while `ensemble.yaml` stays the same.
- W&B logged nothing in P5 (see Task 6). The likely fix: attach the `WANDB_API_KEY` secret
  to each GPU notebook in the Kaggle editor.

## Process lessons

- Long local runs go to the background: the shell's foreground limit is 10 minutes.
- Kaggle output downloads can fail without an error message, so the watchers retry once.
  They download only the small outputs (`--file-pattern`), never the 1.5 GB of fold
  models.
- A watcher may wait for a Kaggle run and check its output, but a submission is sent only
  in the foreground, right after an explicit OK.

## How to retrace P5 yourself

- The 8 changes, one per line: `git log --oneline 5dd91a5..602c2e6`.
- One task's diff, with `git show <commit>`: `35fac60` (Task 1), `708e63a` (2), `7cfe066`
  (3), `02e0917` (4), `f3064b3` (5), `5d76c5c` (6), `e22148f` (7, the submission) and
  `602c2e6` (7, the results).
- The pull requests keep the original commits and the CI runs:
  [#22](https://github.com/Ryqn0/name-that-llm/pull/22) to
  [#29](https://github.com/Ryqn0/name-that-llm/pull/29).
- The tests: `uv run --all-extras pytest -q`. The model tests need torch:
  `uv run --all-extras --with torch --with transformers pytest -q tests/test_encoder.py`.
- The notebooks, with their outputs and Findings, are on Kaggle (links in Tasks 4 to 6).
- The full conversation: see [the full sessions](README.md#the-full-sessions).
