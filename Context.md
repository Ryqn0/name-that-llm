# Context: Name That LLM (read me first)

> The project's living brief, for Claude and for humans. Keep it short and current:
> update **Status** and **Next steps** at the end of every session. The full detail lives
> in the [design](docs/superpowers/specs/2026-10-23-name-that-llm-design.md).

## Purpose

Learn the full production ML/DL/LLMOps lifecycle by building it for real on a live Kaggle
competition, and score high while doing it (target: a medal, i.e. the top 10%). The
outputs are a public repo, a progress diary, and a live demo app.

## Competition facts

- [Name That LLM](https://www.kaggle.com/competitions/name-that-llm): which of 18 LLMs wrote
  a passage? Metric: balanced accuracy; random guessing scores ≈ 0.056.
- `train.csv`: 2,699 rows (`id, prompt_id, text, model`) covering 150 prompts. 149 of those
  prompts were answered by all 18 models exactly once. The hidden test has ≈ 4,000 rows,
  all from prompts not in train.
- Code competition: an offline notebook (≤ 9 h) that writes `submission.parquet`
  (`id, model`). Limits: 5 submissions per day, 2 final picks.
- Deadlines: entry and team merge 2027-01-06; final submission 2027-01-13 23:59 UTC.
- All facts and rules: [design §3](docs/superpowers/specs/2026-10-23-name-that-llm-design.md#3-facts-verified-2026-10-23).
- **Kaggle environment** (EDA run, 2026-10-24):
  - Versions: Python 3.13.15, pandas 2.3.3, numpy 2.1.3, scikit-learn 1.6.1. CI's
    `kaggle-env` job tests exactly these.
  - Data path: `/kaggle/input/competitions/name-that-llm`.
  - **pandera is not preinstalled**; typeguard, typing_inspect and mypy_extensions are.
  - **GPU image** (LM-features run, 2026-10-24): torch 2.11.0+cu128, CUDA 12.8,
    transformers 5.16.1, Tesla T4. Extracting every layer for the 2,699 training passages
    took 1.4 / 2.7 / 6.2 min for Qwen 3 0.6B / 1.7B / 4B base, all finite in float16.
- **Label noise:** gemini-3.7-flash vs gemini-3.8-flash is barely separable (0.56 two-class
  accuracy, where a coin flip scores 0.50). Treat these two as near-indistinguishable.
- **EDA results:** [the notebook on Kaggle](https://www.kaggle.com/code/ryqn00/name-that-llm-eda)
  (its section 9 summarizes the findings).

## Status

- Phase: **P7 in progress** (LLM and ensemble:
  [design](docs/superpowers/specs/2026-10-27-p7-llm-ensemble-design.md),
  [plan](docs/superpowers/plans/2026-10-27-p7-llm-ensemble.md)). Tasks 1–6 of 9 are done.
- **P6** (the MLOps loop) was done on 2026-10-28:
  [plan](docs/superpowers/plans/2026-10-27-p6-mlops-loop.md),
  [journal](docs/journal/p6-mlops-loop.md),
  [ADR 0007](docs/adr/0007-watching-a-static-site.md). The final review's fixes shipped in
  #60.
- **P4** (App v1 and CD) was done on 2026-10-27
  ([plan](docs/superpowers/plans/2026-10-26-p4-app.md), [journal](docs/journal/p4-app.md)).
  - **Monitoring (P6):**
    - the live demo sends anonymous events (never the text) to the Grafana Cloud stack;
    - [the dashboard](https://sincerebagel1076.grafana.net/d/name-that-llm-demo), with
      alerts by email: more than 5 real errors in an hour, or the robot silent 12 hours;
    - a robot visitor every 6 hours (`robot.yml`);
    - a weekly export to DagsHub, Mondays (`weekly-export.yml`);
    - a weekly retraining that may propose a challenger by pull request, Sundays
      (`retrain.yml`);
    - **shadow mode is off since 2026-10-30** (`app/demo/CHALLENGER_RUN` is empty).
      - The last challenger, `lr` + the pool (MLflow run `f59454cd`, CV 0.4186), trained
        on the pool's old version.
      - That version included 2 sources its maintainer removed in the "clean" release.
        Some removals were for licence or provenance risk, and nothing published says
        which reason applied to ours.
      - The weekly retraining reads the clean pool, so its next proposal brings back a
        clean challenger.
  - **P7 so far:**
    - Sinkhorn balancing was dropped: its 95% CI includes 0 (I-027).
    - The pool of public generated passages is built, with its safeguards. With 30,000 of
      its passages, `lr` alone goes from CV 0.3212 to **0.4205** (+0.099, 95% CI [+0.080,
      +0.120]), as high as the whole champion ensemble's CV.
    - Only 5 of the pool's 56 sources are kept, as an allowlist in `configs/pool.yaml`.
      The other 51 lowered `lr` when added alone, but mostly through class imbalance
      (I-028).
    - `llm.py` is built and tested on CPU: LoRA on 4-bit Qwen, with two folds at once on
      Kaggle's two T4s.
    - **The first Qwen3-8B run** (notebook 10, pool-free, seed 7) took 4 h on two T4s:
      - the LLM alone scored **0.3641**, against the encoder's 0.3590;
      - the best 4-member blend scored **0.4523**, against the ensemble's 0.4293. That's
        in-sample on seed 7; Task 7 judges it on fresh seeds.
    - **The pool lifts the LLM too** (notebook 11, on proxy folds 0 and 1 of seed 7):
      - **0.373 → 0.444** with 5,000 pool passages per fold (+0.070, and both folds
        gained);
      - fewer blocks or shorter inputs only hurt;
      - `configs/llm.yaml` now has full depth, 512 tokens, and `pool_size` 5000.
      - **The cost:** 420 minutes per pair of folds with the pool. A full 5-fold run takes
        about 21 GPU-hours, which changes Task 7's budget.
    - **The pool in the ensemble's `lr` too (I-029, CPU only): promoted.** CV **0.4705**
      against the champion's 0.4219 on seed 42 (95% CI [+0.0315, +0.0652]), and +0.0456
      and +0.0393 on seeds 43 and 44. The weights are 0.75 / 0.10 / 0.15, chosen on seed 7.
    - **Task 7, session 1** (notebook 12): 1 epoch loses 0.045 (0.399 against 0.444), so
      the LLM keeps 2 epochs. One epoch took 122 minutes, so a pair of folds may take
      about 4 hours, not 7.
  - **The free public demo is live:** <https://huggingface.co/spaces/Ryqn/name-that-llm>.
    It's a static Space: the page runs `lr` in the visitor's browser (`app/demo/ntl.js`,
    parity-tested against scikit-learn), so the text never leaves the device. Its model is
    the MLflow run pinned in `app/demo/MODEL_RUN`. Every merge touching `app/demo/`
    redeploys and smoke-tests it ([ADR 0005](docs/adr/0005-public-demo-runs-in-the-browser.md)).
  - **The full champion** is `name-that-llm` v1 `@champion` in the private DagsHub
    registry. The FastAPI app and its Docker image run it on CPU, about 3.7 s a passage
    ([ADR 0004](docs/adr/0004-serve-the-champion-on-cpu.md)). It runs locally only: since
    2026, Hugging Face needs PRO for Docker Spaces (I-026).
    - **To run it:** load `C:\Users\dungr\.dagshub.env`, then run
      `uv run --all-extras --with torch --with transformers uvicorn name_that_llm.serve:app --port 7860`,
      and open <http://localhost:7860>.
    - **Measured on 2026-10-28:** ready after 410 s on the first start (downloads), then
      3.9 s a passage.
    - It loads whatever is `@champion` in the registry, so a new champion appears on the
      next restart.
  - Before that, P5 (fine-tuned encoders, which gave the champion LB 0.410) and P3 (prompt
    assignment, then LM features) were completed. Kaggle's T4 accelerator provides
    **2 GPUs**; we use one so far (I-019).
- **Lessons so far:**
  - P0–P2: public LB **0.320** against CV 0.321, so **the backtest tracks the leaderboard**.
  - P3a: per-prompt assignment lifted CV from 0.321 to 0.406, but the LB stayed at 0.320.
    The gain depends on the test's layout (+0.084 at 18 answers per prompt, about 0 at
    1–3), so prompt-structure tricks (I-001, I-007) are shelved. The base model is the
    lever.
  - P3b: Qwen 3 hidden states work. Blended with `lr`, they reach a CV of 0.402 (the mean
    of three fresh seeds) and **public LB 0.389** (+0.069). The CV–LB gap widened from
    −0.001 to −0.013, so watch it. Optuna didn't beat the hand-made sweep (I-012, I-018).
  - P5: a fine-tuned ModernBERT-base is weaker alone (0.353) but makes different mistakes,
    so a 15% share lifts the blend: +0.021 on fresh folds, and +0.021 on the public LB. The
    CV–LB gap held at −0.013. Fine-tuning is noisy (identical runs differed by 0.014 on two
    folds), so a small tuning gain needs a second seed (I-022).
  - P4: hosting terms change, so check them early: free Docker Spaces disappeared behind a
    paywall. A model small enough for the browser costs nothing to serve. And look at the
    page: a headless browser found two layout bugs that every HTTP check had passed.
- **Leaderboard snapshot, 2026-10-26:** 3,387 teams; median 0.532; top-10% cutoff 0.641;
  best 0.710. With 0.410 we're about 2,236th (about 2,219th on 2026-10-25, with 0.389).
- Champion: **`ensemble`** (since 2026-10-26), a weighted mean of three models'
  probabilities, 0.70 / 0.15 / 0.15:
  1. `lr`, the word + character TF-IDF logistic regression;
  2. the LM head, a logistic regression on Qwen 3 4B base layers 6–9 (together with `lr`,
     the previous champion `lm_blend`);
  3. ModernBERT-base fine-tuned on our 18 labels: the average of its 5 fold models.

  It beat `lm_blend` on fresh folds (+0.017 on seed 42, 95% CI [+0.004, +0.030]) and on the
  public LB (0.410 vs 0.389). The offline submission runs it on a T4 GPU in about 30
  minutes.

CV = the seed-42 folds, the same for every row.

| date | model | CV | LB | W&B run |
|---|---|---|---|---|
| 2026-10-29 | ensemble + pool (lr with 30,000 pool passages + LM head + ModernBERT, 0.75 / 0.10 / 0.15) | 0.4705 | — | — |
| 2026-10-26 | **ensemble** (lr + LM head + ModernBERT, 0.70 / 0.15 / 0.15) | 0.4219 | **0.410** | — |
| 2026-10-26 | encoder (ModernBERT-base, tuned) | 0.3501 | — | — |
| 2026-10-25 | lm_blend (lr + Qwen 3 4B layers 6–9) | 0.4052 | **0.389** | — |
| 2026-10-25 | lm_blend_tuned (Optuna's best, not promoted) | 0.3963 | — | — |
| 2026-10-24 | lr | 0.3212 | **0.320** | r0lx5cja |
| 2026-10-24 | lr + assign (topic groups) | 0.4056 | 0.320 | to8unxny |
| 2026-10-24 | nbsvm | 0.2971 | — | sxhd09jf |
| 2026-10-24 | mnb | 0.2852 | — | 7f8bv2zr |
| 2026-10-24 | svm | 0.2848 | — | 5p2yum3r |
| 2026-10-24 | cnb | 0.2474 | — | xzuywz5t |

## Decisions

- Design-time calls: [design §12](docs/superpowers/specs/2026-10-23-name-that-llm-design.md#12-calls-made-by-claude-to-check).
- Each significant new decision gets a file `docs/adr/NNNN-title.md`, covering context,
  decision and consequences.

## Conventions

- **Layout:**
  - `src/name_that_llm/`: the shared package
  - `tests/`: tests, using fake data only
  - `notebooks/<nn-name>/`: Kaggle notebooks plus their `kernel-metadata.json`
  - `scripts/` and `docs/`
- **Commands:** `uv sync --all-extras` · `uv run --all-extras pytest -q` ·
  `uv run pre-commit run --all-files`.
- **Local data (gitignored):** `kaggle competitions download name-that-llm -p data/raw`, then
  `Expand-Archive data/raw/name-that-llm.zip data/raw`. Load it with
  `name_that_llm.data.load_train()` and `load_test()`, never with a bare `pd.read_csv`:
  those functions keep ids as strings and validate the data.
- **Code style:** comments explain the **why**, generously. This is a learning project and
  that's the user's standing request.
- **Notebook sync:** Claude edits notebooks in the repo, then runs
  `kaggle kernels push -p notebooks/<dir>`. If the user edits a notebook on Kaggle, run
  `kaggle kernels pull <id> -p notebooks/<dir>` before Claude edits it again.
- **Ship procedure** (from Task 4 on):
  1. Create a branch and commit.
  2. Push, then `gh pr create --fill`.
  3. `gh pr checks --watch`.
  4. `gh pr merge --squash --delete-branch`.
- **Release (CD):** pushing a tag `vX.Y.Z` triggers the GitHub Actions workflow
  `release-lib`, which publishes a new version of the Kaggle dataset
  [ryqn00/name-that-llm-lib](https://www.kaggle.com/datasets/ryqn00/name-that-llm-lib):
  our wheel plus pandera and its helpers. The offline submission notebook installs from
  it.
  - The dataset's license field is `other`, because Kaggle's license list has no MIT and
    rejected `"MIT"` with a generic server error. The description states MIT.
  - The workflow uses the secret `KAGGLE_API_TOKEN`, a **dedicated** Kaggle token for
    GitHub Actions, so revoking it never breaks the local token.
- **Demo deploy (CD):** merging a change under `app/demo/` (or to `scripts/deploy_space.py`
  or its workflow) redeploys the static Space through `deploy-app`, which ends with a smoke
  test of the live site.
  - **A new demo model:** run `python -X utf8 scripts/export_browser_model.py`
    (`--dry-run` first), then commit the `app/demo/MODEL_RUN` it writes. Update the
    numbers in `app/demo/README.md` by hand.
  - **A new challenger** comes from the weekly retraining's pull request, which sets
    `app/demo/CHALLENGER_RUN`. Merging it is a person's decision.
  - **Promoting the challenger to champion:** a pull request that moves its run id from
    `CHALLENGER_RUN` into `MODEL_RUN` and empties `CHALLENGER_RUN`. Its description gives
    two sets of numbers: the offline gate's, from the challenger's MLflow run, and the live
    ones, from the dashboard and the `live-feedback` export. Update `app/demo/README.md`'s
    numbers by hand.
    - **Before promoting, re-measure the unusual-text limits with the challenger's own
      recipe** (P6's final review). `browser.unusual_thresholds` fits each fold's
      vocabulary on our passages only. For `lr` + the pool, add the pool to each fold's
      training rows. A larger vocabulary knows more words, so the unknown-word limit
      changes.
- **Monitoring as code:** edit `monitoring/dashboard.json` or `monitoring/alerts.json`, then
  run `python -X utf8 scripts/push_monitoring.py` with `C:\Users\dungr\.grafana.env` loaded.
  Every push to Grafana needs the user's OK.
- **A new champion version:** `python -X utf8 scripts/register_champion.py` with the
  DagsHub env loaded. MLflow prints an emoji, which Windows' cp1252 can't write without
  `-X utf8`.
- **Lesson (P3a):** before trusting any trick that exploits prompt structure, re-measure
  it on subsampled layouts (m = 1, 2, 3, 6 answers per prompt), because the hidden test
  is not laid out like train.
- **Journal:** before a plan's scratch workspace is deleted, its step-by-step record goes
  to `docs/journal/<phase>.md`: the steps, how each was checked, what broke and the fix,
  and every judgment call. [docs/journal/README.md](docs/journal/README.md) maps where
  each part of the record lives, including the session transcripts.
- **Attribution:** commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PR bodies end with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## Environment

- **Windows with Avast HTTPS scanning:** run `scripts/setup-local.ps1` once in each
  PowerShell version, and re-run it if Avast updates its certificate. It fixes three
  problems:
  - the Python crash caused by `SSLKEYLOGFILE`
  - uv's TLS errors, via `UV_SYSTEM_CERTS`
  - Python's TLS errors, via a combined certificate bundle at `~/.certs/combined-ca.pem`
- Kaggle notebooks and GitHub Actions aren't affected.
- **Avast also cuts large transfers over localhost** (a 13.5 MB file, 4 times out of 4). To
  look at a local page, drive headless Edge with Playwright and answer its requests from
  the folder (`uv run --no-project --with playwright`, `channel="msedge"`), rather than
  serving it.
- **Sleep pauses long local runs.** An overnight CPU check stood still while the PC slept,
  and resumed on waking. To check a run, compare its CPU time with its age, not just
  whether it's alive.
- **With uv on Windows, the venv's `python.exe` is only a launcher.** The work happens in a
  child process, `python3.11.exe`: measure that one.
- **A process started through WMI (`Win32_Process.Create`) survives the end of a Claude
  session.** Processes started from the session's own shell don't.
- **Package floors are kept loose on purpose:** Kaggle's preinstalled versions are older,
  and the offline notebook can't upgrade them (see the comment in `pyproject.toml`).

## Accounts and secrets (names only, never values)

- Kaggle `ryqn00`: token in `~/.kaggle/access_token`.
- GitHub `Ryqn0`: repo `Ryqn0/name-that-llm`, public.
- W&B: project `name-that-llm`, which must stay **private** because its `train-data`
  artifact holds competition data. Locally the key comes from `wandb login`.
  - **Kaggle doesn't hand secrets to runs pushed through its API**, so the GPU notebooks
    log nothing to W&B, even with the Kaggle Secret `WANDB_API_KEY` attached. Their
    results are files, logged from the PC.
  - It lives in the one-member team `dungryan0-ryan-dung` with visibility "Team", which is
    effectively private. **Before adding any teammate, switch the project to Private.**
  - Runs: [forge.coreweave.com/wandb/dungryan0-ryan-dung/name-that-llm](https://forge.coreweave.com/wandb/dungryan0-ryan-dung/name-that-llm).
    W&B now runs on CoreWeave's domain, and wandb 0.31 stores its login under
    `api.forge.coreweave.com` in `~/_netrc`.
- DagsHub `Ryqn0`: the **private** repo `Ryqn0/name-that-llm`, home of the MLflow registry
  (`name-that-llm` v1 `@champion`) and the experiment `browser-demo`. Locally, the
  `MLFLOW_TRACKING_URI`, `_USERNAME` and `_PASSWORD` variables come from
  `C:\Users\dungr\.dagshub.env`, outside the repo. In GitHub: the secret `DAGSHUB_TOKEN`
  and the variables `DAGSHUB_USER` and `MLFLOW_TRACKING_URI`.
  - **To load it, and `.grafana.env`,** use a `while read` loop that also reads a last line
    without a newline, and strips the Windows carriage returns:
    `while IFS='=' read -r k v || [ -n "$k" ]; do [ -n "$k" ] && export "$k=${v%$'\r'}"; done < FILE`.
    Without the strip, DagsHub answers 401.
- Hugging Face `Ryqn` (not `Ryqn0`): the public static Space `Ryqn/name-that-llm`. In
  GitHub: the secret `HF_TOKEN` and the variable `HF_SPACE`. The variable `ENCODER_FOLDS`
  is left over from the Docker deploy, and unused.
- Grafana Cloud: the stack `sincerebagel1076` (<https://sincerebagel1076.grafana.net>).
  - A free account gets one stack. A 14-day trial, then the free tier, with logs kept
    about 2 weeks.
  - The Frontend Observability app is `name-that-llm-demo`, and its allowed origin is the
    Space's host.
  - Locally, `C:\Users\dungr\.grafana.env` holds:
    - `GRAFANA_URL`;
    - `GRAFANA_TOKEN`, from the service account `push-monitoring`, role Editor;
    - `GRAFANA_LOGS_URL`, `GRAFANA_LOGS_USER` and `GRAFANA_LOGS_TOKEN` (an access policy
      `read-demo-logs`, `logs:read`).
  - In GitHub: the variables `GRAFANA_COLLECTOR_URL`, `GRAFANA_LOGS_URL` and
    `GRAFANA_LOGS_USER`, and the secret `GRAFANA_LOGS_TOKEN`.
  - Alerts go to the contact point `email`, the default policy's.
- GitHub secret `RETRAIN_PR_TOKEN`: a fine-grained token (this repo only; contents and pull
  requests, write) for the weekly retraining's pull requests. A pull request opened with the
  workflow's own token wouldn't run CI.

## Rules checklist

- Never commit competition data: `data/` is gitignored, and the tests use fake data.
- Committed notebooks have their outputs stripped (pre-commit does this) and show only
  aggregate numbers.
- Public code must also be shared on Kaggle: Foundational Rules §6.b, *"on the discussion
  forum or notebooks associated specifically with the Competition"*. **Status: covered** by
  the public EDA notebook
  [ryqn00/name-that-llm-eda](https://www.kaggle.com/code/ryqn00/name-that-llm-eda), which is
  attached to the competition and links the repo in its first cell. A forum post is
  optional extra visibility.
- The final ensemble notebook stays private until 2027-01-13, then gets published.
- Kaggle pushes, dataset creation and submissions each need the user's explicit OK.

## Next steps

State at the end of session 10 (2026-10-29): **P7 Tasks 1–6 are done, and I-029 is
promoted and ready to submit.** Task 7's session 2 is running on Kaggle (notebook 12,
version 3, pushed 17:32, at most 10.5 hours): seed 42 at 2 epochs, as many folds as fit,
keeping the adapters. The ledger: `.superpowers/sdd/2026-10-27-p7-llm-ensemble/progress.md`.

1. **Submit I-029's ensemble** (CV 0.4705, against 0.4219; with the usual −0.013 gap,
   about 0.457 on the LB, against 0.410).
   - The code (#64) and lib v0.9.0 are ready: `notebooks/90-submit` selects
     `ensemble_pool`.
   - Push it, then submit; each needs the user's OK. Its run needs about an hour of GPU.
   - Do it once session 2 has ended, so the quota left is known (the user had 11+ hours
     before session 2).
2. **Session 2's results:**
   - download them: `kaggle kernels output ryqn00/name-that-llm-llm-final -p
     data/kaggle-outputs/llm-final-s2`;
   - log them with `scripts/log_gpu_run.py`.
   - **Keep every adapter:** a later version of the notebook replaces this one's outputs,
     and the submission needs all 5 folds. The folds left go to session 3.
3. **Task 7's ensemble, on the PC:** the LLM's seed-42 OOF, plus `lr` with the pool, the
   LM head and the encoder.
   - Choose the weights by nested CV inside seed 42, since the LLM has no other seed.
   - Then the A/B gate against the I-029 ensemble.
4. **Task 8:** the four-member submission.
5. **Now and then:**
   - look at the dashboard;
   - answer "Do you know who wrote it?" on texts whose author you know;
   - weekly: the retraining may open a challenger PR, which is a person's decision.
6. **Suggested, your setting:** branch protection on main (pull requests only, CI
   required). The retraining's token can write to the repo, and protection keeps main
   reachable through reviewed, tested pull requests only.
7. Cheap follow-ups:
   - I-015: a calibrated or log blend, in P7 Task 7;
   - I-017: TF without IDF;
   - I-018: the weight grid inside each Optuna trial;
   - I-020 and I-022: encoder tuning (more epochs, and the noise floor);
   - I-028: a fairer check of the pool's other 51 sources, only if the pool runs short.
