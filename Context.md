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

- Phase: **P4 done** (2026-10-27): App v1 and CD
  ([plan](docs/superpowers/plans/2026-10-26-p4-app.md),
  [journal](docs/journal/p4-app.md)). **Next:** the user picks P6 (the MLOps loop) or P7
  (LLM and ensemble).
  - **The free public demo is live:** <https://huggingface.co/spaces/Ryqn/name-that-llm>.
    It's a static Space: the page runs `lr` in the visitor's browser (`app/demo/ntl.js`,
    parity-tested against scikit-learn), so the text never leaves the device. Its model is
    the MLflow run pinned in `app/demo/MODEL_RUN`. Every merge touching `app/demo/`
    redeploys and smoke-tests it ([ADR 0005](docs/adr/0005-public-demo-runs-in-the-browser.md)).
  - **The full champion** is `name-that-llm` v1 `@champion` in the private DagsHub
    registry. The FastAPI app and its Docker image run it on CPU, about 3.7 s a passage
    ([ADR 0004](docs/adr/0004-serve-the-champion-on-cpu.md)). It runs locally only: since
    2026, Hugging Face needs PRO for Docker Spaces (I-026).
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
- **Package floors are kept loose on purpose:** Kaggle's preinstalled versions are older,
  and the offline notebook can't upgrade them (see the comment in `pyproject.toml`).

## Accounts and secrets (names only, never values)

- Kaggle `ryqn00`: token in `~/.kaggle/access_token`.
- GitHub `Ryqn0`: repo `Ryqn0/name-that-llm`, public.
- W&B: project `name-that-llm`, which must stay **private** because its `train-data`
  artifact holds competition data. Locally the key comes from `wandb login`; in Kaggle
  notebooks, from the Kaggle Secret `WANDB_API_KEY`.
  - It lives in the one-member team `dungryan0-ryan-dung` with visibility "Team", which is
    effectively private. **Before adding any teammate, switch the project to Private.**
  - Runs: [forge.coreweave.com/wandb/dungryan0-ryan-dung/name-that-llm](https://forge.coreweave.com/wandb/dungryan0-ryan-dung/name-that-llm).
    W&B now runs on CoreWeave's domain, and wandb 0.31 stores its login under
    `api.forge.coreweave.com` in `~/_netrc`.
- DagsHub `Ryqn0`: the **private** repo `Ryqn0/name-that-llm`, home of the MLflow registry
  (`name-that-llm` v1 `@champion`) and the experiment `browser-demo`. Locally, the
  `MLFLOW_TRACKING_URI`, `_USERNAME` and `_PASSWORD` variables come from
  `C:\Users\dungr\.dagshub.env`, outside the repo; load it with a `while read` loop that
  also reads a last line without a newline. In GitHub: the secret `DAGSHUB_TOKEN` and the
  variables `DAGSHUB_USER` and `MLFLOW_TRACKING_URI`.
- Hugging Face `Ryqn` (not `Ryqn0`): the public static Space `Ryqn/name-that-llm`. In
  GitHub: the secret `HF_TOKEN` and the variable `HF_SPACE`. The variable `ENCODER_FOLDS`
  is left over from the Docker deploy, and unused.
- Later: Grafana Cloud.

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

State on 2026-10-27: **P4 is done**
([journal](docs/journal/p4-app.md)). The free demo is live, and the champion (LB 0.410)
is registered and runs locally.

1. **Choose the next phase:**
   - **P7, LLM and ensemble:** the score lever. The top-10% cutoff is 0.641, against our
     0.410.
   - **P6, the MLOps loop:** monitoring, online A/B, continuous training. Its monitoring
     was designed around a hosted server, but the public demo now runs in the browser and
     has none. So P6 needs a rethink first: monitor the full app locally, or host it
     (I-026). I-024 and I-025 belong there.
2. **Fix W&B in the GPU notebooks:** every pushed P5 run printed "W&B tracking is OFF
   (ConnectionError)", so P5 has no W&B runs. Most likely the Kaggle Secret
   `WANDB_API_KEY` must be attached to each notebook once, in the Kaggle editor
   (Add-ons → Secrets).
3. Cheap follow-ups: I-015 (calibrated or log blend, in P7's M6 ensemble), I-017 (TF
   without IDF), I-018 (the weight grid inside each Optuna trial), I-020 and I-022 (encoder
   tuning: more epochs, and the noise floor).
4. **Optional:** tag `v0.8.0` (`pyproject.toml` already says 0.8.0). It would publish the
   Kaggle lib dataset, which needs the user's OK, but the offline notebook needs nothing
   from P4.
