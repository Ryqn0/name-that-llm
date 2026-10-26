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

- Phase: **P5 in progress**: fine-tuned encoders, DeBERTa-v3-base vs ModernBERT-base
  ([plan](docs/superpowers/plans/2026-10-25-p5-encoders.md)). The user chose the order
  P5, then P4, on 2026-10-25.
  - **Task 4 (2026-10-26):** ModernBERT-base wins the encoder head-to-head, at seed-7 CV
    **0.344** against DeBERTa-v3-base's 0.206 and TF-IDF `lr`'s 0.330. P5's exit
    criterion is met.
    - DeBERTa underfit at its default lr, so notebook 08 re-checks it at ModernBERT's
      settings before tuning.
    - Kaggle's T4 accelerator provides **2 GPUs**; we use one so far (I-019).
  - Before that, P3 was completed: P3a (prompt assignment) and P3b (LM features, then
    tuning, [plan](docs/superpowers/plans/2026-10-24-p3b-lm-features.md)).
- **Lessons so far:**
  - P0–P2: public LB **0.320** against CV 0.321, so **the backtest tracks the leaderboard**.
  - P3a: per-prompt assignment lifted CV from 0.321 to 0.406, but the LB stayed at 0.320.
    The gain depends on the test's layout (+0.084 at 18 answers per prompt, about 0 at
    1–3), so prompt-structure tricks (I-001, I-007) are shelved. The base model is the
    lever.
  - P3b: Qwen 3 hidden states work. Blended with `lr`, they reach a CV of 0.402 (the mean
    of three fresh seeds) and **public LB 0.389** (+0.069). The CV–LB gap widened from
    −0.001 to −0.013, so watch it. Optuna didn't beat the hand-made sweep (I-012, I-018).
- **Leaderboard snapshot, 2026-10-25:** 3,231 teams; median 0.526; top-10% cutoff 0.639;
  best 0.703. We're about 2,219th (from about 2,687th on 2026-10-24).
- Champion: **`lm_blend`** (since 2026-10-25): `lr`, the word + character TF-IDF logistic
  regression, blended with a logistic regression on Qwen 3 4B base layers 6–9 at LM weight
  0.1.
  - It beat `lr` on fresh folds (+0.084, 95% CI [+0.066, +0.101]) and on the public LB
    (0.389 vs 0.320).
  - The offline submission runs it on a T4 GPU in about 20 minutes.

CV = the seed-42 folds, the same for every row.

| date | model | CV | LB | W&B run |
|---|---|---|---|---|
| 2026-10-25 | **lm_blend** (lr + Qwen 3 4B layers 6–9) | 0.4052 | **0.389** | — |
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
- **Lesson (P3a):** before trusting any trick that exploits prompt structure, re-measure
  it on subsampled layouts (m = 1, 2, 3, 6 answers per prompt), because the hidden test
  is not laid out like train.
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
- Later: DagsHub (MLflow), Hugging Face (`HF_TOKEN`), Grafana Cloud.

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

State on 2026-10-26: P5 Tasks 1–4 are done. The library is on `main` (PRs #22–#24), and
07-encoder-select ran: ModernBERT-base wins. Progress lives in the ledger,
`.superpowers/sdd/2026-10-25-p5-encoders/progress.md`, and the notebook builders are in its
`scripts/` folder.

1. **Tasks 5 and 6**, the tuning (08) and final (09) GPU notebooks:
   - 08 first re-runs DeBERTa at ModernBERT's settings, then tunes whichever scores
     higher.
   - Push only after the user's OK in the current session: approvals don't carry over
     between sessions.
   - The submission (Task 7) always needs its own OK.
2. **Then P4, App v1 and CD:** real-time inference, Docker, CD, a model card, and a live
   demo serving the champion.
3. Cheap follow-ups: I-015 (calibrated or log blend, in P7's M6 ensemble), I-017 (TF
   without IDF), I-018 (the weight grid inside each Optuna trial).
