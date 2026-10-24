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
- **Label noise:** gemini-3.7-flash vs gemini-3.8-flash is barely separable (0.56 two-class
  accuracy, where a coin flip scores 0.50). Treat these two as near-indistinguishable.
- **EDA results:** [the notebook on Kaggle](https://www.kaggle.com/code/ryqn00/name-that-llm-eda)
  (its section 9 summarizes the findings).

## Status

- Phase: **P2 Backtest and baselines**.
  - Done: P0, and P1 (the data layer and EDA). From P2: the five baseline models (Task 7)
    and the backtesting core (Task 6: prompt-grouped folds, metrics, and the paired prompt
    bootstrap written by the user).
  - Next: the CV runner and W&B tracking (Task 8).
  - Plan: [P0–P2](docs/superpowers/plans/2026-10-24-p0-p2-foundations-to-first-submission.md).
- Champion: none yet.

| date | model | CV | LB | W&B run |
|---|---|---|---|---|

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

1. The user writes the `prompt_bootstrap_ci` body (Task 6, user contribution).
2. Task 8: CV runner and W&B tracking (the user creates a free W&B account).
