# P0–P2: Foundations to First Leaderboard Submission — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Go from an empty folder to a public, CI-tested repo with a data layer, backtesting,
five baseline models tracked in W&B, and a first scored Kaggle submission produced by an
offline notebook.

**Architecture:** One small Python package (`src/name_that_llm`) holds the code: loading,
validation, cleaning, folds, metrics, models, training and submission. Kaggle notebooks import
that same package. Notebooks with internet on install it from a GitHub tag. The offline
submission notebook installs it from wheel files published as a Kaggle Dataset. Tests run on a
tiny *fake* dataset, so competition data never enters the repo.

**Tech Stack:** uv, Python ≥ 3.11, pandas, numpy, pyarrow, scikit-learn, pandera ≥ 0.24,
pydantic v2, PyYAML, wandb (optional extra), pytest, ruff, pre-commit, nbstripout, GitHub
Actions, Kaggle CLI ≥ 2.2.3.

**Spec:** `docs/superpowers/specs/2026-10-23-name-that-llm-design.md`. This plan covers
phases P0, P1 and P2 of §10. Later phases get their own plans, written when we reach them and
shaped by what these phases teach us.

## Global Constraints

These apply to every task.

**Identifiers**
- Repo: `github.com/Ryqn0/name-that-llm`, public, MIT license.
- Kaggle user: `ryqn00`. Competition slug: `name-that-llm`.
- Package: distribution `name-that-llm`, import name `name_that_llm`. `requires-python = ">=3.11"`.

**The 18 labels (exact strings):**
`claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5,
gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, glm-5.3,
glm-5.3-flash, gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, kimi-k3,
qwen3.8-max`.

**Lab of each model:** `claude-*` is anthropic, `gemini-*` is google, `glm-*` is zai, `gpt-*`
is openai, `kimi-*` is moonshot, `qwen*` is qwen.

**Competition rules**
- Metric: balanced accuracy, the average of each class's recall.
- Submission: `/kaggle/working/submission.parquet` with columns `id` and `model`. Every test
  `id` appears exactly once, with no nulls and only the 18 labels. The submission notebook runs
  with internet off and must finish within 9 hours.

**Data rules**
- Never commit competition data. `data/` is gitignored.
- Notebooks are committed with outputs stripped (nbstripout).
- Committed notebooks show aggregate statistics only, never raw passages.

**Cleaning:** apply only Unicode NFC and trimming of outer whitespace. Keep em dashes, curly
versus straight quotes, punctuation spacing and casing.

**Validation:** GroupKFold on `prompt_id`, 5 folds, seed 42. Every experiment produces
out-of-fold (OOF) probabilities for every training row.

**Tracking:** W&B, project `name-that-llm`, kept **private** because it holds competition
data. W&B is optional at runtime: code must run without it.

**Docs and comments:** comments explain the *why*, heavily. That's the user's standing request
for this learning project. Every session ends with a `DIARY.md` entry and a `Context.md`
status update.

**Account-visible actions:** pushing a Kaggle notebook, creating a Kaggle dataset, or
submitting needs the user's explicit "yes" in the session, given just before running it.
State the exact command and its effect when asking. The user posts the Kaggle forum post
themselves.

**Ship procedure** (Tasks 4–11, once CI exists):
1. `git switch -c <branch>`
2. Commit. Messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
3. `git push -u origin <branch>`
4. `gh pr create --fill`. PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
5. `gh pr checks --watch`, then continue only if CI is green.
6. `gh pr merge --squash --delete-branch`, then `git switch main` and `git pull`.

**Tagging:** bump the version inside the PR with `uv version --bump minor`. After the merge,
run `git tag vX.Y.Z` and `git push origin vX.Y.Z` on `main`.

**Avast in Claude's shell:** until Claude Code restarts after Task 1, prefix Claude's local
Python, uv and Kaggle commands with
`Remove-Item env:SSLKEYLOGFILE -ea 0; $env:UV_SYSTEM_CERTS='1'; $env:REQUESTS_CA_BUNDLE="$HOME\.certs\combined-ca.pem"; $env:SSL_CERT_FILE=$env:REQUESTS_CA_BUNDLE;`.
The user's own terminals are fixed by Task 1.

## Review Focus

These are five inputs that break things quietly. Each has a test in the task that owns the code.

1. **Ids that look like numbers.** An id like `000123` must stay `"000123"` from loading
   through to the submission. pandas would turn it into the integer `123`. Tested in Task 4
   (`test_load_train_keeps_ids_as_strings`) and Task 10
   (`test_predict_submission_is_valid_and_keeps_test_order`).
2. **Unusual rows in the hidden test.** Empty, missing, very long, emoji or non-English text
   must still get a valid label and must not crash the scoring run. Tested in Task 4
   (`test_load_test_survives_missing_text`) and Task 10
   (`test_predict_submission_survives_odd_texts`).
3. **A model missing from a training fold.** If a fold never trains on some model, the
   probability columns must not shift. Tested in Task 8
   (`test_run_cv_keeps_columns_aligned_when_a_class_is_missing`).
4. **The Kaggle data path.** It may be `/kaggle/input/name-that-llm` or
   `/kaggle/input/competitions/name-that-llm`. Loading must find either, accept an
   `NTL_DATA_DIR` override, and say what it tried when it fails. Tested in Task 4
   (`test_find_data_dir_*`); Task 5 prints the real path.
5. **W&B unavailable.** Training and submission must still work offline, with no API key, or
   when the network is broken by Avast. Tested in Task 8 (`test_main_runs_without_tracking`)
   and Task 10 (`test_submit_and_train_never_import_wandb`).

---

### Task 1: Local environment fix (Avast) and Kaggle CLI

**Files:**
- Create: `scripts/setup-local.ps1`

**Interfaces:**
- Produces: a user-level environment where `kaggle`, `uv` and Python HTTPS work in a new
  terminal. It creates the combined certificate bundle at `$HOME\.certs\combined-ca.pem`.

- [ ] **Step 1: Write `scripts/setup-local.ps1`.** It must be idempotent: running it twice
  changes nothing. It prints each change it makes and does the following, in order:
  1. If `C:\ProgramData\Avast Software\Avast\wscert.pem` is missing, print "No Avast
     certificate found, nothing to do" and exit 0.
  2. In the current process, remove `SSLKEYLOGFILE` and set `UV_SYSTEM_CERTS=1`.
  3. Find certifi's bundle with
     `uv run --no-project --with certifi python -I -c "import certifi; print(certifi.where())"`.
  4. Write certifi's bundle followed by Avast's `wscert.pem` to `$HOME\.certs\combined-ca.pem`.
  5. Set the user-level environment variables `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` to that
     file, and `UV_SYSTEM_CERTS=1`, with
     `[Environment]::SetEnvironmentVariable(..., "User")`.
  6. Create `$PROFILE` if it's missing. Append
     `Remove-Item Env:SSLKEYLOGFILE -ErrorAction SilentlyContinue  # Avast injects this; it crashes Python's ssl module`
     unless that line is already there.
  7. Run `uv tool install "kaggle>=2.2.3"`.
  8. Print: "Open a NEW terminal. If Avast updates its certificate, re-run this script."

  Comment heavily, explaining each Avast symptom and its fix (spec §9.4).

- [ ] **Step 2: Ask the user to run the script.** It changes their persistent environment, so
  it's their call. Command:
  `powershell -ExecutionPolicy Bypass -File scripts\setup-local.ps1`.
  Ask them to run it in every PowerShell version they use (5.1 and 7 have different profiles).

- [ ] **Step 3: Verify in a NEW terminal** (the user runs these):
  - `kaggle competitions files name-that-llm`. Expected: lists `sample_submission.parquet`,
    `test.csv` and `train.csv`.
  - `uv run --no-project --with requests python -c "import requests; print(requests.get('https://storage.googleapis.com').status_code)"`.
    Expected: an HTTP status code, with no `SSLError` and no `OPENSSL_Uplink` crash.

- [ ] **Step 4: No commit yet.** The repo starts in Task 2, and this script is committed
  there.

---

### Task 2: Repo foundation and public GitHub repo

**Files:**
- Create: `pyproject.toml` and `uv.lock` (via uv), `.python-version`, `.gitignore`,
  `LICENSE`, `README.md`, `CLAUDE.md`, `Context.md`, `DIARY.md`, `IDEAS.md`,
  `.pre-commit-config.yaml`, `src/name_that_llm/__init__.py`, `tests/test_smoke.py`
- Already present, and committed here: `docs/superpowers/specs/…-design.md`, this plan,
  `scripts/setup-local.ps1`

**Interfaces:**
- Produces: `name_that_llm.__version__: str`, read from installed package metadata. Also an
  installed dev environment (`uv sync`) and an `origin` remote on GitHub.

- [ ] **Step 1: Scaffold the project with uv.** From the repo root:
  - `uv init --package --name name-that-llm --python 3.11` (this also runs `git init`)
  - `uv add numpy pandas pyarrow scikit-learn "pandera[pandas]>=0.24" "pydantic>=2.5" pyyaml`
  - `uv add --optional track "wandb>=0.19"`
  - `uv add --dev pytest ruff pre-commit`

  Expected: `uv.lock` exists, and `uv run python -c "import name_that_llm"` exits 0.

- [ ] **Step 2: Write the failing smoke test** in `tests/test_smoke.py`:

```python
import importlib.metadata

import name_that_llm


def test_package_imports_and_reports_its_version():
    # The version comes from installed metadata, so it can never drift from pyproject.toml.
    assert name_that_llm.__version__ == importlib.metadata.version("name-that-llm")
```

- [ ] **Step 3: Run it.** `uv run pytest -q`. Expected: FAIL, because `__version__` is not
  defined yet.

- [ ] **Step 4: Implement `__version__` in `src/name_that_llm/__init__.py`** with
  `importlib.metadata.version("name-that-llm")`.

- [ ] **Step 5: Configure tools in `pyproject.toml`:**
  - `[tool.ruff]`: `line-length = 100`, `target-version = "py311"`
  - `[tool.ruff.lint]`: `select = ["E", "F", "I", "UP", "B", "SIM"]`
  - `[tool.pytest.ini_options]`: `testpaths = ["tests"]`

- [ ] **Step 6: Write `.pre-commit-config.yaml`** with these hooks, pinned by running
  `uv run pre-commit autoupdate`:
  - `astral-sh/ruff-pre-commit`: `ruff-check --fix` and `ruff-format`
  - `kynan/nbstripout`: `nbstripout`
  - `pre-commit/pre-commit-hooks`: `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`,
    `check-added-large-files` (`--maxkb=1000`), `detect-private-key`

  Then run `uv run pre-commit install`.

- [ ] **Step 7: Write the docs.**
  - **`.gitignore`:** append `data/`, `models/`, `wandb/`, `dist/`, `dist-kaggle/`, `*.zip`,
    `submission.parquet`, `.env`, `.ipynb_checkpoints/`.
  - **`LICENSE`:** MIT, "Copyright (c) 2026 Ryqn0".
  - **`CLAUDE.md`:** exactly one line, `@Context.md`.
  - **`Context.md`** sections:
    - *Purpose* (2 lines)
    - *Competition facts* (short, linking spec §3)
    - *Status*: phase P0; champion: none; a CV and leaderboard table with columns `date`,
      `model`, `CV`, `LB`, `W&B run`
    - *Decisions* (link to spec §12; significant new decisions go in `docs/adr/NNNN-title.md`)
    - *Conventions*: layout; commands; heavy "why" comments; notebook sync rule (edit in the
      repo, then `kaggle kernels push`; if the user edits on Kaggle, run
      `kaggle kernels pull` before Claude edits again); the Ship procedure
    - *Environment*: the Avast fix (Task 1)
    - *Accounts and secrets*: names only, never values
    - *Rules checklist*: no data in git; Kaggle forum post; final ensemble notebook stays
      private until 2027-01-13
    - *Next steps*
  - **`DIARY.md`:** a first entry for 2026-10-23/24 (research, Avast diagnosis, data facts,
    rules, design approved), in the format **Did / Learned / Scores / Next**.
  - **`IDEAS.md`:** a table with columns ID, Idea, From, Verdict, Expected gain, Cost, Status,
    Result, seeded with:
    - I-001 prompt grouping plus assignment
    - I-002 early-layer Qwen hidden states
    - I-003 external generated data (blocked until the host rules on it)
    - I-004 gemini-3.7/3.8 label-noise check
    - I-005 NB-SVM baseline
    - I-006 style-only clustering
  - **`README.md`:** title; one-paragraph summary; links to the design, `DIARY.md` and the
    competition; quickstart (`uv sync`, `uv run pytest`); a note that the data isn't included
    and comes from Kaggle; license; disclaimer ("not affiliated with Kaggle or Google; not a
    general AI-text detector").

- [ ] **Step 8: Run all checks.**
  - `uv run pytest -q`. Expected: `1 passed`.
  - `uv run pre-commit run --all-files`. Expected: all hooks pass. Re-run once if a hook
    auto-fixed files.

- [ ] **Step 9: Make the first commit and publish.**
  - `git add -A` and `git commit`. The message ends with the Co-Authored-By line.
  - `gh repo create Ryqn0/name-that-llm --public --source . --remote origin --push --description "End-to-end MLOps project for the Kaggle Name That LLM competition"`.

  Expected: the repo URL is printed.

- [ ] **Step 10: Verify secret protection.**
  `gh api repos/Ryqn0/name-that-llm --jq .security_and_analysis`. Expected: both
  `secret_scanning` and `secret_scanning_push_protection` show `"enabled"`. If either
  doesn't, enable it with
  `gh api -X PATCH repos/Ryqn0/name-that-llm -F "security_and_analysis[secret_scanning][status]=enabled" -F "security_and_analysis[secret_scanning_push_protection][status]=enabled"`.

---

### Task 3: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md` (add the CI badge)

**Interfaces:**
- Produces: a `ci` check on every push and pull request, which the Ship procedure waits on.

- [ ] **Step 1: Write `ci.yml`.**
  - Triggers: `push` and `pull_request`. One job on `ubuntu-latest`.
  - Steps:
    1. `actions/checkout@v5`
    2. `astral-sh/setup-uv@v6` with `python-version: "3.11"`
    3. `uv sync --locked --all-extras`
    4. `uv run ruff check .`
    5. `uv run ruff format --check .`
    6. `uv run pytest -q`
  - The Docker build step that spec §6.11 lists for `ci.yml` is added by the P4 plan, which
    creates the Dockerfile.

- [ ] **Step 2: Add the badge to `README.md`:**
  `![CI](https://github.com/Ryqn0/name-that-llm/actions/workflows/ci.yml/badge.svg)`.

- [ ] **Step 3: Commit and push directly to `main`.** This is the last direct push; PRs start
  in Task 4. Then run `gh run watch`. Expected: the `ci` run completes green.

---

### Task 4: Data layer (labels, loading, cleaning, schemas)

**Files:**
- Create: `src/name_that_llm/data.py`, `src/name_that_llm/schemas.py`, `tests/conftest.py`,
  `tests/test_data.py`, `tests/test_schemas.py`

**Interfaces:**
- Produces, in `name_that_llm.data`:
  - `LABELS: list[str]`: the 18 labels, sorted.
  - `LAB_OF: dict[str, str]`: each model's lab.
  - `normalize_text(text: str) -> str`
  - `find_data_dir(candidates: Sequence[Path] | None = None) -> Path`
  - `load_train(data_dir: Path | None = None) -> pd.DataFrame`, with columns
    `id, prompt_id, text, model`, all `str`, text normalized, validated.
  - `load_test(data_dir: Path | None = None) -> pd.DataFrame`, with columns `id, text`;
    missing text becomes `""`; validated.
- Produces, in `name_that_llm.schemas`:
  - `TrainSchema`, `TestSchema`, `SubmissionSchema` (pandera `DataFrameModel`)
  - `validate_submission(sub: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame`

- [ ] **Step 1: Write the fixtures in `tests/conftest.py`.** They are fake data, safe to
  commit:

```python
import pandas as pd
import pytest

from name_that_llm.data import LABELS, load_train


@pytest.fixture
def fake_data_dir(tmp_path):
    """A tiny FAKE competition folder: 6 prompts x 18 models, plus 3 test rows.

    Each model gets its own marker word, so classifiers have something to learn. The ids
    look numeric on purpose ("000000"), to catch pandas dropping leading zeros.
    """
    rows = [
        {
            "id": f"{p:03d}{m:03d}",
            "prompt_id": f"prompt{p}",
            "text": f"topic{p} words about subject {p}. marker{m} marker{m} \u2014 style{m}",
            "model": label,
        }
        for p in range(6)
        for m, label in enumerate(LABELS)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(
        {"id": ["000001", "a1b2c3", "999999"], "text": ["marker3 text", None, "marker5 text"]}
    ).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


@pytest.fixture
def fake_train(fake_data_dir):
    return load_train(fake_data_dir)
```

- [ ] **Step 2: Write the failing tests in `tests/test_data.py`.**

```python
import pandas as pd
import pytest
from pandera.errors import SchemaError

from name_that_llm.data import LABELS, LAB_OF, find_data_dir, load_test, load_train, normalize_text

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
    assert LABELS == sorted(EXPECTED) and len(LABELS) == 18


def test_every_label_has_one_of_six_labs():
    assert set(LAB_OF) == EXPECTED
    assert set(LAB_OF.values()) == {"anthropic", "google", "moonshot", "openai", "qwen", "zai"}


def test_normalize_text_keeps_style_marks():
    raw = "  He said \u201chi\u201d \u2014 it\u2019s fine\u2026 ok\u00a0then  "
    assert normalize_text(raw) == "He said \u201chi\u201d \u2014 it\u2019s fine\u2026 ok\u00a0then"


def test_normalize_text_composes_accents():
    assert normalize_text("cafe\u0301") == "caf\u00e9"


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
```

- [ ] **Step 3: Write the failing tests in `tests/test_schemas.py`.**

```python
import pandas as pd
import pytest
from pandera.errors import SchemaError

from name_that_llm.schemas import validate_submission

TEST_IDS = pd.Series(["a", "b", "c"])


def _sub(ids, models):
    return pd.DataFrame({"id": ids, "model": models})


def test_validate_submission_accepts_a_valid_file():
    sub = _sub(["a", "b", "c"], ["gpt-5.5", "kimi-k3", "glm-5.3"])
    assert validate_submission(sub, TEST_IDS).equals(sub)


@pytest.mark.parametrize(
    "ids,models",
    [
        (["a", "b"], ["gpt-5.5", "gpt-5.5"]),  # a test id is missing
        (["a", "a", "b", "c"], ["gpt-5.5"] * 4),  # duplicate id
        (["a", "b", "c"], ["gpt-5.5", "gpt-9", "gpt-5.5"]),  # unknown label
        (["a", "b", "c"], ["gpt-5.5", None, "gpt-5.5"]),  # null label
    ],
)
def test_validate_submission_rejects_bad_files(ids, models):
    with pytest.raises((SchemaError, ValueError)):
        validate_submission(_sub(ids, models), TEST_IDS)
```

- [ ] **Step 4: Run the tests.** `uv run pytest -q`. Expected: FAIL with import errors.

- [ ] **Step 5: Implement `data.py`.**
  - Read the CSVs with `pd.read_csv(path, dtype=str)`. Explain why in a comment: pandas would
    otherwise turn `000123` into `123`.
  - `normalize_text` is `unicodedata.normalize("NFC", text).strip()`. Explain in a comment why
    not NFKC: NFKC would rewrite `…` and the non-breaking space, both style fingerprints.
  - `load_test` fills missing text with `""` before validating.
  - `find_data_dir` checks, in order:
    1. `NTL_DATA_DIR`, if set
    2. `/kaggle/input/name-that-llm`
    3. `/kaggle/input/competitions/name-that-llm`
    4. `<repo root>/data/raw`, where the repo root is `Path(__file__).resolve().parents[2]`

    It returns the first path that contains `train.csv`, or the `NTL_DATA_DIR` directory when
    that is set. Otherwise it raises `FileNotFoundError` listing every path it tried.

- [ ] **Step 6: Implement `schemas.py`** using `import pandera.pandas as pa`:
  - `TrainSchema`: `id` unique; `text` at least 1 character; `model` in `LABELS`;
    `Config.unique = ["prompt_id", "model"]`; `Config.strict = True`.
  - `TestSchema`: `id` unique and non-null; `text` is a string (may be empty).
  - `SubmissionSchema`: columns exactly `id, model`; no nulls; `id` unique; `model` in
    `LABELS`.
  - `validate_submission` runs `SubmissionSchema.validate`, then raises
    `ValueError("submission ids do not match test ids")` unless `set(sub.id) == set(test_ids)`
    and `len(sub) == len(test_ids)`.

- [ ] **Step 7: Run the tests.** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 8: Check against the real data locally.**
  - `kaggle competitions download name-that-llm -p data/raw`, then
    `Expand-Archive data/raw/name-that-llm.zip data/raw`.
  - `uv run python -c "from name_that_llm.data import load_train, load_test; print(load_train().shape, load_test().shape)"`.

  Expected: `(2699, 4) (3, 2)`.

- [ ] **Step 9: Update docs.** Add the download commands to Context.md's *Conventions*. Add a
  DIARY.md entry, and update the status to P1.

- [ ] **Step 10: Ship.**
  - Branch `task-04-data`. Bump the version to `0.1.0` inside the PR.
  - After the merge, tag `v0.1.0`.

- [ ] **Step 11: Draft the Kaggle forum post.** The repo now contains competition code, so the
  rules require sharing it on Kaggle.
  - Give the user a title and a short body: what the repo is (learning-in-public, end-to-end
    MLOps), the link, and the MIT license.
  - The user posts it in the competition's Discussion tab.
  - Record the post's URL in Context.md.

---

### Task 5: EDA notebook on Kaggle

**Files:**
- Create: `notebooks/01-eda/eda.ipynb`, `notebooks/01-eda/kernel-metadata.json`

**Interfaces:**
- Consumes: `load_train`, `LABELS` and `LAB_OF` from Task 4, installed through
  `pip install "name-that-llm @ git+https://github.com/Ryqn0/name-that-llm@v0.1.0"`.
- Produces:
  - Context.md facts: Kaggle's Python major.minor version (Task 11 needs it), the real data
    path, and the label-noise verdict.
  - Findings in DIARY.md, plus IDEAS.md updates for I-004 and I-006.

- [ ] **Step 1: Load the `dataviz` skill** before writing any chart code.

- [ ] **Step 2: Write `kernel-metadata.json`:**
  - `"id": "ryqn00/name-that-llm-eda"`, `"title": "Name That LLM - EDA"`
  - `"code_file": "eda.ipynb"`, `"language": "python"`, `"kernel_type": "notebook"`
  - `"is_private": false`, `"enable_gpu": false`, `"enable_internet": true`
  - `"competition_sources": ["name-that-llm"]`

- [ ] **Step 3: Write `eda.ipynb`.** Each section gets a markdown cell explaining *why* we look
  at it, and code cells with heavy comments. Show aggregates only, never raw passages. Sections:
  1. **Setup:** install the package; print `sys.version`, the pandas and scikit-learn
     versions, and the contents of `/kaggle/input`. Print which of `pandera`, `typeguard`,
     `typing_inspect` and `mypy_extensions` were already installed *before* our install (check
     with `importlib.util.find_spec` in a cell that runs before `pip install`). Then
     `train = load_train()`.
  2. **Shape and balance:** rows per model, rows per prompt, and a check that 149 prompts have
     18 rows and 1 has 17.
  3. **Length:** words per passage, as a histogram and as box plots per model colored by lab.
  4. **Punctuation habits:** for each model, the rate per 1,000 characters of `— – - ; : ( ) " ' “ ” ‘ ’ … ! ?`
     and digits, shown as a heatmap of z-scores (models × marks).
  5. **Discriminative words:** the top 10 χ² word uni- and bigrams per lab, as tables of short
     n-grams.
  6. **Model similarity:** a character 1–5-gram TF-IDF centroid per model, an 18×18 cosine
     similarity heatmap, and a dendrogram.
  7. **Label-noise check (I-004):** a two-class logistic regression on character TF-IDF,
     gemini-3.7-flash versus gemini-3.8-flash, validated with prompt-grouped 5-fold CV. Report
     its accuracy next to the same test for gemini-3.6-flash versus gemini-3.8-flash.
  8. **Style-only clustering (I-006):** per passage, punctuation rates plus frequencies of
     `sklearn` English stop words, averaged per model, then hierarchical clustering. Compare
     the clusters with lab membership.
  9. **Findings:** a markdown summary, filled in after the run.

- [ ] **Step 4: Ask the user's OK, then push.** The notebook will be public on their Kaggle
  profile. Run `kaggle kernels push -p notebooks/01-eda`, then poll
  `kaggle kernels status ryqn00/name-that-llm-eda` until it reports `complete`. If it reports
  `error`, fetch the log with `kaggle kernels output ryqn00/name-that-llm-eda -p <scratchpad>`
  and fix the cause.

- [ ] **Step 5: Record the findings.**
  - Context.md *Competition facts*: Python version, data path, label-noise verdict.
  - DIARY.md: the key charts, described in words.
  - IDEAS.md: update I-004 and I-006.

- [ ] **Step 6: Ship.** Branch `task-05-eda`. Commit the notebook with outputs stripped (the
  pre-commit hook does this).

---

### Task 6: Backtesting core (folds, metrics, prompt bootstrap)

**Files:**
- Create: `src/name_that_llm/cv.py`, `src/name_that_llm/metrics.py`, `tests/test_cv.py`,
  `tests/test_metrics.py`

**Interfaces:**
- Consumes: `LABELS` and `LAB_OF` (Task 4).
- Produces:
  - `make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.Series`: one fold
    number from 0 to n_splits−1 per row, on the same index as `df`.
  - `score(y_true, y_pred) -> float`: balanced accuracy.
  - `per_class_recall(y_true, y_pred) -> pd.Series`: indexed by `LABELS`, with NaN for labels
    absent from `y_true`.
  - `lab_accuracy(y_true, y_pred) -> float`
  - `prompt_bootstrap_ci(y_true, pred_a, pred_b, groups, n_boot=2000, seed=0, alpha=0.05) -> tuple[float, float, float]`:
    returns the observed difference `score(b) − score(a)` and the low and high bounds of its
    confidence interval.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_cv.py
import numpy as np
import pandas as pd
from name_that_llm.cv import make_folds


def _prompts(n):
    return pd.DataFrame({"prompt_id": np.repeat([f"p{i}" for i in range(n)], 2)})


def test_make_folds_never_splits_a_prompt(fake_train):
    folds = make_folds(fake_train, n_splits=3, seed=42)
    assert (folds.groupby(fake_train["prompt_id"]).nunique() == 1).all()


def test_make_folds_is_deterministic_and_seeded():
    df = _prompts(30)
    assert make_folds(df, 5, 42).equals(make_folds(df, 5, 42))
    assert not make_folds(df, 5, 42).equals(make_folds(df, 5, 7))


def test_make_folds_balances_prompts():
    df = _prompts(30)
    assert df.groupby(make_folds(df, 5, 42))["prompt_id"].nunique().tolist() == [6] * 5
```

```python
# tests/test_metrics.py
import numpy as np
import pytest
from name_that_llm.data import LABELS
from name_that_llm.metrics import lab_accuracy, per_class_recall, prompt_bootstrap_ci, score


def test_score_is_mean_per_class_recall():
    y, p = np.array(["a", "a", "a", "b"]), np.array(["a", "a", "b", "b"])
    assert score(y, p) == pytest.approx((2 / 3 + 1) / 2)


def test_lab_accuracy_counts_sibling_models_as_correct():
    y = np.array(["gpt-5.5", "claude-opus-5"])
    p = np.array(["gpt-6-astra", "gemini-3.6-flash"])
    assert lab_accuracy(y, p) == 0.5


def test_per_class_recall_is_indexed_by_all_labels():
    r = per_class_recall(np.array(["gpt-5.5"]), np.array(["gpt-5.5"]))
    assert list(r.index) == LABELS and r["gpt-5.5"] == 1.0 and np.isnan(r["kimi-k3"])


def _toy(n_prompts=20):
    return np.tile(np.array(LABELS), n_prompts), np.repeat(np.arange(n_prompts), 18)


def test_bootstrap_identical_predictions_gives_zero():
    y, g = _toy()
    assert prompt_bootstrap_ci(y, y, y, g, n_boot=200) == (0.0, 0.0, 0.0)


def test_bootstrap_clearly_better_model_has_interval_above_zero():
    y, g = _toy()
    pred_a = y.copy()
    pred_a[g % 2 == 0] = "kimi-k3"  # model A is wrong on half the prompts
    diff, lo, hi = prompt_bootstrap_ci(y, pred_a, y, g, n_boot=500, seed=0)
    assert diff > 0 and lo > 0 and lo <= diff <= hi


def test_bootstrap_is_reproducible():
    y, g = _toy()
    pred_a = y.copy()
    pred_a[::3] = "kimi-k3"
    assert prompt_bootstrap_ci(y, pred_a, y, g, seed=1) == prompt_bootstrap_ci(
        y, pred_a, y, g, seed=1
    )
```

- [ ] **Step 2: Run the tests.** `uv run pytest tests/test_cv.py tests/test_metrics.py -q`.
  Expected: FAIL with import errors.

- [ ] **Step 3: Implement `make_folds`.**
  1. Sort the unique `prompt_id`s.
  2. Shuffle them with `np.random.default_rng(seed).permutation`.
  3. Give the i-th shuffled prompt fold `i % n_splits`.
  4. Map that fold back onto each row.

  Don't use `GroupKFold`'s shuffle option. It only exists in scikit-learn ≥ 1.6, and Kaggle's
  version may be older.

- [ ] **Step 4: Implement `score`, `per_class_recall` and `lab_accuracy`.**
  - `score`: `balanced_accuracy_score`.
  - `per_class_recall`: `recall_score(labels=LABELS, average=None, zero_division=np.nan)`.
  - `lab_accuracy`: map both arrays through `LAB_OF`, then take the mean of matches.

- [ ] **Step 5: USER CONTRIBUTION, the body of `prompt_bootstrap_ci`.**
  - Claude writes the signature and a docstring that explains why we resample *prompts*: the
    18 answers to one prompt are correlated, and every prompt contains all 18 classes. The
    function body is left as `raise NotImplementedError` with a TODO.
  - The user writes the 5–10 line body: resample prompt ids with replacement `n_boot` times,
    compute `score(b) − score(a)` on each sample, and return the observed difference with the
    `alpha/2` and `1−alpha/2` quantiles.
  - Guidance to give: use the tests above, `rng.choice` over the unique groups, and a
    precomputed index for each group.
  - Claude reviews the code with the user.

- [ ] **Step 6: Run the tests.** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 7: Ship.** Branch `task-06-backtesting`. DIARY.md gets an entry on why prompt-level
  resampling matters.

---

### Task 7: Baseline models and configs

**Files:**
- Create: `src/name_that_llm/config.py`, `src/name_that_llm/models.py`,
  `src/name_that_llm/configs/{mnb,cnb,lr,svm,nbsvm}.yaml`, `tests/test_config.py`,
  `tests/test_models.py`

**Interfaces:**
- Consumes: `LABELS` (Task 4).
- Produces:
  - `ExperimentConfig` (pydantic, `extra="forbid"`), with these fields and defaults:
    - `name: str`
    - `model: Literal["mnb", "cnb", "lr", "svm", "nbsvm"]`
    - `word_ngrams: tuple[int, int] = (1, 2)`
    - `char_ngrams: tuple[int, int] = (1, 5)`
    - `min_df: int = 2`
    - `max_features: int | None = 200_000`
    - `C: float = 1.0`
    - `alpha: float = 0.1`
    - `n_splits: int = 5`
    - `seed: int = 42`
  - `load_config(name_or_path: str | Path) -> ExperimentConfig`
  - `build_vectorizer(cfg) -> FeatureUnion`
  - `NBSVMClassifier(C: float = 1.0, alpha: float = 1.0)`, a scikit-learn classifier
  - `build_model(cfg) -> Pipeline`, with steps `features` and `clf`

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_config.py
import pytest
from pydantic import ValidationError
from name_that_llm.config import ExperimentConfig, load_config


def test_load_config_by_name_and_by_path(tmp_path):
    cfg = load_config("lr")
    assert cfg.model == "lr" and cfg.n_splits == 5 and cfg.seed == 42
    path = tmp_path / "x.yaml"
    path.write_text("name: x\nmodel: mnb\n")
    assert load_config(path).model == "mnb"


def test_config_rejects_unknown_model_and_typos():
    with pytest.raises(ValidationError):
        ExperimentConfig(name="bad", model="xgboost")
    with pytest.raises(ValidationError):
        ExperimentConfig(name="bad", model="lr", c=1.0)  # typo for C
```

```python
# tests/test_models.py
import numpy as np
import pytest
from name_that_llm.config import load_config
from name_that_llm.models import build_model, build_vectorizer


def _cfg(name):
    return load_config(name).model_copy(update={"min_df": 1})


@pytest.mark.parametrize("name", ["mnb", "cnb", "lr", "svm", "nbsvm"])
def test_every_baseline_predicts_valid_probabilities(name, fake_train):
    model = build_model(_cfg(name)).fit(fake_train["text"], fake_train["model"])
    proba = model.predict_proba(fake_train["text"])
    assert proba.shape == (len(fake_train), 18)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_char_features_keep_case():
    X = build_vectorizer(_cfg("lr")).fit_transform(["Hello World", "hello world"]).toarray()
    assert not np.allclose(X[0], X[1])


def test_nbsvm_learns_marker_tokens(fake_train):
    model = build_model(_cfg("nbsvm")).fit(fake_train["text"], fake_train["model"])
    assert (model.predict(fake_train["text"]) == fake_train["model"]).mean() >= 0.9
```

- [ ] **Step 2: Run the tests.** `uv run pytest tests/test_config.py tests/test_models.py -q`.
  Expected: FAIL with import errors.

- [ ] **Step 3: Implement `config.py`.**
  - `load_config` reads a YAML file when given a path ending in `.yaml`/`.yml` or an existing
    path.
  - Otherwise it reads `importlib.resources.files("name_that_llm") / "configs" / f"{name}.yaml"`.

  Write the five YAML files. Each sets `name` and `model`. `lr`, `svm` and `nbsvm` use
  `C: 4.0`; `mnb` and `cnb` use `alpha: 0.1`.

- [ ] **Step 4: Implement `build_vectorizer`.** It's a `FeatureUnion` of two vectorizers:
  - Word `TfidfVectorizer`: `ngram_range=cfg.word_ngrams`, `lowercase=True`.
  - Character `TfidfVectorizer`: `analyzer="char"`, `ngram_range=cfg.char_ngrams`,
    `lowercase=False`. Comment why: casing and punctuation spacing are style.

  Both use `sublinear_tf=True`, `min_df=cfg.min_df` and `max_features=cfg.max_features`.

- [ ] **Step 5: Implement `NBSVMClassifier`.** This is the algorithm, since the tests don't fix
  it:

```python
class NBSVMClassifier(ClassifierMixin, BaseEstimator):
    """NB-SVM (Wang & Manning, 2012): weight each feature by its Naive Bayes log-count ratio,
    then fit one-vs-rest logistic regressions on the reweighted features."""

    def __init__(self, C=1.0, alpha=1.0):
        self.C, self.alpha = C, alpha

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_, self.clfs_ = [], []
        for c in self.classes_:
            pos = y == c
            p = self.alpha + X[pos].sum(axis=0)
            q = self.alpha + X[~pos].sum(axis=0)
            r = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
            clf = LogisticRegression(C=self.C, max_iter=2000).fit(X.multiply(r).tocsr(), pos)
            self.r_.append(r)
            self.clfs_.append(clf)
        return self

    def predict_proba(self, X):
        s = np.column_stack(
            [clf.predict_proba(X.multiply(r).tocsr())[:, 1] for r, clf in zip(self.r_, self.clfs_)]
        )
        return s / s.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
```

- [ ] **Step 6: Implement `build_model`** as `Pipeline([("features", build_vectorizer(cfg)), ("clf", <classifier>)])`.
  The classifier by name:
  - `mnb`: `MultinomialNB(alpha)`
  - `cnb`: `ComplementNB(alpha)`
  - `lr`: `LogisticRegression(C, max_iter=2000)`
  - `svm`: `CalibratedClassifierCV(estimator=LinearSVC(C=C), cv=3, method="sigmoid")`
  - `nbsvm`: `NBSVMClassifier(C, alpha)`

  Comment why the SVM is wrapped: we need probabilities for ensembling and for the assignment
  step later.

- [ ] **Step 7: Run the tests.** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 8: Write the project's first ADR, `docs/adr/0001-configs-inside-package.md`.**
  Spec §9.1 puts `configs/` at the repo root. The configs live inside the package instead, so
  the wheel used by the offline submission notebook carries them. Cover the context, the
  decision and its consequences in about 10 lines.

- [ ] **Step 9: Ship.** Branch `task-07-baselines`.

---

### Task 8: CV runner and W&B tracking

**Files:**
- Create: `src/name_that_llm/train.py`, `src/name_that_llm/tracking.py`,
  `tests/test_train.py`, `tests/test_tracking.py`

**Interfaces:**
- Consumes: `make_folds`, `score` (Task 6), `build_model`, `load_config` (Task 7), `LABELS`
  and `load_train` (Task 4).
- Produces:
  - `CVResult`, a dataclass with these fields:
    - `oof: np.ndarray`, of shape (n_rows, 18), with columns in `LABELS` order
    - `pred: np.ndarray`, with one label per row
    - `folds: np.ndarray`, with one fold number per row
    - `fold_scores: list[float]`
    - `score: float`
  - `run_cv(cfg: ExperimentConfig, df: pd.DataFrame) -> CVResult`
  - `main(argv: list[str] | None = None) -> int`. Usage:
    `python -m name_that_llm.train --config <name|path> [--data-dir DIR] [--track]`. It prints
    `balanced accuracy: X.XXXX` and the per-fold scores.
  - `log_cv_run(cfg, result, df) -> str` returns the W&B run id. It imports `wandb` inside the
    function and logs to project `name-that-llm`:
    - `cfg.model_dump()` as the run config
    - `cv_balanced_accuracy`, `lab_accuracy`, `oof_log_loss` (`log_loss` with `labels=LABELS`)
      and the fold scores
    - per-class recall as a `wandb.Table`
    - a confusion matrix
    - the training data, as artifact `train-data` of type `dataset`, through
      `run.use_artifact`. This gives lineage: every run records which data version it used,
      and W&B stores identical content only once. This is fine because the project is private
      (spec §6.1).
    - an artifact `oof-{cfg.name}` of type `oof`: a parquet file with `id`, `fold`, and one
      column per label. Storing the fold numbers here is how the folds become a W&B artifact
      (spec §6.6).

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_train.py
import subprocess
import sys

import numpy as np
import pytest
from name_that_llm.config import load_config
from name_that_llm.data import LABELS
from name_that_llm.metrics import score
from name_that_llm.train import main, run_cv


def _cfg(name):
    return load_config(name).model_copy(update={"min_df": 1, "n_splits": 3})


def test_run_cv_returns_oof_for_every_row(fake_train):
    result = run_cv(_cfg("lr"), fake_train)
    assert result.oof.shape == (len(fake_train), 18)
    assert np.allclose(result.oof.sum(axis=1), 1.0)
    assert len(result.fold_scores) == 3
    assert result.score == pytest.approx(score(fake_train["model"], result.pred))


def test_run_cv_keeps_columns_aligned_when_a_class_is_missing(fake_train):
    # keep kimi-k3 only in prompt0: the fold that validates prompt0 never trains on kimi-k3
    df = fake_train[(fake_train["model"] != "kimi-k3") | (fake_train["prompt_id"] == "prompt0")]
    df = df.reset_index(drop=True)
    result = run_cv(_cfg("mnb"), df)
    in_prompt0 = (df["prompt_id"] == "prompt0").to_numpy()
    assert np.allclose(result.oof[in_prompt0, LABELS.index("kimi-k3")], 0.0)
    assert np.allclose(result.oof.sum(axis=1), 1.0)


def test_main_runs_without_tracking(fake_data_dir, tmp_path, capsys):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text("name: tiny\nmodel: mnb\nmin_df: 1\nn_splits: 3\n")
    assert main(["--config", str(cfg), "--data-dir", str(fake_data_dir)]) == 0
    assert "balanced accuracy" in capsys.readouterr().out


def test_train_never_imports_wandb():
    code = "import sys, name_that_llm.train; assert 'wandb' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
```

```python
# tests/test_tracking.py
import pytest
from name_that_llm.config import load_config
from name_that_llm.tracking import log_cv_run
from name_that_llm.train import run_cv


def test_log_cv_run_works_offline(fake_train, tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    cfg = load_config("mnb").model_copy(update={"min_df": 1, "n_splits": 3})
    run_id = log_cv_run(cfg, run_cv(cfg, fake_train), fake_train)
    assert isinstance(run_id, str) and run_id
```

- [ ] **Step 2: Run the tests.** `uv run pytest tests/test_train.py tests/test_tracking.py -q`.
  Expected: FAIL with import errors.

- [ ] **Step 3: Implement `run_cv`.** For each fold:
  1. Fit `build_model(cfg)` on the other folds.
  2. Call `predict_proba` on the fold.
  3. Write the probabilities into `oof` at the columns `[LABELS.index(c) for c in model.classes_]`.
     This is Review Focus 3: never assume `classes_` is all 18 labels.

  Then `pred = np.array(LABELS)[oof.argmax(1)]`. Fold scores and the overall score use
  `score`.

- [ ] **Step 4: Implement `main`.**
  - Parse the arguments with argparse, then load the config and the training data.
  - Call `run_cv` and print the scores.
  - If `--track` is set, import and call `log_cv_run`.
  - Return 0.

- [ ] **Step 5: Implement `log_cv_run`.** The interface block above fixes its outputs.

- [ ] **Step 6: Run the tests.** `uv run pytest -q`. Expected: all pass.
  `test_tracking` runs because the dev environment installs the `track` extra.

- [ ] **Step 7: Set up W&B (user).**
  - Create a free W&B account.
  - Run `uv run wandb login` locally.
  - Create the project `name-that-llm` with visibility **private**.
  - Add the API key as a Kaggle Secret named `WANDB_API_KEY` (Kaggle → Settings → Secrets, or
    a notebook's Add-ons → Secrets).

- [ ] **Step 8: Smoke-test on the real data locally.**
  `uv run python -m name_that_llm.train --config mnb --track`.
  Expected: it prints a balanced accuracy clearly above chance (0.056), and a W&B run appears
  in the private project.

- [ ] **Step 9: Ship.**
  - Branch `task-08-train`. Bump the version to `0.2.0` inside the PR.
  - After the merge, tag `v0.2.0`.
  - Context.md: list the W&B project, by name only.

---

### Task 9: Baselines notebook on Kaggle (the first offline A/B test)

**Files:**
- Create: `notebooks/02-baselines/baselines.ipynb`, `notebooks/02-baselines/kernel-metadata.json`

**Interfaces:**
- Consumes: `run_cv`, `load_config`, `log_cv_run`, `prompt_bootstrap_ci` and `load_train` from
  package `v0.2.0`.
- Produces:
  - The **baseline champion**: a config name, recorded in Context.md *Status*.
  - Five OOF artifacts in W&B.

- [ ] **Step 1: Write `kernel-metadata.json`.** Same shape as Task 5, with
  `"id": "ryqn00/name-that-llm-baselines"` and `"title": "Name That LLM - Baselines"`. Internet
  on, CPU only.

- [ ] **Step 2: Write `baselines.ipynb`.**
  1. Install `@v0.2.0`.
  2. Read `WANDB_API_KEY` with `kaggle_secrets.UserSecretsClient` inside a `try`. If it's
     missing, run without tracking and print a clear note (Review Focus 5).
  3. Run all five configs, each with `run_cv` and `log_cv_run`.
  4. Show a comparison table: CV, standard deviation across folds, lab accuracy, runtime.
  5. Run `prompt_bootstrap_ci` for the best model against the runner-up.
  6. A markdown cell explains how to read a confidence interval that crosses 0.

- [ ] **Step 3: Ask the user's OK, then push.** Run
  `kaggle kernels push -p notebooks/02-baselines` and poll
  `kaggle kernels status ryqn00/name-that-llm-baselines` until it reports `complete`. If W&B
  didn't log, ask the user to attach the `WANDB_API_KEY` secret to this notebook in the Kaggle
  editor once, then re-run.

- [ ] **Step 4: DECISION with the user: choose the baseline champion.**
  - Go through the table and the bootstrap interval together.
  - If the interval crosses 0, prefer the simpler or faster model.
  - Record the choice and the reason in Context.md and DIARY.md.

- [ ] **Step 5: Ship.** Branch `task-09-baselines-notebook`. IDEAS.md: update I-005 (NB-SVM)
  with its result.

---

### Task 10: Submission builder

**Files:**
- Create: `src/name_that_llm/submit.py`, `tests/test_submit.py`

**Interfaces:**
- Consumes: `build_model` (Task 7), `validate_submission` and `LABELS` (Task 4).
- Produces: `predict_submission(cfg: ExperimentConfig, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame`.
  It fits on all of `train`, predicts `test`, and returns `id, model` in the order of `test`,
  already validated.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_submit.py
import subprocess
import sys

import pandas as pd
from name_that_llm.config import load_config
from name_that_llm.data import LABELS, load_test
from name_that_llm.submit import predict_submission


def _cfg():
    return load_config("lr").model_copy(update={"min_df": 1})


def test_predict_submission_is_valid_and_keeps_test_order(fake_train, fake_data_dir):
    test = load_test(fake_data_dir)
    sub = predict_submission(_cfg(), fake_train, test)
    assert sub["id"].tolist() == ["000001", "a1b2c3", "999999"]  # leading zeros kept
    assert sub["model"].isin(LABELS).all()


def test_predict_submission_survives_odd_texts(fake_train):
    odd = pd.DataFrame(
        {
            "id": ["empty", "space", "long", "emoji", "ja"],
            "text": ["", " ", "word " * 5000, "\U0001f642" * 3, "日本語のテキストです"],
        }
    )
    sub = predict_submission(_cfg(), fake_train, odd)
    assert len(sub) == 5 and sub["model"].isin(LABELS).all()


def test_submit_and_train_never_import_wandb():
    code = (
        "import sys, name_that_llm.submit, name_that_llm.train; assert 'wandb' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run the tests.** `uv run pytest tests/test_submit.py -q`. Expected: FAIL with
  an import error.

- [ ] **Step 3: Implement `predict_submission`.** Fit `build_model(cfg)` on `train.text` and
  `train.model`, predict `test.text`, build the DataFrame, and return
  `validate_submission(sub, test["id"])`.

- [ ] **Step 4: Run the tests.** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 5: Ship.** Branch `task-10-submit`. Don't tag yet; Task 11 does.

---

### Task 11: Release to Kaggle, submission notebook, first leaderboard score

**Files:**
- Create:
  - `.github/workflows/release-lib.yml`
  - `kaggle/lib-dataset/dataset-metadata.json`
  - `notebooks/90-submit/submit.ipynb`
  - `notebooks/90-submit/kernel-metadata.json`

**Interfaces:**
- Consumes:
  - `load_train`, `load_test`, `load_config` and `predict_submission` from the published wheel.
  - The list of packages Kaggle was missing, from Task 5's setup cell.
- Produces:
  - The Kaggle Dataset `ryqn00/name-that-llm-lib`: our wheel plus the pure-Python wheels
    Kaggle lacks.
  - A scored submission, with its CV and leaderboard scores recorded.

- [ ] **Step 1: Write `kaggle/lib-dataset/dataset-metadata.json`:**
  `{"title": "name-that-llm-lib", "id": "ryqn00/name-that-llm-lib", "licenses": [{"name": "MIT"}]}`.
  If Kaggle rejects `"MIT"` in Step 4, use `"other"` and note it in Context.md.

- [ ] **Step 2: Write `release-lib.yml`.**
  - Trigger: `push: tags: ["v*"]`. Runs on `ubuntu-latest`.
  - Steps:
    1. Checkout, then `setup-uv`.
    2. `uv build --wheel --out-dir dist-kaggle`.
    3. `uvx pip download --no-deps --dest dist-kaggle pandera typeguard typing-inspect mypy-extensions`.
       These are pure-Python wheels, so they install on any Python version and platform.
       Vendor all four even if Task 5 found some already installed: pip leaves satisfied
       packages alone. numpy, pandas, pydantic, pyarrow, scikit-learn and pyyaml are already
       part of Kaggle's image.
    4. `cp kaggle/lib-dataset/dataset-metadata.json dist-kaggle/`.
    5. `uvx --from "kaggle>=2.2.3" kaggle datasets version -p dist-kaggle -m "$GITHUB_REF_NAME"`,
       with `KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}`.

- [ ] **Step 3: Add the GitHub secret.** This needs the user's OK, because it gives GitHub
  Actions their Kaggle token.
  `Get-Content -Raw $HOME\.kaggle\access_token | gh secret set KAGGLE_API_TOKEN --repo Ryqn0/name-that-llm`.
  The value never appears in output.

- [ ] **Step 4: Create the dataset once, locally.** Ask the user's OK first: it creates a
  public dataset on their profile.
  1. Build `dist-kaggle` with Steps 2.2–2.4, run locally with the same commands.
  2. Run `kaggle datasets create -p dist-kaggle --public`.
  3. Check that the wheel contains the configs:
     `python -m zipfile -l (Get-Item dist-kaggle\name_that_llm-*.whl).FullName` must list
     `name_that_llm/configs/lr.yaml`. PowerShell doesn't expand `*` for native commands, hence
     `Get-Item`.

  This first dataset version is a placeholder. Step 7's tagged release replaces its files.

- [ ] **Step 5: Write `notebooks/90-submit/kernel-metadata.json`.**
  - `"id": "ryqn00/name-that-llm-submit"`, `"title": "Name That LLM - Submit"`
  - `"enable_internet": false`, `"enable_gpu": false`
  - `"competition_sources": ["name-that-llm"]`
  - `"dataset_sources": ["ryqn00/name-that-llm-lib"]`

- [ ] **Step 6: Write `submit.ipynb`.**
  1. `!pip install --no-index --find-links /kaggle/input/name-that-llm-lib name-that-llm`.
  2. `CHAMPION = "<baseline champion from Task 9>"`.
  3. Load the data, call `predict_submission`, and write
     `/kaggle/working/submission.parquet`.
  4. **Timing cell:** repeat the training texts until there are 4,000 rows, time `predict`,
     and print the projected runtime. It must be far below 9 hours.

  Comment heavily on why the notebook works offline and why it validates before writing.

- [ ] **Step 7: Release.**
  - Ship as branch `task-11-release`. Bump the version to `0.3.0` and set `CHAMPION` inside
    the PR.
  - After the merge, tag and push `v0.3.0`.
  - Watch the run with `gh run watch`. Expected: `release-lib` is green and
    `kaggle datasets files ryqn00/name-that-llm-lib` lists `name_that_llm-0.3.0-*.whl`.

- [ ] **Step 8: Ask the user's OK, then push the submission notebook.**
  - Run `kaggle kernels push -p notebooks/90-submit`.
  - Poll until it reports `complete`.
  - `kaggle kernels output ryqn00/name-that-llm-submit -p <scratchpad>/submit-out` must
    contain `submission.parquet` with 3 rows.

- [ ] **Step 9: Ask the user's OK, then submit.** This uses 1 of today's 5 submissions.
  - `kaggle competitions submit name-that-llm -k ryqn00/name-that-llm-submit -v <version> -f submission.parquet -m "baseline <CHAMPION> CV=<score>"`.
  - Poll `kaggle competitions submissions name-that-llm` until the public score appears.

- [ ] **Step 10: Record the result.**
  - Context.md *Status* table: date, model, CV, leaderboard score, W&B run.
  - Add the leaderboard score to the champion's W&B run summary (`lb_public`).
  - DIARY.md: compare CV with the leaderboard score, and what the gap tells us.
  - README: the current best score.
  - Ship as branch `task-11-results`.
