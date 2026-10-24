# Progress diary

Newest entry first. Each entry follows the same format: **Did / Learned / Scores / Next**.

## 2026-10-24 · Session 2d: CV runner and the first tracked experiment

**Did**
- `run_cv` produces out-of-fold (OOF) probabilities for every row, through a CLI:
  `python -m name_that_llm.train --config mnb --track`.
- `log_cv_run` sends the config, the metrics, the per-class recall, the confusion matrix and
  two artifacts to W&B. The artifacts are `train-data` (the input, for lineage) and
  `oof-mnb` (the output).
- **First tracked run** ([W&B](https://forge.coreweave.com/wandb/dungryan0-ryan-dung/name-that-llm/runs/9i018wmv)),
  verified through the API: metrics present, `train-data:v0` recorded as input, `oof-mnb:v0`
  as output.
- Compared our EDA with five public ones. The best new ideas (subtracting each prompt's
  average, opening and closing phrases, classifier weights) go to IDEAS.md.

**Learned**
- **Lab accuracy is 0.66, model accuracy 0.29:** two thirds of passages get the right lab,
  but under a third get the right model. "The lab is easy, the model is hard", now measured.
- **Log loss 2.13 vs 2.89 for uniform guessing:** the probabilities carry information, but
  are far from confident. This matters for calibration and the I-001 assignment.
- **W&B can't declare inputs offline**, so the code uses `use_artifact` online and
  `log_artifact` offline. The offline test alone wouldn't have exercised the online path,
  which is why the PR waited for this live check.
- **Privacy has to hold before the first upload.** The W&B "Team" visibility is private only
  while the team has one member.

**Scores:** mnb CV balanced accuracy **0.2852** (folds 0.270–0.294). Not submitted.

**Next:** Task 9, all five baselines in a Kaggle notebook, and the first offline A/B test.

## 2026-10-24 · Session 2c: backtesting core (with my first contribution)

**Did**
- `make_folds`: deals the 150 prompts into 5 folds like cards, so a prompt never spans two
  folds. The seed is fixed, so every model is scored on identical splits.
- Metrics: `score` (balanced accuracy), `per_class_recall`, and `lab_accuracy` (right lab,
  possibly the wrong model).
- **I wrote the body of `prompt_bootstrap_ci`**, the statistical core of every A/B test from
  now on. Claude's review found one bug: pandas Series with a non-default index. A new test
  reproduced it (KeyError), and a one-line `np.asarray` fix turned it green.

**Learned**
- **Why resample prompts, not rows:** the 18 answers to one prompt are linked, sharing a
  topic and its difficulty. Treating 2,699 rows as independent would make the confidence
  interval too narrow. We'd "see" improvements that are really luck.
- **Paired beats unpaired:** scoring both models on the *same* resample cancels out "this
  sample happened to be easy". Only the *difference* between the models varies.
- **The pandas indexing trap:** `series[[0, 1, 2]]` uses index *labels*, not positions. It
  works by accident while the index is 0..n-1, then breaks silently after filtering.

**Scores:** none yet.

**Next:** the CV runner and W&B tracking (Task 8).

## 2026-10-24 · Session 2b: EDA on Kaggle

**Did**
- Published the EDA notebook
  [ryqn00/name-that-llm-eda](https://www.kaggle.com/code/ryqn00/name-that-llm-eda). It ran
  in about 100 s on Kaggle's CPU.
- Smoke-tested every cell locally first, then re-ran the results locally with Kaggle's exact
  library versions so the written findings match the run.
- Added a CI job that tests Kaggle's real runtime (Python 3.13 plus Kaggle's pandas, numpy
  and scikit-learn versions).

**Learned**
- **Kaggle's environment isn't what I assumed:** Python 3.13 (not 3.11) and *older*
  libraries than my local setup (pandas 2.3, not 3.1). Testing against production's real
  versions beats guessing them.
- **The data path was the *second* candidate**, `/kaggle/input/competitions/...`, so building
  the loader to try several paths paid off on its first real run.
- **Charts, in words:**
  - The length box plots overlap heavily: length is a weak signal.
  - The punctuation heatmap has a few loud cells. Two gemini models use **no em dashes at
    all**, glm-5.3 uses the most, gpt-5.6-luna stands out for parentheses and kimi-k3 for
    semicolons.
  - The similarity heatmap is dark along the sibling blocks: the gemini flashes, the gpt-5.6
    trio, the two GLMs.
- **Two-class checks:** gemini-3.7 vs 3.8 scores **0.56** (near a coin flip), sibling pairs
  0.64–0.67, and a cross-lab pair 0.89. The 3.7 vs 3.8 result fits the forum's
  request-routing report.
- **Style-only clustering** (adjusted Rand index 0.49): Google forms one clean cluster,
  Anthropic and ZAI share one (GLM writes like Claude), and OpenAI spreads over three.
- **Tooling:** pre-commit run from the main repo silently skips files in another worktree,
  while still reporting "passed". Always run hooks from inside the checkout that owns the
  file.

**Scores:** none yet. The first submission comes in Task 11.

**Next:** the user's bootstrap (Task 6), then the CV runner with W&B (Task 8).

## 2026-10-24 · Session 2: foundation, CI, data layer

**Did**
- Created the public repo [Ryqn0/name-that-llm](https://github.com/Ryqn0/name-that-llm) with
  the uv package, pre-commit hooks, docs and an MIT license.
- Set up CI on GitHub Actions: lint, format check and tests on Python 3.11, all green.
- Built the data layer (`data.py`, `schemas.py`, `labels.py`) test-first: 16 tests, all on
  fake data.
- Checked it against the real data: `(2699, 4)` train and `(3, 2)` test.

**Learned**
- **pandas silently turns `000123` into `123`** unless you pass `dtype=str`. Ids are matched
  against Kaggle's, so that one argument decides whether a submission scores at all.
- **NFC versus NFKC normalization:** NFKC would turn `…` into `...` and erase the
  non-breaking space. Those are style fingerprints, so we only apply NFC.
- **Circular imports:** `data` needs the schemas, and the schemas need the labels. Moving the
  labels into their own dependency-free module breaks the cycle.
- **Test data validation strictly, but the test set leniently:** an empty passage in the
  hidden test should cost one prediction, not crash the whole 9-hour scoring run.
- **Line endings:** Git for Windows converts files to CRLF while Linux CI uses LF. A
  one-line `.gitattributes` keeps every machine identical.

**Scores:** none yet.

**Next:** the Kaggle forum post (a rule requirement, now that the code is public), then the
EDA notebook.

## 2026-10-23 → 24 · Session 1: kickoff, research, design

**Did**
- Researched the competition: first through public pages, then through the official rules
  and data pages (via the Kaggle API), plus the forum.
- Found why Python, uv and the Kaggle CLI kept failing on this PC: Avast's HTTPS scanning,
  which caused three separate symptoms. The fix is `scripts/setup-local.ps1`.
- Checked the data locally with a throwaway script: structure, class balance, formatting.
- Wrote and approved the [design](docs/superpowers/specs/2026-10-23-name-that-llm-design.md),
  then the first plan, covering phases P0–P2.
- Started P0: the uv package skeleton, tooling and docs.

**Learned**
- **Each prompt was answered by all 18 models exactly once** (149 of 150 prompts). A
  per-prompt "assignment puzzle" trick might therefore work on the hidden test. Not yet
  verified.
- Test prompts never appear in train, so validation must hold out whole prompts
  (GroupKFold on `prompt_id`). A random split lets the model memorize topics.
- The host stripped all formatting (no newlines, bold or bullets), yet 39% of passages
  contain an em dash (—). Punctuation is a fingerprint, so cleaning must not touch it.
- **Rules:** the data is CC BY-SA 4.0 (usable for any purpose) but may not be
  redistributed, so it never goes into git. Code that's made public must also be shared on
  Kaggle.
- **Forum (unverified claims):**
  - a public CPU baseline scores 0.353 on the leaderboard
  - early-layer features from an open Qwen model score ≈ 0.39 in CV
  - gemini-3.7-flash and gemini-3.8-flash labels may be mixed up
- **Tooling:** `uv add` sets each dependency's minimum version to the newest release. That's
  too strict for Kaggle's older preinstalled packages, so I loosened the floors by hand.

**Scores:** none yet.

**Next:** finish P0 (the user runs the setup script; CI), then the data layer and EDA.
