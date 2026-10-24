# P3a: Prompt Grouping and Per-Prompt Assignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the training data's structure (each prompt is answered by every model exactly
once) into a post-processing step. Rebuild the prompt groups from the text alone, then let
each model take at most one passage per group, scored by prompt-grouped CV and by a
leaderboard A/B test.

**Architecture:** Two small modules in the package:
- `grouping.topic_groups` clusters passages by topic, using TF-IDF over content words,
  cosine distance and agglomerative clustering.
- `assign.assign_labels` solves one assignment problem per group with
  `scipy.optimize.linear_sum_assignment`, using the classifier's log-probabilities.

`train.evaluate_assignment` adds the "puzzle backtest" to CV. `submit.predict_submission`
gains an `assign` switch, so the leaderboard A/B test is a single notebook constant.

**Tech Stack:** unchanged: scikit-learn, scipy (already a scikit-learn dependency), pandas,
W&B.

**Spec:** `docs/superpowers/specs/2026-10-23-name-that-llm-design.md` covers this in §6.6
(the puzzle backtest), §6.7 (M2) and §10 (P3).

**Evidence (spike, 2026-10-24, real data, `lr` OOF probabilities):**

| | CV balanced accuracy |
|---|---|
| plain | 0.3212 |
| oracle groups (true `prompt_id`) + assignment | 0.4134 |
| topic groups (adjusted Rand index 0.909, sizes 1–54) + assignment | 0.4056 |

**Scope:** P3 is split into two plans, to get the biggest gain first:
- **P3a (this plan):** I-001, prompt grouping and assignment.
- **P3b (next plan):** I-012 (tuning), M3 and I-008 (style features, lab-then-model
  hierarchy), I-009 (near-duplicates).

I-007 (topic subtraction) needs dense features, so it moves to the embeddings phase (P5).
I-014, studying the top public notebooks, waits until after all phases (the user's call).

## Global Constraints

Plan 1's Global Constraints all still apply: the labels, the data rules, cleaning, the
folds (seed 42), privacy, account-visible OKs, the Ship procedure, tagging, and the Avast
prefix for Claude's shell. In addition:
- **Group size** is 18, the number of models: `len(LABELS)`.
- **The assignment is optional everywhere** (`assign=False` by default). The plain path must
  keep working unchanged, because the hidden test might not share train's structure.
- **No new dependencies.** scipy already ships with scikit-learn.

## Review Focus

1. **Odd test sizes.** The interactive run has only 3 test rows, and the hidden rerun's
   ~4,000 rows isn't a multiple of 18. Grouping and assignment must still give every row
   one valid label, with no "empty vocabulary" error and no `n_clusters > n_samples` error.
   Tests: `test_topic_groups_handles_tiny_inputs`, `test_predict_submission_with_assign_survives_three_rows`.
2. **Clusters larger than 18** (two prompts merged into one group) must allow a model to
   repeat up to `ceil(n / 18)` times, never crash. Test: `test_assign_allows_repeats_in_oversized_groups`.
3. **Zero probabilities** (a model never seen in a fold) must not produce `-inf` costs that
   break the solver. Test: `test_assign_handles_zero_probabilities`.
4. **Row order:** labels come back in the input row order, never in group order. Test:
   `test_assign_keeps_row_order`.
5. **A different hidden structure:** if the hidden test has no "one answer per model per
   prompt" structure, the leaderboard A/B test exposes it, and `assign=False` restores the
   old behavior. This is covered by Task 4's A/B submission, not by a unit test.

---

### Task 1: Topic grouping

**Files:**
- Create: `src/name_that_llm/grouping.py`, `tests/test_grouping.py`

**Interfaces:**
- Produces: `topic_groups(texts: Sequence[str], group_size: int = 18) -> np.ndarray`. It
  returns one int group id per text, in input order, with
  `n_groups = max(1, round(len(texts) / group_size))`.

- [ ] **Step 1: Write the failing tests** (`tests/test_grouping.py`):

```python
import numpy as np
from sklearn.metrics import adjusted_rand_score

from name_that_llm.grouping import topic_groups


def test_topic_groups_recovers_distinct_topics():
    topics = [
        "volcano lava magma eruption",
        "piano melody chord rhythm",
        "comet orbit planet telescope",
    ]
    texts = [f"{t} answer number {i} with style{i}" for t in topics for i in range(18)]
    truth = np.repeat([0, 1, 2], 18)
    assert adjusted_rand_score(truth, topic_groups(texts)) == 1.0


def test_topic_groups_handles_tiny_inputs():
    assert topic_groups(["only one passage"]).tolist() == [0]
    groups = topic_groups(["a cat sat", "the dog ran", "a bird flew"])
    assert len(groups) == 3 and set(groups) == {0}  # round(3 / 18) -> a single group
```

- [ ] **Step 2: Run them.** `uv run --all-extras pytest tests/test_grouping.py -q`. Expected:
  FAIL with an import error.

- [ ] **Step 3: Implement `topic_groups`** with heavy "why" comments:
  1. If `n_groups == 1`, return zeros.
  2. Vectorize with `TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2 if len(texts) >= 100 else 1)`.
     Topic lives in content words, so stop words go. On tiny inputs, `min_df=2` could prune
     every term.
  3. Compute `D = cosine_distances(X)`, which is dense n×n: about 128 MB for 4,000 texts.
  4. Return `AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average").fit_predict(D)`.

- [ ] **Step 4: Run the tests.** `uv run --all-extras pytest -q`. Expected: all pass.

- [ ] **Step 5: Check against the real data.**
  `uv run python -c "from name_that_llm.data import load_train; from name_that_llm.grouping import topic_groups; from sklearn.metrics import adjusted_rand_score as a; t = load_train(); print(round(a(t.prompt_id, topic_groups(t.text.tolist())), 3))"`.
  Expected: about 0.91, at least 0.85.

- [ ] **Step 6: Ship.** Branch `p3a-01-grouping`.

---

### Task 2: Per-group assignment

**Files:**
- Create: `src/name_that_llm/assign.py`, `tests/test_assign.py`

**Interfaces:**
- Consumes: `LABELS` (`name_that_llm.labels`).
- Produces: `assign_labels(proba: np.ndarray, groups: np.ndarray) -> np.ndarray`.
  - `proba` has shape (n, 18), with columns in `LABELS` order.
  - It returns n label strings, in input row order.
  - Within each group, each label is used at most `ceil(group_size / 18)` times, and the
    choice minimizes the total `-log(proba)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_assign.py`):

```python
import numpy as np

from name_that_llm.assign import assign_labels
from name_that_llm.labels import LABELS


def _proba(true_idx, confusions=()):
    """Rows mostly confident in their true label. Each (row, other) pair moves extra mass
    onto a wrong label, so plain argmax picks `other` for that row."""
    p = np.full((len(true_idx), 18), 0.01)
    p[np.arange(len(true_idx)), true_idx] = 0.5
    for row, other in confusions:
        p[row, other] = 0.6
    return p / p.sum(axis=1, keepdims=True)


def test_assign_fixes_duplicate_argmax_within_a_prompt():
    p = _proba(np.arange(18), confusions=[(0, 1)])  # row 0's argmax is label 1, which row 1 owns
    out = assign_labels(p, np.zeros(18, dtype=int))
    assert list(out) == LABELS


def test_assign_keeps_row_order():
    order = np.random.default_rng(0).permutation(18)
    out = assign_labels(_proba(order), np.zeros(18, dtype=int))
    assert list(out) == [LABELS[i] for i in order]


def test_assign_allows_repeats_in_oversized_groups():
    out = assign_labels(_proba(np.arange(20) % 18), np.zeros(20, dtype=int))
    assert len(out) == 20 and max(list(out).count(lab) for lab in LABELS) <= 2


def test_assign_handles_zero_probabilities():
    p = _proba(np.arange(18))
    p[:, 5] = 0.0
    p = p / p.sum(axis=1, keepdims=True)
    out = assign_labels(p, np.zeros(18, dtype=int))
    assert len(set(out)) == 18  # still a valid one-of-each assignment


def test_assign_single_row_group_is_argmax():
    p = _proba(np.array([7]))
    assert assign_labels(p, np.array([3])).tolist() == [LABELS[7]]
```

- [ ] **Step 2: Run them.** Expected: FAIL with an import error.

- [ ] **Step 3: Implement `assign_labels`.** This is the algorithm:

```python
cost = -np.log(np.clip(proba, 1e-12, None))
out = np.empty(len(proba), dtype=object)
for g in np.unique(groups):
    rows = np.flatnonzero(groups == g)
    k = -(-len(rows) // len(LABELS))  # ceil: label "slots" per model
    r, c = linear_sum_assignment(np.tile(cost[rows], (1, k)))
    out[rows[r]] = np.asarray(LABELS)[c % len(LABELS)]
return out.astype(str)
```

  Explain in comments: why log-probabilities (they turn "probability of the whole
  assignment" into a sum), why slots, and why clip.

- [ ] **Step 4: Run the tests.** `uv run --all-extras pytest -q`. Expected: all pass.

- [ ] **Step 5: Ship.** Branch `p3a-02-assign`.

---

### Task 3: Puzzle backtest in CV and tracking

**Files:**
- Modify: `src/name_that_llm/train.py` (add a function and a CLI flag),
  `src/name_that_llm/tracking.py` (log the extra metrics)
- Test: `tests/test_train.py` (append)

**Interfaces:**
- Consumes: `topic_groups` (Task 1), `assign_labels` (Task 2), `CVResult` and `score`.
- Produces:
  - `evaluate_assignment(result: CVResult, df: pd.DataFrame) -> dict[str, float]`, with
    keys `plain`, `assigned` (topic groups), `oracle` (true `prompt_id` groups) and
    `grouping_ari`.
  - The CLI flag `--assign`, which prints these four numbers.
  - `log_cv_run(cfg, result, df, extra: dict[str, float] | None = None)`. The `extra` metrics
    go into the run summary.

- [ ] **Step 1: Write the failing test** (append to `tests/test_train.py`):

```python
from name_that_llm.train import evaluate_assignment


def test_evaluate_assignment_reports_all_four_numbers(fake_train):
    report = evaluate_assignment(run_cv(_cfg("mnb"), fake_train), fake_train)
    assert set(report) == {"plain", "assigned", "oracle", "grouping_ari"}
    assert (
        report["oracle"] >= report["plain"] - 1e-9
    )  # assignment never hurts with true groups here
```

- [ ] **Step 2: Run it.** Expected: FAIL with an import error.

- [ ] **Step 3: Implement.**
  - `evaluate_assignment`: `plain = result.score`. `assigned` uses
    `topic_groups(df.text.tolist())`. `oracle` uses `df.prompt_id` factorized to ints.
    `grouping_ari` comes from `adjusted_rand_score`.
  - `main` with `--assign`: print the four values. When `--track` is also set, pass them as
    `extra` to `log_cv_run`.
  - `log_cv_run`: `run.summary.update(extra or {})`.

- [ ] **Step 4: Run the tests.** `uv run --all-extras pytest -q`. Expected: all pass.

- [ ] **Step 5: Run on the real data, tracked.**
  `uv run --all-extras python -m name_that_llm.train --config lr --assign --track`.
  Expected: plain ≈ 0.321, assigned ≈ 0.40, oracle ≈ 0.41, ARI ≈ 0.91, and a W&B run with
  those summary fields.

- [ ] **Step 6: Ship.** Branch `p3a-03-puzzle-backtest`. Bump the version to 0.4.0, but don't
  tag yet. Docs:
  - DIARY.md: the puzzle backtest numbers.
  - IDEAS.md: I-001 gets its CV result. Add I-014, "Study the top public notebooks and the
    five shared EDAs, after all phases (user's call)".

---

### Task 4: Submission with assignment, and the leaderboard A/B test

**Files:**
- Modify: `src/name_that_llm/submit.py`, `tests/test_submit.py`,
  `notebooks/90-submit/submit.ipynb` (regenerate it with the builder, adding `ASSIGN = True`)

**Interfaces:**
- Consumes: `topic_groups`, `assign_labels`, `LABELS`.
- Produces: `predict_submission(cfg, train, test, assign: bool = False) -> pd.DataFrame`.
  With `assign=False` the behavior is unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_submit.py`):

```python
def test_predict_submission_with_assign_survives_three_rows(fake_train, fake_data_dir):
    test = load_test(fake_data_dir)
    sub = predict_submission(_cfg(), fake_train, test, assign=True)
    assert sub["id"].tolist() == ["000001", "a1b2c3", "999999"]
    assert sub["model"].isin(LABELS).all()


def test_predict_submission_assign_gives_one_of_each_per_topic(fake_train):
    # 18 fresh passages on ONE new topic: assignment must use every label exactly once
    test = fake_train[fake_train["prompt_id"] == "prompt0"][["id", "text"]].reset_index(drop=True)
    sub = predict_submission(_cfg(), fake_train, test, assign=True)
    assert sorted(sub["model"]) == sorted(LABELS)
```

- [ ] **Step 2: Run them.** Expected: FAIL with `TypeError` (an unexpected `assign` argument).

- [ ] **Step 3: Implement.**
  1. Fit as before, then take `predict_proba`, mapping its columns onto `LABELS` the same
     way as `run_cv`.
  2. If `assign`: `labels = assign_labels(proba, topic_groups(test.text.tolist()))`.
     Otherwise, take the argmax.
  3. Validate as before.

- [ ] **Step 4: Run the tests.** `uv run --all-extras pytest -q`. Expected: all pass.

- [ ] **Step 5: Update the submit notebook.**
  - Add `ASSIGN = True` next to `CHAMPION`, and pass it to `predict_submission`.
  - Its markdown cell explains the A/B test: this submission against the earlier 0.320,
    with the same model and only the post-processing changed.
  - The timing cell also uses `assign=ASSIGN`.

- [ ] **Step 6: Release.**
  1. Ship as branch `p3a-04-submit-assign`.
  2. After the merge, tag `v0.4.0` and watch `release-lib` until it's green.
  3. Check that `kaggle datasets files ryqn00/name-that-llm-lib` lists the 0.4.0 wheel.

- [ ] **Step 7: Ask the user's OK, then push and submit.**
  - Push the notebook, poll until it reports `complete`, and check the 3-row output.
  - Submit with the message `"lr + prompt assignment (A/B vs 0.320), v0.4.0"`, then poll for
    the public score.
  - Expected: noticeably above 0.320 if the hidden test shares train's structure.

- [ ] **Step 8: Record the A/B result.** Context.md's table gets the new LB, and the decision:
  keep `ASSIGN = True` only if the LB improved. W&B gets `lb_public` on the Task 3 run.
  Also update DIARY.md and README. Ship as branch `p3a-05-results`.
