# P6: The MLOps Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Watch the public demo and its models. The static page reports anonymous events to
Grafana Cloud. A dashboard and email alerts sit on top, with a robot visitor. Visitors can
answer "It was actually…". A challenger scores in shadow mode, and a weekly job retrains it
and proposes it by pull request. Unusual texts are flagged. The answers are archived weekly
on DagsHub.

**Architecture:**
- **In the browser:**
  - a new torch-free, DOM-free module `app/demo/telemetry.js` builds and sends events.
    Its allowlist makes leaking the text impossible by construction, and Node tests prove
    it;
  - `ntl.js` gains the unusual-text checks;
  - `index.html` wires the events, the "It was actually…" row, the notice, and a challenger
    loaded lazily.
- **In Python:**
  - `browser.py` measures the unusual-text thresholds;
  - `live.py` turns exported events into live accuracy;
  - `retrain.py` holds the retraining gate.
- **The deploy** stages the challenger and a `config.json` (the collector's URL).
- **Three scheduled workflows:** the robot, the weekly export, and retraining.
- **The dashboard and alerts** are JSON in `monitoring/`, pushed by a script.

**Tech Stack:**
- the existing stack (Node for the JS tests, Playwright for the robot, MLflow on DagsHub);
- Grafana Cloud's free tier: the frontend collector, Loki, dashboards, alerting, and the
  Infinity data source;
- GitHub Actions schedules.

**Spec:** `docs/superpowers/specs/2026-10-27-p6-mlops-loop-design.md`. It runs alongside
`docs/superpowers/plans/2026-10-27-p7-llm-ensemble.md`.

## Global Constraints

The P7 constraints on Ship procedure, merges, the Avast prefix and never printing a passage
still apply. Also, from the spec:

- **"The pasted text leaving the browser,** in any form: no excerpt, hash or word list" is an
  anti-goal. Every event field is allowlisted and typed (Task 1).
- **"No identifiers, no cookies, no device storage."**
- **"Every send is fire-and-forget."** A failure is swallowed. The page works exactly as
  today without a collector URL, or with the collector down.
- **The robot's events carry `synthetic: true`,** and are never counted as visits.
- **"It only opens pull requests,"** and every merge stays a person's.
- **Competition passages** never appear in public logs or artifacts. The repo, and so its
  Actions logs, is public.
- **Free services only.**
- **Accounts and secrets are created by the user.** Claude handles names only. The names
  used here:
  - GitHub variables `GRAFANA_COLLECTOR_URL`, `GRAFANA_URL`, `GRAFANA_LOGS_URL` and
    `GRAFANA_LOGS_USER`;
  - GitHub secrets `GRAFANA_TOKEN`, `GRAFANA_LOGS_TOKEN` and `RETRAIN_PR_TOKEN`;
  - the existing `KAGGLE_API_TOKEN` and `DAGSHUB_TOKEN`.
- **Approvals, asked in the session that uses them:** the first deploy of the instrumented
  page, every push to Grafana, every Kaggle dataset action, and turning on each scheduled
  workflow (merging it).
- **Ordering with P7:** these tasks run while P7's GPU notebooks train. Task 8 consumes P7
  Task 3's `run_cv_pool`, so it starts only once that is merged.

## Review Focus

1. **The collector blocked or down:** ad blockers block such collectors. The page works
   unchanged, and no error reaches the visitor. Test: `test_send_never_throws` (Task 1),
   and the page check with a failing collector (Task 4).
2. **An error message that contains the visitor's text:** it must be redacted before it
   leaves. Test: `test_string_fields_never_carry_words_of_the_text` (Task 1).
3. **The challenger missing, or failing to load:** the page answers as before, with no
   challenger label. Step: the page check without challenger files (Task 4).
4. **A second "It was actually…" for the same answer, or one sent after a new text:** only
   one answer per result, tied to that result's labels. Test:
   `test_feedback_belongs_to_its_result` (Task 4).
5. **A retraining run with no challenger, or one without a logged CV:** it's treated as "no
   challenger", and the gate against the champion still decides. Test:
   `test_propose_without_a_challenger` (Task 8).

---

### Task 1: The telemetry module, private by construction

**Files:**
- Create: `app/demo/telemetry.js`, `tests/test_telemetry.py`
- Modify: `app/demo/cli.mjs` (a new command, `event`)

**Interfaces:**
- Produces (ES module, no DOM):
  - `makeEvent(name, fields, ctx) -> {name, fields}`.
    - `ctx` is `{labels, bands, text}`. `text` is used only to redact.
    - An unknown `name` throws. Unknown fields are dropped.
    - Numbers must be finite, labels must be in `ctx.labels` (or, for `actual`, also
      "another model" or "a person"), and enums must be valid. Anything else is dropped.
    - Any string field that contains a word of `text` (4 letters or more) becomes
      `"[redacted]"`.
  - `wordBucket(n) -> number`: `25 * floor(n / 25)`.
  - `toCollectorPayload(events, meta) -> object`: the collector's JSON. `meta` is
    `{commit, model_run, challenger_run, synthetic}`. Every value goes out as a string
    attribute.
  - `send(payload, url, fetchImpl = fetch) -> Promise<void>`: a no-op without `url`, POST
    with `keepalive`, and it never rejects.
- **The events and their fields** (the spec's §4.1):
  - `page_load`: `load_ms`, `ok`, `what` (`champion` or `challenger`);
  - `answer`: `answer_ms`, `words` (bucketed), `label`, `band`, `challenger_label`
    (label or null), `unusual` (a subset of `short`, `unknown_words`, `non_latin`);
  - `feedback`: `actual`, `label`, `challenger_label`, `band`;
  - `error`: `where` (`load`, `answer` or `feedback`), `kind` (≤ 40 characters),
    `message` (≤ 120 characters).

- [ ] **Step 1: Read the collector's payload format** in Grafana's Frontend Observability
  docs: the JSON accepted at the collector URL, and how events and their attributes appear
  in Loki. Ledger the shape that `toCollectorPayload` must produce.
- [ ] **Step 2: Write the failing tests.** They run the module through `cli.mjs event`
  (`run_engine`), with Node required:
  - `test_events_keep_only_allowlisted_typed_fields`: an `answer` event with an extra
    `"text"` field, a non-finite `answer_ms` and an unknown label loses all three.
  - `test_unknown_event_names_are_refused`.
  - `test_string_fields_never_carry_words_of_the_text`: an `error` whose `message` quotes
    the passage comes out `"[redacted]"`.
  - `test_payload_of_a_real_flow_contains_no_word_of_the_passage`:
    1. predict a passage written for the test with the fake exported model (as in
       `test_browser.py`);
    2. build `answer`, `feedback` and `error` events from that flow, and the payload;
    3. no word (4+ letters) of the passage appears in `JSON.stringify(payload)`.
  - `test_word_counts_are_bucketed`: 0, 24, 25 and 199 give 0, 0, 25 and 175.
  - `test_send_never_throws`: an unreachable URL and an empty one both resolve.
- [ ] **Step 3: Run them.** Expected: FAIL (the module doesn't exist).
- [ ] **Step 4: Implement `telemetry.js`** and the `event` command. Comments explain the
  allowlist and why the text can't leak.
- [ ] **Step 5: Run them.** Expected: PASS. Then run the whole suite. Commit, and ship
  (branch `p6-01-telemetry`).

---

### Task 2: Unusual texts: thresholds from our passages, checks in the browser

**Files:**
- Modify:
  - `src/name_that_llm/browser.py`: `unusual_thresholds`, `unknown_share`;
  - `app/demo/ntl.js`: `textStats`, `unusualReasons`;
  - `app/demo/cli.mjs`: a new command, `stats`;
  - `scripts/export_browser_model.py`: writes `meta["unusual"]`;
  - `app/demo/MODEL_RUN`: the re-exported champion;
  - `tests/test_browser.py`.

**Interfaces:**
- Produces:
  - `unknown_share(texts, vocabulary: set[str]) -> np.ndarray`: the share of each
    flattened text's word tokens (`lr`'s word analyzer, unigrams, lowercased) that aren't
    in `vocabulary`.
  - `unusual_thresholds(cfg: ExperimentConfig, df: pd.DataFrame, quantile: float = 0.99) -> dict`:
    `{"min_words": <the shortest training passage, in words after flattening>,
    "max_unknown_share": <the quantile of HELD-OUT unknown shares>,
    "max_non_latin_share": 0.2}`.
    - "Held-out" means each passage is measured against the vocabulary of its fold's
      training rows (`make_folds`).
    - 0.2 is a fixed rule, since our passages are English (the spec's §4.6).
  - **In `ntl.js`:**
    - `textStats(model, text) -> {words, unknown_share, non_latin_share}`;
    - `unusualReasons(model, text) -> string[]`, with reasons from `short`,
      `unknown_words` and `non_latin`. It returns `[]` when `model.meta.unusual` is
      absent.

- [ ] **Step 1: Write the failing tests:**
  - `test_js_text_stats_match_python`: on the fake model, the JS `unknown_share` equals
    `unknown_share(...)` (atol 1e-9) for 6 texts, accents and markdown included.
  - `test_thresholds_come_from_held_out_passages`: on fake data, `max_unknown_share` is above
    0 (held-out vocabularies miss words), and `min_words` equals the shortest passage.
  - `test_unusual_reasons`: a 25-word text gives `short` when `min_words` is 36; a
    Japanese passage gives `non_latin`; a passage of unknown words gives `unknown_words`; a
    training-like passage gives `[]`.
  - `test_no_thresholds_means_no_reasons`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement**, with the JS letters counted by `\p{L}` and `\p{Script=Latin}`.
- [ ] **Step 4: Run them.** Expected: PASS, along with the whole suite.
- [ ] **Step 5: Re-export the champion** (`scripts/export_browser_model.py`, now writing
  `meta["unusual"]`, with the DagsHub env loaded). Expected: the same parity (50 of 50), a
  new run, and `MODEL_RUN` updated. Ledger the thresholds and the share of our passages
  they flag: about 1% on held-out passages.
- [ ] **Step 6: Commit and ship** (branch `p6-02-unusual`). Merging redeploys the demo with
  the new model files. That's the same page, so it's covered by the deploy procedure the
  user approved in P4.

---

### Task 3: Shadow mode in the deploy: the challenger and the page's config

**Files:**
- Create: `app/demo/CHALLENGER_RUN` (empty for now)
- Modify:
  - `scripts/deploy_space.py`;
  - `tests/test_deploy_space.py`;
  - `.github/workflows/deploy-app.yml`: the variable `GRAFANA_COLLECTOR_URL`.

**Interfaces:**
- Produces:
  - `stage_space(repo_root, model_dir, out_dir, commit, challenger_dir: Path | None = None, collector_url: str = "") -> list[Path]`.
    It also stages `telemetry.js`, plus `challenger/model.json` and `challenger/weights.bin`
    when `challenger_dir` is given. `version.json` becomes
    `{commit, model_run, challenger_run}` (`challenger_run` is "" when absent).
    `config.json` is `{"collector_url": collector_url}`.
  - `challenger_run(repo_root) -> str`: "" when the file is empty.
  - `main()` downloads the challenger's run when there is one. `smoke_test` checks every
    staged file, `challenger/` included, and runs the engine on both models.

- [ ] **Step 1: Write the failing tests:**
  - `test_stage_space_with_a_challenger_and_a_collector`: the staged list adds
    `challenger/model.json`, `challenger/weights.bin`, `config.json` and `telemetry.js`;
    `version.json` names both runs.
  - `test_stage_space_without_a_challenger`: no `challenger/` folder,
    `"challenger_run": ""`, and `config.json`'s URL is "".
  - The existing tests are updated for `telemetry.js` and `config.json`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** `smoke_test` walks `staged.rglob("*")`. The skipped card and
  the page's insertion rule are unchanged.
- [ ] **Step 4: Run them.** Expected: PASS, along with the whole suite. Commit, and ship
  (branch `p6-03-shadow-deploy`). An empty `CHALLENGER_RUN` and an empty URL redeploy the
  same page, so there's no new approval.

---

### Task 4: The instrumented page

**Files:**
- Modify:
  - `app/demo/index.html`;
  - `app/demo/README.md`: what's collected;
  - `tests/test_deploy_space.py`, if the staged list changes.
- Create: `app/demo/feedback.js`, a tiny pure module, and `tests/test_feedback.py`.

**Interfaces:**
- Consumes: Task 1's `makeEvent`, `toCollectorPayload`, `send` and `wordBucket`; Task 2's
  `unusualReasons` and `textStats`; Task 3's `config.json` and `version.json`.
- Produces: `feedback.js`'s `newResult(labels) -> {id, label, challenger_label, band, answered: false}`
  and `answer(result, actual) -> event fields | null`. The second answer for the same
  result returns null.

**The page:**
- **On load:** read `config.json` and `version.json`. `synthetic` is true when the URL has
  `?synthetic=1`. Send `page_load`.
- **On each answer:** send `answer`. If `unusualReasons` isn't empty, show the note "This
  text looks unlike what the model learned from: treat the answer with extra care."
- **After the first answer:** load `challenger/` lazily, if `version.challenger_run` is set,
  and send its `page_load`. From then on, every answer also gets the challenger's label.
- **After each result:** the optional row "Do you know who wrote it?". It offers the 18
  models, "another model" and "a person", with a Send button. Sending fires `feedback` and
  thanks the visitor.
- **Errors:** each `catch` sends an `error` event with `where`, `kind` and `message`.
- **The notice**, under the warning box: "Anonymous numbers are sent to help us keep the
  site working: load and answer times, errors, the predicted model, and your optional
  answer. Your text never leaves your browser." The model card repeats it.

- [ ] **Step 1: Write the failing tests:**
  - `test_feedback_belongs_to_its_result` (Review Focus 4): one answer per result, with
    that result's labels;
  - `test_feedback_accepts_only_known_choices`.
- [ ] **Step 2: Run them** (expected: FAIL), **implement `feedback.js`**, **run them**
  (expected: PASS).
- [ ] **Step 3: Wire the page.**
- [ ] **Step 4: Check the page in headless Edge** (P4's Playwright driver, requests
  answered from the staged folder). Three cases, all with no console errors:
  1. **A fake collector route records the POSTs:**
     - a paste gives `page_load` and `answer`, then a challenger `page_load`;
     - feedback gives one `feedback`;
     - no recorded body contains a word of the passage;
     - `?synthetic=1` is carried.
  2. **The collector route returns 500, or is aborted:** the page behaves exactly the
     same (Review Focus 1).
  3. **No challenger files:** answers come, with no challenger label (Review Focus 3).
  - Screenshots at 900 and 390 px.
- [ ] **Step 5: Commit, open the PR, and get CI green.** Then ask the user's OK for the
  first deploy of the instrumented page, and merge. The deploy's smoke test passes, and the
  live page sends nothing until `GRAFANA_COLLECTOR_URL` is set (Task 5).

---

### Task 5: Grafana: the collector, the dashboard and alerts as code

**Files:**
- Create:
  - `monitoring/dashboard.json`;
  - `monitoring/alerts.json`;
  - `monitoring/model_history.csv`: date, model, cv and lb, from Context's table;
  - `scripts/push_monitoring.py`;
  - `tests/test_monitoring.py`.

**Interfaces:**
- Produces: `push_monitoring.py` with `build_requests(dashboard: dict, alerts: list[dict], folder_uid: str) -> list[tuple[str, str, dict]]`
  (method, path, JSON body), and `main()`, which sends them with `GRAFANA_URL` and
  `GRAFANA_TOKEN`.
- **The dashboard's panels** (spec §4.5):
  - visits, load and answer times, errors (with `synthetic=false`);
  - the robot's line;
  - the live comparison: answers and accuracy per run id, champion and challenger;
  - the drift signals: word buckets, unusual reasons, the mix of predicted labels, bands;
  - models over time, from `model_history.csv` through the Infinity data source.
- **The alerts:** real `error` events above 5 an hour; no synthetic `answer` for 12 hours.
  Both go to the email contact point.

- [ ] **Step 1: The user creates the accounts.**
  1. A Grafana Cloud account and a Frontend Observability app. Its collector URL goes in
     `GRAFANA_COLLECTOR_URL`, with the allowed origin set to the Space's host.
  2. A service-account token goes in `GRAFANA_TOKEN`, and the stack URL in `GRAFANA_URL`.
  3. Loki read access for the weekly export: `GRAFANA_LOGS_URL`, `GRAFANA_LOGS_USER` and
     `GRAFANA_LOGS_TOKEN`.
  4. The alert email, in an email contact point.
  5. The Infinity plugin, if the stack doesn't already have it.

  Ledger the names (no values) and the logs retention the free tier states.
- [ ] **Step 2: Write the failing tests:**
  - `test_dashboard_panels_all_have_a_query`;
  - `test_live_panels_exclude_synthetic_events`: every visit, time and error query filters
    `synthetic="false"`;
  - `test_alerts_cover_errors_and_robot_silence`;
  - `test_build_requests_targets_the_dashboard_and_alert_apis`: the paths
    `/api/dashboards/db` and `/api/v1/provisioning/alert-rules`, with the token never
    placed in a body.
- [ ] **Step 3: Run them** (expected: FAIL), **implement**, **run them** (expected: PASS).
- [ ] **Step 4: Ask the user's OK, then push** (`python scripts/push_monitoring.py`).
  Re-deploy the page, which picks up the variable through `deploy-app`'s
  `workflow_dispatch`. Then visit the live page once, and see the events arrive on the
  dashboard.
- [ ] **Step 5: Commit and ship** (branch `p6-05-grafana`).

---

### Task 6: The robot visitor

**Files:**
- Create: `scripts/robot_visit.py` and `.github/workflows/robot.yml`
  (`schedule: "0 */6 * * *"`, plus `workflow_dispatch`)

**Interfaces:**
- Produces: `robot_visit.py --url URL`. It opens `URL?synthetic=1` in headless Chromium,
  waits up to 90 s for Identify (one retry), pastes `ROBOT_TEXT` (written for the purpose,
  not `SMOKE_TEXT`), and expects a result within 10 s. It prints the load and answer times,
  and exits 1 on failure.

- [ ] **Step 1: Write `robot_visit.py`, then run it locally against the live site.**
  Expected: the times printed, exit 0, and a synthetic `answer` on the dashboard.
- [ ] **Step 2: Write `robot.yml`:**
  - `pip install playwright`, then `playwright install --with-deps chromium`, then run the
    script;
  - `timeout-minutes: 10`;
  - `permissions: contents: read`.
- [ ] **Step 3: Ask the user's OK to turn on the schedule, then merge** (branch
  `p6-06-robot`). Run it once with `workflow_dispatch`. Expected: green, and its events on
  the dashboard.

---

### Task 7: The weekly export of events to DagsHub

**Files:**
- Create:
  - `src/name_that_llm/live.py`;
  - `scripts/export_events.py`;
  - `.github/workflows/weekly-export.yml` (Mondays 06:00 UTC, plus `workflow_dispatch`);
  - `tests/test_live.py`.

**Interfaces:**
- Produces:
  - `parse_loki(response: dict) -> pd.DataFrame`: one row per event, with the attributes as
    columns.
  - `live_accuracy(events: pd.DataFrame) -> pd.DataFrame`. Per run id, champion and
    challenger, it reports `n` and `accuracy` with the Wilson 95% interval `lo` and `hi`.
    It counts only real `feedback` events whose `actual` is one of the 18 models (not
    "another model" or "a person").
  - `paired(events) -> dict`: on feedback rows where both labels exist, how often only
    the champion is right, and how often only the challenger is.
  - `export_events.py` queries the last 7 days from `GRAFANA_LOGS_URL`, then logs one MLflow
    run in the experiment `live-feedback`: the events CSV as an artifact, and the counts and
    accuracies as metrics.

- [ ] **Step 1: Write the failing tests:**
  - `test_parse_loki_reads_events`: from a fake response shaped like Task 1's ledgered
    format;
  - `test_live_accuracy_ignores_synthetic_and_unknown_authors`;
  - `test_wilson_interval_is_honest_for_small_n`: 1 of 1 gives an interval wider than
    [0.2, 1.0];
  - `test_paired_counts_only_rows_with_both_labels`.
- [ ] **Step 2: Run them** (expected: FAIL), **implement**, **run them** (expected: PASS).
- [ ] **Step 3: Run `export_events.py` locally** with the env loaded. Expected: a run in
  `live-feedback` on DagsHub. Ask the user's OK to turn on the schedule, then merge
  (branch `p6-07-export`).

---

### Task 8: Weekly retraining that proposes a challenger

**Files:**
- Create:
  - `src/name_that_llm/retrain.py`;
  - `scripts/retrain_challenger.py`;
  - `.github/workflows/retrain.yml` (Sundays 03:00 UTC, plus `workflow_dispatch`;
    `timeout-minutes: 300`);
  - `tests/test_retrain.py`.

**Interfaces:**
- Consumes:
  - P7's `build_pool` and `load_pool_sources` (merged) and `run_cv_pool` (P7 Task 3);
  - `export_browser_model` and `unusual_thresholds`;
  - `run_engine`.
- Produces: `propose(y, groups, champion_pred, candidate_pred, candidate_cv: float, challenger_cv: float | None) -> tuple[bool, dict]`.
  - It's True only if the bootstrap (`prompt_bootstrap_ci`, candidate minus champion) has a
    lower bound above 0, *and*, when `challenger_cv` is given, `candidate_cv >= challenger_cv`.
  - The dict holds the numbers for the PR body.

**`retrain_challenger.py` steps:**
1. Download the competition data and the pool, under `$RUNNER_TEMP` only.
2. Run `run_cv_pool` (seed 42) for the champion recipe (`lr`) and for the candidate (`lr`
   plus the kept pool, capped at 30,000).
3. Read the challenger's logged `cv_balanced_accuracy` from its MLflow run. If the file is
   empty, or the metric is missing, there's no challenger.
4. If `propose` says yes:
   - fit the candidate on all of train plus the capped pool;
   - export it, with `meta.unusual`, and check parity on 50 passages under Node;
   - log the run (experiment `browser-demo`);
   - write `CHALLENGER_RUN`, then `git push` a branch and run `gh pr create`, with
     `RETRAIN_PR_TOKEN`.
5. Print scores and counts only, and upload no artifact.

- [ ] **Step 1: Write the failing tests:**
  - `test_propose_needs_a_significant_win`;
  - `test_propose_without_a_challenger` (Review Focus 5);
  - `test_propose_refuses_a_candidate_below_the_challenger`;
  - `test_the_body_reports_both_scores_and_the_interval`.
- [ ] **Step 2: Run them** (expected: FAIL), **implement `retrain.py`**, **run them**
  (expected: PASS).
- [ ] **Step 3: Write the script and the workflow.**
- [ ] **Step 4: The user creates `RETRAIN_PR_TOKEN`:** a fine-grained token with Contents
  and Pull requests write, on this repo only.
- [ ] **Step 5: Ask the user's OK, merge, and run it once with `workflow_dispatch`.**
  Expected:
  - if the pool helps `lr` (P7 Task 3), a PR that sets the first challenger, with CI green;
  - otherwise, a "no proposal" log.

  Merging that PR deploys the challenger in shadow mode, and its events appear on the
  dashboard.

---

### Task 9: Record and ship

**Files:**
- Create:
  - `docs/adr/0007-watching-a-static-site.md`: events from the page, shadow mode, and
    proposals by pull request;
  - `docs/journal/p6-mlops-loop.md`, written from the ledger.
- Modify:
  - `DIARY.md`;
  - `Context.md`: status, the accounts' names, how to promote a challenger;
  - `IDEAS.md`: I-024 and I-025 done;
  - `docs/journal/README.md`.

- [ ] **Step 1: Write them**, then run the suite and the hooks. Expected: PASS.
- [ ] **Step 2: Commit and ship** (branch `p6-09-results`). Then the final review, the
  journal, deleting the workspace, and finishing the branch.
