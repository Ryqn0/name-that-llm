# Journal: P6, the MLOps loop: watching the site and the models (2026-10-27 → 28)

**Goal:** in the user's words, see "when you see some issues on the site or notice that
there might be some lags, slows or any connection error", and "check occasionally on if
some models will overtake others". The demo is a static page with no server of ours (ADR
0005), so the page itself reports what happens. Visitors who know the author of a text
turn the site into a live test set.

**Result:**
- The live demo sends anonymous events to the user's Grafana Cloud stack: loads, answers,
  errors, and the optional "Do you know who wrote it?" answer. The text never leaves the
  browser.
- A dashboard as code, and two email alerts.
- A robot visitor every 6 hours.
- A weekly export of the events to DagsHub, with live accuracy and honest intervals.
- A weekly retraining that may propose a better model as the demo's challenger, by pull
  request. The challenger runs in shadow mode.
- The decisions are in [ADR 0007](../adr/0007-watching-a-static-site.md).

**In numbers:**
- 10 pull requests (#48 to #57): 32 files, 2,787 lines added and 59 removed in
  `3b8becd..4c75a3e`, 580 of the added lines in tests.
- Then #58, the first challenger the retraining proposed, and #60, the final review's fix
  pass: 14 files, 588 lines added and 97 removed, with 16 new tests.
- The full suite grew from 158 to 180 tests by #57. When #60 merged, it stood at 210,
  P7's tests included.
- The live chain was proved end to end. The robot's visit reached Loki through the
  collector. The user's own visit followed: page load, answer, and "It was actually…
  qwen3.8-max".
- The retraining's dry run reproduced P7 Task 3 exactly: `lr` 0.3212, against `lr` plus
  the pool 0.4205 (+0.0993, 95% interval [+0.0801, +0.1196]).

## How the work was run

- **Native execution** of the [plan](../superpowers/plans/2026-10-27-p6-mlops-loop.md),
  from the approved [design](../superpowers/specs/2026-10-27-p6-mlops-loop-design.md).
  Claude implemented each task itself, test first. One fresh reviewer read the whole
  branch at the end.
- **Interleaved with P7.** The P6 tasks ran on CPU while P7's first GPU run trained.
  P6 branches were cut only when no P7 run depended on the working tree.
- **One branch and one pull request per task**, merged only after CI passed on the PR's
  head commit. The PR's test counts were compared with main's, to prove the new tests ran.
- **The user's OK was asked in the session, each time:**
  - the first deploy of the instrumented page;
  - every push to Grafana;
  - each scheduled workflow switched on (robot, export, retraining).

  The user created every account and token themselves. Only their names appear here.

## Step by step

### Task 1: the telemetry module (PR #48)

`app/demo/telemetry.js`, a module with no DOM, so the tests can run it under Node:
- `makeEvent`: the allowlist. Each event name has a spec of fields and types: a finite
  number, a known label, a valid choice, a short string. Anything else is dropped.
- **Redaction:** a free-text field becomes `[redacted]` if any run of 4+ letters in it
  occurs anywhere in the visitor's text.
- `wordBucket`: steps of 25.
- `toCollectorPayload`: the Faro format. Events carry string attributes, and the
  context (model runs, synthetic) goes in each event's attributes.
- `send`: fire-and-forget, never rejects.

**Checked:** 7 tests through `cli.mjs`, seen failing first:
- the allowlist: a stray `text` field, an infinite number and an unknown label are all
  dropped;
- an unknown event name is refused;
- an error that quotes the passage is redacted;
- in a real flow (predict, then answer, feedback and error events), no word of the passage
  appears in the payload;
- word buckets;
- the payload's shape;
- `send` never throws.

The collector's format was read beforehand from Grafana's docs and the Alloy receiver's
source.

### Task 2: unusual texts (PR #49)

- `browser.unusual_thresholds` measures the limits on our own passages:
  - the shortest passage, 36 words;
  - the 99th percentile of **held-out** unknown-word shares, 22.9%. Each passage is measured
    against its fold's training vocabulary, as a visitor's text would be. Against the full
    vocabulary, the limit would be 0;
  - a fixed 20% of non-Latin letters.
- `ntl.js` gained `textStats` and `unusualReasons`.
- The export writes `meta.unusual`.

**Checked:** 4 tests:
- JS and Python agree on unknown shares to 1e-9;
- the thresholds come from held-out vocabularies;
- each reason appears for its kind of text;
- a model exported without limits flags nothing.

The champion's browser model was re-exported (MLflow run `d527eec4…`): parity 50 of 50,
largest difference 4.4e-9, and none of the 50 training passages flagged.

**What broke:** the first real export got **401** from DagsHub. The credentials file has
Windows line endings, so the loader kept a carriage return on the URI and the username.
The fix is a loader that strips `\r`.

### Task 3: shadow mode in the deploy (PR #50)

`stage_space` now also stages:
- `telemetry.js`;
- `config.json`, with the collector's address from the variable `GRAFANA_COLLECTOR_URL`;
- the challenger's model files under `challenger/`, when `CHALLENGER_RUN` pins a run.

`version.json` names both runs. The smoke test walks every staged file, and runs the engine
on each model.

**Checked:** 2 new tests, plus the file list updated. The deploy after the merge uploaded
8 files, and the smoke test passed.

### Task 4: the instrumented page (PR #51)

- **The page sends events.** It reports `page_load`, `answer`, `feedback` and `error`,
  never awaiting them, and every `catch` sends an `error`.
- **It shows the unusual note**, when a text trips one of the limits.
- **It asks "Do you know who wrote it?"** after each result, offering the 18 models,
  "another model" and "a person". `feedback.js` allows one answer per result, tied to that
  result's labels.
- **It loads the challenger lazily**, after the first answer.
- **The notice, and the question, appear only when a collector is configured.** Until then,
  the page claims nothing, and asks for nothing that would go nowhere.
- **The model card** says what's sent.

**Checked:**
- 2 feedback tests.
- `scripts/check_demo_page.py` drives the staged page in headless Edge, with a fake
  collector on the page's own origin. It covers four cases:
  1. **Recording:** the events come in order, exactly one feedback is sent, Send turns
     off once answered, no word of the passage is sent, and `?synthetic=1` is carried.
  2. **A collector answering 500, or aborted:** the same answer, and no JavaScript error.
  3. **A missing challenger:** answers still come, and the failure is reported once.
  4. **No collector:** nothing is shown or sent.
- Screenshots at 900 and 390 px were checked.
- After the user's OK, the deploy and a visit to the live site in headless Edge: the
  answer came, and nothing was posted.

### Task 6: the robot visitor (PR #52)

`scripts/robot_visit.py` opens the live page with `?synthetic=1`. It waits up to 90 s for
the model and expects an answer within 10 s. It retries once, then exits 1.
`robot.yml` runs it every 6 hours. It needs no secret, and Playwright is pinned to 1.64.1.

**Checked:** against the live site (loaded in 3.5 s, answered in 0.08 s), and against an
unreachable address (two attempts, then exit 1). With the user's OK, the schedule is on.
The first run on GitHub loaded in 1.2 s.

### Task 7: the weekly export (PRs #53 and #56)

`name_that_llm.live` provides:
- `parse_loki`: Loki's streams back into one row per event;
- `live_accuracy`: per model run, with Wilson 95% intervals. One right out of one gives
  [0.21, 1.0], not 100%;
- `paired`: on texts both models scored, how often only one of them was right;
- `event_counts`.

Only real visitors' answers that name one of our 18 models count.

`scripts/export_events.py` reads 7 days, page by page, and logs a run in the DagsHub
experiment `live-feedback`.

**Checked:** 6 tests:
- parsing;
- live accuracy, ignoring the robot and "a person";
- the exact Wilson interval;
- paired counts;
- Loki paging;
- **an empty week**. That one crashed before its fix: a likely case on a small demo.

A real run from this PC logged its first run.

**What broke:** the first GitHub run timed out reading the week (httpx's default timeout is
5 s). It now waits 60 s (#56). The re-run read 3 events, and logged.

### Task 5: Grafana, and the real stack's surprises (PRs #54, #56, #57)

The dashboard (17 panels) and the alerts were written as JSON, by a generator kept outside
the repo, and `scripts/push_monitoring.py` sends them. The user then created, guided step by
step:
- the stack `sincerebagel1076`. A free account gets one stack, created with the org's
  name;
- the Frontend Observability app `name-that-llm-demo`, with the Space as its allowed
  origin;
- a service-account token, and a `logs:read` token;
- the `email` contact point, set as the default policy's receiver;
- the Infinity data source.

**One synthetic test event**, sent and read back before anything was pushed, found what the
docs hadn't said:
1. **The collector refuses a request without `X-Faro-Session-Id`.** The page now sends a
   fresh random id per request. It satisfies the collector, and links nothing, not even
   two events of one visit.
2. **It refuses a payload that names no SDK** ("no SDK version"). `meta.sdk` now names our
   module, `name-that-llm-telemetry` 1.0.0.
3. **Loki labels the app `service_name`, not `app_name`.** The queries were fixed. All 17
   of them ran on the real Loki without error before the push, and a test pins the label.

Also caught along the way:
- **The local file held the example stack address** from the instructions. It was fixed in
  place, leaving the token lines untouched.
- **A token was saved as a Variable.** The log token went into GitHub Variables, which are
  plain text. Its printed prefix held only the org number and the token's name, so it
  wasn't exposed. The user moved it to Secrets.

**The push:**
- The folder and the dashboard went in. The alert rules came back **403** the first time:
  Grafana caches permissions, and the folder the token had just created wasn't in its
  rights yet. A minute later, both rules were created (201). The script now retries once
  after 30 s.
- With the user's OK, #54 switched telemetry on. The live `config.json` names the
  collector, and the robot's events reached Loki from GitHub's Chromium: `page_load`,
  943 ms, and `answer`, 8 ms, 75 words, `claude-opus-5`, "Fairly sure".
- The robot alert, created paused, was switched on once those events flowed (#56).

**The dashboard showed nothing**, while the user's visit was in Loki. Each link was checked
in turn: browser, collector, Loki, dashboard. The stack has three Loki data sources (logs,
alert state history, usage insights), and the dashboard's picker took the first one
alphabetically. The picker now offers only the one ending in `-logs` (#57), and a test pins
it.

### Task 8: retraining that proposes (PRs #55 and #56)

`name_that_llm.retrain.propose` decides. A candidate is proposed only if the prompt
bootstrap's 95% interval of (candidate − champion) lies above 0, and it isn't below the
current challenger. `pr_body` writes the evidence and what merging does.
`scripts/retrain_challenger.py` and `retrain.yml` (Sundays) carry it out.

**Checked:**
- 4 tests;
- a full dry run on this PC, with the pool rebuilt from the allowlist (48,083 passages). It
  gave **0.3212 → 0.4205**, exactly P7 Task 3's numbers, with parity 50 of 50.
- The user created `RETRAIN_PR_TOKEN`, and the schedule is on.

**What broke:** the first GitHub run couldn't find the competition data. The pool step
dedupes against our passages, and finds them through `NTL_DATA_DIR`, which only the next
step had. Fixed in #56, and re-run (see Open issues).

### Task 9: record and ship

This journal, [ADR 0007](../adr/0007-watching-a-static-site.md), the diary, `Context.md`,
`IDEAS.md` (I-024 and I-025 done) and the journal index.

## Every judgment call

From the ledger, in order, each with its cost if wrong:
1. **No git worktree.** One branch per task, cut only when no P7 run used the working tree.
   Cost: none.
2. **The spec and plan were committed at the end of session 8**, in a docs pull request
   (#47), not with Task 1. They were approved, and existed only on this PC. Cost: none.
3. **Redaction covers free text only** (an error's kind and message). Labels and choices
   come from fixed lists, and redacting them would erase a correct label whenever a passage
   names a model. Cost: none.
4. **Redaction matches any 4-letter run as a substring of the text**, not only whole words.
   A message cut mid-word, or a fragment of a script written without spaces, is still
   caught. Cost: errors redacted more often than needed.
5. **Small format choices:**
   - numbers are rounded to integers;
   - null values are left out of the attributes;
   - lists go out as "a,b";
   - the CLI answers `{"error"}` for an unknown event name;
   - one test beyond the plan pins the payload's shape.

   Cost: none.
6. **A missing `CHALLENGER_RUN` counts as empty**, so an older checkout deploys unchanged.
   Cost: none.
7. **The notice and the question appear only with a collector.** Cost: none.
8. **`feedback.answer()` takes the model's labels.** The page check is committed, its fake
   collector sits on the page's own origin (no CORS preflight), and it adds a fourth case,
   no collector. The browser's own "Failed to load resource" lines don't fail a case.
   Cost: none.
9. **The dashboard reaches its data sources through variables, and the robot alert started
   paused.** Every query filters synthetic events one way or the other. Cost: small.
10. **The robot runs on uv with Playwright pinned**, not pip. Cost: none.
11. **The Wilson test checks the exact interval for 1 of 1**, [0.207, 1.0]. The plan's
    "wider than [0.2, 1.0]" was approximate. Cost: none.
12. **Two export tests beyond the plan:** Loki paging, and an empty week. Cost: none.
13. **The candidate's unusual-text limits come from our passages' vocabularies**, not the
    pool's. The candidate knows more words, so the limit flags a little less, never more.
    Cost: a few unusual texts unflagged.
14. **The retraining tests use 20 prompts instead of 40**, to halve the bootstrap's time.
    Cost: none.
15. **A per-request random session id**, rather than one per visit, as the collector
    required. Cost: Grafana's own "sessions" view is meaningless for this app, and nothing
    here uses it.

## Open issues

- **The retraining's second run** (after #56) proposed `lr` plus the pool as the demo's
  challenger: CV 0.4186 on the runner, where the local dry run gave 0.4205. The user merged
  it (#58), and shadow mode is live.
- **GitHub switches off scheduled workflows** in a public repo after 60 days without
  activity. After the competition, the robot will stop, and its silence alert will email
  every repeat interval. ADR 0007 says what to do.
- **Branch protection on main** (pull requests only, CI required) is suggested: the
  retraining's token can write to the repo. It's the user's setting to switch on.
- **Grafana Cloud's trial** turns into the free tier on its own after 14 days. Logs are
  kept about two weeks, which is why the weekly export exists.
- **The collector's address is public**, so anyone could send fake events. Browsers are
  held to the allowed origin; scripts aren't. Filtering by page commit is the defense, if
  it's ever needed.

## The final review

- **The reviewer:** the user chose a fresh reviewer agent. It read the whole branch
  (`3b8becd..4c75a3e`, #48 to #57) against the plan and the design.
- **Its verdict:** "with fixes". Nothing was Critical, 6 findings were Important, and 10
  were Minor (A to J). In its words: "The privacy and robustness promises hold, and nothing
  is Critical."
- **Facts checked before the triage:**
  - the live challenger's download is 35.1 MB (4.7 + 30.4), and the champion's 15.3 MB;
  - the collector adds `page_id`, and `browser_mobile` (read from the User-Agent), to each
    stored event. It stores no IP address, location or browser name.
- **The triage** graded each finding by what a person using the site would get. All six
  Important findings were fixed. Two Minor ones were upgraded:
  - **B**, because the model card promises that a message quoting the text is redacted;
  - **C**, because anyone can send events to the public collector, and one junk value
    crashed the weekly export.

**The fixes**, in one pull request (#60). Each has a test that failed first.

| # | The finding | The fix | Its test |
|---|---|---|---|
| 1 | The retraining would open the same proposal every week: the candidate only had to be no worse than the challenger | A margin of 0.005 over both pinned models' logged CVs | `test_propose_needs_a_margin_over_every_pinned_model` |
| 2 | The write token stayed in the checkout for every step, and `kaggle` wasn't pinned | The checkout keeps no credentials; the token is set up only in the pull-request step (`gh auth setup-git`); `kaggle==2.2.5` | `test_the_retraining_workflow_keeps_its_write_token_in_one_step`, `test_the_pull_request_step_sets_up_git_credentials_before_pushing` |
| 3 | The live panels counted answers no model can get right ("a person", "another model"), and compared champion and challenger on different texts | Only answers naming one of the 18; per model run, with n beside; champion vs challenger on the same texts, as counts | `test_live_accuracy_counts_only_answers_naming_a_model`, `test_live_numbers_are_per_run_on_the_same_texts_and_show_n`, `test_a_model_with_no_right_answer_shows_zero` |
| 4 | The bar charts needed an x field the queries didn't give, so they didn't draw | Bar gauges, one bar per series | `test_no_panel_depends_on_an_x_field` |
| 5 | The export's week ended "now": a late, repeated or missed run left gaps or overlaps | Fixed Monday-to-Monday weeks (UTC), and `--end` to redo a missed week | `test_export_weeks_are_fixed_monday_windows` |
| 6 | The challenger was scored before the answer was painted, inside the champion's `try` | After paint, in its own `try`, with its own time; skipped when the browser asks to save data or on 2g; once loaded, it scores the text on screen | `test_answer_events_carry_the_challengers_own_time`, and the page check |
| B | Redaction missed the same letters in another Unicode form | NFKC normalization on both sides | `test_redaction_sees_through_unicode_forms` |
| C | One junk event could sink the export, or blank a panel | A logfmt parser instead of shlex; missing fields count as zero; safe metric names; `__error__=""` after `unwrap` | `test_parse_loki_survives_junk_values`, `test_a_window_without_the_synthetic_field_gives_zeros`, `test_metric_names_are_safe_whatever_the_run_id`, `test_unwrapped_values_drop_parse_errors` |
| H | The page check never ran a second answer | It does, and checks the challenger's label and time | `scripts/check_demo_page.py` |

**The render check** came after the fixed dashboard was pushed (with the user's OK). Three
panels were rendered as images through Grafana's render API:
- the stat panel read "Value #A 0.00", and each bar gauge showed one bar with no name;
- **the causes:** a Loki instant query reaches a stat panel as one table, which it reduces
  to a single unnamed number; and Grafana hides the name of a panel's only value, which is
  common on a quiet demo;
- **the fix:** range queries reduced to their last point, and display-name templates. Two
  tests failed against the committed dashboard, then passed;
- **re-pushed and re-rendered:** "share right (d527eec…) 0%", "answers 2", and bars named
  qwen3.8-max and claude-opus-5. That's the user's visit: they said qwen3.8-max, and the
  champion said claude-opus-5.

**The rulings:**
- **Task 8's ruling on the unusual-text limits is overturned in part,** as the reviewer
  argued. Before a challenger is promoted, its limits are re-measured with its own recipe.
  This is now a step of the promotion procedure in `Context.md`. Cost if wrong: none while
  it's only a challenger, since its flags are never shown.
- **`propose()` now takes `pinned_cvs`**, both pinned models' CVs, as the reviewer
  proposed. Cost: none.
- **The points the reviewer declined to judge:**
  - the collector's stored fields were checked, and the model card now lists what it adds;
  - the panels were checked by rendering them (above);
  - branch protection on main is suggested to the user;
  - the rest stand: keepalive in Firefox and Safari, the alert thresholds, action tags
    rather than commit SHAs, code from before P6, and data retention.

  Cost if wrong: small.

**Docs, as the reviewer asked:**
- the model card gives the 35 MB download, the "phone or not" fact the collector adds, and
  that servers see the visitor's IP address, which the stored events don't include;
- ADR 0007 gives the retraining's baseline after a promotion, the 60-day rule (G), and
  where the intervals live.

**Deferred minors**, not fixed:
- **A, the rest:** if the visitor edits the text without pressing Identify again, their
  answer still attaches to the result on screen.
- **D:**
  - the export's paging restarts just after a page's newest timestamp, so lines sharing that
    millisecond can be dropped;
  - `paired()` pools every pair of runs.
- **E:**
  - the dashboard has no page-commit filter;
  - "another model" and "a person" answers don't feed the unusual-text counts;
  - the length buckets sort as text;
  - the reference text is copied by hand.
- **F:** the retraining's branch name (`retrain/<date>`) collides on a same-day re-run,
  leaving an orphan MLflow run; there's no concurrency group.
- **I:** `push_monitoring.py` treats any 400 as "already exists", so a real validation
  error surfaces as a 404.
- **J:** keepalive with a custom header needs a CORS preflight, and was checked in Chromium
  only.

**Shipped:**
- #60 (squash `9d6b8bf`), after CI passed;
- the redeploy's smoke test on the live site: champion claude-opus-5 (11%), challenger
  gpt-5.6-terra (19%);
- the whole suite: 210 passed, with torch, transformers and peft.

## Process lessons

- **Test against the real service before building on its docs.** One synthetic event found
  three differences from the documentation (a required header, a required SDK name, a
  different label) before any dashboard was pushed. Every query was then run on the real
  Loki.
- **When something "shows nothing", check each link of the chain separately.** Reading Loki
  directly showed the visit had arrived, so the bug could only be in the dashboard.
- **Never rely on a picker's default.** It depends on what else exists on the stack, which
  no local test sees.
- **A first scheduled run is a test of its own.** Both weekly jobs failed their first run
  on GitHub, for reasons a local run couldn't show: a longer time range, and an environment
  variable the next step had but this one didn't.
- **Give step-by-step instructions with the real values, not examples.** An example address
  ended up in a credentials file.
- **Look at the rendered panel, not only its JSON.** Thirteen tests passed, and the real
  Loki accepted every query, yet two panels showed unnamed numbers. P4 learned the same with
  a headless browser. Grafana's render API turns a panel into an image in one call.

## How to retrace P6 yourself

- The plan and the design: [plan](../superpowers/plans/2026-10-27-p6-mlops-loop.md),
  [design](../superpowers/specs/2026-10-27-p6-mlops-loop-design.md).
- The pull requests #48 to #58, in order, then #60 (the fix pass), with their CI runs.
- **Run the checks:**
  - `uv run --all-extras pytest -q tests/test_telemetry.py tests/test_feedback.py
    tests/test_browser.py tests/test_deploy_space.py tests/test_monitoring.py
    tests/test_live.py tests/test_retrain.py`;
  - the page check:
    `uv run --all-extras --with playwright python -X utf8 scripts/check_demo_page.py`.
- **The live system:**
  - the dashboard at <https://sincerebagel1076.grafana.net/d/name-that-llm-demo>;
  - the robot, export and retraining runs in the repo's Actions tab;
  - the `live-feedback` experiment on DagsHub.
