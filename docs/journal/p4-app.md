# Journal: P4, the app and continuous deployment (2026-10-26 → 27)

**Goal:** a web page where anyone can paste a passage and see which of the 18 models most
likely wrote it, ranked with percentages and an honest "How sure?". It should run the
real champion, deploy itself on every merge, and check itself once live.

**Result:**
- **The full champion is packaged and registered:** `name-that-llm` version 1 is
  `@champion` in a private MLflow registry on DagsHub. A FastAPI app and a Docker image
  run it on CPU, about 3.7 s per passage.
- **The free public demo is live:** <https://huggingface.co/spaces/Ryqn/name-that-llm>.
  Hugging Face started charging for Docker Spaces, so the public page runs the lighter
  `lr` model inside the visitor's browser. It's a static Space, deployed and smoke-tested
  on every merge.

**In numbers:**
- 12 tasks (the plan's 9; Task 8 became 8a, 8b and 8c after the hosting change);
- 12 pull requests (#31 to #42);
- one registered model version, and one pinned model run for the demo.

The design: [2026-10-26-p4-app-design.md](../superpowers/specs/2026-10-26-p4-app-design.md)
(the hosting change is its §12). The plan:
[2026-10-26-p4-app.md](../superpowers/plans/2026-10-26-p4-app.md).

## How the work was run

- **Native execution, with a ledger:** the same as P5. Claude implemented each task in the
  session, test first, and logged every step, result and judgment call in a git-ignored
  ledger. That ledger is folded into this journal.
- **Shipping:** one branch and one pull request per task, squash-merged only when CI is
  green. Each merge was gated on the PR's own CI run (its commit had to be the PR's head).
  Its pytest counts were compared with `main`'s, to prove that new tests had run rather
  than skipped.
- **Approvals, asked in the session that used them:**
  - registering the champion (Task 7);
  - the first public deploy (asked twice: once on 2026-10-26 for the Docker Space, and
    again on 2026-10-27 for the static Space, since approvals don't carry over between
    sessions).
  - **Hosting:** when the deploy hit Hugging Face's paywall, the choice of hosting was
    yours.
- **Secrets:** you created the DagsHub and Hugging Face accounts, the tokens, the env file
  (`C:\Users\dungr\.dagshub.env`, outside the repo), and the GitHub secrets and variables.
  Claude only ever printed key names and status codes.
- **The final review** was a self-review (sub-agents run only when you ask for them). It
  found no critical or important issue; its five minor findings are listed in
  [the final review](#the-final-review).

## Step by step

### Task 1: flatten pasted markdown (PR #31)

Chatbot answers arrive with markdown, while the competition's passages had all of it
removed. So `data.flatten_text` turns a pasted answer into the same kind of plain
paragraph. Its rules come from the data:
- at the start of a line: headings, quotes, bullets, numbering, code fences and table rules
  go;
- anywhere: `**`, `__`, backticks and pipes go, and links keep only their text. A single
  `*emphasis*` stays, because 14 training passages contain one;
- then whitespace is collapsed, and the text is normalized to NFC.

**Checked:** none of the 2,699 training passages changes. For each rule, Claude worked out
that dropping it makes at least one of the 3 tests fail.

### Task 2: load only the Qwen blocks the LM head reads (PR #32)

`lm.load_lm_layers` loads Qwen 3 4B base cut to its first **10** blocks, in float32 on CPU.
The LM head reads hidden states 6 to 9. A 9-block cut would return layer 9 *after* the
final normalization, which the LM head never saw: the "9-block trap". Hence the rule: keep
the last layer read, plus one. The tests use a tiny random Qwen.

### Task 3: the app's predictor and the "How sure?" bands (PR #33)

`champion.Champion.predict_proba` flattens the text, then blends the three models with the
leaderboard weights, 0.70 / 0.15 / 0.15. It's defensive: a non-finite LM feature falls back
to the training mean, and a non-finite encoder output to the uniform 1/18. The bands are
the terciles of the top probability on out-of-fold predictions, each with its measured
accuracy.

- **A ruling:** the plan's NaN test poisoned the last kept block, but block 10's output
  feeds only the final state, which the LM head never reads, so the test could not fail.
  The test poisons block 6 instead.

### Task 4: package the champion with MLflow (PR #34)

`registry.py` turns the champion into an MLflow `pyfunc` model: the joblib files, the 5
encoder folds, and a `meta.json` holding the weights, the bands and Qwen's pinned
revision. `load_registered(name, alias)` resolves the alias once and returns the predictor
and its info. The requirements are listed explicitly; otherwise MLflow would load Qwen
just to infer them.

`scripts/register_champion.py` does every check before registering (see Task 7).

- **A ruling:** the plan said to drop the vectorizers' `stop_words_` before saving, but
  scikit-learn 1.9.1 no longer creates that attribute. Code to drop it would be dead, and
  its test could never fail.

### Task 5: the web app and its page (PR #35)

`api.create_app` is a FastAPI app that works with any predictor (the tests use a fake one):
- **Loading:** the model loads in the background. `/health` answers 503 while loading, 200
  when ready, 500 if loading failed.
- **Limits:** one prediction at a time with 4 waiting, then 503; 10 requests a minute per
  visitor; 20 to 1,000 words after flattening (422); 32,000 bytes (413).
- **Nothing stored:** the text is never logged.
- **`/version`** names the build's commit and the model version.

The page follows the approved mockup (layout B): the best guess, the "How sure?" pill, the
top 5 bars with "Show all 18", and the warning box. Checked over HTTP with the real server
and a fake model. The visual check waited for a browser, which came in Task 8c.

### Task 6: the Docker image (PR #36)

`app/Dockerfile`:
- `python:3.13-slim`, then `uv sync --frozen --no-editable`;
- CPU-only torch 2.11.0 and transformers 5.16.1;
- a non-root user, on port 7860.

A new CI job builds the image on every pull request (72 s). It fails if:
- the versions inside aren't exactly `2.11.0+cpu 5.16.1`;
- a file planted in `data/raw` before the build is found anywhere in the image.

- **Rulings:**
  - the plan's test that grepped `.dockerignore` became that planted-file check, which
    tests the behavior, not the text;
  - the local build became the CI build, since Docker's daemon wasn't running on this PC.

### Task 7: register the champion (no code)

1. **Download the inputs:** the GPU features (511 MB) and the 5 fold models, from the P3b
   and P5 notebooks.
2. **Dry run, every check:**
   - Qwen pinned at revision `906bfd4b…`;
   - Hub CPU features against Kaggle GPU features: median relative difference 0.0010,
     max 0.0052, the same LM-head top label 20 of 20;
   - the bands (below);
   - parity 50 of 50, max probability difference 0.0001.
3. **Your accounts:** a standalone *private* DagsHub repo (a ruling: a mirror of the
   public GitHub repo would have inherited its public visibility), and the token in an env
   file. The first env loader skipped the token: the file's last line had no newline, and a
   plain `while read` loop drops such a line. The fix: `read … || [ -n "$key" ]`.
4. **Your OK, then the registration.** Version 1 was created, but the script then crashed
   on MLflow's closing print: an emoji that Windows' cp1252 can't encode. Since the version
   was complete, it was kept (a ruling): the run was marked finished and the alias set by
   hand. Then the full check: `load_registered("name-that-llm", "champion")` from DagsHub
   took 76 s, and one prediction 3.71 s on 2 threads. Lesson: run with `python -X utf8`.

The champion's bands:

| How sure? | top probability | right |
|---|---|---|
| Fairly sure | ≥ 0.268 | 61.8% |
| Unsure | ≥ 0.197 | 40.3% |
| Very unsure | below | 24.5% |

### Task 8: deploy the Docker app, then the paywall (PR #37)

`scripts/deploy_space.py` and the `deploy-app` workflow:
1. stage the app;
2. create the Space and set its secrets;
3. upload;
4. wait for `/version` to name the commit;
5. call `/predict` once.

With your OK, the PR was merged and the deploy ran:
- **403:** the Hugging Face namespace is `Ryqn`, not `Ryqn0`. The GitHub variable was
  fixed and the deploy re-run.
- **402:** *"Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
  free cpu-basic requires a PRO subscription."* Nothing was created or charged. Claude
  stopped and asked.

You chose the **lighter free demo** over Hugging Face PRO, a free VM, or local only. The
design gained §12; the plan's Task 8 became 8a to 8c; the deploy workflow became
manual-only until 8c.

### Task 8a: the browser engine (PR #38)

`app/demo/ntl.js` ports to JavaScript, with no DOM:
- `flatten_text`;
- scikit-learn's TF-IDF for exactly our settings (tokens are runs of letters, digits and
  `_`, at least 2 code points; the character analyzer squeezes whitespace; sublinear term
  frequency, idf, l2 norm);
- the softmax.

`name_that_llm.browser.export_browser_model` writes `model.json` (the labels, each
vectorizer's vocabulary in column order, the bands) and `weights.bin` (float32). It refuses
any setting the port doesn't implement. Node runs the engine in the tests: flattening must
match exactly, and probabilities within 1e-5. CI went from 98 to 102 passed with the
skipped count unchanged, so the new tests really ran there.

- **Two traps:**
  - Node loads `.js` as CommonJS unless `package.json` says `"type": "module"`;
  - a raw U+2028 in a regex literal is a line break, so it's written as `\u2028`.

### Task 8b: export the demo model and pin it (PR #39)

`scripts/export_browser_model.py`:
1. `lr` from version 1's files;
2. **its own bands** on out-of-fold predictions (seed 42);
3. the export;
4. parity on 50 real passages, Node against scikit-learn: 50 of 50, largest difference
   4.4e-9;
5. an MLflow run on DagsHub, whose id goes in `app/demo/MODEL_RUN`.

Downloading the run back gave byte-identical files.

| How sure? (`lr` alone) | top probability | right |
|---|---|---|
| Fairly sure | ≥ 0.152 | 47.2% |
| Unsure | ≥ 0.117 | 30.3% |
| Very unsure | below | 18.8% |

Its balanced accuracy is 0.3212, the same as in the results table. Alone, `lr` spreads its
probability thinly, so the champion's thresholds would have called almost every answer
"Very unsure".

### Task 8c: the static page and its deploy (PRs #40 and #41)

**The page** (`app/demo/index.html`) is layout B, running `ntl.js`: the model loads once
(15 MB), and each answer takes about 0.01 s on the visitor's device. **The model card**
(`app/demo/README.md`, `sdk: static`) gives `lr`'s numbers next to the full model's.

**The deploy script**, rewritten:
1. download the pinned run;
2. stage exactly six files;
3. upload;
4. wait until the live `version.json` names the commit;
5. smoke-test.

- **Where to wait:** Hugging Face serves static Spaces on `*.static.hf.space`, not on
  Docker's `*.hf.space`. So the script asks the API (`space_info().host`) instead of
  guessing (checked on two public static Spaces).

**The visual check, at last.** Headless Edge, driven by Playwright from a throwaway
environment, rendered the staged page on a 900 px light screen and a 390 px dark one.
(Avast on this PC cuts 13.5 MB transfers over localhost, 4 times out of 4, so Playwright
answered the page's requests from the staged folder instead.)

It found **two layout bugs, shared by both pages**:
- the top 5 and the other 13 were separate grids, so their bars didn't line up;
- on a phone the lower bars shrank to 1–3 px.

The fixes:
- one grid for all 18 rows, with rows 6 to 18 hidden by
  `.bars.collapsed > :nth-child(n+21)` (4 cells per row, so the 6th row starts at cell 21);
- below 480 px, no company column.

Measured after the fix: every bar starts at the same x, and phone bars are 99 px wide.

**The first public deploy (#40, with your OK)** went live, then the smoke test stopped.
The site serves `README.md` rendered as HTML, and it adds tags to the page's `<head>` (a
script with the Space's variables, and social-preview `<meta>` tags). The engine and the
model files were byte-identical. Claude drove the live page in Edge, which worked. **The
fix (#41):** skip the card, which the page never loads, and accept the page if every line
of ours is there, in order (`served_intact`). The redeploy passed end to end.

### Task 9: record and ship (PR #42)

- ADR 0004 (the champion on CPU, from a registry) and ADR 0005 (the public demo in the
  browser);
- this journal;
- DIARY session 7, Context and IDEAS (I-024 to I-026);
- the Docker app's card (its speed, measured locally);
- the `-X utf8` note in the registration script.

## Every judgment call

The plan didn't settle these, so Claude decided, and logged each with its cost if wrong.

1. No git worktree: one branch per task, shipped by PR. Cost if wrong: none.
2. The design, its mockup and the plan went in with Task 1. Cost if wrong: none.
3. The LM-fallback test poisons block 6, not the last kept block (Task 3). Cost if wrong:
   none.
4. No `stop_words_` dropping: scikit-learn 1.9.1 doesn't create it (Task 4). Cost if wrong:
   a bigger pickle with scikit-learn < 1.9.
5. The no-words test uses fence lines, which flattening drops, instead of `# ` lines,
   which it keeps (Task 5). Cost if wrong: none.
6. The page's visual check moved to the deploy (Task 5, done in 8c). Cost if wrong: a
   layout bug found later, which is what happened.
7. A planted file instead of grepping `.dockerignore`, and the build in CI instead of on
   this PC (Task 6). Cost if wrong: a CI round trip per failed build.
8. Task 7 ran its downloads and dry run before asking for your account. Task 8's code was
   written while Task 7 waited. Cost if wrong: none.
9. A standalone private DagsHub repo, not a mirror of the public repo (Task 7). Cost if
   wrong: none.
10. Version 1 kept after the crash in MLflow's closing print (Task 7). Cost if wrong: a
    bad version would have failed the end-to-end load; it didn't.
11. The lighter free demo: your choice, which replaced the Docker deploy with 8a to 8c.
12. `package.json` with `"type": "module"`, and escaped line separators (8a). Cost if
    wrong: none.
13. The tests' Node call moved into the package as `run_engine`, so the export and the
    deploy reuse it (8b). Cost if wrong: none.
14. The Space's address comes from the API; `space_host` was deleted (8c). Cost if
    wrong: none.
15. The smoke test compares the live files with the staged ones, then runs the engine on
    the staged files rather than on copies of the live ones (8c). Cost if wrong: none.
16. The DagsHub credentials are passed under MLflow's own names; `ENCODER_FOLDS` was
    dropped from the workflow (8c). Cost if wrong: none.
17. The layout fix went into both pages, but only the demo's was driven in a browser
    (8c). Cost if wrong: a glitch in the local-only full app.
18. The smoke-test fix redeployed under the approval you'd given for the first deploy:
    the same files, re-uploaded (8c). Cost if wrong: none.
19. Two ADRs, where the plan named one: the hosting change was a second decision of its
    own (Task 9). Cost if wrong: none.

## Open issues

- **The full app's page** got the layout fix without being driven in a browser.
- **The demo's model card** quotes `lr`'s numbers by hand. A new `MODEL_RUN` needs the
  card updated too.
- **The smoke test relies on how Hugging Face serves static Spaces today** (the card
  rendered, tags added in `<head>`). If that changes, deploys stop with a clear message
  rather than passing silently.
- **No `v0.8.0` tag.** `pyproject.toml` says 0.8.0, but the offline notebook needs nothing
  from P4. A tag would publish the Kaggle dataset, which needs your OK.
- **Leftover settings:** the GitHub variable `ENCODER_FOLDS` is no longer used, and W&B
  is still off in the GPU notebooks (from P5).

## The final review

A self-review of all of P4 (`8f89cf6..9e51d83`, plus this task's docs). The six tests
behind the plan's Review Focus all exist and pass.

- **Strengths:**
  - parity is tested on every pull request, and was checked on 50 real passages;
  - CI proves no data reaches the Docker image, by planting a file in `data/`;
  - the public model is pinned by a run id, and the deploy's smoke test runs against the
    live site;
  - the registry alias is resolved once, so `/version` names what's really running;
  - both pages render with DOM calls only, never raw HTML.
- **No critical or important finding.**
- **Five minor findings, deferred.** Each is a ruling for you to accept or reverse:
  1. The full app reads the whole request body before its 32 KB check, and before the
     rate limit. Once it's public, a huge upload could exhaust memory.
  2. The full app's rate limit trusts the *first* `X-Forwarded-For` address, which a
     client can set itself. Rotating fake addresses would bypass the limit and grow its
     table without bound.
  3. The demo's flattening uses JavaScript's idea of whitespace, not Python's. U+FEFF (a
     byte-order mark) counts as a space only in the browser, and U+001F only in Python.
     For a text containing them, the browser's answer differs slightly from
     scikit-learn's. Probed directly; the other unusual spaces agree.
  4. `/health` shows the model-loading error's text publicly.
  5. A transient Hugging Face API error during the deploy's wait fails the deploy; it's
     re-run by hand.

  Findings 1, 2 and 4 matter only once the full app is hosted, so they're written into
  I-026 as prerequisites.

## Process lessons

- **Check a host's current terms before building for it.** A 402 at deploy time cost the
  Docker deploy; the paywall was a few weeks old.
- **Test what visitors actually get.** A static host can rewrite files. The smoke test now
  compares the live files with ours and allows only what Hugging Face adds.
- **Look at the page.** The HTTP checks passed in Task 5 while two layout bugs waited to be
  seen. A headless browser and two screenshots found them in minutes.
- **On this PC:**
  - Avast cuts large localhost transfers, so drive local pages with Playwright's request
    routing rather than a local server;
  - Python's `write_text` writes CRLF, which git normalizes, so readers should strip;
  - MLflow prints emoji, so run with `python -X utf8`.
- **`gh pr checks --watch` returns at once if CI already ran on the push.** Check that the
  run's commit is the PR's head before trusting it.

## How to retrace P4 yourself

- The 12 changes, one per line: `git log --oneline 8f89cf6..HEAD` (on `main`, after #42).
- One task's diff, with `git show <commit>`:

  | Task | Commit |
  |---|---|
  | 1 | `91d3bd8` |
  | 2 | `2c707b1` |
  | 3 | `6b74b0b` |
  | 4 | `17e76a0` |
  | 5 | `7fc1b33` |
  | 6 | `3d476d8` |
  | 8 | `e168464` |
  | 8a | `a8b519a` |
  | 8b | `54a9c2d` |
  | 8c | `804fe3f` and `9e51d83` |
- The pull requests keep the original commits and the CI runs:
  [#31](https://github.com/Ryqn0/name-that-llm/pull/31) to
  [#42](https://github.com/Ryqn0/name-that-llm/pull/42).
- **The tests:** `uv run --all-extras pytest -q`.
  - The model tests need torch: add `--with torch --with transformers`.
  - The browser tests need Node.
- **The demo:** <https://huggingface.co/spaces/Ryqn/name-that-llm>.
- **The full champion, locally**, with the three `MLFLOW_TRACKING_*` variables set:
  `uv run --all-extras --with torch --with transformers uvicorn name_that_llm.serve:app --port 7860`,
  then open <http://localhost:7860>. Or build `app/Dockerfile`.
- **A new demo model:** `python -X utf8 scripts/export_browser_model.py` (try `--dry-run`
  first), then commit the new `app/demo/MODEL_RUN`. Merging it redeploys.
- **The full conversation:** see [the full sessions](README.md#the-full-sessions).
