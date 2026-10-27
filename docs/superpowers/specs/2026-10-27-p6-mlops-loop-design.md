# P6: the MLOps loop: watching the site and the models — design

**Date:** 2026-10-27 · **Status:** draft, for the user's approval · **Phase:** P6 ("MLOps
loop" in the [project design](2026-10-23-name-that-llm-design.md)). It's built on CPU while
P7 trains on Kaggle's GPUs.

## 1. Intent

In the user's words, P6 is for two moments:

1. **"When you see some issues on the site or notice that there might be some lags, slows
   or any connection error":** debugging, and watching the site's activity.
2. **"To check occasionally if some models will overtake others":** watching model
   performance over time.

What changed since the project design: the public demo is a **static page** (P4, ADR 0005).
`lr` runs in the visitor's browser, and there's no server of ours to instrument. So the page
itself reports what happens, and model performance on real texts comes from visitors who
know which chatbot wrote what they pasted.

## 2. Decisions

**The user's:**
- Monitor the public site (activity, speed, errors) and the models (which overtakes which).
- **The page asks an optional question after each answer: "It was actually…".** Visitors
  who know the author turn the site into a live test set.
- The approach played back on 2026-10-27, which the user accepted, including:
  - Grafana Cloud's free tier;
  - anonymous numbers, never the text;
  - a robot visitor;
  - shadow mode;
  - retraining that opens pull requests;
  - flagging unusual texts.

**Claude's calls** (each is marked again where it applies):
- **Shadow mode instead of a 50/50 A/B test** (I-024). A demo gets little traffic, so a split
  would take far too long to tell two models apart. In shadow mode, every visitor sees the
  champion's answer, a challenger silently scores the same text, and the visitor's answer
  judges both, on the same texts.
- **Grafana Cloud** (free tier) for the dashboard, the stored events and the alerts. It's
  the service the project design named.
- **No Evidently report on GitHub Pages.** The dashboard shows the same drift signals (text
  length, unknown-word share, confidence, the mix of predicted models) next to their
  values on our own passages.
- **The local full app (FastAPI) isn't monitored in P6.** Nobody visits it. Monitoring it
  waits for hosting (I-026).

## 3. Goals, non-goals, anti-goals

**Goals**
1. **The site reports anonymous events**, and a Grafana dashboard shows activity, load
   time, answer time and errors over time.
2. **Alerts:** an email when errors spike, or when the site stops answering.
3. **A robot visitor:** a scheduled job opens the live page, pastes a passage written for
   the purpose, and checks the answer and its speed. It catches a broken or slow site with
   no visitors around. Its events are tagged and never counted as visits.
4. **"It was actually…":** an optional answer after each result, with live accuracy per
   model on the dashboard.
5. **Shadow mode:** a challenger model scores every text silently, so champion and challenger
   are compared on the same texts.
6. **Retraining on a schedule:**
   - a weekly job rebuilds the pool and retrains the browser model;
   - if the new model wins on our folds, the job opens a pull request that makes it the
     challenger;
   - promoting a challenger to champion is a separate pull request, decided by a person.
7. **Unusual texts (I-025):** the page notices a text unlike the training passages, tells
   the visitor, and counts it on the dashboard.
8. **Models over time:** one dashboard view lines up every model's CV and leaderboard score
   by date (from what W&B and MLflow already record), alongside the live results of
   goals 4 and 5.

**Non-goals**
- Monitoring the local FastAPI app; hosting the full champion (I-026).
- A 50/50 A/B split (shadow mode replaces it, see §2).
- An Evidently report and GitHub Pages.
- Any paid service.

**Anti-goals** (failures even if everything runs)
- **The pasted text leaving the browser,** in any form: no excerpt, hash or word list.
- Cookies, anything stored on the visitor's device, or any identifier that follows a
  visitor.
- **Telemetry that can break or slow the page.** If the monitoring service is down, the
  page works exactly as today, silently.
- The robot's visits counted as real ones.
- **A retraining job that changes the site by itself.** It only opens pull requests, and
  every merge stays a person's.
- Competition passages in public logs or artifacts. The repo is public, so its Actions logs
  are too.
- The page getting slower for visitors because of the challenger.

## 4. The approach

### 4.1 Events from the page

The page sends small JSON events to Grafana Cloud's frontend collector. Each event carries:
- the event name;
- the page's commit;
- the champion's and the challenger's MLflow run ids;
- `synthetic` (true only for the robot);
- the event's own fields:

| event | fields |
|---|---|
| `page_load` | model download and load time (ms), whether it loaded |
| `answer` | answer time (ms), word count (in buckets of 25), champion's label and band, challenger's label (if loaded), unusual-text reasons (if any) |
| `feedback` | the visitor's answer: one of the 18, "another model", or "a person"; plus the champion's and challenger's labels for that text |
| `error` | where (load, answer), the error's type and message (never the text) |

- **No identifiers, no cookies, no device storage.** Claude's call: send plain events, or
  use Grafana's Faro SDK only with its session tracking off.
- **Every send is fire-and-forget.** A failure is swallowed, so the page never waits for
  monitoring or breaks because of it.
- **A notice under the warning box,** repeated in the model card, lists exactly what's sent
  and says the text never is.
- **The collector's address comes from a GitHub variable** that the deploy writes into the
  page's config (Claude's call): never hard-coded, and changeable without a code change.
- **The answers outlive the free tier** (Claude's call). Grafana's free tier keeps data for
  about two weeks (to confirm once the account exists), and "It was actually…" answers are
  rare. So a weekly job exports the week's events, never any text, to the private DagsHub
  tracking (an MLflow experiment `live-feedback`). The long-run live comparison is
  computed from those exports.

### 4.2 "It was actually…"

After each answer, a small optional row asks: "Do you know who wrote it?" The choices are
the 18 models, "another model" and "a person". One answer per result.
- The "another model" and "a person" answers also feed the unusual-text counts.
- The choices in that row are Claude's call.

### 4.3 Shadow mode

- `app/demo/CHALLENGER_RUN` pins the challenger's MLflow run, next to `MODEL_RUN`. Empty
  means no challenger.
- **The challenger loads only after the champion has answered once** (Claude's call). The
  visitor's first answer is never slower, and texts scored before it loads simply have no
  challenger label.
- The deploy stages and smoke-tests the challenger's files like the champion's.
- **The first challenger comes from P7:** `lr` trained with the pool, if P7 Task 3 shows
  that the pool helps `lr`. Otherwise there's no challenger until the retraining finds one.

### 4.4 The robot visitor

- A scheduled GitHub Actions workflow drives a headless browser (Playwright, as P4's visual
  check did). Every 6 hours (Claude's call), it opens the live page and pastes a passage
  written for the purpose (never a competition passage). It checks that an answer appears
  in time, and sends its events with `synthetic: true`.
- **It fails loudly when the page is broken or slow.** The workflow's failure emails the
  repo owner, and its slow or failed runs show on the dashboard.

### 4.5 The dashboard and alerts

- **The dashboard is code:** its JSON lives in the repo, and a script pushes it to Grafana.
  It shows:
  - real visits, load and answer times, and errors over time (robot excluded);
  - the robot's own line;
  - the live comparison: per model, how many "It was actually…" answers it got and its
    accuracy on them, with an interval, champion and challenger side by side;
  - drift signals next to their values on our passages: word count, unknown-word share,
    top probability, the mix of predicted models, and the unusual-text counts;
  - the models over time (goal 8).
- **Alerts** (Grafana alerting, by email): errors above a threshold, or no robot success for
  12 hours.

### 4.6 Unusual texts

The page checks each text against thresholds measured on our own passages: very few words,
a high share of words outside the model's vocabulary, or mostly non-Latin letters. It tells
the visitor ("this text looks unlike what the model learned from: treat the answer with
extra care") and sends the reasons with the event.
- The thresholds come from held-out predictions, so about 1% of our own passages would be
  flagged.
- The exact checks are the builder's.

### 4.7 Retraining

A weekly GitHub Actions workflow, which can also be started by hand:
1. downloads the competition data (Kaggle API) and the pool, into the runner's temporary
   space only;
2. rebuilds the pool (P7's `build_pool`), with its safeguards;
3. trains the candidate (`lr` on our passages plus the pool), and runs the same CV on our
   seed-42 folds for the candidate and for the champion (`lr` on our passages alone). That
   is a paired comparison, on the same folds;
4. **if the candidate beats the champion with the bootstrap's lower bound above 0, and its
   CV isn't below the current challenger's** (the CV logged with the challenger's MLflow
   run, from the same folds; Claude's call):
   - it exports the candidate for the browser, with the 50-passage parity check under Node;
   - it logs the export as an MLflow run, with its CV;
   - it opens a pull request that sets `CHALLENGER_RUN`, with the numbers in its body.

   The pull request runs CI like any other. That needs a token other than the workflow's
   own, because pull requests opened with `GITHUB_TOKEN` don't trigger workflows.
5. It prints counts and scores only, and uploads no artifact.

**Promotion:** a person merges a pull request that moves the challenger into `MODEL_RUN`,
with the offline gate and the live numbers in its description. The deploy runs as today.

## 5. Constraints

- **Free services only:** Grafana Cloud's free tier, and GitHub Actions (free for a public
  repo).
- **Accounts and secrets are created by the user,** as in P4. Claude only ever handles
  names, never values:
  - a Grafana Cloud account with a frontend-observability app (its collector URL is
    public by design);
  - a Grafana token for pushing the dashboard and the alerts;
  - the alert email;
  - a GitHub token for the retraining job's pull requests.
- **The collector's URL is public.** Anyone could send fake events. Those can't harm the
  site, only the numbers. The collector accepts only the Space's origin, and the dashboard
  can filter by page commit.
- **The competition data stays private:**
  - in GitHub Actions it lives only in the runner's temporary space;
  - it never appears in logs, artifacts or commits;
  - the pool follows P7's rules;
  - the retraining job reuses the dedicated `KAGGLE_API_TOKEN` secret of the release
    workflow.
- **Approvals, asked in the session that uses them:** the first deploy of the instrumented
  page, every Kaggle dataset action, and turning on the scheduled workflows.
- **Timing:** built alongside P7, aiming for 2026-11-20 as well.

## 6. Left to the builder

- Faro SDK or hand-written events, within §4.1's rules.
- The dashboard's layout and queries, and the alert thresholds.
- The unusual-text checks and their thresholds (§4.6).
- How the "models over time" view gets the W&B and MLflow history.
- The robot's passage, and its time limit for an answer.

## 7. Risks

| Risk | Answer |
|---|---|
| Hardly anyone answers "It was actually…" | The live numbers show counts and intervals, never a verdict; promotion still rests on the offline gate |
| Fake events from the public collector URL | Origin restriction, the page commit, and the robot's own line as a reference |
| Grafana's free tier changes (as Hugging Face's did) | The page doesn't depend on it; the events format is ours, so another collector can take over |
| The weekly job is too slow on a free runner | The pool is capped as in P7, and the job can run by hand |
| A privacy slip, for example an error message containing text | The anti-goal is tested: the payload builder runs under Node on a passage, and no word of the passage may appear in any event |
