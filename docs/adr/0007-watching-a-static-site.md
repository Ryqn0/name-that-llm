# ADR 0007: Watching a static site: events from the page, shadow mode, proposals by PR

- **Status:** accepted (2026-10-27 → 28, P6; designed with the user)
- **Context:**
  - The user wanted two things from P6, in their words:
    - to see "when you see some issues on the site or notice that there might be some lags,
      slows or any connection error";
    - to "check occasionally on if some models will overtake others".
  - The public demo is a static page (ADR 0005): `lr` runs in the visitor's browser, and no
    server of ours sees a request. There is nothing to instrument but the page itself.
  - The constraints, from the project and the spec:
    - free services only;
    - the pasted text never leaves the browser, in any form;
    - no cookies, no device storage, no identifier that follows a visitor;
    - monitoring must never break or slow the page;
    - the competition's passages never appear in public logs (the repo, and so its Actions
      logs, are public);
    - every change to the site is a person's merge.
  - A 50/50 A/B test needs traffic the demo doesn't have.
- **Decision:**
  - **The page reports anonymous events.** `app/demo/telemetry.js` sends `page_load`,
    `answer`, `feedback` and `error` events to Grafana Cloud's free Frontend Observability
    collector.
    - Each event keeps only the fields its allowlist names, each type-checked. Anything else
      is dropped.
    - Free text (an error's kind and message) becomes `[redacted]` if it holds any run of 4
      letters that also occurs in the visitor's text.
    - Word counts go out in steps of 25.
    - The collector requires a session id header and an SDK name. The page sends a fresh
      random id with every request, which links nothing, and names this module as the SDK.
    - Sending is fire-and-forget, and never awaited.
    - The collector's address comes from a GitHub variable through `config.json`. If it's
      empty, the page sends nothing, shows no notice and asks nothing.
  - **"Do you know who wrote it?"** Visitors' optional answers make the site a live test
    set. `feedback.js` allows one answer per result, tied to that result's labels.
  - **Shadow mode instead of a split.** A challenger pinned by `app/demo/CHALLENGER_RUN`
    loads after the first answer, and scores the same texts silently. Visitors always see
    the champion. The comparison is paired: on the same texts, how often only one of the
    two was right.
    - The challenger scores only once the champion's answer is on screen, in its own
      `try`, so it can't delay or break that answer.
    - Its 35 MB download is skipped when the browser asks to save data, or on a 2g
      connection.
  - **Unusual texts get a note.** The limits are measured on held-out passages: under 36
    words, over 22.9% unknown words, or over 20% non-Latin letters.
  - **Monitoring as code.** `monitoring/dashboard.json` and `monitoring/alerts.json` are
    pushed by `scripts/push_monitoring.py`.
    - The dashboard shows real visits, times and errors, the robot's line, live accuracy,
      drift, and every model's scores over time.
    - Live accuracy counts only answers naming one of the 18 models, per model run, with
      the number of answers beside it. Champion and challenger are compared on the same
      texts, as counts.
    - Every query filters the robot's events in or out, never both.
    - Two alerts go by email: more than 5 real errors in an hour, and no robot answer for
      12 hours.
  - **A robot visitor.** A GitHub Actions job opens the live page every 6 hours in a
    headless browser, with `?synthetic=1`. Its failure emails the repo's owner.
  - **A weekly export.** Grafana's free tier keeps logs about two weeks, so the week's
    events go to the private DagsHub MLflow, with live accuracy and Wilson intervals.
    Weeks are fixed, Monday 00:00 UTC to the next Monday, so a late or repeated run
    neither skips events nor counts them twice. `--end` redoes a missed week.
  - **Retraining that only proposes.** A weekly job compares `lr` plus the pool with `lr`
    on the same folds. It opens a pull request that sets `CHALLENGER_RUN` only if both
    hold:
    - the prompt bootstrap's interval is above 0;
    - the candidate's CV beats both pinned models' logged CVs (`MODEL_RUN`'s and
      `CHALLENGER_RUN`'s) by 0.005.

    A person merges it. Promotion to champion is another pull request.
- **Consequences:**
  - The monitoring is free, and can't hurt the page. A blocked, failing or slow collector
    changes nothing for the visitor: this was checked in headless Edge with a collector
    answering 500 and one aborting.
  - Privacy holds by construction, and the model card lists what's sent.
  - The collector's address is public by design, so anyone could send fake events.
    Browsers are held to the allowed origin, scripts aren't. Each event carries the page's
    commit (`app_version`), so a filter by commit is the defense if fakes ever appear. It
    isn't built yet.
  - Visitors' answers will be rare. The live numbers never decide alone: promotion still
    rests on the offline gate.
    - The intervals live in the weekly export, on DagsHub.
    - The dashboard shows counts, with the number of answers beside each share.
  - **The retraining's bar moves with each promotion.** A candidate must beat both pinned
    runs by 0.005, so after a promotion the new champion's CV is the bar. An equally good
    candidate is never proposed again.
  - **GitHub switches off scheduled workflows in a public repo after 60 days without
    activity.** After the competition, the robot will stop, and its silence alert will
    email every repeat interval. When the project goes quiet, pause that alert in Grafana,
    or re-enable the workflows from the Actions tab.
  - The setup depends on details of Grafana's pipeline, and three of them differed from
    the docs we read:
    - the session id header was required;
    - so was the SDK name;
    - the app's label is `service_name`, not `app_name`.

    All three were checked on the real stack, and tests pin them. Another free tier change,
    as Hugging Face's was, would cost a collector swap: the event format is ours.
  - The retraining job needs a fine-grained token (`RETRAIN_PR_TOKEN`), so that its pull
    request runs CI. Only the pull-request step sees it; the checkout keeps no
    credentials. The job handles the competition data in the runner's temporary space only,
    and prints counts and scores only.
