# ADR 0005: The public demo runs in the visitor's browser

- **Status:** accepted (2026-10-26 → 27, P4; the user's choice)
- **Context:**
  - The first deploy of the Docker app (ADR 0004) got `402 Payment Required` from Hugging
    Face: "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free
    cpu-basic requires a PRO subscription."
  - The options were Hugging Face PRO (about $9 a month), the app on a free cloud VM, the
    app only locally for now, or a lighter demo that a free static Space can host. The
    user chose the lighter free demo.
  - A static Space serves files and runs nothing. So the model has to run in the browser.
- **Decision:**
  - **The demo runs `lr` alone**: TF-IDF on words (1- and 2-grams, lowercased) and on
    characters (1- to 5-grams), then a logistic regression. It's the champion's own `lr`,
    taken from registered version 1, with its **own** "How sure?" bands. Alone it's less
    sure of itself, so its thresholds sit lower (0.152 and 0.117, against 0.268 and 0.197).
  - **A JavaScript port, checked against Python.** `app/demo/ntl.js` re-implements
    `flatten_text`, scikit-learn's TF-IDF for exactly our settings, and the softmax. The
    export (`name_that_llm.browser`) refuses any setting the port doesn't implement. Tests
    run the port under Node on every pull request. At export, 50 real passages were
    checked: the same top label on all 50, with the largest difference 4.4e-9.
  - **The model is pinned.** Its files (`model.json` 1.8 MB, `weights.bin` 13.5 MB of
    float32) are an MLflow run on DagsHub (experiment `browser-demo`), and
    `app/demo/MODEL_RUN` holds the run's id. The deploy downloads exactly that run, so
    the public model changes only through a pull request.
  - **Continuous deployment** on every merge that touches `app/demo/`, the deploy script
    or its workflow:
    1. stage six files;
    2. create or update the Space;
    3. wait until the live `version.json` names this commit (Hugging Face says where a
       static Space is served: `*.static.hf.space`);
    4. smoke-test. The engine, the model and `version.json` must be served byte for byte.
       The page must be served whole, with only Hugging Face's added `<head>` tags. The
       model card is skipped, since the site serves it rendered as HTML. Then the engine
       runs on those files under Node.
- **Consequences:**
  - Free, with no server to run or secure. **The text never leaves the visitor's
    browser**, and an answer takes about 0.01 s after a one-time 15 MB download.
  - A weaker model in public: `lr` scores LB 0.320 (about 3 right in 10) against the
    champion's 0.410 (about 4 in 10). The page says so, and points to the repo for the
    full model.
  - The model's vocabulary (words, word pairs and short character runs) is public, as
    model parameters. No passage is.
  - Two implementations of the same model must stay in step. The parity tests guard that,
    and the export refuses what the port can't do.
  - Hugging Face changes some files it serves (the README rendered, tags added to the
    page), which the smoke test allows for, narrowly.
  - The full champion stays runnable locally (ADR 0004). Hosting it publicly is a later
    decision (I-026).
