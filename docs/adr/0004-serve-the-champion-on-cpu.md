# ADR 0004: Serve the champion on CPU, from a model registry

- **Status:** accepted (2026-10-26, P4)
- **Context:**
  - The champion averages three models: `lr` (TF-IDF and logistic regression), the LM
    head (a logistic regression on Qwen 3 4B base, layers 6 to 9), and 5 fine-tuned
    ModernBERT-base fold models. Its features were computed on Kaggle's T4 GPUs.
  - The app should answer one pasted passage "not too long" after it's sent (the user's
    words), on a CPU: a free host, or the user's own PC. No competition passage may ship
    with it.
  - The app must run exactly the model that scored public LB 0.410, not an approximation.
- **Decision:**
  - **Load only the Qwen blocks the LM head reads** (`lm.load_lm_layers`). Layers 6 to 9
    are the outputs of blocks 6 to 9, so the app keeps the first **10** blocks, not 9. The
    last hidden state Transformers returns is passed through the final normalization, so
    a 9-block model would hand back a normalized layer 9 (the "9-block trap"). Rule:
    keep the last layer read, plus one.
  - **CPU, in float32**, with the Hub revision pinned (`906bfd4b…`). Before registering,
    the script checks that these features match the Kaggle GPU ones the LM head learned
    from: a median relative difference of 0.0010 (max 0.0052), and the same LM-head top
    label on 20 of 20 passages.
  - **A model registry.** The champion is an MLflow `pyfunc` model, registered as
    `name-that-llm` version 1 in a private DagsHub registry, with the alias `@champion`.
    The app loads it by alias, which resolves to a version once, at start-up.
  - **Parity before registering:** the champion rebuilt from the saved files, exactly as
    the app runs it, against the offline computation, on 50 passages: the same top label
    on all 50, and probabilities within 0.0001.
  - **"How sure?" comes from data:** the top probability's terciles on out-of-fold
    predictions, each with its measured accuracy.
  - **One prediction at a time:** 1 running and 4 waiting, then `503`; 10 requests a
    minute per visitor; torch on 2 threads. That's about 3.7 s per passage on this PC,
    after about 76 s to load the model from DagsHub.
  - **A Docker image:** `python:3.13-slim`, CPU torch 2.11.0, transformers 5.16.1. CI
    builds it on every pull request, and fails if a file planted in `data/` reaches it.
- **Consequences:**
  - The full champion runs anywhere Docker runs, given a few GB of memory, and the app
    names exactly which registered version answers (`/version`).
  - Changing the model is a registry operation (a new version, then the alias moved), not
    a code change.
  - Hosting: since 2026, Hugging Face needs a PRO subscription for Docker Spaces, so the
    full app isn't hosted publicly; ADR 0005 covers the free public demo instead.
  - Given up, to add when needed: batching several passages, GPU serving, and int8
    quantization (less memory, faster on CPU, but the parity check would have to be
    redone).
