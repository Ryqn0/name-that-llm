# Progress diary

Newest entry first. Each entry follows the same format: **Did / Learned / Scores / Next**.

## 2026-10-28 · Session 9: the site watched, the models compared live, and the first LLM run

**Did**
- **P6, start to finish:**
  - the demo page now sends anonymous events to your Grafana Cloud stack: loads, answers,
    errors, and the optional "Do you know who wrote it?" answer;
  - a dashboard and two email alerts, as code;
  - a robot visitor every 6 hours;
  - a weekly export of the events to DagsHub;
  - a weekly retraining. Its first run proposed `lr` + the pool (CV 0.4186) as the demo's
    challenger, and with your OK it now scores visitors' texts in shadow mode.
- **You set up Grafana Cloud** (stack, collector, tokens, email, Infinity), guided step by
  step. Then you visited the demo, and your visit showed on the dashboard.
- **P6's final review** by a fresh reviewer: nothing critical, 6 important findings. All
  were fixed in #60, each with a test that failed first:
  - the retraining won't re-propose an equally good model every week;
  - the challenger can't slow or break the visitor's answer;
  - the live panels compare the two models on the same texts.
- **P7 Task 5:** the first Qwen3-8B run on Kaggle's two T4s. The first try ran out of
  memory, which was fixed, and the second ran for 4 hours.
- **P7 Task 6 started:** notebook 11 explores depth, length and the pool for the LLM.
- **Late in the session:**
  - you asked for the better model on your own PC: the full champion ran locally (ready
    after 410 s the first time, then 3.9 s a passage);
  - notebook 10's run was logged to W&B from the PC;
  - **I-029 started:** the pool inside the ensemble's `lr`, a CPU check that runs
    overnight.

**Learned**
- **Test against the real service before building on its docs.** One synthetic test event
  showed that Grafana's collector requires a session header and an SDK name, and that Loki
  names the app `service_name`, all before the dashboard was pushed.
- **When something "shows nothing", check each link of the chain.** Your visit was in Loki
  all along. The dashboard's data source picker had defaulted to the wrong Loki, of three.
- **Kaggle doesn't give secrets to runs pushed through its API**, which is why W&B never
  worked in our GPU notebooks. Their results will be logged from the PC instead.
- **A freed model can stay on the GPU** inside a long-lived process. Running every GPU step
  in a process of its own returns its memory for sure.
- **A first scheduled run is a test of its own.** Both weekly jobs failed their first GitHub
  run, for reasons a local run couldn't show.
- **Look at the rendered panel.** Thirteen tests passed and Loki accepted every query, yet
  two panels showed unnamed numbers. Rendering them as images showed it at once.

**Scores** (seed-7 out-of-fold, balanced accuracy)

| model | CV |
|---|---|
| **Qwen3-8B + LoRA, alone** (pool-free) | 0.3641 |
| ModernBERT encoder | 0.3590 |
| current ensemble (3 members) | 0.4293 |
| **ensemble + the LLM** (in-sample weights) | **0.4523** |

**Next**
- Notebook 11's results: the LLM's depth, length, and above all whether the pool helps it.
- P7 Task 7: the final LLM run, the four-member ensemble, and the A/B gate on fresh seeds.

## 2026-10-27 · Session 8: P7 begins with generated data, and P6 is planned

**Did**
- **Planned two phases**, both approved:
  - P7: an LLM classifier and generated training data, the score lever;
  - P6: watching the public site and the models over time.
- **Studied the leaders' public 0.651 notebooks** (your 10 links). Their recipe is LoRA'd
  8–14B LLMs, about 250k generated passages, and Sinkhorn balancing. We use none of their
  code or weights.
- **P7 Tasks 1–4:**
  - **Sinkhorn class balancing:** measured, and dropped (I-027).
  - **The pool:** 246,945 public, open-licensed generated passages, cleaned and
    deduplicated against ours. Each is matched to our prompts by topic, so no fold ever
    trains on its own topics.
  - **The pool check** on our folds (below).
  - **`llm.py`:** Qwen 3 with LoRA on 4-bit weights, two folds at once on Kaggle's two
    GPUs, tested on CPU with a tiny random Qwen.
- **Pushed three placeholder notebooks to Kaggle**, so the W&B secret can be attached to
  them.

**Learned**
- **More data is the biggest lever yet.** 30,000 pool passages lift `lr` alone from 0.321
  to 0.421 on CV, as high as the whole three-model ensemble.
- **A per-source check on an unweighted model mixes quality up with balance.** 51 of 56
  sources "hurt" when added alone, but most cover only a few of the 18 models, which pulls
  an unweighted `lr` toward them. We kept the 5 that helped.
- **For data that's re-downloaded, an allowlist beats a blocklist:** a new, unchecked
  source stays out by default.

**Scores**

| model | CV (seed 42) | LB |
|---|---|---|
| `lr` + 30,000 pool passages | 0.4205 | — |
| champion `ensemble` | 0.4219 | 0.410 |
| `lr` alone | 0.3212 | 0.320 |

**Next**
- You: attach `WANDB_API_KEY` to the three new notebooks on Kaggle.
- P7 Task 5, the first Qwen3-8B GPU run; P6 Task 1 during its wait.

## 2026-10-26 → 27 · Session 7: the app, the registry, and a free demo in the browser (P4)

**Did**
- **Built the app around the real champion:**
  - `flatten_text` turns pasted chatbot markdown into a plain paragraph;
  - Qwen is cut to the 10 blocks the LM head reads;
  - the `Champion` predictor gives data-driven "How sure?" bands;
  - a FastAPI app serves the page (layout B), and its Docker image is built in CI.
- **Registered the champion** in a private MLflow registry on DagsHub (`name-that-llm`
  v1 `@champion`), after every check passed. Qwen's CPU features match the Kaggle GPU
  ones (median difference 0.0010, the same top label 20 of 20), and parity was 50 of 50.
- **The first deploy hit Hugging Face's new paywall** for Docker Spaces (402). You chose
  the lighter free demo: a static Space whose page runs `lr` in the visitor's browser.
- **Built that demo:**
  - ported `lr` to JavaScript (`ntl.js`), with parity tests under Node;
  - exported it as a pinned MLflow run;
  - built the static page and its continuous deployment, which ends in a smoke test of
    the live site.
- **Looked at the page** in headless Edge (Playwright), which found two layout bugs, both
  fixed. Then published it: <https://huggingface.co/spaces/Ryqn/name-that-llm>.

**Learned**
- **The 9-block trap:** the last hidden state Transformers returns is normalized, so
  reading layer 9 means keeping 10 blocks.
- **CPU float32 features match Kaggle's GPU float16 ones** within 0.1% (median), with the
  same LM-head top label. So the app runs the leaderboard model, not an approximation.
- **Hosting terms change.** Free Docker Spaces now need PRO; static Spaces are still free.
  A model small enough for the browser costs nothing to serve, keeps every text on the
  visitor's device, and answers in 0.01 s.
- **`lr` alone is far less sure of itself than the champion**, so it needs its own bands.
  Its "Fairly sure" starts at a top probability of 0.152 (right 47% of the time), against
  the champion's 0.268.
- **A static host rewrites files.** Hugging Face serves `README.md` rendered as HTML and
  adds tags to the page's `<head>`. The smoke test now allows exactly that, and nothing
  more.
- **Look at the page.** Every HTTP check passed while the bars were misaligned, and nearly
  invisible on phones. Two screenshots found both bugs.

**Scores:** unchanged. The champion `ensemble` stays at public LB 0.410. The public demo
runs `lr` (LB 0.320, CV 0.325).

**Next:** choose the next phase. P7 (LLM and ensemble) is the score lever. P6 (the MLOps
loop) needs a rethink first, since the public demo has no server to monitor.

## 2026-10-25 → 26 · Session 6: a fine-tuned encoder joins the ensemble (P5)

**Did**
- Wrote `encoder.py`: a hand-written PyTorch fine-tuning loop (ADR 0003) with mixed
  precision, a warmup then linear decay, gradient clipping, and a hard stop on a
  non-finite loss. CV trains a fresh pretrained model per fold, and the offline helpers
  find and average the 5 saved fold models.
- Added an Optuna study for encoders (warm start, fold-level pruning), an exact search for
  N blend weights, `EncoderConfig` and `EnsembleConfig`, and the ensemble submission. A new
  CI job installs CPU-only torch, so the 21 model tests run on every PR.
- Ran three GPU notebooks on Kaggle: the head-to-head
  ([07](https://www.kaggle.com/code/ryqn00/name-that-llm-encoder-select)), the tuning
  ([08](https://www.kaggle.com/code/ryqn00/name-that-llm-encoder-tune)), and the final
  training on four fold seeds with the A/B gate
  ([09](https://www.kaggle.com/code/ryqn00/name-that-llm-encoder-final)).
- Released v0.7.0 and submitted the three-model ensemble.

**Learned**
- **ModernBERT-base beat DeBERTa-v3-base clearly:** seed-7 CV 0.344 against 0.206. Even
  at ModernBERT's settings DeBERTa scored only 0.171, and one of its folds never left the
  uniform guess.
- **A weaker model can still help.** Alone, the tuned encoder scores 0.353 on fresh folds,
  between `lr` (0.325) and `lm_blend` (0.402). Yet a 15% share lifts the blend by +0.021,
  because its mistakes differ: it reads the raw text end to end, while the other two are
  linear models on fixed features.
- **The leaderboard confirms it: public LB 0.410**, +0.021 over 0.389, the same gain as on
  fresh folds. The CV–LB gap held at −0.013.
- **Fine-tuning is noisy.** The same config, folds and seed scored 0.374 in the tuning run
  and 0.360 in the final one: GPU training isn't bit-for-bit reproducible. So half of the
  tuning's +0.030 was luck (new idea I-022).
- **Pruning paid off, but the sampler never learned.** 7 of 10 trials stopped after one
  fold. TPE draws its first 10 trials at random by default, so a 10-trial search is a
  random search (I-020).
- **W&B was off in every GPU run:** reading the Kaggle Secret raised `ConnectionError`, so
  P5 has no W&B runs. Tracking is optional by design, so nothing broke, but it needs a fix.

**Scores:** `ensemble`: CV 0.4231 (mean of seeds 42–44), **public LB 0.410**, the new
champion. Leaderboard on 2026-10-26: 3,387 teams, median 0.532, top-10% cutoff 0.641, best
0.710. We're about 2,236th. P5 used about 4.1 GPU-hours.

**Next:** P4, App v1 and CD.

## 2026-10-25 · Session 5: the LM blend reaches the leaderboard (P3b, tasks 5–6)

**Did**
- Released v0.6.0. The offline submission notebook now extracts Qwen 3 4B's layers 6–9 on
  Kaggle's T4 GPU, for train and test in the same run, and blends them with `lr`.
- Submitted it. The GPU run took 22 minutes, and its runtime check estimated the hidden
  rerun at 18.6 minutes (the limit is 9 hours).
- Ran 60 Optuna trials over the blend's five knobs (81 minutes on CPU), then re-measured
  the winner on fresh folds
  ([tuning notebook](https://www.kaggle.com/code/ryqn00/name-that-llm-tuning)).

**Learned**
- **The leaderboard confirms the LM features: public LB 0.389**, up from 0.320 (+0.069).
  CV had promised +0.084, so it still ranks the models correctly.
- **The CV–LB gap widened**, from −0.001 for `lr` to −0.013 for the blend (CV 0.402, the
  mean of three fresh seeds). One number can't tell leaderboard noise from a real shift,
  so keep watching the gap as the models get more complex.
- **Tuning lost to the hand-made sweep.** Optuna's best scored 0.4012 on the seed-7 folds,
  below `lm_blend`'s 0.4038 there, and the gap held on fresh folds (0.3978 vs 0.4023). The
  budgets were similar (67 CVs against 60 trials), but the sweep chose the blend weight on
  a nearly free grid, while Optuna spent whole trials on bad weights (new idea I-018).
- **Confirmed: the LM head is far more confident than `lr`.** Its average top probability
  is 0.666, against 0.147 for `lr`, which is why a weight of 0.1 is enough (I-015).
- **Selection bias was small:** `lm_blend` lost only 0.0015 between the folds that chose
  it and fresh ones.
- **The permission system blocked an unattended auto-submit.** A submission is visible
  and uses a daily quota, so it's sent only in the foreground, right after an explicit OK.

**Scores:** `lm_blend`: CV 0.4023 (mean of seeds 42–44), **public LB 0.389**, the new
champion. Leaderboard on 2026-10-25: 3,231 teams, median 0.526, top-10% cutoff 0.639, best
0.703. We're about 2,219th, up from about 2,687th.

**Next:** P3b is done. The median (0.526) is still far ahead, so the next plan must find
much bigger gains.

## 2026-10-24 → 25 · Session 4: LLM hidden states beat TF-IDF (P3b, tasks 1–4)

**Did**
- `lm.py` reads open Qwen 3 base models and averages each layer's hidden states over the
  tokens of a passage. torch and transformers are imported only inside its functions, so
  the text-only path never needs them (ADR 0002).
- Added dense-feature CV (`run_cv_features`), probability blending (`blend.py`) and
  `BlendConfig`.
- A GPU notebook extracted every layer of Qwen 3 0.6B, 1.7B and 4B base for the 2,699
  training passages in 13 minutes on one T4.
- A chained CPU notebook swept the layers, refined the best one, chose the blend weight and
  ran the A/B test on fresh folds (73 minutes,
  [lm-layer-sweep](https://www.kaggle.com/code/ryqn00/name-that-llm-lm-layer-sweep)).

**Learned**
- **Every model size beats TF-IDF on its own.** The best single layer of each size, on
  seed-7 folds: 0.6B 0.345, 1.7B 0.377 and 4B 0.380, against `lr` at 0.330. Bigger helps,
  with diminishing returns from 1.7B to 4B. The forum's "≈ 0.39 for early layers" holds:
  4B layers 6–9 reach 0.386.
- **Early layers win.** The best layers sit at 11–22% of the depth, and the last layers
  are the worst (0.28–0.32). They specialize in predicting the next token, while early
  layers still describe the surface of the text, where style lives.
- **Even layer 0 matches TF-IDF.** It's the plain average of the input token embeddings,
  with no context at all, and it scores 0.33–0.36. A guess: an average weights tokens by
  frequency, and frequent tokens (function words, punctuation) carry style, while TF-IDF
  boosts rare, topical words. New idea I-017.
- **A small blend weight can hide a big effect.** The best weight on the LM side is only
  0.1, yet the blend beats both of its parts. A weighted mean mixes probabilities, so a
  very confident model moves the result even at 10%. That's a hypothesis, to check during
  tuning (I-015).
- **The A/B on folds never used for any choice (seed 42):** blend minus `lr` = **+0.084**,
  95% CI [+0.066, +0.101]. Promoted to the leaderboard A/B test.
- **Chained notebooks** keep GPU time for the step that needs a GPU: 13 GPU-minutes for the
  extraction, then 73 minutes of free CPU for the CVs.
- Kaggle names a notebook's URL after its *title* on the first push, not after the
  metadata `id`.

**Scores:** `lm_blend`: CV ≈ 0.405 (seed 42, = `lr` 0.3212 + 0.0841). LB pending.

**Next:** Task 5, the offline GPU submission with the blend, kept only if the public LB
beats 0.320. Then Optuna tuning (Task 6).

## 2026-10-24 · Session 3c: why the assignment trick failed

**Did**
- Read forum thread 750213, *"Test-time prompt grouping: simulation vs. LB"*. Its claim:
  the trick's gain depends on how many answers each hidden prompt has.
- **Verified it with our own code:** I subsampled each training prompt to m random answers
  and re-measured the assignment gain on lr's out-of-fold probabilities.

**Learned**
- **Gain by layout:** +0.084 at m = 18 (train's layout), +0.003 at m = 6, and about
  0 to −0.01 at m = 1–3. Even with the TRUE groups, m = 1–3 gives only +0.000 to +0.004.
- **So the hidden test most likely has few answers per prompt**, and our zero LB gain is
  exactly what that layout predicts. No bug, just a wrong assumption about production data.
- **The MLOps lesson:** a validation set must mirror production's *structure*, not just
  its distribution. Our backtest copied train's layout, which the test doesn't share.
- **Topic subtraction (I-007) is doubtful too:** with small groups, the group mean contains
  each passage's own style, so subtracting it removes signal.

**Scores:** unchanged (public LB 0.320).

**Next:** P3b, a better base model: tuning, style features, lab-then-model.

## 2026-10-24 · Session 3b: the leaderboard A/B test says "no"

**Did**
- Released v0.4.0 (tag → CI → Kaggle dataset) and submitted the same `lr` model with
  `ASSIGN = True`. The Kaggle log confirms the assignment really ran.

**Learned**
- **Public LB: 0.320 with assignment, 0.320 without.** CV had promised +0.085. An exact
  tie to three decimals means the post-processing had essentially no effect on the
  hidden test.
- **This is why we A/B test on the real target.** The puzzle backtest only proves the trick
  works on data with *train's* structure. Two explanations remain: the hidden test may not
  have one answer per model per prompt, or our topic grouping may fail on it.
- Following the rule we set *before* seeing the result: keep `ASSIGN` only if the LB
  improves. It didn't, so the default goes back to `False`, the simpler path.

**Scores:** `lr` + assignment: CV 0.4056, public LB **0.320**.

**Next:** diagnose I-001, then P3b (a better base model).

## 2026-10-24 · Session 3a: the puzzle backtest (P3a, tasks 1–3)

**Did**
- `topic_groups`: rebuilds the hidden prompts by clustering content-word TF-IDF (cosine
  distance, average linkage). On train, the groups match the true prompts with an adjusted
  Rand index of 0.909.
- `assign_labels`: one exact assignment per group (`linear_sum_assignment` on −log p), so
  each model is used at most once per prompt.
- `--assign` in the CLI adds the puzzle backtest to CV, logged to W&B (run `to8unxny`).

**Learned**
- **Using the data's structure beats a better model, for now:** plain 0.321 → **0.406**
  with the rebuilt groups, and 0.413 with the true groups. That's +0.085 from
  post-processing alone.
- **The fake test data was the bug, not the algorithm:** passages that share a rare word
  across topics (e.g. `style10`) look more alike than passages sharing the topic, because
  rare words carry a high IDF weight.
- **Process:** a merge must wait for CI. The merge now runs only via
  `gh pr checks --watch && gh pr merge`.

**Scores:** CV `lr` + assignment **0.4056** (not yet on the leaderboard).

**Next:** Task 4, a submission with assignment as a leaderboard A/B test against 0.320.

## 2026-10-24 · Session 2f: the first leaderboard submission

**Did**
- Created a **dedicated** Kaggle token for GitHub Actions (secret `KAGGLE_API_TOKEN`), so
  the CI credential can be revoked without touching my own.
- Created the public dataset
  [ryqn00/name-that-llm-lib](https://www.kaggle.com/datasets/ryqn00/name-that-llm-lib).
  Tagging `v0.3.0` made GitHub Actions publish a new version of it automatically: CD works.
- The offline notebook [ryqn00/name-that-llm-submit](https://www.kaggle.com/code/ryqn00/name-that-llm-submit)
  installed everything with internet off, predicted with `lr`, and validated before
  writing. Then the first submission (ref 57601064).

**Learned**
- **CV 0.321 vs public LB 0.320.** The prompt-grouped backtest predicts the leaderboard
  almost exactly, so we can trust local CV and spend fewer submissions.
- **Kaggle mounted our dataset at `/kaggle/input/datasets/ryqn00/...`**, not the path the
  plan assumed. Searching for the wheel instead of hard-coding the path saved the run.
- **Kaggle's license list has no MIT.** It rejected `"MIT"` with a vague server error. The
  plan's fallback was `other`, with MIT stated in the description.
- **The leaderboard is far ahead:** median 0.512, top 10% at 0.632, best 0.703, among 3,012
  teams. We're about 2,687th. A TF-IDF baseline is nowhere near enough. The big ideas
  (prompt assignment, topic subtraction, transformer and LLM features) are the next plan.

**Scores:** `lr`: CV **0.3212**, public LB **0.320**.

**Next:** review this whole first plan, then write the P3 plan around the biggest gains.

## 2026-10-24 · Session 2e: five baselines and the first offline A/B test

**Did**
- Ran all five baselines on identical prompt-grouped folds: in the Kaggle notebook
  [ryqn00/name-that-llm-baselines](https://www.kaggle.com/code/ryqn00/name-that-llm-baselines),
  and logged to W&B from my PC (runs r0lx5cja, sxhd09jf, 7f8bv2zr, 5p2yum3r, xzuywz5t).
- **First offline A/B test, with my bootstrap:** lr minus nbsvm = +0.024, 95% CI
  [+0.008, +0.040], so **lr is the baseline champion**.
- Built the submission builder (Task 10) and prepared the release pipeline (Task 11).

**Learned**
- The ranking: lr 0.321 > nbsvm 0.297 > mnb 0.285 ≈ svm 0.285 > cnb 0.247. The fold spread
  is ±0.005–0.010, so gaps smaller than about 0.01 are noise. That's why the decision used
  the bootstrap and not the raw ranking.
- **Same code, same folds, same numbers:** the local run, the W&B runs and the notebook all
  match exactly. That's reproducibility in practice.
- The classifier's top weights confirm the EDA: punctuation for Claude, GPT, GLM and Kimi;
  vocabulary for Gemini and Qwen.
- A public logistic regression reaches 0.356, so our feature settings have headroom
  (idea I-012).

**Scores:** champion lr, CV **0.3212** (not yet submitted).

**Next:** Task 11, which publishes the package as a Kaggle dataset, runs the offline
submission notebook, and gives us the first leaderboard score.

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
