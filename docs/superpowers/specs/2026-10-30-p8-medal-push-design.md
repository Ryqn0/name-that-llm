# P8: our own strong models on the clean pool, for a medal: design

**Date:** 2026-10-30 · **Status:** draft, for the user's approval · **Phase:** P8, after
P7 ([design](2026-10-27-p7-llm-ensemble-design.md)), whose last tasks it starts by
finishing.

## 1. Intent

**Reach a medal: the top 10% of the leaderboard by the final deadline, 2027-01-13.** The
user chose this aim on 2026-10-30. The line was 0.658 that day, and it will likely rise as
more people fork the public 0.66 notebooks (my guess).

**The way, in the user's words: "ensembling and also training models".** We train strong
models ourselves and blend them. The project is also a lesson in that craft: training a
large encoder and an LLM on about 200,000 passages, on free GPUs, and combining them
honestly.

**What failure looks like, in the user's words: "if the score doesn't improve".** Every
step must move the score, or it stops.

## 2. What we know (recon, 2026-10-30)

### 2.1 The leaderboard

| | Public LB |
|---|---|
| best | 0.732 |
| top-10% cutoff (the medal line) | 0.658 |
| the best public notebooks (many forks) | 0.660–0.663 |
| median of 3,954 teams | 0.549 |
| **ours**: `ensemble_pool` (about 2,462nd) | **0.454** |

The medal line sits just below the public notebooks: forking one reaches it. A medal means
beating what's public.

### 2.2 What the top public notebooks are made of

17 original notebooks were read (the top by score and by votes; metadata only, none run):

- **Their passage data is mostly the clean pool.** Of the 5 passage datasets they attach, 4
  are in the clean pool. The fifth, `prvsiyan/prvsiyans-extended-name-that-llm-dataset`,
  was removed by the pool's maintainer in the clean release. It states CC BY 4.0, but the
  reason for its removal isn't published.
- **Their edge is blending many trained models, mostly trained by other participants.**
  - LoRA adapters on 7–14B LLMs: Qwen3 8B and 14B, Qwen2.5 7B and 14B, GLM-4-9B and
    Llama-3.1-8B;
  - encoders: DeBERTa-v3-large, ELECTRA-large, and a custom 380M one.

  All are published as datasets or notebook outputs. Few of these notebooks train anything
  themselves.
- **Recurring ideas in their titles:**
  - "two-stage LoRA";
  - "remove the prompt", "the prompt is not the author" and "adaptive centering": taking
    the topic out of the signal so the style remains;
  - "Hungarian": the one-model-per-prompt assignment we shelved in P3a.

### 2.3 The clean pool

`crtproduction/name-that-llm-open-pool`, version 10 (2026-10-30), CC BY-SA 4.0:

- **198,507 passages from 54 sources.** Every passage passed the maintainer's quality, genre
  and licence checks. Version 9 removed 70,558 rows: junk, off-genre sources, licence and
  provenance risks, and answers to the competition's training prompts.
- **The maintainer's results:**
  - DeBERTa-v3-large trained on the clean v9 (193k passages) scored **0.587** public LB,
    against 0.585 on the raw v7 (254k);
  - at equal size, clean data beat raw by +0.027 balanced accuracy, with **every label
    weighted equally**.
- **It changes upstream without notice.** Between two of our runs, it lost 2 of our 5
  allowlisted sources (P7's ledger, 2026-10-30).

### 2.4 Our own evidence

All balanced accuracy on the seed-42 folds:

- **`lr`:**
  - with the clean pool's 3 allowlisted sources (28,048 passages): **0.4338**;
  - with the same number drawn at random from all 54 sources: **0.3623**, probably because
    the draw skews the labels (the biggest source has 25,000 passages for 4 models).
- **The pooled Qwen3-8B LoRA** (5,000 pool passages per fold, 2 epochs): about 0.468 on folds
  0–3. One epoch loses 0.045.
- **The four-member blend** (`lr` + pool, LM head, encoder, LLM; weights chosen by nested CV
  on folds 0–3): **0.516**, against 0.4705 for I-029's three members. The encoder's weight
  falls to 0 once the LLM is in.
- **The pool's gain held on the leaderboard:** +0.049 in CV became +0.044 on hidden prompts
  (0.410 → 0.454). The CV–LB gap is about −0.016.
- **Speed on Kaggle's T4:** Qwen3-8B LoRA takes about 1 second per passage per epoch (122
  minutes per epoch on about 7,160 passages). DeBERTa-v3-large hasn't been timed.

### 2.5 The rules on generated data

**Still no host answer** (forum, 2026-10-30): the "Section 2.6" thread has one reply, from a
participant. Participants treat published generated data as allowed. P7's three safeguards
stand (its design, §3):
1. openly licensed public datasets only;
2. the pool only trains, and every score is measured on our own passages;
3. one of the two final picks stays pool-free.

## 3. Decisions

**The user's (2026-10-30):**
- **Aim for a medal.**
- **Train our own models,** and blend them. No other participants' published weights or
  adapters, at first.
  - **The checkpoint:** if our best public LB is still below the medal line on 2026-12-15,
    Claude brings back the question of adding published models. The date is Claude's
    call.
- **Claude makes the technical calls and explains them.**
- **The score must improve:** a step that doesn't move the gate stops.
- **Public code, private weights, until 2027-01-13.**
  - The code, the configs and the diary stay public, as the project always was.
  - The trained weights, and the Kaggle notebooks that use them, are private until the
    deadline, then published. That includes the submit notebook.
  - The recipe isn't secret (the 0.66 notebooks are public); the GPU hours are the real
    barrier, and the weights are what they buy.
- **If the pool drops a source later:**
  - new runs use only the sources still in the latest release;
  - a member already trained on a dropped source is retrained only if that source is
    shown to break its licence, not merely because it was dropped;
  - a weekly check compares each new release with our pinned copy, source by source.
- Unchanged from P7: free compute only, and no paying for data.
- **Unchanged from the project design (§6.9): CV decides,** and the public leaderboard alone
  never does. The hedge pick (P7's safeguard 3) is the pool-free `ensemble`, LB 0.410
  (P7's ledger).

**Claude's calls:**
- **Finish P7 first:** fold 4, the four-member gate, the four-member submission, and P7's
  record.
- **The first new member is DeBERTa-v3-large on the whole clean pool.**
  - It's the maintainer's proven recipe (0.587 alone), and fast enough for one session.
  - It's another architecture than the LLM, so its mistakes should differ.
- **The second is the LLM trained on far more of the pool,** after DeBERTa's verdict.
- **Every label weighted equally,** whatever the member: the fix for what sank `lr`'s random
  draw.
- **One honest run instead of five folds,** for every pool-trained member: §5.2.
- **Pin the pool:** a private copy of v10, and per-source counts logged in every run
  (I-034). Creating the copy needs the user's OK.
- The checkpoint date, 2026-12-15.

## 4. Goals, non-goals, anti-goals

**Goals**
1. **Finish P7:**
   - seed 42's fold 4;
   - the four-member gate on all 5 folds;
   - the four-member submission (P7 Task 8, now lib v0.10.0);
   - P7's journal.
2. **A DeBERTa-v3-large member** trained on the whole clean pool, that passes the gate and
   lifts the public LB.
3. **An LLM member trained on far more of the pool,** that passes the gate.
4. **The leaders' cheap ideas, each measured:** first, taking the prompt's topic out of
   the signal (centering).
5. **An offline submission that runs every member in under 9 hours** on Kaggle's T4s.
6. **A top-10% finish** on the private leaderboard, by 2027-01-13.

**Non-goals**
- Other participants' trained weights or adapters, unless the checkpoint (§3) decides
  otherwise.
- Generating data ourselves; paid compute.
- One-model-per-prompt assignment (Hungarian), unless it passes P3a's layout check (m = 1,
  2, 3 and 6 answers per prompt).
- Changes to the public demo, and to P6's monitoring.

**Anti-goals** (failures even if everything runs)
- **A GPU week spent on a member that doesn't move the gate.** Each member starts with a
  speed probe and a budget, and stops on a failed gate.
- **A score lifted by leakage:** a pool passage about one of our prompts, judged on that
  prompt's passages.
- **A submission over 9 hours**, or one that fails offline.
- **Data of unclear licence or provenance.**
- **A pool that changes under a run** (it happened in P7): pin it, and log its counts.
- **Both final picks depending on the pool** (P7's safeguard 3).

## 5. The approach

### 5.1 Finish P7 (week 1)

1. **Session 3:** notebook 12 with `JOBS = [("seed42-f4", 42, (4,), {}, True)]`, about 4
   GPU-hours, after the weekly reset.
2. **The gate on all 5 folds:** `scripts/llm_ensemble_check.py`, with `lr` on the clean
   pool.
3. **The four-member submission:** Task 8, with the LLM's inference added to the offline
   notebook. The submission needs folds 0–3's adapters, from
   `data/kaggle-outputs/llm-final-s2`, together with fold 4's.
4. **P7's journal.**

### 5.2 The data, and one honest run per member

- **The pool:** v10, pinned, with every source.
- **No pool passage about one of our 150 training prompts ever trains.** That's every
  passage that `pool.match_prompts` matches to one (about 15% in notebook 11).
- **Our own 2,699 passages never train either.**

**The pool's releases are watched.** Each week, a check compares the latest release with
our pinned copy, source by source. New runs drop any source the latest release no longer
has (§3).

So every one of our passages is unseen by the model, about a prompt it never trained on.
That's the setting of the hidden test (new prompts), and it makes the model's predictions
on our passages honest **for all 2,699 rows at once.** One run replaces five folds, which
is what makes a 200,000-passage member affordable. The same model then predicts the test.

**Every label weighted equally**, by balanced sampling or a weighted loss (§7).

### 5.3 Member 1: DeBERTa-v3-large

1. **A speed probe:** a few hundred steps on a T4 to measure seconds per passage, and check
   the training's numeric stability in half precision.
2. **One run,** sized by the probe to fit a session on T4 × 2:
   - trained on §5.2's data;
   - checkpoints saved, so a run can resume in the next session;
   - its predictions saved for our 2,699 passages.
3. **The gate (§5.6).**
4. **If it passes,** the submission adds its inference: minutes for 4,000 passages.

### 5.4 Member 2: the LLM on far more of the pool

After DeBERTa's verdict, Qwen3-8B LoRA (or a 14B base, if the budget allows) trains with
§5.2's protocol on as much of the pool as the GPU budget buys. At about 1 second per passage
per epoch, 2 epochs on 30,000 passages take about 17 hours: two sessions. The amount is
chosen by budget and by the gain per hour.

### 5.5 The cheap ideas

- **Centering:** take each test prompt group's average signal out of every passage, so the
  topic no longer counts. The groups are found by topic (`grouping.topic_groups`), since the
  test's prompts aren't given.
- **P3a's layout check is mandatory:** the hidden test isn't laid out like train.
- These cost CPU time, or a little GPU, and are tried while the big runs train.

### 5.6 The gate and promotion

- **The new candidate is the current best blend plus the new member.** Its weights are
  chosen by nested cross-validation on the seed-42 folds: each fold is scored with weights
  chosen on the other folds. That's `scripts/llm_ensemble_check.py`'s method.
- **PROMOTE if the prompt bootstrap's 95% lower bound is above 0.** Then a submission, with
  the user's OK, confirms it on the leaderboard.
- The CV–LB gap is watched at each submission (−0.016 now).

### 5.7 The submission

- **It runs every member offline, within 9 hours on T4s.** Its runtime check projects the
  hidden run, on a 4,000-passage stand-in, as now.
- Every member's weights are attached as pinned, private Kaggle datasets or notebook
  outputs.
- **The submit notebook goes private** for P8's versions, until 2027-01-13 (§3).
- One final pick stays pool-free (P7's safeguard 3): `ensemble`, LB 0.410.
- **On 2027-01-13, after the final submission:** the weights, the training notebooks and the
  submit notebook are published, as the project design promised.

### 5.8 Order, budget and the checkpoint

About 30 GPU-hours a week (Kaggle's weekly quota), for about 10 weeks:

| When | What | GPU |
|---|---|---|
| Week 1 | P7's fold 4 and the four-member submission | about 5 h |
| Weeks 1–2 | DeBERTa-v3-large: the probe, then one run | about 8–12 h |
| Weeks 2–4 | the LLM on more of the pool | about 20–35 h |
| Throughout | centering and the other cheap ideas | CPU, a little GPU |
| **2026-12-15** | **the checkpoint:** below the line → the user decides about published models | — |
| Until 2027-01-13 | more members or more data, by measured gain per hour; the final picks | the rest |

## 6. Constraints

- **Kaggle's T4 × 2:** 16 GB each, no bfloat16, 12-hour sessions, about 30 GPU-hours a week.
  API-pushed runs get no secrets: W&B is logged from the PC.
- **The submission:** offline, at most 9 hours; 5 submissions a day; 2 final picks.
- **Data:** openly licensed, published Kaggle datasets only. Never commit competition data,
  and the tests use fake data.
- **Private until 2027-01-13:** the trained weights, and every Kaggle notebook that uses them,
  the submit notebook included. The code, the configs and the diary stay public (§3).
- **The user's explicit OK** for every Kaggle push, dataset creation, submission and release
  tag.
- The P7 conventions: the ledger, PRs gated by CI, the journal before a workspace is deleted.

## 7. Left to the builder

- How every label is weighted equally (balanced sampling or a weighted loss).
- DeBERTa's settings: length, learning rate, epochs and batch, within the probe's budget.
  Also the precision: fp16 with care, or fp32 if fp16 isn't stable.
- The LLM's base (8B or 14B) and its share of the pool, by budget.
- Where the weights live for the submission: private datasets or notebook outputs.
- The centering variant, and its features (logits or embeddings).

## 8. Risks

- **DeBERTa-v3-large in half precision on T4s** can overflow. The probe checks this first,
  and fp32 is the fallback, at about half the speed.
- **The prompt matcher can miss passages about our prompts.** If so, some topic would leak
  into the honest predictions and flatter the new member. The leaderboard gap at each
  submission is the check.
- **The medal line will rise** as people fork the public notebooks.
- **A late host ruling** against generated data: P7's safeguard 3 covers it.
- **A run that overruns a session:** checkpoints let it resume, and each push has a budget
  under the quota left.
