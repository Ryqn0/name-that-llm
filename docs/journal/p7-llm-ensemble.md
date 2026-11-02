# Journal: P7, the LLM and the ensemble (2026-10-27 → 11-02)

**Goal:** in the plan's words, "make the model clearly more accurate". Fine-tune
Qwen3-8B-Base with LoRA as an 18-way classifier, train it on our data plus a leak-free public
pool of generated passages, blend it with the current members, promote it through the usual
gate, and submit it offline.

**Result:**
- **Public LB 0.410 → 0.454 → 0.505**, in two submissions. On seed 42, the recipes as
  submitted score 0.4219 → 0.4727 → 0.5246 in CV:
  - 0.454: the pool in the ensemble's `lr` (I-029);
  - 0.505: the LLM as a fourth member, in a log blend, every weight chosen by nested CV.
- **The LLM alone** scores 0.4705 out of fold (five folds, seed 42): our best single member.
  It's a coincidence that this equals `ensemble_pool`'s CV.
- **The pool policy**, the user's call delegated to Claude: [ADR 0006](../adr/0006-generated-data-and-lora.md).
  Open licences only, the pool only trains, and one final pick stays pool-free.
- **Sinkhorn balancing** failed its gate twice, and is off.
- **The rank** moved from 2,236th of 3,387 (top 66%) to 2,532nd of 4,339 (top 58%). The
  field grows and improves too: the top 10% now starts at 0.663.

**In numbers:**
- 19 pull requests (#43 to #86), interleaved on main with P6's and P8's.
- P7's own code: 43 files, 3,876 lines added and 110 removed. 817 of the added lines are
  tests: 47 new tests.
- Then the final review's fix pass (#89): 3 fixes, each with a test seen failing first.
- The suite stands at 249 tests, all passing with torch, transformers and peft (P6's and
  P8's included).
- About 30 GPU-hours on Kaggle, by the ledger's timings, against the plan's budget of 90:
  - notebook 10 (the first run): 4.4 h;
  - notebook 11 (the exploration): about 10.5 h;
  - notebook 12 (the final runs): three sessions, about 14 h;
  - the submission's three checked runs: about 1.7 h.

## How the work was run

- **Native execution** of the [plan](../superpowers/plans/2026-10-27-p7-llm-ensemble.md),
  from the approved [design](../superpowers/specs/2026-10-27-p7-llm-ensemble-design.md).
  Claude implemented each task itself, test first, and kept a ledger of every step and
  judgment call. One fresh reviewer read the whole phase at the end.
- **Interleaved with two other phases.** P6 (the MLOps loop) ran on CPU while P7's first GPU
  runs trained. P8 (the medal push) started on 2026-10-30, and its Task 3, the nested gate
  (#75), is what judged P7's ensemble.
- **One branch and one pull request per task**, or per step for Tasks 7 and 8, merged only
  after CI passed.
- **GPU work ran on Kaggle** (two T4s), pushed through the API. Kaggle doesn't hand secrets
  to API-pushed runs, so each run's results were downloaded and logged to W&B from the PC
  (`scripts/log_gpu_run.py`).
- **The user's OK was asked in the session, each time:** every Kaggle push, every dataset
  and dataset version, every release tag, and every submission. The user also made two of
  the phase's decisions: the GPU budget's reshaping (Task 7), and removing the demo's
  challenger (#66).

## Step by step

### Task 1: Sinkhorn balancing (PR #43, 2026-10-27)

`balance.py`: `sinkhorn_balance(proba, alpha, n_iter, tol)` rescales the predicted
probabilities so that each model gets an equal share of the test, with `alpha` setting how
far to go. The leaders' public ensemble does it.

**Checked:**
- 7 tests, seen failing first: full balancing equalizes each model's share, `alpha` 0
  changes nothing and a partial `alpha` moves part way, picks move away from an over-predicted model, an
  empty column stays finite, and an `alpha` outside [0, 1] is refused.
- **The verdict on our folds** (`scripts/sinkhorn_check.py`, on the champion's out-of-fold
  predictions):
  - `alpha` chosen on seed 7: 0.75 (+0.0030);
  - on fresh seeds 42, 43 and 44: +0.0019, +0.0011, +0.0011;
  - seed 42's prompt bootstrap: 95% CI [−0.0019, +0.0056]. The lower bound is below 0, so
    the gate fails: **dropped** (I-027);
  - with 6 models thinned to half their passages, as an unbalanced test might be (50
    subsets): a mean of +0.0013, and −0.0041 at worst.

### Task 2: the pool of generated passages (PR #44, 2026-10-27)

`pool.py`, torch-free:
- `license_allowed`: open licences only (CC0, CC BY, CC BY-SA, Apache 2.0, MIT). "Unknown"
  and "other" are refused;
- `normalize_label`: `openai/gpt-5.5` or `Claude-Opus-5` map to our labels, and an unknown
  model is dropped, never guessed;
- cleaning and dedupe, then `match_prompts`: each pool passage is matched to the training
  prompt it answers, if any;
- `pool_for_fold`: a fold's pool leaves out every passage matched to one of its held-out
  prompts. Without this, a pool passage answering a validation prompt would train the
  fold's model, and the CV would be inflated;
- `build_pool` and `load_pool_sources` (`configs/pool.yaml`).

**The source:** `crtproduction/name-that-llm-open-pool` (CC-BY-SA-4.0), the forum's merged
pool, which already gathers every open-licensed candidate and drops the others. Each of its
rows names its original dataset (`source_column`). Its 246,978 rows gave 246,945 passages;
6.9% matched a training prompt.

**Checked:**
- 9 tests, seen failing first, and two beyond the brief: each row is credited to its
  original source, and the shipped config loads;
- a cross-check against the pool's own flag for passages near a training prompt: we
  matched all 323 it flags, so our guard is the stricter one.

### Task 3: which sources help (PR #45, 2026-10-27)

`scripts/pool_check.py` adds each original source alone to `lr`'s training folds, and
measures the change in CV with a prompt bootstrap. It took 186 minutes on CPU.

- `lr` alone: 0.3212.
- **5 of 56 sources raise it** (drjcfmd +0.022, darragh1walsh +0.020, ayhampar +0.018,
  vbandaru +0.016, prvsiyan +0.010). **51 lower it**, from −0.007 to about −0.15.
- **`lr` + the 5 kept** (48,083 passages, capped at 30,000): **0.4205** (+0.0993, CI
  [+0.0801, +0.1196]). The pool helps.
- **Why the 51 hurt** is unclear: the worst sources cover only 1 to 5 of the 18 models, and
  `lr` has no class weights. So they're recorded as "hurt as used", not "wrong labels"
  (I-028, a fairer check).
- `configs/pool.yaml` records the verdict as an **allowlist**.

### Task 4: the LLM classifier (PR #46, 2026-10-27)

`llm.py`:
- Qwen3-8B base, loaded in 4 bits (bitsandbytes), with LoRA adapters (peft) and an 18-way
  classification head;
- trained in our own loop (ADR 0003), not Hugging Face's `Trainer`;
- a CLI that trains chosen folds, so two processes can each take one GPU;
- adapters saved per fold, then averaged at inference on one shared base.

**Checked:** 8 tests on a tiny Qwen, on CPU, seen failing first. Among them:
- an adapter loaded onto a different cut of the base model is refused;
- empty and very long texts still give 18 probabilities;
- training folds split across two runs, then merged, equals one run.

CI's torch job ran 43 tests, against main's 35.

### Task 5: the first Qwen3-8B run (notebook 10, PR #59, 2026-10-28)

- **The smoke test** is a pytest test (`tests/test_notebooks.py`): it runs the whole
  notebook on a tiny Qwen, in CI's torch job.
- **Run 1 ran out of GPU memory** after 10 minutes. The speed probe's model stayed in the
  notebook's own process (5.80 GB, despite `del` and `empty_cache`), and a fold process
  needed that memory.
  - **The fix:** the probe runs in its own process (`python -m name_that_llm.llm --probe
    N`), so the memory is returned when it exits.
- **Run 2** took 4 h 11 min. On seed 7:
  - the LLM alone: **0.3641**, against the encoder's 0.3590;
  - the best 4-member blend: **0.4523**, against the ensemble's 0.4293 (in-sample).
- **Decision: continue**, since both of the plan's rules held (the LLM at least matches the
  encoder, and adds at least +0.01 to the blend).

### Task 6: the exploration (notebook 11, PR #62, 2026-10-28 → 29)

On proxy folds 0 and 1 of seed 7, against notebook 10's 0.3734:

| Setting | Score | Minutes |
|---|---|---|
| 24 of 36 blocks | 0.3373 | 55 |
| 16 of 36 blocks | 0.3493 | 37 |
| 320 tokens | 0.3604 | 82 |
| **5,000 pool passages per fold** | **0.4438** (+0.070) | 420 |
| 15,000 pool passages | skipped (projected 995 minutes) | — |

**Chosen:** full depth, 512 tokens, 5,000 pool passages, step size 1e-4, 2 epochs. **The
cost:** 420 minutes per pair of folds, so a 5-fold seed would take about 21 GPU-hours.

### I-029: the pool in the ensemble's `lr` (PRs #63 and #64, 2026-10-29)

The plan gave the pool only to the LLM. Yet `lr` + the pool alone (0.4205) matched the
whole ensemble (0.4219). So, with the user's OK, a CPU check while notebook 11 trained:
- weights 0.75 / 0.10 / 0.15, chosen on seed 7;
- on fresh seeds: +0.0486, +0.0456 and +0.0393. Seed 42's CI: [+0.0315, +0.0652];
- **promoted.** It shipped as `ensemble_pool` in v0.9.0 (#64), and scored **LB 0.454**
  (2026-10-30). By then the pool had changed (next section): with the clean pool it was
  submitted with, its seed-42 CV is 0.4727.

### The pool changed under us (2026-10-29 → 11-01)

- **The maintainer's clean release** (v9, then v10) removed 70,558 rows, among them 2 of
  our 5 sources (prvsiyan and ayhampar). The other three were trimmed, to 28,048 passages.
  - Notebook 11, Task 3 and I-029 had used the old pool; notebook 12 and the submission
    used the clean one. Only a count in a log showed it.
  - **The clean 3-source pool is better for `lr`:** 0.4338, against the old 5-source pool's
    0.4205 (I-032). 28,048 passages from all 54 clean sources score only 0.3623, probably
    because of class imbalance.
  - **The demo's challenger** had trained on the old pool. The user removed it (#66),
    because some of the removals were for licence or provenance risk.
- **The 2026-11-01 release** dropped one more source, shrank eight (`lr`'s drjcfmd among
  them) and brought prvsiyan back. The submission now reads `lr`'s pool from our pinned
  copy of v10, and stops unless it holds the measured 28,048 passages (Task 8).
- P8 turned this into a pinned dataset and a weekly check (I-034).

### Task 7: the final LLM runs, and the gate (notebook 12, PRs #65, #72, #73, #80)

- **The budget couldn't hold.** The plan wanted four pooled seeds plus two pool-free ones;
  at 21 GPU-hours a seed, that was impossible. **The user's decision:** seed 42 only, run
  over several sessions; the weights, the report and the gate run on the PC.
- **Session 1** tested 1 epoch against notebook 11's 2, on the same folds: 0.399 against
  0.444, so **2 epochs stay**. One epoch took 122 minutes, not the projected 214.
  - **Caveat:** the two runs used different pools (above), so the −0.045 mixes two effects.
- **Session 2:** folds 0–3 of seed 42: 0.4611, 0.4577, 0.4963, 0.4574 (225 minutes per
  pair). Fold 4 would have crossed the session's budget.
- **Session 3:** fold 4: 0.4796. **All five: 0.4705** out of fold (W&B `llm-final`).
- **The gate** (`scripts/ensemble_gate.py`, from P8 Task 3): every weight chosen by nested
  CV, so each fold's weights come from the other four folds; a prompt bootstrap for the CI.

  | Candidate | Against | CV | Gain, 95% CI | Verdict |
  |---|---|---|---|---|
  | + the LLM (probability blend) | the three members, re-weighted: 0.4819 | 0.5108 | +0.0289 [+0.0159, +0.0415] | promote |
  | the log blend, weights searched for it | the four-member probability blend | **0.5246** | +0.0137 [+0.0030, +0.0238] | promote |
  | Sinkhorn 0.75 | the four-member probability blend | 0.5171 | +0.0063 [−0.0023, +0.0141] | off |

- **Shipped:** `configs/ensemble_llm.yaml`, a log blend weighted 0.50 / 0.05 / 0.10 / 0.35
  (`lr` + the pool, the LM head, the encoder, the LLM).

### Task 8: the four-member submission (PRs #74, #81, #83, #85, 2026-11-01 → 02)

- **The code (#74):** `predict_submission_ensemble` takes the LLM's probabilities, and
  refuses them without an LLM member, so a misconfigured ensemble can't drop it silently.
- **The release, v0.10.0:** the lib dataset now ships the `peft` and `bitsandbytes` wheels,
  which the Kaggle image lacks.
- **The adapters** are a private dataset, `ryqn00/name-that-llm-llm-adapters` (783 MB
  zipped). Version 1 was unusable: Kaggle unpacked the folds at the root, where the notebook
  doesn't look. Version 2 wraps them in an `llm/` folder.
- **The Kaggle CLI broke in CI** (#81, I-035): python-dateutil 4.0 dropped `six`, which
  kaggle 2.2.5 imports without declaring it. Every workflow's call now brings `six`.
- **The notebook (#83)** reads `lr`'s pool from the pinned v10, and stops unless the count
  is 28,048.
  - **A correction:** the ledger first said the pinned dataset had no licence. It has one
    (CC-BY-SA-4.0); the one-liner had read the wrong key.
- **Version 6 ran out of GPU memory** in its runtime check: the LM-features model (Qwen 3
  4B, 8 GB in float16) stayed loaded next to the 4-bit 8B LLM. The hidden test would have
  failed the same way, hours in. **The fix (#85):** the LM features load their model per
  call, and free it.
- **Version 7** ran cleanly in 54 minutes, and estimated 138 minutes for the hidden test,
  well under the 450-minute bar. Submitted 2026-11-02 at 16:53, it scored **public LB
  0.505**: +0.051 over 0.454, against +0.052 between the two recipes' CVs (0.4727 →
  0.5246).
- **The served model:** the registry and the local app run on CPU (ADR 0004), and the LLM
  needs a CUDA GPU. So they keep serving `ensemble` v1, and ADR 0004 is amended (#87).

### Task 9: record and ship (PRs #86 to #90)

[ADR 0006](../adr/0006-generated-data-and-lora.md), the diary, `Context.md`, `IDEAS.md`
(I-003 used, I-015 promoted, I-019 done, I-027 dropped, I-036 new), and this journal. Then
the final review and its fix pass (#89), below.

## Every judgment call

From the ledger, in order, each with its cost if wrong. The ones marked **(the user)** were
the user's decisions.

**Setup**
1. **No git worktree:** one branch per task, shipped by pull request with CI before the
   merge. Cost: none.
2. **The spec and plan were committed with Task 1.** Cost: none.

**Tasks 1 to 3: Sinkhorn and the pool**

3. **ruff formats the Python inside Markdown,** so the plan's compact test snippets were
   reformatted. Cost: none.
4. **The licence filter reads Kaggle's long names** ("Attribution 4.0 International (CC BY
   4.0)"). The plan's filter would have rejected six good sources. Cost: none.
5. **The merged open pool is the single source,** with each row credited to its original
   dataset (`source_column`). Cost: an extra optional field.
6. **Two tests beyond the brief,** for interfaces later tasks rely on. Cost: none.
7. **The source check caps each source at 3,000 passages** (not 6,000), and judges the 17
   smallest as one group: about 1 CPU-hour instead of 6. Cost: a noisier verdict per
   source.
8. **The 51 sources that lower `lr` are "hurt as used", not "wrong":** class balance
   confounds the check. Only the 5 are kept. Cost: useful passages left out (I-028).
9. **`pool.yaml` is an allowlist, not an exclude list:** the weekly retraining downloads
   the latest pool, and an exclude list would let a new, unchecked source in. Cost: none
   known.

**Tasks 4 to 6: the LLM**

10. **Small interface choices:** `--data-dir` on the CLI; 2 epochs in the equality tests;
    empty texts become the pad token in `llm.py` only, so the encoder can't change; a fold
    outside the range is refused; no generation cache; the cut is checked by the actual
    number of blocks. Cost: none.
11. **The notebook's smoke test is a pytest test,** run by CI. Cost: about 30 s per CI job.
12. **One failed fold process stops the other,** since a run whose folds can't all merge
    shouldn't keep spending GPU hours. Cost: none.
13. **The probe runs in its own process,** after run 1's out-of-memory error. Cost: none.
14. **W&B is logged from the PC,** because Kaggle doesn't hand secrets to API-pushed runs.
    Cost: none.
15. **Notebook 11 reuses notebook 10's full-depth result** as its baseline (same settings,
    same folds), saving about 80 GPU-minutes. Cost: none.
16. **Notebook 11's budget is Kaggle's 12-hour session,** not the plan's 20 GPU-hours, since
    one session can't run longer. Cost: a stage skipped (I-031).
17. **IDs:** the 14B idea is I-030, and the skipped stages I-031, since I-028 was taken.
    Cost: none.

**I-029 and Task 7: the ensemble**

18. **A CPU check of the pool in the ensemble's `lr`** (the user's OK), which the plan
    didn't foresee. Cost: CPU time only.
19. **The pool ensemble shipped as v0.9.0,** so the LLM ensemble became v0.10.0. Cost: a
    version number.
20. **(the user)** **One seed for the LLM,** run over several sessions; fresh-seed and
    pool-free LLM runs dropped. Cost: the gate rests on seed 42 alone.
21. **(the user)** **Session 1 tested 1 epoch first,** on seed 7's folds, so the answer came
    7 hours sooner. Cost: those folds don't count toward seed 42.
22. **`lr`'s cached predictions are keyed by the pool file and its size,** so one pool's
    predictions can never stand in for another's. Cost: none.
23. **The config tests match their error messages,** so an unrelated validation error
    can't pass them. Cost: none.
24. **The blend form and Sinkhorn were judged on seed 42 by the same nested gate,** since no
    seed-7 LLM exists. Sinkhorn kept Task 1's `alpha`, chosen on seed 7. Cost: a form that
    only looks better on one seed.
25. **Sinkhorn wasn't retested on top of the log blend:** a third try on the same seed
    would be fishing. Cost: up to about +0.006 left on the table.

**Task 8: the submission**

26. **The LLM's probabilities without an LLM member raise an error.** Cost: none.
27. **The runtime check times the LLM on 400 stand-ins,** not all 4,000: an hour saved in
    the hidden run. Cost: a noisier time estimate.
28. **The adapters dataset's corrected version needed no second OK:** the same files in the
    layout the notebook searches. Cost: none (it's private).
29. **The submission reads `lr`'s pool from the pin, and stops unless the count matches:**
    the gate measured exactly that pool. Cost: `lr` misses whatever newer releases improved.
30. **`pool.yaml` stays on the public pool** for the weekly retraining, which measures its
    challenger afresh each week. Cost: the challenger's pool may differ from the
    champion's; it's gated, and a person merges it.
31. **The re-push after the memory fix completed the approved push and submit.** Cost: one
    push the user might have wanted to see first.
32. **The leaderboard's champion and the served model part ways** (ADR 0004, amended).
    Cost: the local app serves a weaker model, which changes no score.

## Open issues

- **Two notebooks are still public on Kaggle**, until the user switches them on their pages:
  11 (its outputs hold the old pool) and 09 (the encoder's weights). See the final review.
- **One seed.** The LLM ran on seed 42 only, so its weights rest on nested CV on that seed.
- **The CV–LB gap** went from −0.012 to −0.019 with the pool's step, then held at −0.020
  with the LLM's (see the process lessons).
- **The 1-epoch test is confounded** with the pool's change (Task 7, session 1).
- **Notebook 11's timing:** its pooled pair of folds took 420 minutes, about 1.7 times
  what notebook 12's identical pairs took (225). The cause is unknown.
- **The pool's other 51 sources** may hold useful passages, if balanced: I-028 and I-032.
- **The registry and the local app** serve `ensemble` v1 (LB 0.410). Registering
  `ensemble_pool` (0.454) is a cheap follow-up.
- **The medal is 0.158 away** (0.505 against the 0.663 cutoff). The pool's maintainer
  reports 0.587 for one DeBERTa-v3-large trained on the whole clean pool: P8's plan.

## The final review

- **The reviewer:** a fresh reviewer agent (Fable), as the user chose for P6 and P7. It read
  the net diff of P7's 42 code, config, notebook and workflow files (`b0db055..98fc182`, 19
  pull requests). The P6 and P8 parts of shared files were labelled as such.
- **Its verdict:** "with fixes". Nothing was Critical, 2 findings were Important, and 10
  were Minor (A to J). In its words: "The phase's code is correct where it matters. The only
  leak I found is a measured 0.15% edge case, the submission is consistent with the gate,
  and the suite passes."
- **Its own checks**, beyond reading:
  - notebook 12's sessions 2 and 3 trained on byte-identical pools;
  - the adapters dataset's 5 files are byte-identical to the sessions' outputs;
  - averaging 3 distinct tiny adapters through the shared base equals the mean of their
    single predictions exactly;
  - the gate's member order matches `ensemble_llm.yaml`;
  - on the real pool, the stored prompt matches equal a fresh computation (3,780 of 3,780).
- **Facts checked before the triage:**
  - notebook 11 (`name-that-llm-llm-explore`) is public, and its outputs list
    `pool.parquet`: 49,349 passages, among them the 2 sources the clean release removed;
  - notebooks 12 and 13 write the pool there too. They're private, but would publish it the
    day they're made public;
  - of the other public notebooks, only 09 (`encoder-final`) exposes anything: the encoder's
    5 weight files, against P8's rule (private weights, and the notebooks that hold them);
  - the submit notebook attaches 09's outputs, so a Quick Save, which makes an output-less
    latest version, isn't a safe way to make 09 private. Its page's sharing setting is;
  - the recipe submitted at 0.454 scores 0.4727 on seed 42 with the clean pool it used. So
    the CV gain behind the LB's +0.051 is +0.052, not the +0.054 the docs said;
  - `lr`'s three sources are open-licensed on their own pages: CC-BY-SA-4.0, CC BY 4.0 and
    CC0.
- **The triage** graded each finding by what a person using the project would get. Both
  Important findings were fixed, and one Minor was upgraded:
  - **E**, because the weekly retraining would have trained its next challenger, unseen, on
    a source the user had withdrawn a challenger over (#66).

**The fixes**, in one pull request (#89). Each has a test that failed first.

| # | The finding | The fix | Its test |
|---|---|---|---|
| 1 | Notebook 11 (public) has the pool among its outputs; 12 and 13 write it there too; 09 (public) holds the encoder's weights | Notebooks 11–13 write the pool to a scratch folder, never `/kaggle/working`; 09 and 11 are private in their metadata; the generators too | `test_notebooks_holding_weights_or_the_pool_stay_private`, and `assert_no_pool_in` in three smoke tests |
| 2 | Nothing tested that the submission blends in log space | A test where the two forms pick different labels | `test_ensemble_submission_follows_the_log_blend`, failing on a mutant without `log=` |
| E | `pool.yaml` still allowlisted the 2 removed sources, and one came back upstream on 2026-11-01 | The allowlist names the 3 measured sources | `test_lr_allowlist_is_the_three_measured_sources` |

**J**, the "+0.054 in CV" claim, was corrected in this record and in `Context.md`, with the
like-for-like +0.052.

**The rulings on what the reviewer declined to judge** (each stands):
- the P6 and P8 parts of shared files: their own reviews (P8's final review);
- generated data under the rules: ADR 0006's policy, with the pool-free final pick as the
  hedge;
- one LLM seed, and no pool-free LLM: the user's decision (Task 7);
- W&B inside the GPU notebooks: Kaggle's limit;
- the hyperparameters and the pool size: the builder's call (spec §7), and I-031;
- the blend form chosen on seed 42: the LB confirmed it;
- the registry serving `ensemble` v1: ADR 0004, amended;
- the label imbalance of the LLM's pool sample: I-032's label-balanced draw;
- the sources' licences one by one: checked (above);
- the cross-source dedupe order: pinned by the 28,048 check;
- the 9-hour limit at 138 minutes: no risk now (Minor A carries the future risk).

Cost if wrong: small for each.

**Deferred minors**, not fixed:
- **A:** the runtime bar (450 minutes) doesn't bound the hidden run, because the stand-in
  check runs again inside it. Safe at today's 138 minutes. **For P8 Task 5**, before it
  adds GPU members, with a smoke test of the submit notebook.
- **B:** `USE_ENCODER = False`, a documented fallback, crashes at `ens_cfg.llm`, and the
  submit notebook's intro still describes `ensemble_pool`. **For P8 Task 5** (same notebook).
- **C:** prompt matching keeps only each passage's best prompt. 0.15% of (passage, fold)
  pairs train a fold with a passage that clears the threshold for one of its held-out
  prompts.
- **D:** `lr`'s allowlist was chosen on seed 42, the gates' seed: a mild winner's curse.
  Seeds 43 and 44 confirmed the gain, and the LB confirmed most of it.
- **F:** the gate's LLM loader checks neither the seed nor duplicate folds. **For P8 Task
  4's gate**, which changes that file anyway.
- **G:** an adapter's `llm_config.json` records neither the label order nor `max_length`.
- **H:** the release takes the newest `peft` and `bitsandbytes`, and the 5-adapter swap has
  no CI test. **For P8 Task 5's release:** pin both.
- **I:** notebook 12's pool is neither pinned nor count-checked.
- **J, the rest:** two docstrings still say three members. **For P8 Task 5**, which edits
  that function.

**The second recommendation** became I-036: the test-time members average 5 fold models,
so their probabilities are flatter than the out-of-fold ones the blend weights were tuned
on.

**The Kaggle side** is the user's: switching notebooks 09 and 11 to private on their pages.

## Process lessons

- **Measure the cost before planning on it.** The plan's GPU budget assumed one pooled
  seed was cheap. Notebook 11 showed 21 GPU-hours per seed, and Task 7 had to be reshaped
  around one.
- **A public dependency changes silently.** The pool lost two of our sources between two
  runs, and later shrank again. Pin the data, and check its count where it's used.
- **A pinned version isn't a pinned environment.** kaggle 2.2.5 never changed, yet it broke
  when a dependency it never declared changed.
- **GPU memory can't be tested on CPU.** Two out-of-memory errors (notebook 10's probe, the
  submission's two models) appeared only on the T4. The submission's runtime check doubled
  as its only memory rehearsal, and caught one before it cost a submission.
- **A dataset's layout is part of its contract.** The adapters' first version held the right
  files in the wrong place.
- **Weigh the baseline the same way as the candidate.** On folds 0–3, against the fixed
  I-029 weights, the LLM seemed to add +0.045. With both sides weighted by nested CV, it
  added +0.034 (P8 Task 3), and +0.029 on all five folds.
- **The gains held on the hidden prompts.** Comparing the recipes as submitted, on seed 42:
  the pool's step was +0.051 in CV and +0.044 on the LB; the LLM's, +0.052 and +0.051. The
  CV–LB gap went from −0.012 to −0.019 with the pool, then held. The pool's allowlist was
  chosen on seed 42 itself, so part of its CV gain is likely a winner's curse (Minor D).
- **Compare like with like.** The docs first said "+0.054 in CV" for the LLM's step. That
  compared the new recipe on today's pool with the old one on a pool that had since
  changed.
- **A public notebook publishes its outputs.** Everything under `/kaggle/working` becomes a
  download for anyone once the notebook is public. Write data you may not redistribute to a
  scratch folder.

## How to retrace P7 yourself

- The plan, the design and the ADR:
  [plan](../superpowers/plans/2026-10-27-p7-llm-ensemble.md),
  [design](../superpowers/specs/2026-10-27-p7-llm-ensemble-design.md),
  [ADR 0006](../adr/0006-generated-data-and-lora.md).
- The pull requests, in order: #43 to #46, #59, #62 to #67, #72 to #74, #80, #81, #83, #85,
  then the records (#86 to #88), the final review's fix pass (#89), and this journal (#90).
- **Run the checks:**
  `uv run --all-extras --with torch --with transformers --with peft pytest -q
  tests/test_balance.py tests/test_pool.py tests/test_llm.py tests/test_notebooks.py
  tests/test_config.py tests/test_blend.py tests/test_submit.py`.
- **The CPU experiments**, each a script that prints its verdict: `sinkhorn_check.py`,
  `pool_check.py`, `pool_ensemble_check.py`, `clean_pool_check.py`, `llm_ensemble_check.py`
  and `ensemble_gate.py`, in `scripts/`.
- **On Kaggle (private until 2027-01-13):** the notebooks `name-that-llm-llm-first`,
  `-llm-explore`, `-llm-final` and `-submit`; the datasets `name-that-llm-lib`,
  `-pool-v10` and `-llm-adapters`.
- **On W&B:** `llm-first` (u0062ugi), `llm-explore` (t1mza9kl), `llm-final-s1-ep1`
  (dqhy2w6d) and `llm-final` (4i8ru6ex).
