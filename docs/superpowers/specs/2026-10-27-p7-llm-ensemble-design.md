# P7: a fine-tuned LLM and a bigger ensemble — design

**Date:** 2026-10-27 · **Status:** draft, for the user's approval · **Phase:** P7 ("LLM and
ensemble" in the [project design](2026-10-23-name-that-llm-design.md)), done before P6 at the
user's choice.

## 1. Intent

Make the model **clearly more accurate**: in the user's words, P7 is worth it if "it would be
more accurate". The champion scores public LB 0.410. The medal line (top 10%) was 0.641 on
2026-10-26, and a public notebook now scores 0.651, so the line will climb as people submit
it.

P7 is also where the project learns the LLM side of the lifecycle: fine-tuning a
billion-parameter model with LoRA on 4-bit quantized weights, on free GPUs, and serving it
offline.

## 2. What we know (recon, 2026-10-27)

### 2.1 Where the champion loses points

Out-of-fold predictions on the seed-42 folds (balanced accuracy 0.422; `lr` 0.321, the LM
head 0.387, the encoder 0.350 alone):

- **The right model is usually close.** It's first for 42% of passages, in the top 2 for 62%,
  the top 3 for 74%, and the top 5 for 88%.
- **The lab is right for 74% of passages, and 54% of the wrong answers stay within the right
  lab.** The worst confusions are between siblings: the three GPT-5.6 variants (terra, sol,
  luna), glm-5.3 against glm-5.3-flash, claude-opus-4-8 against claude-sonnet-5, and the
  Gemini flash family (3.7 against 3.8 is the known near-coin-flip pair).
- **A few cross-lab confusions:** claude-fable-5 is right only 15% of the time, mostly taken
  for a GLM model; kimi-k3 is often taken for gpt-5.5.
- **Short passages are hardest:** 0.36 for 36–74 words, 0.43 for 75–136, 0.48 for 137–274.

So the remaining errors are fine distinctions, which more model capacity and more examples
buy.

### 2.2 What the leaders do

The user shared 9 high-scoring public notebooks. All are forks of one public ensemble that
scores **0.651** (Dhruv Tibarewal's `nlm-v27`). Its core, read from its code:

1. **LoRA fine-tunes of decoder LLMs as 18-way classifiers** (a classification head on top):
   Qwen3-14B-Base, Qwen3-8B-Base and Llama-3.1-8B, each with 5 fold models plus a full fit,
   weighted 3:2:1. Several models are cut to their early layers, as we do with Qwen in the
   LM head.
2. **A pool of generated passages:** text produced by the 18 models through their APIs,
   posted on Kaggle by participants. One merged pool holds 240k+ passages for all 18 models,
   about 90 times our 2,699.
3. **Lighter members** trained on our data plus the pool (TF-IDF and SVC heads, stylometry),
   mixed in at small weights.
4. **"Sinkhorn" balancing** of the final probabilities: it pushes the predictions toward an
   equal share of the test set for each of the 18 models (with a strength setting).

### 2.3 The rules on generated data

The rules allow external data that is free and publicly available, or of "minimal cost"
under their Reasonableness Standard. Whether passages generated through the 18 models' APIs
count has been asked on the forum since 2026-10-14 (the top thread, 127 votes), and **no
organizer has answered**. Participants read it as allowed, provided the data is published.
The project design (§12) said: test such datasets offline at any time, but submit with them
only after the host's ruling.

## 3. Decisions

**The user's:**
- P7 comes before P6.
- Success means a more accurate model, measured as always: CV on our folds, then the public
  leaderboard.
- **We train our own models.** No submitting the public notebook, and no borrowing other
  participants' trained weights or adapters.
- **Free compute only:** Kaggle's GPUs, within the weekly quota. No paid GPUs.

**Claude's call** (the user delegated it: "as you wish"). It replaces the project design's
"only after the host's ruling" from P7 on:
- **We use the public pool of generated passages, with three safeguards:**
  1. only free, public Kaggle datasets with an explicit open license on their page (such as
     CC0, CC BY, CC BY-SA, Apache 2.0 or MIT; "unknown" or "other" is excluded); we never pay
     to generate data;
  2. the pool only ever trains: every score is measured on our own passages, held out by
     prompt, after removing pool passages that duplicate them;
  3. one of the two final submissions stays pool-free, so a late ruling against generated
     data can't sink us. The forum stays watched.

## 4. Goals, non-goals, anti-goals

**Goals**
1. A fine-tuned LLM classifier (LoRA) that beats the champion's parts on our folds.
2. A cleaned, leak-free training pool, with the evidence that it helps.
3. A new champion: the LLM classifier blended with the current members, promoted by the usual
   gate (fresh folds, then the leaderboard).
4. A tested verdict on Sinkhorn balancing.
5. An offline submission that runs the new champion in under 9 hours on Kaggle's T4s.
6. W&B tracking working in the GPU notebooks again (it was off for all of P5).

**Non-goals**
- Generating data ourselves, or paying for anything.
- Submitting, forking or blending the public 0.651 notebook or its trained parts.
- P6 (monitoring, online A/B, continuous training), and changes to the public demo.

**Anti-goals** (failures even if everything runs)
- **A CV score inflated by the pool.** Pool passages written for the same prompts as a
  validation fold would leak topics into training, exactly the trap P3a's lesson warns about.
- A submission that times out, or fails offline.
- A "gain" inside the noise: fine-tuning is noisy (P5: identical runs differed by 0.014 on
  two folds).
- Data of unclear origin or license in the training set.
- Both final picks depending on the pool.

## 5. The approach

### 5.1 The pool

- **Sources:** free, public Kaggle datasets of generated passages, open license only (the
  candidates appear in the leaders' notebooks and in forum thread 752766).
- **Same shape as the competition data:** flattened with `flatten_text` (the community saw
  unflattened external data hurt), labels mapped to our 18, empty or off-length passages
  dropped (the training passages are 36 to 274 words).
- **No leakage:** exact and near duplicates of our passages are removed. Every pool passage
  belongs to a prompt group: its source prompt where the dataset gives one, else a topic
  match against our prompts. Passages in the same group as a validation fold never train
  the model scored on that fold.
- **Quality, per source:** a model trained on one source alone, then scored on our passages,
  shows whether that source's labels and style match. Sources that hurt are dropped.
- **Evidence that it helps:** `lr` trained with and without the pool, on the same folds.
  It's a cheap first test before any GPU time.

### 5.2 The fine-tuned classifier

- **First model: Qwen3-8B-Base**, LoRA on 4-bit quantized weights, with an 18-way
  classification head, on one or both of Kaggle's T4s. It's the family whose features already
  give our best single member (the LM head on Qwen3-4B, layers 6–9).
- **The first run trains on our passages alone.** It measures what the LLM adds by itself,
  before the pool, and it's the pool-free candidate for the hedge pick (§3).
- **Tools** (Claude's call): peft for the LoRA layers and bitsandbytes for the 4-bit weights,
  inside our own training loop (ADR 0003), not Hugging Face's `Trainer`. Every step stays
  visible, as in P5.
- **To explore, in the order the evidence suggests:**
  - keeping only the early layers (faster, and where style lives);
  - sequence length (passages reach about 400 tokens);
  - pool size per epoch (the full pool won't fit the GPU budget);
  - then a bigger or different model (Qwen3-14B-Base, Llama-3.1-8B) if 8B pays off.
- **Weights come from Kaggle Models** (Qwen and Llama are published there), so the offline
  notebook can attach them.

### 5.3 Sinkhorn balancing

- Measured first on out-of-fold predictions, where every model has about 150 passages.
- The test's per-model counts aren't known: a participant believes they're unequal. So it's
  also measured on deliberately unbalanced subsets of the folds (some models thinned out), as
  P3a's lesson requires for any trick that leans on the test's layout.
- It's kept only if it helps on fresh folds and survives the unbalanced check, and then
  confirmed on the leaderboard.

### 5.4 The ensemble and promotion

- The LLM classifier joins `lr`, the LM head and the encoder. The weights are chosen on
  out-of-fold predictions, as in P5. A calibrated or log-probability blend is tested too
  (I-015).
- **Promotion is unchanged:** better on fresh folds (seeds 42, 43, 44), the bootstrap gate's
  lower bound above 0, then a better public LB.
- **Two final picks:** the best overall, and the best pool-free.

### 5.5 The submission

- It is the same offline notebook, extended. It attaches the fine-tuned adapters (outputs of
  our own Kaggle notebooks) and the base model, and runs in under 9 hours on T4s with internet
  off.
- Libraries the Kaggle image lacks (for example, for 4-bit loading) ship in our lib dataset,
  as pandera does now.

## 6. Constraints

- **Hardware:** Kaggle's free GPUs, two T4s of 16 GB, within the weekly quota (about 30 hours,
  shared with nothing else during P7). Local runs are CPU-only.
- **Time box** (Claude's call): about three weeks of GPU quota (some 90 GPU-hours), aiming to
  finish by 2026-11-20. P6 is designed and built on CPU meanwhile, and December and early
  January stay free for P8, the finals.
- **Llama is gated:** its license must be accepted on Kaggle by the user, and only if Llama
  is ever chosen.
- **The submission:** offline, at most 9 hours, 5 submissions a day, 2 final picks.
- **Kaggle's stack:** transformers 5.16.1 and torch 2.11 on the GPU image. Anything else
  must be installable offline.
- **Approvals, asked in the session that uses them:** every Kaggle push of a GPU notebook,
  every dataset creation or new version, every submission, and a library release (the tag
  that publishes the lib dataset).
- **The W&B fix needs the user:** attaching the `WANDB_API_KEY` secret to each GPU notebook,
  once, in the Kaggle editor.
- Everything else as before: no competition data in the repo, notebook outputs stripped,
  every experiment comparable on the same folds.

## 7. Left to the builder

- Which pool datasets, after the license and quality checks.
- The LoRA setup: rank, learning rate, epochs, sequence length, layers kept, batch size and
  memory tricks.
- How folds are spent within the GPU budget. For example, tune on a subset of folds, then
  train all 5 for the out-of-fold predictions the blend needs.
- Whether a second LLM member (14B or Llama) is worth its GPU hours.
- The Sinkhorn strength, and the blend weights.

## 8. Risks

| Risk | Answer |
|---|---|
| The pool's labels are noisy (proxies can route to the wrong model) | Per-source quality checks; noisy sources dropped |
| The GPU quota runs out mid-phase | Cheap CPU tests first (the pool with `lr`); stop and re-plan rather than overspend |
| The host rules against generated data | The pool-free final pick; the forum watched |
| 4-bit training is unstable or too slow on T4 | Start small (early layers, short sequences), measure, then scale |
| The new member doesn't add to the blend | Report it honestly, as P3a's no-gain result was; the champion stays |
