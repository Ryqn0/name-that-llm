---
title: Name That LLM
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Which of 18 LLMs wrote this passage? A Kaggle model, live.
---

# Name That LLM

Paste a passage written by an AI chatbot, and the app ranks which of 18 language models most
likely wrote it, with a percentage for each.

> **Not hosted for now.** Hugging Face needs a PRO subscription for Docker Spaces, so this
> full app runs locally, from the repo or its Docker image. The free public demo runs the
> lighter word-and-character model in your browser:
> [Ryqn/name-that-llm](https://huggingface.co/spaces/Ryqn/name-that-llm).

It runs the best model of a learning-in-public project built for the Kaggle competition
[Name That LLM](https://www.kaggle.com/competitions/name-that-llm). The code, the experiments
and a diary of every step are on [GitHub](https://github.com/Ryqn0/name-that-llm).

## Before you trust it

- **It is not an AI detector.** It only knows the 18 models below, and it always picks one of
  them, even for text written by a human or by another AI.
- **It is often unsure.** On passages it has never seen, it names the exact model about
  4 times in 10. Random guessing would manage 1 in 18. Models from the same company write
  alike, so its second and third guesses matter too.
- **It learned from a small, specific dataset:** 2,699 passages (about 150 per model)
  answering 150 prompts. Other topics or writing styles can fool it.
- **Your text is not stored.** It isn't kept or logged; the logs only record its length, the
  time taken and the answer.

## The 18 models

| Company | Models |
|---|---|
| Anthropic | claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5 |
| Google | gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash |
| OpenAI | gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra |
| Z.ai | glm-5.3, glm-5.3-flash |
| Moonshot | kimi-k3 |
| Qwen | qwen3.8-max |

## How it works

Three models each read the passage, and their probabilities are averaged with weights
70% / 15% / 15%:

1. **Word and character patterns**: a logistic regression on TF-IDF features. It catches
   vocabulary and punctuation habits (curly quotes, em dashes, semicolons).
2. **An LLM's view of the text**: the hidden states of layers 6 to 9 of
   [Qwen 3 4B base](https://huggingface.co/Qwen/Qwen3-4B-Base), averaged over the passage,
   then a logistic regression. Early layers still carry the surface of the text, where
   style lives. Only those first blocks of Qwen are loaded.
3. **A fine-tuned encoder**: [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base),
   fine-tuned to name the 18 models. Five copies, one per cross-validation fold, are averaged.

Chatbot answers usually arrive with markdown (headings, bullets, bold). The competition's
passages had all of that removed, so the app flattens pasted text the same way first.

The "How sure?" line uses the best guess's probability, but the wording comes from data. On
held-out predictions we measured how often the best guess is right at each confidence
level, and the page reports that rate.

## How good it is

| Measure | Score |
|---|---|
| Balanced accuracy, cross-validation (the mean of 3 fresh fold seeds) | 0.423 |
| Balanced accuracy, Kaggle public leaderboard | 0.410 |
| Random guessing | 0.056 |

Balanced accuracy is the average, over the 18 models, of the share of each model's
passages that are named correctly.

## Speed

About 4 seconds per passage on 2 CPU threads, one passage at a time (measured: 3.7 s on a
laptop). At start-up, loading the models from the registry takes about 75 seconds.

## Limitations

- Only the 18 models above, and only in English.
- gemini-3.7-flash and gemini-3.8-flash are nearly impossible to tell apart: even a model
  trained on just those two barely beats a coin flip.
- The two neural models read only about the first 350 to 400 words (512 tokens). The word
  and character model reads everything.
- The training passages are 36 to 274 words long. The app works best on 50 to 300 words.

## Licenses

- **The code:** MIT.
- **Qwen 3 and ModernBERT:** Apache 2.0.
- **The competition data:** CC BY-SA 4.0. The models were trained on it, but none of its
  passages appear here.
