---
title: Name That LLM
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: Which of 18 LLMs wrote this passage? Runs in your browser.
---

# Name That LLM: the free demo

Paste a passage written by an AI chatbot, and the page ranks which of 18 language models most
likely wrote it, with a percentage for each. **It runs entirely in your browser:** your text
never leaves your device.

It comes from a learning-in-public project built for the Kaggle competition
[Name That LLM](https://www.kaggle.com/competitions/name-that-llm). The code, the experiments
and a diary of every step are on [GitHub](https://github.com/Ryqn0/name-that-llm).

## Before you trust it

- **It is not an AI detector.** It only knows the 18 models below, and it always picks one of
  them, even for text written by a human or by another AI.
- **It is often unsure.** On passages it has never seen, it names the exact model about
  3 times in 10. Random guessing would manage 1 in 18. Models from the same company write
  alike, so its second and third guesses matter too.
- **This demo runs the lighter of the project's models.** The full model averages three
  models, two of them neural networks, and names the exact model about 4 times in 10. It
  needs a server with several GB of memory, so it isn't hosted here; the repo has it, with a
  Docker image to run it yourself.
- **It learned from a small, specific dataset:** 2,699 passages (about 150 per model)
  answering 150 prompts. Other topics or writing styles can fool it.

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

A logistic regression reads TF-IDF features of the passage: its **words** (single words and
pairs, lowercased) and its **characters** (runs of 1 to 5, case kept). It catches vocabulary
and punctuation habits (curly quotes, em dashes, semicolons). That makes 178,212 features;
the model's files weigh about 15 MB, which your browser downloads once.

The page runs a JavaScript copy of exactly what scikit-learn computes (`ntl.js`). When the
model was exported, the copy was checked against Python on 50 training passages: the same
best guess on all 50, and probabilities within 0.00001. Tests check it again on every change.

Chatbot answers usually arrive with markdown (headings, bullets, bold). The competition's
passages had all of that removed, so the page flattens pasted text the same way first.

The "How sure?" line uses the best guess's probability, but the wording comes from data. On
held-out predictions, we measured how often the best guess is right at each level:

| How sure? | The best guess's probability | Right |
|---|---|---|
| Fairly sure | 15% or more | 47% of the time |
| Unsure | 12% to 15% | 30% |
| Very unsure | below 12% | 19% |

## How good it is

| Balanced accuracy | This demo | The full model |
|---|---|---|
| Cross-validation (the mean of 3 fresh fold seeds) | 0.325 | 0.423 |
| Kaggle public leaderboard | 0.320 | 0.410 |
| Random guessing | 0.056 | 0.056 |

Balanced accuracy is the average, over the 18 models, of the share of each model's
passages that are named correctly.

## What the page sends

When monitoring is switched on, the page sends small anonymous events, so we can see whether
the site works and how the models do on real texts:

- **when it loads:** how long the model took to load, and whether it loaded;
- **after each answer:** how long it took, the passage's length rounded down to 25 words,
  the predicted model and its "How sure?" level, and why the text looked unusual, if it did.
  When a second model is being tested, it scores the same text silently, and its guess is
  sent too; you always see the same model's answer;
- **if you answer "Do you know who wrote it?":** your answer, with what the page predicted;
- **errors:** where they happened, and their message. A message that quotes your text is
  replaced by "[redacted]".

**Your text never leaves your browser,** in any form: no excerpt, no fingerprint, no word
list. There are no cookies, nothing is stored on your device, and nothing identifies you or
links two visits. Each event even carries a fresh random number, which the monitoring
service requires and which links nothing.

Two more things, to be complete:
- **The monitoring service (Grafana Cloud) adds one fact** to each stored event: whether
  your browser runs on a phone, read from the standard header every browser sends. Like
  any website, the servers that receive the page's requests, Hugging Face's and
  Grafana's, see your IP address. The stored events don't include it.
- **When a second model is being tested,** your browser downloads it after your first
  answer, in the background: about 35 MB more. It's skipped when your browser asks to save
  data, or on a slow connection.

## Limitations

- Only the 18 models above, and only in English. A text unlike the training passages (under
  36 words, with many words the model has never seen, or mostly non-Latin letters) gets a
  note asking you to treat the answer with extra care.
- gemini-3.7-flash and gemini-3.8-flash are nearly impossible to tell apart: even a model
  trained on just those two barely beats a coin flip.
- The training passages are 36 to 274 words long. The page works best on 50 to 300 words.

## Licenses

- **The code:** MIT.
- **The competition data:** CC BY-SA 4.0. The model was trained on it, but none of its
  passages appear here. Its vocabulary (words, word pairs and short runs of characters) is
  part of the model's parameters.
