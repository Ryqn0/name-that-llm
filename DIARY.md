# Progress diary

Newest entry first. Each entry follows the same format: **Did / Learned / Scores / Next**.

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
