# ADR 0002: LM features without making torch a package dependency

- **Status:** accepted (2026-10-24, P3b)
- **Context:**
  - The language-model features (`name_that_llm.lm`) need torch and transformers.
  - Both are large: PyPI's Linux torch wheel ships CUDA and weighs gigabytes.
  - Kaggle's GPU image already provides both, and the offline submission notebook can't
    install anything anyway.
  - The text-only path (`train`, `submit`, `blend`) never needs them.
- **Decision:**
  - torch and transformers are **not** declared in `pyproject.toml`, not even as an
    optional extra. `lm.py` imports them inside the functions that use them.
  - The model tests in `tests/test_lm.py` use a tiny random Qwen 3 model built offline,
    with `pytest.importorskip`, so CI skips them. Locally they run with
    `uv run --all-extras --with torch --with transformers pytest tests/test_lm.py`.
- **Consequences:**
  - CI stays fast and light.
  - The LM code is tested on the developer's machine, not in CI. The real integration check
    is the Kaggle extraction notebook, which asserts the transformers version it needs
    (>= 4.51, for Qwen 3).
  - If the LM code grows, add a CI job that installs CPU-only torch from
    `https://download.pytorch.org/whl/cpu` and runs `tests/test_lm.py`.
