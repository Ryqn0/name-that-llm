# ADR 0001: Experiment configs live inside the package

- **Status:** accepted (2026-10-24)
- **Context:** the design (spec §9.1) put experiment configs in `configs/` at the repo root.
  But the final Kaggle submission notebook runs **offline** and installs our code from a
  wheel published as a Kaggle Dataset. Files at the repo root aren't part of that wheel, so
  the notebook couldn't load `lr.yaml` by name.
- **Decision:** the named configs live in `src/name_that_llm/configs/*.yaml` and are read
  with `importlib.resources`. `load_config("lr")` loads a shipped config by name, and
  `load_config("path/to/x.yaml")` still accepts any file.
- **Consequences:** a config always travels with the exact code version it was written for,
  so a tag like v0.2.0 pins both. Changing a shipped config means releasing a new package
  version, which is good for reproducibility. One-off experiments can still use a YAML file
  anywhere, by path. Task 11 verifies that the configs are actually inside the built wheel.
