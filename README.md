# Name That LLM: an end-to-end MLOps project

![CI](https://github.com/Ryqn0/name-that-llm/actions/workflows/ci.yml/badge.svg)

Which of 18 frontier LLMs wrote this paragraph? This is a learning-in-public project for the
Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm). It
builds a real, production-style ML system one stage at a time, with every step explained:

- EDA, data ingestion and validation
- feature engineering, training and backtesting
- A/B testing
- batch and real-time inference
- CI/CD, monitoring, and a live demo app

**Links**
- **Design:** [docs/superpowers/specs/2026-10-23-name-that-llm-design.md](docs/superpowers/specs/2026-10-23-name-that-llm-design.md)
- **Progress diary:** [DIARY.md](DIARY.md). **Ideas backlog:** [IDEAS.md](IDEAS.md)
- **Current status:** [Context.md](Context.md)

## Quickstart

```bash
uv sync --all-extras            # create the environment (Python 3.11)
uv run --all-extras pytest -q   # run the tests (they use a tiny fake dataset)
```

On Windows with Avast HTTPS scanning, run `scripts/setup-local.ps1` first.

## Data

The competition data is **not** included, because the rules forbid redistributing it.
Accept the rules on the competition page, then run:

```bash
kaggle competitions download name-that-llm -p data/raw
```

## License

MIT, see [LICENSE](LICENSE).

*Not affiliated with Kaggle or Google. This is not a general AI-text detector: it only
tells apart the 18 models in this competition.*
