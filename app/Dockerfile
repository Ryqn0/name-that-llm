# The live app's image (P4): FastAPI serving the champion, on a free Hugging Face Space.
#
# Built from the repo root (CI: `docker build -f app/Dockerfile .`), or from the Space's own
# repo, where the deploy (scripts/deploy_space.py) puts this file at the root next to the
# same files. Both contexts hold the same paths, so the COPY lines work in both.
FROM python:3.13-slim

# uv, the same tool and version as on our machines, copied from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

# Hugging Face runs Docker Spaces as user 1000. Create that user, with a home where the
# Hugging Face cache (Qwen's weights, downloaded at startup) can be written.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TORCH_THREADS=2 \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /home/user/app

# Only the named files go in: never data/ (the competition rules forbid redistributing it;
# CI checks that no data file reaches the image), nor tests or notebooks.
COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
# --frozen: exactly the locked versions. --no-editable: install the package, don't link it.
RUN uv sync --frozen --no-dev --extra serve --no-editable

# torch and transformers aren't package dependencies (ADR 0002). Install Kaggle's
# versions, the ones the champion was trained and checked with, and torch's CPU build:
# the Space has no GPU, and the default wheel carries ~2.5 GB of CUDA it would never use.
RUN uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install transformers==5.16.1

# 7860 is the port Hugging Face Spaces expect (app_port in the Space's README).
EXPOSE 7860
CMD ["uvicorn", "name_that_llm.serve:app", "--host", "0.0.0.0", "--port", "7860"]
