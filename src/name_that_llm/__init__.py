"""name_that_llm: shared code for the Kaggle "Name That LLM" competition project.

One package, used everywhere: local tests, Kaggle notebooks (installed from a git
tag, or from an offline wheel), and later the demo app. Training and serving run
the same code, so they can't silently disagree ("training/serving skew").
"""

import importlib.metadata

# Read the version from the installed package metadata instead of hard-coding it
# here. pyproject.toml then stays the single source of truth, and `uv version --bump`
# updates everything at once.
__version__ = importlib.metadata.version("name-that-llm")
