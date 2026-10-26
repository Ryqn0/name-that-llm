"""The 18 candidate models (the classes) and the lab that built each one.

This module has no dependencies on purpose. Both `data` (loading) and `schemas`
(validation) need the labels, and keeping them here avoids a circular import
between those two. Other code imports them via `name_that_llm.data`.
"""

# Sorted alphabetically. scikit-learn orders `model.classes_` the same way, so
# probability columns line up with this list by default (but see train.run_cv, which
# still maps columns explicitly instead of trusting that).
LABELS: list[str] = sorted(
    [
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "glm-5.3",
        "glm-5.3-flash",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
        "kimi-k3",
        "qwen3.8-max",
    ]
)

# Each model name starts with its family prefix, which identifies the lab. "Lab
# accuracy" (right lab, possibly wrong model) is a useful second metric: it shows
# whether mistakes stay within a family (e.g. gpt-5.6-luna vs gpt-5.6-sol) or are wild.
_LAB_BY_PREFIX = {
    "claude": "anthropic",
    "gemini": "google",
    "glm": "zai",
    "gpt": "openai",
    "kimi": "moonshot",
    "qwen": "qwen",
}

LAB_OF: dict[str, str] = {
    model: next(lab for prefix, lab in _LAB_BY_PREFIX.items() if model.startswith(prefix))
    for model in LABELS
}

# How pages name each lab (LAB_OF uses short lowercase keys). Shared by the web app (api.py)
# and the browser demo's export (browser.py), so both pages say the same thing.
LAB_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "google": "Google",
    "zai": "Z.ai",
    "openai": "OpenAI",
    "moonshot": "Moonshot",
    "qwen": "Qwen",
}
