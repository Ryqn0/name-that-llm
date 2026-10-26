"""Data ingestion and cleaning: find the competition files, load them, validate them.

Everything that reads the competition data goes through `load_train`/`load_test`,
so every notebook, test and (later) the app sees the data exactly the same way.
"""

import os
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from name_that_llm.labels import LAB_OF, LABELS
from name_that_llm.schemas import TestSchema, TrainSchema

# LABELS and LAB_OF live in labels.py (to avoid a circular import with schemas.py)
# but are part of this module's public interface.
__all__ = [
    "LABELS",
    "LAB_OF",
    "find_data_dir",
    "flatten_text",
    "load_test",
    "load_train",
    "normalize_text",
]

# src/name_that_llm/data.py -> parents[2] is the repo root (only meaningful in a
# local checkout; on Kaggle the package is installed elsewhere and this path won't exist).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the data can live, in order of preference. Kaggle's mount path has changed
# over the years, so we accept both known layouts (Review Focus 4).
DEFAULT_CANDIDATES: tuple[Path, ...] = (
    Path("/kaggle/input/name-that-llm"),
    Path("/kaggle/input/competitions/name-that-llm"),
    REPO_ROOT / "data" / "raw",
)


def normalize_text(text: str) -> str:
    """The ONLY cleaning we apply: Unicode NFC, plus trimming outer whitespace.

    Why so little? In authorship attribution the "noise" is the signal. Em dashes,
    curly quotes, ellipses and spacing habits are exactly how the models differ, so
    we must not "clean" them away.

    Why NFC and not NFKC? Both merge different byte sequences for the same visible
    character (e.g. "e" + combining accent -> "é"), which is safe. But NFKC also
    rewrites "compatibility" characters: "…" becomes "...", and the non-breaking space
    becomes a normal space. Those are real style fingerprints, so NFKC would erase them.
    """
    return unicodedata.normalize("NFC", text).strip()


# Markdown that only makes sense at the START of a line (after its indentation).
_FENCE = re.compile(r"^(```|~~~)")  # a code fence line: dropped, the code itself is kept
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")  # |---|---|
_QUOTE = re.compile(r"^(>\s?)+")  # "> quote", also nested ">> quote"
_LIST_OR_HEADING = re.compile(r"^(#{1,6}|[-*+]|\d+[.)])\s+")  # "# h", "- b", "* b", "1. n"
# Markdown that can appear ANYWHERE in a line.
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")  # ![alt](url) -> alt
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) -> text


def flatten_text(text: str) -> str:
    """Turn pasted chatbot markdown into ONE plain paragraph, like the training passages.

    Why: the competition's host flattened every passage before we got it. None of the
    2,699 has a line break, a heading, `**bold**`, a backtick or a list marker. A visitor
    pastes raw chatbot output, full of exactly those. Feeding that to models that never
    saw it would be a train/serve skew: the input's FORMAT would differ from training,
    not just its content. So the app flattens first, the way the host did.

    What the training data tells us about the host's rules (P4 design §6):
      - line breaks, headings, bold, backticks, bullets and list numbers are gone;
      - single `*emphasis*` SURVIVES (14 training passages still contain it), so we keep
        single asterisks and underscores and remove only the doubled ones;
      - a number ending a sentence ("in 2023. The") is ordinary text, so list numbers
        are removed only at the start of a line.
    Contract: training-style text (one paragraph, no markdown) comes back unchanged.
    The local check in P4 Task 1 confirms it on all 2,699 real passages.
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if _FENCE.match(line) or _TABLE_RULE.match(line):
            continue  # pure syntax: nothing to keep
        line = _QUOTE.sub("", line)  # a quote can hold a list item: strip it first
        line = _LIST_OR_HEADING.sub("", line)
        lines.append(line)
    flat = " ".join(lines)
    flat = _LINK.sub(r"\1", _IMAGE.sub(r"\1", flat))
    # Bold and inline code markers go; table cell borders become word gaps.
    flat = flat.replace("**", "").replace("__", "").replace("`", "").replace("|", " ")
    # Every run of whitespace (newlines, tabs, double spaces) becomes a single space.
    return normalize_text(re.sub(r"\s+", " ", flat))


def find_data_dir(candidates: Sequence[Path] | None = None) -> Path:
    """Return the folder holding the competition CSVs.

    The NTL_DATA_DIR environment variable wins if set (handy for tests and odd
    setups). Otherwise we try each candidate and return the first one that contains
    train.csv. If none does, the error lists every path tried, so the fix is obvious.
    """
    override = os.environ.get("NTL_DATA_DIR")
    if override:
        return Path(override)
    tried = list(DEFAULT_CANDIDATES if candidates is None else candidates)
    for folder in tried:
        if (Path(folder) / "train.csv").is_file():
            return Path(folder)
    raise FileNotFoundError(
        "Competition data not found. Set NTL_DATA_DIR, or put the files in one of: "
        + ", ".join(str(p) for p in tried)
    )


def _read_csv(path: Path) -> pd.DataFrame:
    # dtype=str: without it, pandas guesses types and turns an id like "000123" into
    # the integer 123. That id then never matches Kaggle's (Review Focus 1).
    # na_filter=False: also keep strings like "NA" or "null" verbatim instead of
    # turning them into NaN. A missing field simply becomes "".
    return pd.read_csv(path, dtype=str, na_filter=False)


def load_train(data_dir: Path | None = None) -> pd.DataFrame:
    """Load train.csv (id, prompt_id, text, model): cleaned and validated."""
    folder = Path(data_dir) if data_dir is not None else find_data_dir()
    df = _read_csv(folder / "train.csv")
    df["text"] = df["text"].map(normalize_text)
    return TrainSchema.validate(df)


def load_test(data_dir: Path | None = None) -> pd.DataFrame:
    """Load test.csv (id, text): cleaned and validated. A missing text becomes ""."""
    folder = Path(data_dir) if data_dir is not None else find_data_dir()
    df = _read_csv(folder / "test.csv")
    df["text"] = df["text"].map(normalize_text)
    return TestSchema.validate(df)
