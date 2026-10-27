"""The pool of generated passages (P7): extra training data, made safe to use.

Why: we have 2,699 passages. Participants have generated about 250k more through the 18
models' APIs and shared them on Kaggle, and more examples are what the fine distinctions
need (P7's error analysis: most misses are between sibling models).

The safeguards of the design (§3) live here:
  - licenses: only an explicit open license (license_allowed);
  - the competition's shape: flattened like the training passages, 30 to 320 words;
  - no copies: pool passages that duplicate ours are removed, and so is every text that
    two sources label differently (one of them is wrong, and we can't tell which);
  - no leaks: each pool passage is matched to the training PROMPT it's about, by topic,
    and pool_for_fold keeps a fold's validation prompts out of its training pool. Without
    that, a pool answer to the same question as a validation passage would teach the
    model the topic and inflate the score: P3a's trap.

Torch-free (scikit-learn and pandas): the GPU notebooks rebuild the pool from the attached
public datasets with this same code, so we never publish the data ourselves.
"""

import re
from collections.abc import Collection, Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from name_that_llm.data import flatten_text, normalize_text
from name_that_llm.labels import LABELS

MIN_WORDS, MAX_WORDS = 30, 320  # a margin around the training passages' 36 to 274 words
# Kaggle's short license ids, once lowercased and dashed: CC0, CC BY, CC BY-SA, Apache, MIT.
# NonCommercial (cc-by-nc-*) and NoDerivatives (cc-by-nd-*) don't match.
_OPEN_LICENSE = re.compile(r"cc0.*|cc-by-\d.*|cc-by-sa-.*|apache-2\.0|mit")
_LABEL_SET = set(LABELS)


class PoolSource(BaseModel):
    """One public Kaggle dataset of generated passages, and where its columns are."""

    model_config = ConfigDict(extra="forbid")

    slug: str  # "owner/dataset"
    files: str  # a glob for its data files, e.g. "*.parquet"
    text: str  # the column holding the passage
    model: str  # the column holding the model's name
    # A merged pool credits each row to its original dataset in this column; per-source
    # quality checks then judge the original datasets (P7 Task 3). None: the slug.
    source_column: str | None = None
    keep: bool = True  # False once a quality check shows the source hurts
    # The original datasets (values of source_column) to keep, from P7 Task 3's check;
    # None keeps them all. An allowlist, so a dataset added in a later version of the pool
    # stays out until it's checked (P6's weekly retraining downloads the latest version).
    include: list[str] | None = None


def load_pool_sources(name: str = "pool") -> list[PoolSource]:
    """The pool's sources, from the `sources` list of a YAML shipped in the package."""
    text = (files("name_that_llm") / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
    return [PoolSource(**source) for source in yaml.safe_load(text)["sources"]]


def license_allowed(name: str) -> bool:
    """True for an explicit open license: CC0, CC BY, CC BY-SA, Apache 2.0 or MIT.

    Kaggle reports most licenses by a long name ("Attribution 4.0 International (CC BY
    4.0)"): the short form in the parentheses is the one compared.
    """
    text = (name or "").strip()
    inner = re.search(r"\(([^)]*)\)", text)
    if inner:
        text = inner.group(1)
    key = re.sub(r"[\s_:]+", "-", text.lower()).strip("-")
    return bool(_OPEN_LICENSE.fullmatch(key))


def normalize_label(raw) -> str | None:
    """One of our 18 labels, or None: "openai/gpt-5.5" -> "gpt-5.5"; unknown -> None.

    Never guessed: a name that isn't exactly one of ours (once lowercased, without a
    vendor prefix) is dropped, since a wrong label would teach the model the wrong author.
    """
    name = str(raw).strip().lower().rsplit("/", 1)[-1]
    name = re.sub(r"[\s_]+", "-", name)
    return name if name in _LABEL_SET else None


def clean_pool(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """In the competition's shape: (text, model, source), flattened, 30 to 320 words."""
    text = df["text"].fillna("").astype(str).map(flatten_text)
    model = df["model"].map(normalize_label)
    n_words = text.str.split().str.len()
    keep = model.notna() & n_words.between(MIN_WORDS, MAX_WORDS)
    out = pd.DataFrame({"text": text[keep], "model": model[keep], "source": source})
    return out.reset_index(drop=True)


def _key(text: str) -> str:
    """What makes two passages "the same": equal once flattened and NFC-normalized."""
    return normalize_text(flatten_text(str(text)))


def dedupe(pool: pd.DataFrame, train_texts: Sequence[str]) -> pd.DataFrame:
    """Drop copies of our passages, texts with conflicting labels, and repeats (keep one)."""
    keys = pool["text"].map(_key)
    train_keys = {_key(t) for t in train_texts}
    labels_per_key = pool.groupby(keys)["model"].nunique()
    conflicted = set(labels_per_key.index[labels_per_key > 1])
    keep = ~keys.isin(train_keys) & ~keys.isin(conflicted) & ~keys.duplicated()
    return pool[keep].reset_index(drop=True)


def _topic_space(train_texts: Sequence[str]):
    """TF-IDF on CONTENT words (no stop words), as grouping.py does: topic, not style.

    Returns the vectorizer fitted on our passages, and their rows (unit length).
    """
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X = vectorizer.fit_transform([str(t) for t in train_texts])
    return vectorizer, X


def _membership(prompt_ids: Sequence[str]) -> tuple[np.ndarray, sp.csr_matrix]:
    """The prompt ids, and a (n_prompts, n_rows) 0/1 matrix: which rows answer which prompt."""
    prompts, rows = np.unique(np.asarray(prompt_ids, dtype=str), return_inverse=True)
    n = len(rows)
    G = sp.csr_matrix((np.ones(n), (rows, np.arange(n))), shape=(len(prompts), n))
    return prompts, G


def calibrate_match_threshold(
    train_texts: Sequence[str], train_prompt_ids: Sequence[str], quantile: float = 0.01
) -> float:
    """The topic similarity that `quantile` of our own passages fall below.

    Each training passage is compared with its own prompt's centroid computed WITHOUT it
    (leave-one-out): counting itself would flatter the match. With quantile 0.01, 99% of
    genuine answers to a prompt clear the bar, so few pool answers to a training prompt
    slip through unmatched.
    """
    _, X = _topic_space(train_texts)
    prompts, G = _membership(train_prompt_ids)
    S = G @ X  # each prompt's SUM of row vectors
    own = np.asarray(G.argmax(axis=0)).ravel()  # each row's prompt index
    dots = np.asarray(X.multiply(S[own]).sum(axis=1)).ravel()  # x . S_p
    sq = np.asarray(S.multiply(S).sum(axis=1)).ravel()[own]  # |S_p|^2
    sizes = np.asarray(G.sum(axis=1)).ravel()[own]
    # Rows are unit vectors, so (S_p - x) . x = x . S_p - 1, |S_p - x|^2 = |S_p|^2 - 2 x . S_p + 1
    loo_norm = np.sqrt(np.maximum(sq - 2 * dots + 1, 0.0))
    ok = (sizes > 1) & (loo_norm > 0)
    sims = (dots[ok] - 1) / loo_norm[ok]
    return float(np.quantile(sims, quantile))


def match_prompts(
    pool_texts: Sequence[str],
    train_texts: Sequence[str],
    train_prompt_ids: Sequence[str],
    threshold: float,
    chunk: int = 20_000,
) -> np.ndarray:
    """For each pool passage, the training prompt it's about (by topic), or None.

    The best-matching prompt centroid wins if its cosine similarity reaches `threshold`.
    Pool rows are compared in chunks, so 250k passages fit in memory.
    """
    vectorizer, X = _topic_space(train_texts)
    prompts, G = _membership(train_prompt_ids)
    C = normalize(G @ X)  # unit centroids, one per prompt
    out = np.empty(len(pool_texts), dtype=object)
    texts = [str(t) for t in pool_texts]
    for start in range(0, len(texts), chunk):
        sims = (vectorizer.transform(texts[start : start + chunk]) @ C.T).toarray()
        best = sims.argmax(axis=1)
        hit = sims[np.arange(len(best)), best] >= threshold
        out[start : start + len(best)] = np.where(hit, prompts[best], None)
    return out


def pool_for_fold(
    pool: pd.DataFrame, valid_prompt_ids: Collection[str], max_rows: int | None, seed: int
) -> pd.DataFrame:
    """The pool rows a fold may train on: none matched to its validation prompts.

    Unmatched rows (prompt_id null) are new topics, so always allowed. With max_rows, a
    seeded sample of that size, in the pool's original order (0 rows: pool-free).
    """
    allowed = pool[~pool["prompt_id"].isin(set(valid_prompt_ids))]
    if max_rows is not None and len(allowed) > max_rows:
        rows = np.random.default_rng(seed).choice(len(allowed), size=max_rows, replace=False)
        allowed = allowed.iloc[np.sort(rows)]
    return allowed.reset_index(drop=True)


def _source_files(root: Path, source: PoolSource) -> list[Path]:
    """The source's data files under `root`, inside a folder named after the dataset.

    Works for our downloads (data/external/<owner>__<name>/) and for Kaggle's mounts
    (/kaggle/input/<name>/ or /kaggle/input/datasets/<owner>/<name>/).
    """
    name = source.slug.split("/")[-1]
    return [
        p
        for p in sorted(root.rglob(source.files))
        if any(name in part for part in p.relative_to(root).parts[:-1])
    ]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a data file with pandas only (downloads are untrusted: never executed)."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)[columns]
    raise ValueError(f"unsupported pool file: {path.name}")


def build_pool(sources: Sequence[PoolSource], root: Path, train: pd.DataFrame) -> pd.DataFrame:
    """The training pool: (text, model, source, prompt_id), cleaned, deduped and matched."""
    root, frames, kept = Path(root), [], set()
    for source in sources:
        if not source.keep:
            continue
        paths = _source_files(root, source)
        if not paths:
            raise FileNotFoundError(f"no {source.files} for {source.slug} under {root}")
        columns = [source.text, source.model] + (
            [source.source_column] if source.source_column else []
        )
        raw = pd.concat([_read(p, columns) for p in paths], ignore_index=True)
        raw = raw.rename(columns={source.text: "text", source.model: "model"})
        if source.source_column:
            for name, part in raw.groupby(source.source_column, sort=True):
                frames.append(clean_pool(part, str(name)))
                if source.include is None or str(name) in source.include:
                    kept.add(str(name))
        else:
            frames.append(clean_pool(raw, source.slug))
            kept.add(source.slug)
    # Copies and label conflicts are judged across EVERY source, before the allowlist: a
    # text two sources label differently has a wrong label somewhere, even when one of the
    # two isn't kept. (It's also exactly the pool P7 Task 3's check measured.)
    pool = dedupe(pd.concat(frames, ignore_index=True), train["text"])
    pool = pool[pool["source"].isin(kept)].reset_index(drop=True)
    threshold = calibrate_match_threshold(train["text"], train["prompt_id"])
    pool["prompt_id"] = match_prompts(pool["text"], train["text"], train["prompt_id"], threshold)
    return pool[["text", "model", "source", "prompt_id"]]
