"""Does centering help the ensemble? (P8 Task 7, spec §5.5). Run by hand, on CPU.

Centering takes each topic group's average signal out of its passages, so the topic no longer
counts (blend.center). Prompt tricks have fooled us before: P3a's assignment gained +0.084 in
CV at 18 answers per prompt and nothing on the leaderboard, because the hidden test isn't laid
out like train. So this scores the blend with and without centering on subsampled layouts: m
random answers to each prompt, for m in 1, 2, 3, 6 and 18. The groups come from each layout's
texts alone (grouping.topic_groups), at its default size, since the submission can't know m.

It starts from the blend's nested OOF, so no weight saw the rows it's scored on. A layout counts
as a gain only if the mean over its draws is 2 standard errors above 0. m = 18 is every row, so
one draw, where any gain counts. An alpha is kept only if it gains at every m.
Prints numbers only, never a passage.

Usage (repo root, with the OOF from ensemble_gate.py's --save-oof):
  python -I -X utf8 scripts/centering_check.py data/runs/oof-ensemble-p8-seed42.npy [--draws 300]
"""

import argparse
from pathlib import Path

import numpy as np

from name_that_llm.blend import center
from name_that_llm.data import LABELS, load_train
from name_that_llm.grouping import topic_groups
from name_that_llm.metrics import score

ALPHAS = (0.25, 0.5, 0.75)
LAYOUTS = (1, 2, 3, 6, 18)
LABEL_ARRAY = np.array(LABELS)


def draw_layout(prompts: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """The row numbers of m random answers to each prompt, or all of them if it has fewer."""
    picks = [rng.permutation(np.flatnonzero(prompts == p))[:m] for p in np.unique(prompts)]
    return np.sort(np.concatenate(picks))


def acc(y: np.ndarray, proba: np.ndarray) -> float:
    return score(y, LABEL_ARRAY[proba.argmax(axis=1)])


def layout_gains(oof, y, prompts, texts, m: int, draws: int) -> tuple[dict, int, float]:
    """Each alpha's gains over no centering, one per draw; the rows; the mean score without."""
    gains, bases = {a: [] for a in ALPHAS}, []
    for i in range(1 if m == max(LAYOUTS) else draws):
        rows = draw_layout(prompts, m, np.random.default_rng(i))
        groups = topic_groups(texts[rows].tolist())
        bases.append(acc(y[rows], oof[rows]))
        for a in ALPHAS:
            gains[a].append(acc(y[rows], center(oof[rows], groups, a)) - bases[-1])
    return {a: np.array(g) for a, g in gains.items()}, len(rows), float(np.mean(bases))


def standard_error(gains: np.ndarray) -> float:
    return float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else 0.0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("oof", type=Path, help="the blend's nested (n, 18) OOF")
    # 300: 30 draws showed +0.0026 at m = 1, where 300 show about 0. A layout of 150 rows is noisy.
    parser.add_argument("--draws", type=int, default=300, help="random layouts for each m < 18")
    args = parser.parse_args(argv)

    train = load_train()
    y, prompts = train["model"].to_numpy(), train["prompt_id"].to_numpy()
    texts = train["text"].astype(str).to_numpy()
    oof = np.load(args.oof)
    if oof.shape != (len(train), len(LABELS)):
        raise SystemExit(f"STOP: the OOF is {oof.shape}, not ({len(train)}, 18)")

    print(f"the blend on every row: {acc(y, oof):.4f}", flush=True)
    kept = dict.fromkeys(ALPHAS, True)
    for m in LAYOUTS:
        gains, n, base = layout_gains(oof, y, prompts, texts, m, args.draws)
        cells = []
        for a, g in gains.items():
            se = standard_error(g)
            kept[a] &= g.mean() - 2 * se > 0
            cells.append(f"alpha {a}: {g.mean():+.4f} (se {se:.4f})")
        print(f"m = {m:>2}, {n:,} rows, base {base:.4f} | " + " | ".join(cells), flush=True)

    good = [a for a, ok in kept.items() if ok]
    print(f"VERDICT: {f'KEEP, alpha in {good}' if good else 'DROP'} (a gain at every m)")


if __name__ == "__main__":
    main()
