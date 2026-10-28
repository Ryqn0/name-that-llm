"""The weekly retraining's decision (P6): may it propose a new challenger for the demo?

Every week, scripts/retrain_challenger.py retrains the demo's recipe (`lr`) on our passages
plus the pool, and compares it with the champion's recipe (`lr` alone) on the same seed-42
folds. It may then open a pull request that makes the new model the demo's challenger, in
shadow mode. It never changes the site itself: a person decides by merging.

propose() is that decision, kept apart so it's tested: a candidate is proposed only if
  - it beats the champion with a margin luck can't explain: the prompt bootstrap's 95%
    interval of (candidate - champion) lies entirely above 0, as for every promotion in
    this project; and
  - it isn't worse than the current challenger, if there is one (their logged CVs come
    from the same folds), so a weaker model never replaces a stronger one.
"""

import numpy as np

from name_that_llm.metrics import prompt_bootstrap_ci, score


def propose(
    y: np.ndarray,
    groups: np.ndarray,
    champion_pred: np.ndarray,
    candidate_pred: np.ndarray,
    candidate_cv: float,
    challenger_cv: float | None,
) -> tuple[bool, dict]:
    """(propose it?, the numbers for the pull request's description)."""
    diff, lo, hi = prompt_bootstrap_ci(y, champion_pred, candidate_pred, groups)
    numbers = {
        "champion_cv": score(y, champion_pred),
        "candidate_cv": candidate_cv,
        "challenger_cv": challenger_cv,
        "diff": diff,
        "lo": lo,
        "hi": hi,
    }
    beats_champion = lo > 0
    beats_challenger = challenger_cv is None or candidate_cv >= challenger_cv
    return beats_champion and beats_challenger, numbers


def pr_body(numbers: dict, run_id: str) -> str:
    """The pull request's description: the evidence, and what merging does."""
    challenger = numbers["challenger_cv"]
    current = "none yet" if challenger is None else f"CV {challenger:.4f}"
    return "\n".join(
        [
            "## A new challenger for the demo (weekly retraining, P6)",
            "",
            "Balanced accuracy on our seed-42 prompt-grouped folds (scores on our passages only):",
            "",
            "| model | CV |",
            "|---|---|",
            f"| the champion's recipe (`lr`) | {numbers['champion_cv']:.4f} |",
            f"| **the candidate** (`lr` + the pool) | **{numbers['candidate_cv']:.4f}** |",
            "",
            f"- The candidate minus the champion: {numbers['diff']:+.4f}, prompt bootstrap 95% "
            f"interval [{numbers['lo']:.4f}, {numbers['hi']:.4f}], entirely above 0.",
            f"- The current challenger: {current}.",
            f"- The candidate's files: MLflow run `{run_id}` (experiment `browser-demo`), with "
            "parity checked on 50 passages under Node.",
            "",
            "**Merging** sets `app/demo/CHALLENGER_RUN`, and the deploy then serves this model "
            "in shadow mode: it scores visitors' texts silently, and the dashboard compares it "
            "with the champion on the same texts. Visitors still see the champion's answer. "
            "Promoting it to champion is a separate pull request, decided by a person.",
            "",
            "🤖 Opened by the weekly retraining workflow (`retrain.yml`).",
        ]
    )
