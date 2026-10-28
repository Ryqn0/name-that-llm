"""The weekly retraining's decision (P6): may it propose a new challenger for the demo?

Every week, scripts/retrain_challenger.py retrains the demo's recipe (`lr`) on our passages
plus the pool, and compares it with the champion's recipe (`lr` alone) on the same seed-42
folds. It may then open a pull request that makes the new model the demo's challenger, in
shadow mode. It never changes the site itself: a person decides by merging.

propose() is that decision, kept apart so it's tested: a candidate is proposed only if
  - it beats the champion with a margin luck can't explain: the prompt bootstrap's 95%
    interval of (candidate - champion) lies entirely above 0, as for every promotion in
    this project; and
  - it beats every model the demo already pins (the champion in MODEL_RUN and the
    challenger in CHALLENGER_RUN, by the CVs their MLflow runs logged on the same folds) by
    at least MARGIN. Retraining the same recipe on the same pool gives the same CV, give
    or take noise (0.4205 on the PC, 0.4186 on GitHub's runner): without a margin, every
    week would reopen the same proposal, and after a promotion it would propose the
    champion's own recipe as its challenger (the final review's finding 1).
"""

from collections.abc import Sequence

import numpy as np

from name_that_llm.metrics import prompt_bootstrap_ci, score

MARGIN = 0.005  # the project's tie: within 0.005, the simpler or the incumbent stays


def propose(
    y: np.ndarray,
    groups: np.ndarray,
    champion_pred: np.ndarray,
    candidate_pred: np.ndarray,
    candidate_cv: float,
    pinned_cvs: Sequence[float],
) -> tuple[bool, dict]:
    """(propose it?, the numbers for the pull request's description).

    pinned_cvs: the CVs logged by the models the demo pins now (none, one or two).
    """
    diff, lo, hi = prompt_bootstrap_ci(y, champion_pred, candidate_pred, groups)
    numbers = {
        "champion_cv": score(y, champion_pred),
        "candidate_cv": candidate_cv,
        "pinned_cvs": list(pinned_cvs),
        "diff": diff,
        "lo": lo,
        "hi": hi,
    }
    beats_recipe = lo > 0
    beats_pinned = all(candidate_cv >= cv + MARGIN for cv in pinned_cvs)
    return beats_recipe and beats_pinned, numbers


def pr_body(numbers: dict, run_id: str) -> str:
    """The pull request's description: the evidence, and what merging does."""
    pinned = numbers["pinned_cvs"]
    current = ", ".join(f"{cv:.4f}" for cv in pinned) if pinned else "none logged"
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
            f"- The models the demo pins now (champion, challenger): CV {current}. The "
            f"candidate beats each by at least {MARGIN}.",
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
