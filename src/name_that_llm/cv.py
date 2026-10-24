"""Backtesting splits: which rows are held out in which validation fold.

The hidden test set contains ONLY prompts that never appear in train.csv. To estimate
honestly how we'll score there, validation must copy that situation: every prompt goes
ENTIRELY into one fold, so a model is always scored on prompts it never saw. A random
row split would put some answers to a prompt in training and others in validation. The
model would then partly recognize the TOPIC instead of the author, and the CV score would
be misleading.
"""

import numpy as np
import pandas as pd


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.Series:
    """Assign each row a fold number in 0..n_splits-1, keeping every prompt in one fold.

    Deterministic: the same seed always gives the same folds. Every experiment therefore
    scores on IDENTICAL splits, which is what makes comparing two models (an A/B test)
    fair.

    We don't use scikit-learn's GroupKFold(shuffle=True): that option only exists in
    scikit-learn >= 1.6, and Kaggle's image may be older. This version is just as simple.
    """
    # Sort first, so the result doesn't depend on the order rows happen to appear in.
    prompts = np.sort(df["prompt_id"].unique())
    shuffled = np.random.default_rng(seed).permutation(prompts)
    # Deal the shuffled prompts out like cards: 0, 1, 2, 3, 4, 0, 1, ... so fold sizes
    # differ by at most one prompt (150 prompts -> 30 per fold).
    fold_of_prompt = {prompt: i % n_splits for i, prompt in enumerate(shuffled)}
    return df["prompt_id"].map(fold_of_prompt).astype(int).rename("fold")
