import numpy as np
import pandas as pd

from name_that_llm.cv import make_folds


def _prompts(n):
    return pd.DataFrame({"prompt_id": np.repeat([f"p{i}" for i in range(n)], 2)})


def test_make_folds_never_splits_a_prompt(fake_train):
    folds = make_folds(fake_train, n_splits=3, seed=42)
    assert (folds.groupby(fake_train["prompt_id"]).nunique() == 1).all()


def test_make_folds_is_deterministic_and_seeded():
    df = _prompts(30)
    assert make_folds(df, 5, 42).equals(make_folds(df, 5, 42))
    assert not make_folds(df, 5, 42).equals(make_folds(df, 5, 7))


def test_make_folds_balances_prompts():
    df = _prompts(30)
    assert df.groupby(make_folds(df, 5, 42))["prompt_id"].nunique().tolist() == [6] * 5
