import numpy as np
import pytest

from name_that_llm.config import load_config
from name_that_llm.models import build_model, build_vectorizer


def _cfg(name):
    # min_df=1: the fake dataset is tiny, so keep every n-gram it has.
    return load_config(name).model_copy(update={"min_df": 1})


@pytest.mark.parametrize("name", ["mnb", "cnb", "lr", "svm", "nbsvm"])
def test_every_baseline_predicts_valid_probabilities(name, fake_train):
    model = build_model(_cfg(name)).fit(fake_train["text"], fake_train["model"])
    proba = model.predict_proba(fake_train["text"])
    assert proba.shape == (len(fake_train), 18)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_char_features_keep_case():
    X = build_vectorizer(_cfg("lr")).fit_transform(["Hello World", "hello world"]).toarray()
    assert not np.allclose(X[0], X[1])


def test_nbsvm_learns_marker_tokens(fake_train):
    model = build_model(_cfg("nbsvm")).fit(fake_train["text"], fake_train["model"])
    assert (model.predict(fake_train["text"]) == fake_train["model"]).mean() >= 0.9
