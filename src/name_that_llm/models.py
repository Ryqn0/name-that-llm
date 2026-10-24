"""Baseline models: TF-IDF features + five classic text classifiers.

Every model is a scikit-learn `Pipeline` with two named steps:
    "features": text -> sparse TF-IDF matrix (word + character n-grams)
    "clf":      matrix -> probabilities over the 18 models

Putting the vectorizer INSIDE the pipeline matters for honest validation: during
cross-validation the vocabulary and IDF weights are learned from the training folds only,
so no information leaks in from the held-out prompts.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from name_that_llm.config import ExperimentConfig


def build_vectorizer(cfg: ExperimentConfig) -> FeatureUnion:
    """Word n-grams + character n-grams, side by side in one feature matrix."""
    shared = {"sublinear_tf": True, "min_df": cfg.min_df, "max_features": cfg.max_features}
    return FeatureUnion(
        [
            # Words: vocabulary habits ("delve", "in summary", ...). Lowercased, so "The"
            # and "the" count as the same word.
            ("words", TfidfVectorizer(ngram_range=cfg.word_ngrams, lowercase=True, **shared)),
            # Characters: style that word tokens throw away. Punctuation ("—", " ; "),
            # spacing, word endings and CASING, so lowercase=False on purpose.
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char", ngram_range=cfg.char_ngrams, lowercase=False, **shared
                ),
            ),
        ]
    )


class NBSVMClassifier(ClassifierMixin, BaseEstimator):
    """NB-SVM (Wang & Manning, 2012): weight each feature by its Naive Bayes log-count ratio,
    then fit one-vs-rest logistic regressions on the reweighted features.

    Intuition: for each model c, the ratio r says how much more often a feature appears in
    c's passages than in everyone else's. Multiplying the features by r tells the linear
    model up front which features matter for c. That's why it often beats plain Naive Bayes
    and plain logistic regression on small text datasets.
    """

    def __init__(self, C=1.0, alpha=1.0):
        self.C, self.alpha = C, alpha

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.r_, self.clfs_ = [], []
        for c in self.classes_:
            pos = y == c
            # Smoothed feature totals inside class c (p) and outside it (q).
            p = self.alpha + X[pos].sum(axis=0)
            q = self.alpha + X[~pos].sum(axis=0)
            r = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
            clf = LogisticRegression(C=self.C, max_iter=2000).fit(X.multiply(r).tocsr(), pos)
            self.r_.append(r)
            self.clfs_.append(clf)
        return self

    def predict_proba(self, X):
        # Each one-vs-rest model gives P(class c | x). Normalize them to sum to 1 per row.
        s = np.column_stack(
            [
                clf.predict_proba(X.multiply(r).tocsr())[:, 1]
                for r, clf in zip(self.r_, self.clfs_, strict=True)
            ]
        )
        return s / s.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_dense_model(C: float) -> Pipeline:
    """Logistic regression on DENSE features, such as LM hidden states (P3b).

    Why scale first: hidden-state dimensions live on wildly different scales (a few take
    huge values, most are near zero). Logistic regression's L2 penalty treats every weight
    the same, so without scaling it would effectively ignore the small-scale dimensions.
    StandardScaler gives each dimension mean 0 and variance 1, learned on the training
    fold only (it sits inside the pipeline).
    """
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=2000))])


def build_model(cfg: ExperimentConfig) -> Pipeline:
    """The full baseline pipeline named by `cfg.model`. It's untrained until you call .fit()."""
    classifiers = {
        "mnb": lambda: MultinomialNB(alpha=cfg.alpha),
        "cnb": lambda: ComplementNB(alpha=cfg.alpha),
        "lr": lambda: LogisticRegression(C=cfg.C, max_iter=2000),
        # A linear SVM outputs scores, not probabilities. We need probabilities for
        # ensembling and for the per-prompt assignment step (idea I-001), so a
        # calibrator learns a score -> probability mapping (sigmoid, 3 internal folds).
        "svm": lambda: CalibratedClassifierCV(estimator=LinearSVC(C=cfg.C), cv=3, method="sigmoid"),
        "nbsvm": lambda: NBSVMClassifier(C=cfg.C, alpha=cfg.alpha),
    }
    return Pipeline([("features", build_vectorizer(cfg)), ("clf", classifiers[cfg.model]())])
