import numpy as np
import pytest

from name_that_llm.blend import best_blend_weight, best_blend_weights, blend, nested_blend_oof
from name_that_llm.labels import LABELS
from name_that_llm.metrics import score


def _random_problem(n_models, seed=0, n=200):
    """Random labels plus `n_models` random probability matrices, all unrelated."""
    rng = np.random.default_rng(seed)
    return rng.choice(LABELS, n), [rng.dirichlet(np.ones(18), n) for _ in range(n_models)]


def test_best_blend_weights_matches_the_two_model_search():
    # With 2 models, the N-model search must return exactly the 2-model search's answer,
    # tie rule included (the smaller weight on the newer model wins).
    y, (a, b) = _random_problem(2)
    w, s = best_blend_weight(a, b, y)
    assert best_blend_weights([a, b], y) == ((round(1 - w, 4), w), s)


def test_best_blend_weights_three_models_returns_a_consistent_point():
    y, probas = _random_problem(3)
    w, s = best_blend_weights(probas, y)
    assert len(w) == 3 and min(w) >= 0 and abs(sum(w) - 1) < 1e-9
    labels = np.asarray(LABELS)
    assert s == score(y, labels[blend(probas, w).argmax(axis=1)])  # the score is w's score
    for corner in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # the grid includes each model alone
        assert s >= score(y, labels[blend(probas, corner).argmax(axis=1)])


def test_best_blend_weights_finds_an_informative_third_model():
    y, (a, b) = _random_problem(2)
    perfect = np.eye(18)[[LABELS.index(v) for v in y]]
    w, s = best_blend_weights([a, b, perfect], y)
    assert s == 1.0 and w[2] > 0


def test_best_blend_weights_gives_a_redundant_newcomer_no_weight():
    # Models 2 and 3 are equally perfect, so many weightings tie at 1.0. The tie rule must
    # keep the newcomer (the LAST model) at 0, because an earlier model already does its job.
    y, (a,) = _random_problem(1)
    perfect = np.eye(18)[[LABELS.index(v) for v in y]]
    w, s = best_blend_weights([a, perfect, perfect.copy()], y)
    assert s == 1.0 and w[1] > 0 and w[2] == 0.0


def test_best_blend_weights_rejects_a_step_that_does_not_divide_one():
    y, probas = _random_problem(2)
    with pytest.raises(ValueError):
        best_blend_weights(probas, y, step=0.3)


def test_blend_normalizes_weights_and_rows_sum_to_one():
    a, b = np.full((2, 18), 1 / 18), np.eye(18)[:2]
    out = blend([a, b], [2, 2])
    assert np.allclose(out.sum(axis=1), 1.0) and np.allclose(out, (a + b) / 2)


def test_best_blend_weight_prefers_the_informative_source():
    y = np.array(LABELS * 3)
    good = np.eye(18)[np.tile(np.arange(18), 3)] * 0.9 + 0.1 / 18
    bad = np.full_like(good, 1 / 18)
    weight, score = best_blend_weight(bad, good, y)
    assert weight > 0 and score == 1.0
    assert best_blend_weight(good, bad, y)[0] == 0.0  # tie -> the smaller weight


def test_log_blend_is_a_normalized_geometric_mean():
    # I-015 (P7 Task 7): the log form is a weighted geometric mean, renormalized. A member
    # sure that a label is wrong then vetoes it, where the plain mean only dilutes it.
    p1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    p2 = np.array([[0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
    out = blend([p1, p2], [0.5, 0.5], log=True)
    expected = np.sqrt(p1 * p2)
    np.testing.assert_allclose(out, expected / expected.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def _members(seed=0):
    """Two random members on 3 folds of 36 rows, each fold holding every label twice: a
    fold missing a label would make balanced accuracy warn about it."""
    rng = np.random.default_rng(seed)
    y = np.array(LABELS * 6)
    probas = [rng.dirichlet(np.ones(18), size=len(y)) for _ in range(2)]
    return probas, y, np.repeat([0, 1, 2], 36)


def test_nested_blend_oof_chooses_each_folds_weights_elsewhere():
    # P8 Task 3: a member that exists on one seed only gets its weights by nested CV. Each
    # fold's weights must come from the OTHER folds' rows only, or its score is flattered.
    probas, y, folds = _members()
    _, weights = nested_blend_oof(probas, y, folds)
    for k in (0, 1, 2):
        rest = folds != k
        assert weights[k] == best_blend_weights([p[rest] for p in probas], y[rest])[0]


def test_nested_blend_oof_leaves_rows_outside_untouched():
    # Rows whose fold isn't done (e.g. an LLM fold still training) get no prediction.
    probas, y, folds = _members()
    out, weights = nested_blend_oof(probas, y, folds, rows=folds < 2)
    assert set(weights) == {0, 1}
    assert not out[folds == 2].any() and out[folds < 2].sum(axis=1) == pytest.approx(1.0)


def test_best_blend_weights_can_search_the_log_blend():
    # P7 Task 7 (I-015): the log blend's weights must be chosen FOR it, since a member's
    # confidence counts differently there; the best probability weights needn't be its best.
    y, probas = _random_problem(3)
    w, s = best_blend_weights(probas, y, log=True)
    assert s == score(y, np.asarray(LABELS)[blend(probas, w, log=True).argmax(axis=1)])


def test_nested_blend_oof_can_blend_in_log_space():
    probas, y, folds = _members()
    out, weights = nested_blend_oof(probas, y, folds, log=True)
    for k in (0, 1, 2):
        rest, here = folds != k, folds == k
        fit = best_blend_weights([p[rest] for p in probas], y[rest], log=True)[0]
        assert weights[k] == fit
        np.testing.assert_allclose(out[here], blend([p[here] for p in probas], fit, log=True))
