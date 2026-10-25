import numpy as np
import pytest

from name_that_llm.blend import best_blend_weight, best_blend_weights, blend
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
