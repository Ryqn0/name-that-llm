import numpy as np

from name_that_llm.blend import best_blend_weight, blend
from name_that_llm.labels import LABELS


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
