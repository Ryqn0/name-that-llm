"""Tests for Sinkhorn class balancing (P7 Task 1).

The leaders' notebooks rebalance their final probabilities so that each of the 18 models
gets an equal share of the test set. These tests pin what that operation must do: equalize
the class mass at full strength, change nothing at strength 0, move part of the way in
between, take picks away from an over-predicted class, and stay finite when a class gets
no probability at all.
"""

import numpy as np
import pytest

from name_that_llm.balance import sinkhorn_balance


def test_full_balancing_equalizes_class_mass():
    p = np.random.default_rng(0).dirichlet(np.r_[8.0, np.ones(17)], size=180)
    b = sinkhorn_balance(p, alpha=1.0)
    np.testing.assert_allclose(b.sum(axis=0), 10.0, atol=1e-3)
    np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-9)


def test_alpha_zero_changes_nothing():
    p = np.random.default_rng(1).dirichlet(np.ones(18), size=50)
    np.testing.assert_allclose(sinkhorn_balance(p, alpha=0.0), p, atol=1e-12)


def test_partial_balancing_moves_part_way():
    p = np.random.default_rng(2).dirichlet(np.r_[8.0, np.ones(17)], size=180)
    spread = [np.ptp(sinkhorn_balance(p, a).sum(axis=0)) for a in (0.0, 0.5, 1.0)]
    assert spread[0] > spread[1] > spread[2]


def test_balancing_moves_picks_away_from_an_over_predicted_class():
    # 18 true classes, 10 rows each; class 0 gets an extra 0.35 everywhere, so before
    # balancing it wins every row.
    y = np.repeat(np.arange(18), 10)
    p = np.full((180, 18), 0.02)
    p[np.arange(180), y] += 0.3
    p[:, 0] += 0.35
    p /= p.sum(axis=1, keepdims=True)
    before = (p.argmax(1) == y).mean()
    after = (sinkhorn_balance(p).argmax(1) == y).mean()
    assert after > before


def test_zero_columns_stay_finite():
    p = np.random.default_rng(3).dirichlet(np.ones(18), size=40)
    p[:, 5] = 0
    p /= p.sum(axis=1, keepdims=True)
    assert np.isfinite(sinkhorn_balance(p)).all()


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_outside_0_1_is_refused(alpha):
    p = np.full((4, 18), 1 / 18)
    with pytest.raises(ValueError, match="alpha"):
        sinkhorn_balance(p, alpha=alpha)
