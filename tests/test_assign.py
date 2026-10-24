import numpy as np

from name_that_llm.assign import assign_labels
from name_that_llm.labels import LABELS


def _proba(true_idx, confusions=()):
    """Rows mostly confident in their true label. Each (row, other) pair moves extra mass
    onto a wrong label, so plain argmax picks `other` for that row."""
    p = np.full((len(true_idx), 18), 0.01)
    p[np.arange(len(true_idx)), true_idx] = 0.5
    for row, other in confusions:
        p[row, other] = 0.6
    return p / p.sum(axis=1, keepdims=True)


def test_assign_fixes_duplicate_argmax_within_a_prompt():
    p = _proba(np.arange(18), confusions=[(0, 1)])  # row 0's argmax is label 1, which row 1 owns
    out = assign_labels(p, np.zeros(18, dtype=int))
    assert list(out) == LABELS


def test_assign_keeps_row_order():
    order = np.random.default_rng(0).permutation(18)
    out = assign_labels(_proba(order), np.zeros(18, dtype=int))
    assert list(out) == [LABELS[i] for i in order]


def test_assign_allows_repeats_in_oversized_groups():
    out = assign_labels(_proba(np.arange(20) % 18), np.zeros(20, dtype=int))
    assert len(out) == 20 and max(list(out).count(lab) for lab in LABELS) <= 2


def test_assign_handles_zero_probabilities():
    p = _proba(np.arange(18))
    p[:, 5] = 0.0
    p = p / p.sum(axis=1, keepdims=True)
    out = assign_labels(p, np.zeros(18, dtype=int))
    assert len(set(out)) == 18  # still a valid one-of-each assignment


def test_assign_single_row_group_is_argmax():
    p = _proba(np.array([7]))
    assert assign_labels(p, np.array([3])).tolist() == [LABELS[7]]
