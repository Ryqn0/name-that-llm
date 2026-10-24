import numpy as np
from sklearn.metrics import adjusted_rand_score

from name_that_llm.grouping import topic_groups


def test_topic_groups_recovers_distinct_topics():
    topics = [
        "volcano lava magma eruption",
        "piano melody chord rhythm",
        "comet orbit planet telescope",
    ]
    # Each fake passage gets its OWN unique extra word. A word shared by a few passages
    # across topics (e.g. "style10" in one passage per topic) carries a high IDF weight
    # and would outweigh the topic, which real answers to different prompts don't do.
    texts = [f"{t} answer note{t[:3]}{i}" for t in topics for i in range(18)]
    truth = np.repeat([0, 1, 2], 18)
    assert adjusted_rand_score(truth, topic_groups(texts)) == 1.0


def test_topic_groups_handles_tiny_inputs():
    assert topic_groups(["only one passage"]).tolist() == [0]
    groups = topic_groups(["a cat sat", "the dog ran", "a bird flew"])
    assert len(groups) == 3 and set(groups) == {0}  # round(3 / 18) -> a single group
