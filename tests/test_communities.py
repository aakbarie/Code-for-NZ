import numpy as np

from morrill.analysis.communities import order_communities


def test_communities_renumbered_by_key_with_missing_last():
    labels = np.array([7, 7, 3, 3, 9, 9])
    depth = np.array([30.0, 32.0, 2.0, 4.0, np.nan, np.nan])
    assert order_communities(labels, depth).tolist() == [2, 2, 1, 1, 3, 3]
