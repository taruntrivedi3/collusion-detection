import numpy as np

from collusion.game import Coalition, Population, simulate
from collusion.estimate import gamma
from collusion.triples import ipf_pairwise, triple_counts, triple_index, triple_stats

LOG2_3 = np.log2(3)


def _entropy_bits(x):
    _, c = np.unique(x, axis=0, return_counts=True)
    p = c / c.sum()
    return -(p * np.log2(p)).sum()


def test_tc_matches_entropy_definition():
    rng = np.random.default_rng(0)
    a = simulate(Population(4, [Coalition("triangle", [0, 1, 2], 0.4)]), 500, rng)
    tc, _ = triple_stats(a, np.zeros(500, dtype=np.int64))
    for t, (i, j, k) in enumerate(triple_index(4)):
        ref = sum(_entropy_bits(a[:, [m]]) for m in (i, j, k)) - _entropy_bits(a[:, [i, j, k]])
        assert np.isclose(tc[t], ref, atol=1e-10)


def test_xor_cp1_tc_is_third_order():
    rng = np.random.default_rng(1)
    a = simulate(Population(3, [Coalition("xor", [0, 1, 2], 1.0)]), 6000, rng)
    tc, e = triple_stats(a, np.zeros(6000, dtype=np.int64))
    assert abs(tc[0] - LOG2_3) < 0.01 and abs(e[0] - LOG2_3) < 0.01
    assert gamma(a).max() < 0.01


def test_pair_inside_triple_has_no_third_order():
    rng = np.random.default_rng(2)
    a = simulate(Population(3, [Coalition("pair", [0, 1], 0.6)]), 6000, rng)
    tc, e = triple_stats(a, np.zeros(6000, dtype=np.int64))
    assert abs(tc[0] - gamma(a)[0, 1]) < 0.01
    assert e[0] < 0.005


def test_independent_bias_matches_chi2_dfs():
    rng = np.random.default_rng(3)
    R = 2000
    tcs, es = [], []
    for _ in range(300):
        a = rng.integers(3, size=(R, 3))
        tc, e = triple_stats(a, np.zeros(R, dtype=np.int64))
        tcs.append(tc[0]); es.append(e[0])
    k = 2 * R * np.log(2)
    assert abs(np.mean(tcs) * k - 20) < 1.5      # chi2_20 mean
    assert abs(np.mean(es) * k - 8) < 1.0        # chi2_8 mean


def test_ipf_matches_pairwise_marginals():
    rng = np.random.default_rng(4)
    a = simulate(Population(3, [Coalition("triangle", [0, 1, 2], 0.5)]), 300, rng)
    C = triple_counts(a, np.zeros(300, dtype=np.int64), triple_index(3))
    q = ipf_pairwise(C)
    for ax in (-1, -2, -3):
        assert np.allclose(q.sum(ax), C.sum(ax), atol=1e-6)


def test_fast_counts_match_bincount_with_states():
    from collusion.estimate import encode_states
    from collusion.triples import triple_counts_bincount
    rng = np.random.default_rng(6)
    a = simulate(Population(6, [Coalition("xor", [0, 1, 2], 0.5)]), 400, rng)
    for kind in ("none", "top", "full"):
        s = encode_states(a, kind)
        tri = triple_index(6)
        assert np.array_equal(triple_counts(a, s, tri), triple_counts_bincount(a, s, tri))
