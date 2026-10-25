import numpy as np

from collusion.game import (Coalition, LeaderReactive, Population, beats, loses_to,
                            payoff_paper3, payoff_round_robin, simulate)
from collusion.estimate import encode_states, gamma, net_influence
from collusion.null import build_context, permute_within_states
from collusion.detect import _bh, detect_bonjour_calibrated, detect_subgraph, edge_z, subset_scores

LOG2_3 = np.log2(3)


def test_move_relations():
    for x in range(3):
        assert (beats(x) - x) % 3 == 1
        assert (x - loses_to(x)) % 3 == 1


def test_payoffs():
    # R, P, S all present -> round robin 0 each; paper -> 1 each
    a = np.array([0, 1, 2])
    assert payoff_round_robin(a).tolist() == [0, 0, 0]
    assert payoff_paper3(a).tolist() == [1, 1, 1]
    a = np.array([0, 0, 1])          # P beats both R
    assert payoff_round_robin(a).tolist() == [-1, -1, 2]
    assert payoff_paper3(a).tolist() == [0, 0, 1]
    assert payoff_paper3(np.array([2, 2, 2])).tolist() == [0, 0, 0]


def test_independent_gamma_small_bias():
    rng = np.random.default_rng(1)
    a = rng.integers(3, size=(2000, 4))
    g = gamma(a)
    off = g[~np.eye(4, dtype=bool)]
    # Miller-Madow bias for a 3x3 table: (3-1)(3-1) / (2 N ln 2) ~ 0.0014 bits at N = 2000
    assert np.all(off >= 0) and off.mean() < 0.005


def test_pair_cp1_is_log2_3():
    rng = np.random.default_rng(2)
    pop = Population(3, [Coalition("pair", [0, 1], 1.0)])
    g = gamma(simulate(pop, 5000, rng))
    assert abs(g[0, 1] - LOG2_3) < 0.01
    assert g[0, 2] < 0.01 and g[1, 2] < 0.01


def test_xor_cp1_pairwise_zero():
    rng = np.random.default_rng(3)
    pop = Population(3, [Coalition("xor", [0, 1, 2], 1.0)])
    g = gamma(simulate(pop, 5000, rng))
    assert g[~np.eye(3, dtype=bool)].max() < 0.01


def test_gamma_symmetric_and_states():
    rng = np.random.default_rng(4)
    pop = Population(5, [Coalition("triangle", [0, 1, 2], 0.5)], {3: LeaderReactive(0.8)})
    a = simulate(pop, 1000, rng)
    for kind in ("none", "top", "order", "full"):
        s = encode_states(a, kind)
        g = gamma(a, s)
        assert np.allclose(g, g.T)
    assert encode_states(a, "order").max() + 1 <= 14      # 13 orderings + round-0 state


def test_net_influence_brute_force():
    rng = np.random.default_rng(5)
    g = rng.random((6, 6)); g = g + g.T
    G = net_influence(g)
    for i in range(6):
        for j in range(6):
            if i != j:
                ref = g[i, j] - max(g[k, j] for k in range(6) if k not in (i, j))
                assert np.isclose(G[i, j], ref)


def test_permutation_preserves_state_marginals():
    rng = np.random.default_rng(6)
    a = rng.integers(3, size=(300, 4)).astype(np.int8)
    s = rng.integers(5, size=300)
    (x,) = list(permute_within_states(a, s, 1, rng))
    for st in range(5):
        for j in range(4):
            assert np.array_equal(np.sort(a[s == st, j]), np.sort(x[0][s == st, j]))
    assert not np.array_equal(a, x[0])


def test_subset_T_matches_brute_force():
    rng = np.random.default_rng(7)
    a = rng.integers(3, size=(200, 5)).astype(np.int8)
    ctx = build_context(a, np.zeros(200, dtype=np.int64), B=30, seed=0)
    zr, zA, _ = edge_z(ctx)
    masks, S, _ = subset_scores(ctx)
    k = 17
    mem = [b for b in range(5) if (int(masks[k]) >> b) & 1]
    T = lambda z: sum(z[i, j] for i in mem for j in mem if i < j)
    TA = np.array([T(z) for z in zA])
    assert np.isclose(S[k], (T(zr) - TA.mean()) / TA.std(), rtol=1e-4)
    assert len(masks) == 2 ** 5 - 5 - 1


def test_bh():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    assert _bh(p, 0.05).tolist() == [True, True, True, False]


def test_subgraph_recovers_strong_triangle():
    rng = np.random.default_rng(8)
    pop = Population(5, [Coalition("triangle", [0, 1, 2], 0.6)])
    a = simulate(pop, 1000, rng)
    ctx = build_context(a, np.zeros(1000, dtype=np.int64), B=100, seed=1)
    assert detect_subgraph(ctx).coalitions == [frozenset({0, 1, 2})]


def test_bonjour_calibrated_recovers_pair():
    rng = np.random.default_rng(9)
    a = simulate(Population(3, [Coalition("pair", [0, 1], 0.5)]), 500, rng)
    ctx = build_context(a, np.zeros(500, dtype=np.int64), B=100, seed=2)
    assert detect_bonjour_calibrated(ctx).coalitions == [frozenset({0, 1})]


def test_clique_alias():
    rng = np.random.default_rng(11)
    a = simulate(Population(4, [Coalition("clique", [0, 1, 2, 3], 1.0)]), 50, rng)
    assert (a == a[:, :1]).all()


def _gamma_reference(a, s):
    """Direct per-pair conditional MI in bits (slow reference)."""
    R, n = a.shape
    g = np.zeros((n, n))
    for st in np.unique(s):
        x = a[s == st]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                C = np.zeros((3, 3))
                np.add.at(C, (x[:, i], x[:, j]), 1)
                P = C / len(x)
                pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
                m = P > 0
                g[i, j] += len(x) / R * (P[m] * np.log2(P[m] / (pi @ pj)[m])).sum()
    return g


def test_gamma_matches_reference_with_states():
    rng = np.random.default_rng(12)
    pop = Population(6, [Coalition("triangle", [0, 1, 2], 0.4)], {3: LeaderReactive(0.7)})
    a = simulate(pop, 400, rng)
    for kind in ("none", "top", "full"):
        s = encode_states(a, kind)
        assert np.allclose(gamma(a, s), _gamma_reference(a, s), atol=1e-10)
