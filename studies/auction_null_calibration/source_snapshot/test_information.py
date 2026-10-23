from itertools import product

import numpy as np

from information import bonjour_scores, information_scores, scan, surrogate_indices, third_order
from ingest import direction_code, parse_agent, quote_value


def test_independent_full_factorial_has_zero_information():
    x = np.array(list(product(range(3), repeat=3)))
    mi, tc, connected, _ = information_scores(x[None], np.zeros(len(x)), prior_mass=0)
    np.testing.assert_allclose(mi, 0, atol=1e-12)
    np.testing.assert_allclose(tc, 0, atol=1e-12)
    np.testing.assert_allclose(connected, 0, atol=1e-12)


def test_triangle_breaks_net_influence_but_has_pair_and_group_signal():
    base = np.array(list(product(range(3), repeat=3)))
    x = np.column_stack([base[:, 0]] * 3 + [base[:, 1], base[:, 2]])
    mi, tc, connected, _ = information_scores(x[None], np.zeros(len(x)), prior_mass=0)
    np.testing.assert_allclose(mi[0, [0, 1, 4]], np.log2(3), atol=1e-12)
    np.testing.assert_allclose(tc[0, 0], 2 * np.log2(3), atol=1e-12)
    np.testing.assert_allclose(connected, 0, atol=1e-12)
    assert np.max(bonjour_scores(mi[0], 5)) <= 1e-12


def test_xor_has_no_pair_signal_but_one_bit_of_tc_and_third_order():
    ab = np.array(list(product(range(2), repeat=2)))
    x = np.column_stack([ab, ab[:, 0] ^ ab[:, 1]])
    mi, tc, connected, _ = information_scores(x[None], np.zeros(len(x)), prior_mass=0)
    np.testing.assert_allclose(mi, 0, atol=1e-12)
    np.testing.assert_allclose(tc, 1, atol=1e-12)
    np.testing.assert_allclose(connected, 1, atol=1e-12)


def test_pairwise_ising_distribution_has_no_irreducible_third_order():
    spins = np.array(list(product([-1, 1], repeat=3)))
    p = np.exp(.3 * spins[:, 0] + .7 * spins[:, 0] * spins[:, 1] - .4 * spins[:, 1] * spins[:, 2])
    connected, _, error = third_order(p.reshape(1, 2, 2, 2), prior_mass=0)
    np.testing.assert_allclose(connected, 0, atol=1e-7)
    assert error <= 1e-8


def test_public_signal_disappears_when_conditioned_on_its_state():
    state = np.repeat([0, 1, 2], 10)
    x = np.column_stack([state] * 3)
    unconditioned = information_scores(x[None], np.zeros(len(x)), prior_mass=0)
    conditioned = information_scores(x[None], state, prior_mass=0)
    assert unconditioned[1][0, 0] > 3
    for values in conditioned[:3]:
        np.testing.assert_allclose(values, 0, atol=1e-12)


def test_state_shuffle_preserves_each_agents_conditional_multiset():
    states = np.array([0, 1, 0, 1, 2, 2, 2])
    idx = surrogate_indices(7, 4, states, 11, np.random.default_rng(4), "state_shuffle")
    np.testing.assert_array_equal(idx[0], np.broadcast_to(np.arange(7)[:, None], (7, 4)))
    for batch in idx[1:]:
        for agent in range(4):
            for state in np.unique(states):
                positions = np.flatnonzero(states == state)
                np.testing.assert_array_equal(np.sort(batch[positions, agent]), positions)


def test_degenerate_constant_scan_is_finite_and_not_an_alarm():
    result = scan({"direction": np.zeros((30, 5), dtype=int)}, np.zeros(30), permutations=29)
    assert result["omnibus_surrogate_p"] == 1
    assert result["omnibus_score"] == 0
    assert not any(x["nondegenerate"] for x in result["candidates"])


def test_scan_recovers_an_exact_xor_triple():
    rng = np.random.default_rng(12)
    x = rng.integers(2, size=(400, 5))
    x[:, 2] = x[:, 0] ^ x[:, 1]
    result = scan({"direction": x}, np.zeros(len(x)), permutations=99, seed=6)
    assert result["omnibus_surrogate_p"] <= .01
    for method in ("tc", "third_order"):
        best = next(c for c in result["candidates"] if c["method"] == method)
        assert best["members"] == [0, 1, 2]
        assert best["p_max_adjusted"] <= .01


def test_json_null_is_not_the_literal_withdraw_command():
    assert quote_value(None) == (None, "no_update")
    assert quote_value("null") == (None, "withdraw")
    assert direction_code(None, "no_update", 96) == 5
    assert direction_code(None, "withdraw", 96) == 3


def test_markdown_parser_does_not_confuse_prompt_json_with_response():
    text = '\n## Prompt: Round 1 - time\n``````\nOutput {"ask": "example"}\n``````\n## Response: Round 1 - time\n````json\n{"ask": 94, "message": "## header"}\n````\n'
    prompts, responses = parse_agent(text)
    assert 'example' in prompts[1]
    assert responses[1]["ask"] == 94
