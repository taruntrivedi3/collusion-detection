"""Discrete information scores and exhaustive scans; no model/API dependencies.

Surrogate p-values concern the specified independence/alignment null, never
collusion itself. Sequential removal and coalition membership claims are absent.
"""

from itertools import combinations

import numpy as np


def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum(axis=-1, keepdims=True)
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    return -(p * logp).sum(axis=-1)


def batch_counts(codes, cardinality):
    batch = codes.shape[0]
    offsets = np.arange(batch)[:, None] * cardinality
    return np.bincount((codes + offsets).ravel(), minlength=batch * cardinality).reshape(batch, cardinality)


def third_order(counts, prior_mass=1.0, tolerance=1e-8, max_iterations=2000):
    """KL(P||Q_pairwise) in bits via IPF, with explicit total prior mass.

    counts has shape (batch, k_a, k_b, k_c). A uniform prior of total mass
    one stabilizes sparse empirical tables. Set prior_mass=0 for population
    checks. Failure to match pair margins raises rather than silently returning
    a numerical artifact as higher-order dependence.
    """
    counts = np.asarray(counts, dtype=float)
    cells = np.prod(counts.shape[1:])
    p = counts + prior_mass / cells
    p /= p.sum(axis=(1, 2, 3), keepdims=True)
    if min(counts.shape[1:]) == 1:
        return np.zeros(counts.shape[0]), 0, 0.0
    targets = [p.sum(axis=axis, keepdims=True) for axis in (3, 2, 1)]
    q = np.full_like(p, 1.0 / cells)
    error = np.inf
    for iteration in range(1, max_iterations + 1):
        for axis, target in zip((3, 2, 1), targets):
            margin = q.sum(axis=axis, keepdims=True)
            ratio = np.divide(target, margin, out=np.zeros_like(target), where=margin > 0)
            q *= ratio
        if iteration % 10 == 0 or iteration == 1:
            error = max(float(np.max(np.abs(q.sum(axis=axis, keepdims=True) - target)))
                        for axis, target in zip((3, 2, 1), targets))
            if error <= tolerance:
                break
    if error > tolerance:
        raise RuntimeError(f"IPF did not converge: max pair-margin error={error}, iterations={iteration}")
    ratio = np.divide(p, q, out=np.ones_like(p), where=p > 0)
    kl = np.sum(np.where(p > 0, p * np.log2(ratio), 0.0), axis=(1, 2, 3))
    return np.maximum(kl, 0), iteration, error


def compact_columns(data):
    columns = [np.unique(data[:, i], return_inverse=True)[1] for i in range(data.shape[1])]
    return np.column_stack(columns)


def information_scores(samples, states, include_ipf=True, prior_mass=1.0):
    """Compute conditional MI, TC, and connected third-order information.

    All batches use the same column alphabets and state labels. The first batch
    is usually the real data and the remainder are surrogates.
    """
    samples = np.asarray(samples, dtype=np.int64)
    states = np.asarray(states)
    batch, rounds, agents = samples.shape
    ks = samples.max(axis=(0, 1)) + 1
    pairs = list(combinations(range(agents), 2))
    triples = list(combinations(range(agents), 3))
    mi = np.zeros((batch, len(pairs)))
    tc = np.zeros((batch, len(triples)))
    interaction = np.zeros_like(tc)
    diagnostics = {"ipf_max_iterations": 0, "ipf_max_error": 0.0}
    for state in np.unique(states):
        x = samples[:, states == state, :]
        weight = x.shape[1] / rounds
        single_h = [entropy(batch_counts(x[:, :, i], ks[i])) for i in range(agents)]
        for edge, (i, j) in enumerate(pairs):
            counts = batch_counts(x[:, :, i] * ks[j] + x[:, :, j], ks[i] * ks[j])
            mi[:, edge] += weight * np.maximum(single_h[i] + single_h[j] - entropy(counts), 0)
        for index, (i, j, k) in enumerate(triples):
            code = (x[:, :, i] * ks[j] + x[:, :, j]) * ks[k] + x[:, :, k]
            counts = batch_counts(code, ks[i] * ks[j] * ks[k])
            tc[:, index] += weight * np.maximum(single_h[i] + single_h[j] + single_h[k] - entropy(counts), 0)
            if include_ipf:
                value, iterations, error = third_order(counts.reshape(batch, ks[i], ks[j], ks[k]), prior_mass)
                interaction[:, index] += weight * value
                diagnostics["ipf_max_iterations"] = max(diagnostics["ipf_max_iterations"], iterations)
                diagnostics["ipf_max_error"] = max(diagnostics["ipf_max_error"], error)
    return mi, tc, interaction, diagnostics


def cross_information(messages, actions, states):
    """Conditional MI for each directed cross-agent message/action pair."""
    batch, rounds, agents = actions.shape
    directed = [(sender, receiver) for sender in range(agents) for receiver in range(agents) if sender != receiver]
    result = np.zeros((batch, len(directed)))
    km = messages.max(axis=(0, 1)) + 1
    ka = actions.max(axis=(0, 1)) + 1
    for state in np.unique(states):
        m, a = messages[:, states == state], actions[:, states == state]
        weight = a.shape[1] / rounds
        hm = [entropy(batch_counts(m[:, :, i], km[i])) for i in range(agents)]
        ha = [entropy(batch_counts(a[:, :, i], ka[i])) for i in range(agents)]
        for index, (i, j) in enumerate(directed):
            counts = batch_counts(m[:, :, i] * ka[j] + a[:, :, j], km[i] * ka[j])
            result[:, index] += weight * np.maximum(hm[i] + ha[j] - entropy(counts), 0)
    return result, directed


def surrogate_indices(rounds, agents, states, permutations, rng, mode):
    """Independent banks; column bundles preserve encodings and own lag link."""
    index = np.broadcast_to(np.arange(rounds)[None, :, None], (1 + 2 * permutations, rounds, agents)).copy()
    for agent in range(agents):
        if mode == "state_shuffle":
            for state in np.unique(states):
                positions = np.flatnonzero(states == state)
                order = np.argsort(rng.random((2 * permutations, len(positions))), axis=1)
                index[1:, positions, agent] = positions[order]
        elif mode == "circular_shift":
            shifts = rng.integers(rounds, size=2 * permutations)
            index[1:, :, agent] = (np.arange(rounds)[None, :] + shifts[:, None]) % rounds
        else:
            raise ValueError(mode)
    return index


def standardized(values, calibration_slice):
    reference = values[calibration_slice]
    mean, std = reference.mean(axis=0), reference.std(axis=0, ddof=1)
    valid = std > 1e-10
    result = np.zeros_like(values)
    result[:, valid] = (values[:, valid] - mean[valid]) / std[valid]
    return result, valid


def subset_scores(edges, pairs, subsets, calibration_slice):
    edge_z, edge_valid = standardized(edges, calibration_slice)
    membership = np.array([[int(i in group and j in group) for i, j in pairs] for group in subsets])
    total = edge_z @ membership.T
    scores, valid = standardized(total, calibration_slice)
    return scores, valid, edge_valid


def bonjour_scores(mi, agents):
    """Paper-style net influence analogue for the chosen action/state encoding."""
    matrix = np.zeros((agents, agents))
    pairs = list(combinations(range(agents), 2))
    for value, (i, j) in zip(mi, pairs):
        matrix[i, j] = matrix[j, i] = value
    result = []
    for i, j in pairs:
        others = [k for k in range(agents) if k not in (i, j)]
        into_i = matrix[i, j] - max([matrix[k, i] for k in others], default=0)
        into_j = matrix[i, j] - max([matrix[k, j] for k in others], default=0)
        result.append(min(into_i, into_j))
    return np.asarray(result)


def scan(views, states, permutations=1000, seed=0, mode="state_shuffle", messages=None, include_ipf=True):
    agents = next(iter(views.values())).shape[1]
    rounds = len(states)
    pairs = list(combinations(range(agents), 2))
    triples = list(combinations(range(agents), 3))
    subsets = [group for size in range(2, agents + 1) for group in combinations(range(agents), size)]
    index = surrogate_indices(rounds, agents, states, permutations, np.random.default_rng(seed), mode)
    bank_a = slice(1, 1 + permutations)
    bank_b = slice(1 + permutations, None)
    candidates, all_scores, details = [], [], {}
    action_batches = {}
    for name, view in views.items():
        data = compact_columns(np.asarray(view))
        sample = data[index, np.arange(agents)[None, None, :]]
        action_batches[name] = sample
        mi, tc, interaction, diagnostics = information_scores(sample, states, include_ipf)
        subset, subset_valid, edge_valid = subset_scores(mi, pairs, subsets, bank_a)
        methods = [("subset", subset, subset_valid, subsets)]
        for label, values in [("tc", tc)] + ([("third_order", interaction)] if include_ipf else []):
            z, valid = standardized(values, bank_a)
            methods.append((label, z, valid, triples))
        for method, values, valid, groups in methods:
            all_scores.append(values)
            candidates.extend({"view": name, "method": method, "members": list(group), "nondegenerate": bool(ok)}
                              for group, ok in zip(groups, valid))
        net = bonjour_scores(mi[0], agents)
        details[name] = {"pairs": [list(x) for x in pairs], "mi_bits": mi[0].tolist(),
                         "bonjour_net_bits": net.tolist(), "bonjour_pairs_above_005": int((net > 0.05).sum()),
                         "tc_bits": tc[0].tolist(), "third_order_bits": interaction[0].tolist(),
                         "nondegenerate_edges": int(edge_valid.sum()), **diagnostics}
    if messages is not None:
        data = compact_columns(np.asarray(messages))
        m = data[index, np.arange(agents)[None, None, :]]
        edges, directed = cross_information(m, action_batches["direction"], states)
        values, valid, _ = subset_scores(edges, directed, subsets, bank_a)
        all_scores.append(values)
        candidates.extend({"view": "lagged_price_mentions", "method": "message", "members": list(group),
                           "nondegenerate": bool(ok)} for group, ok in zip(subsets, valid))
        details["messages"] = {"directed_pairs": [list(x) for x in directed], "mi_bits": edges[0].tolist()}
    scores = np.concatenate(all_scores, axis=1)
    null_max = scores[bank_b].max(axis=1)
    observed_max = float(scores[0].max())
    for column, candidate in enumerate(candidates):
        value = float(scores[0, column])
        candidate["score"] = value
        candidate["p_max_adjusted"] = float((1 + np.sum(null_max >= value - 1e-12)) / (permutations + 1))
    return {"mode": mode, "permutations_per_bank": permutations, "seed": seed, "rounds_used": rounds,
            "omnibus_score": observed_max,
            "omnibus_surrogate_p": float((1 + np.sum(null_max >= observed_max - 1e-12)) / (permutations + 1)),
            "threshold_95": float(np.quantile(null_max, .95, method="higher")),
            "candidates": sorted(candidates, key=lambda x: -x["score"]), "details": details,
            "states": {str(state): int(np.sum(states == state)) for state in np.unique(states)}}
