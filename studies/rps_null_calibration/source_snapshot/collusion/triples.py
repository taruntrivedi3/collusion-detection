"""Triple-level statistics: conditional total correlation (TC) and its third-order component (E).

TC(C | s) = sum_s p(s) [sum_i H(a_i | s) - H(a_C | s)]                 (bits)
E(C | s)  = sum_s p(s) KL(p_s || q_s), q_s = max-entropy table matching all three pairwise
            marginals of p_s (fitted by iterative proportional fitting).
TC = (pairwise part) + E (Amari's hierarchical decomposition for 3 variables).
Under mutual independence, 2N ln2 TC ~ chi2_20 and 2N ln2 E ~ chi2_8 (no state).
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np

from .estimate import _xlogx_table


@lru_cache(maxsize=8)
def triple_index(n):
    """[T, 3] all triples i < j < k."""
    return np.array(list(combinations(range(n), 3)), dtype=np.int64)


def triple_counts(actions, states, triples):
    """Joint counts [D, T, S, 3, 3, 3] for datasets actions [D, R, n] (int), states [R] in 0..S-1.

    Per state: P^T @ X, where P [R_s, pairs*9] is the one-hot of each pair's joint action and
    X [R_s, 3n] the one-hot of single actions, gives counts for every (pair, third agent).
    Triples (i<j<k) are then sub-blocks (pair (i,j), agent k).
    """
    actions = np.asarray(actions)
    if actions.ndim == 2:
        actions = actions[None]
    D, R, n = actions.shape
    S = int(states.max()) + 1
    iu, ju = np.triu_indices(n, 1)
    pid = np.full((n, n), -1)
    pid[iu, ju] = np.arange(len(iu))
    p_of, k_of = pid[triples[:, 0], triples[:, 1]], triples[:, 2]
    out = np.zeros((D, len(triples), S, 3, 3, 3), dtype=np.int64)
    eye3 = np.eye(3, dtype=np.float32)
    for s in range(S):
        rows = np.flatnonzero(states == s)
        if len(rows) == 0:
            continue
        for d in range(D):
            X = eye3[actions[d, rows]]                                   # [Rs, n, 3]
            P = (X[:, iu, :, None] * X[:, ju, None, :]).reshape(len(rows), -1)   # [Rs, pairs*9]
            M = (P.T @ X.reshape(len(rows), -1)).reshape(len(iu), 9, n, 3)
            out[d, :, s] = np.rint(M[p_of, :, k_of, :]).astype(np.int64).reshape(-1, 3, 3, 3)
    return out


def triple_counts_bincount(actions, states, triples):
    """Reference implementation of triple_counts (slower)."""
    actions = np.asarray(actions)
    if actions.ndim == 2:
        actions = actions[None]
    D, R, n = actions.shape
    S = int(states.max()) + 1
    T = len(triples)
    out = np.empty((D, T * S * 27), dtype=np.int64)
    base = (np.arange(T, dtype=np.int64) * S * 27)[None, :] + (states.astype(np.int64) * 27)[:, None]
    for d in range(D):
        a = actions[d].astype(np.int64)
        code = 9 * a[:, triples[:, 0]] + 3 * a[:, triples[:, 1]] + a[:, triples[:, 2]]   # [R, T]
        out[d] = np.bincount((base + code).ravel(), minlength=T * S * 27)
    return out.reshape(D, T, S, 3, 3, 3)


def total_correlation(C, rounds):
    """C [..., S, 3, 3, 3] integer counts -> TC in bits [...], weighted by p(s) = N_s / rounds."""
    f = _xlogx_table(rounds)
    joint = f[C].sum(axis=(-1, -2, -3))
    m0, m1, m2 = C.sum(axis=(-1, -2)), C.sum(axis=(-1, -3)), C.sum(axis=(-2, -3))
    own = f[m0].sum(-1) + f[m1].sum(-1) + f[m2].sum(-1)
    Ns = C.sum(axis=(-1, -2, -3))
    return ((joint - own + 2 * f[Ns]) / rounds).sum(-1)


def _ratio(target, cur):
    return np.divide(target, cur, out=np.zeros_like(cur), where=cur > 0)


def ipf_pairwise(C, iters=200, tol=1e-9):
    """Max-entropy fit q (same shape as C, float) matching the three 2-way marginals of C."""
    C = C.astype(np.float64)
    N = C.sum(axis=(-1, -2, -3), keepdims=True)
    q = np.broadcast_to(N / 27.0, C.shape).copy()
    t01, t02, t12 = C.sum(-1), C.sum(-2), C.sum(-3)
    for _ in range(iters):
        q *= _ratio(t01, q.sum(-1))[..., None]
        q *= _ratio(t02, q.sum(-2))[..., None, :]
        q *= _ratio(t12, q.sum(-3))[..., None, :, :]
        err = max(np.abs(q.sum(-1) - t01).max(), np.abs(q.sum(-2) - t02).max())
        if err < tol * max(1.0, float(N.max())):
            break
    return q


def third_order(C, rounds, q=None):
    """E in bits [...]: sum_s (N_s / rounds) KL(p_s || q_s)."""
    q = ipf_pairwise(C) if q is None else q
    Cf = C.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(Cf > 0, Cf * np.log2(Cf / q), 0.0)
    return t.sum(axis=(-1, -2, -3, -4)) / rounds


def triple_stats(actions, states, triples=None):
    """(TC [D, T], E [D, T]) for datasets [D, R, n] (or [R, n] -> [T])."""
    actions = np.asarray(actions)
    single = actions.ndim == 2
    rounds, n = actions.shape[-2:]
    triples = triple_index(n) if triples is None else triples
    C = triple_counts(actions, states, triples)
    tc, e = total_correlation(C, rounds), third_order(C, rounds)
    return (tc[0], e[0]) if single else (tc, e)
