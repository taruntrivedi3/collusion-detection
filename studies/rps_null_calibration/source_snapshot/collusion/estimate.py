"""State encoders, pairwise conditional mutual information (gamma), net influence (Gamma).

All information quantities are in bits (log base 2).
"""
from __future__ import annotations

import numpy as np

from .game import action_counts

STATE_KINDS = ("none", "top", "order", "full", "profile")


# ---------------------------------------------------------------- states

def encode_states(actions, kind: str):
    """Per-round state ids computed from the PREVIOUS round's action counts.

    Round 0 has no history and gets its own state. Returns compact ids 0..K-1 (int64).
    """
    rounds, n = actions.shape
    if kind == "none":
        return np.zeros(rounds, dtype=np.int64)
    if kind == "profile":
        # previous round's full action profile (3^n states; diagnostics for small n only)
        code = np.concatenate([[-1], actions[:-1].astype(np.int64) @ 3 ** np.arange(n)])
        return np.unique(code, return_inverse=True)[1].astype(np.int64)
    c = action_counts(actions)                      # counts of round t
    prev = np.vstack([np.full((1, 3), -1), c[:-1]])  # counts of round t-1 (row 0 = no history)
    if kind == "top":
        code = ((prev == prev.max(1, keepdims=True)) * np.array([1, 2, 4])).sum(1)
    elif kind == "order":
        s = np.sign(prev[:, [0, 0, 1]] - prev[:, [1, 2, 2]]) + 1   # sign(R-P), sign(R-S), sign(P-S)
        code = s @ np.array([9, 3, 1])
    elif kind == "full":
        code = prev[:, 0] * (n + 1) + prev[:, 1]
    else:
        raise ValueError(kind)
    code = np.where(prev[:, 0] < 0, -1, code)       # round-0 sentinel
    return np.unique(code, return_inverse=True)[1].astype(np.int64)


# ---------------------------------------------------------------- gamma

def onehot(actions):
    """[..., R, n] int actions -> [..., R, 3n] float32 one-hot (agent-major: col = 3*i + a)."""
    a = np.asarray(actions)
    oh = (a[..., None] == np.arange(3)).astype(np.float32)
    return oh.reshape(*a.shape[:-1], a.shape[-1] * 3)


def _xlogx_table(N):
    """t[k] = k * log2(k) for integer k in 0..N (t[0] = 0)."""
    k = np.arange(N + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 0, k * np.log2(np.maximum(k, 1)), 0.0)


def gamma_batch(actions, states):
    """gamma[b, i, j] = I(a_i; a_j | s) in bits for a batch of datasets sharing `states`.

    actions: [B, R, n] (or [R, n]); states: [R] ids. Diagonal set to 0.
    For each state s, X_s^T X_s on the one-hot matrix gives all pairwise 3x3 joint counts C.
    N_s * MI_ij = sum_ab f(C_ab) - sum_a f(r_ia) - sum_b f(r_jb) + f(N_s), f(x) = x log2 x,
    where r_i are agent i's own action counts in state s; f is a lookup on integer counts.
    """
    actions = np.asarray(actions)
    single = actions.ndim == 2
    if single:
        actions = actions[None]
    B, rounds, n = actions.shape
    oh = onehot(actions)                            # [B, R, 3n]
    g = np.zeros((B, n, n), dtype=np.float64)
    order = np.argsort(states, kind="stable")
    bounds = np.flatnonzero(np.diff(states[order])) + 1
    f = _xlogx_table(rounds)
    for idx in np.split(order, bounds):
        Ns = len(idx)
        if Ns < 2:
            continue
        X = oh[:, idx, :]
        C = np.rint(np.matmul(X.transpose(0, 2, 1), X)).astype(np.int64)   # [B, 3n, 3n]
        joint = f[C].reshape(B, n, 3, n, 3).sum(axis=(2, 4))               # [B, n, n]
        own = f[np.diagonal(C, axis1=1, axis2=2)].reshape(B, n, 3).sum(-1)  # [B, n]
        g += (joint - own[:, :, None] - own[:, None, :] + f[Ns]) / rounds
    idx = np.arange(n)
    g[:, idx, idx] = 0.0
    return g[0] if single else g


def gamma(actions, states=None):
    if states is None:
        states = np.zeros(len(actions), dtype=np.int64)
    return gamma_batch(actions, states)


def net_influence(g):
    """Gamma[i, j] = g[i, j] - max_{k != i, j} g[k, j] (Bonjour et al.). Requires n >= 3.

    g may be [n, n] or [B, n, n]. Diagonal is NaN.
    """
    g = np.asarray(g, dtype=np.float64)
    n = g.shape[-1]
    assert n >= 3
    eye = np.eye(n, dtype=bool)
    m = np.where(eye, -np.inf, g)                   # column j, exclude k == j
    srt = np.sort(m, axis=-2)
    top1, top2 = srt[..., -1, :], srt[..., -2, :]   # [.., j]
    is_top = m == top1[..., None, :]
    # if i holds the max of column j, the max over k != i is the second-largest
    # (ties: another k also attains top1, and then top2 == top1 anyway)
    other_max = np.where(is_top, top2[..., None, :], top1[..., None, :])
    G = g - other_max
    return np.where(eye, np.nan, G)
