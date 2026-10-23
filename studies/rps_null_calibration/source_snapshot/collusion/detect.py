"""Collusion detectors. All share one interface:

    detect(actions[R, n], states[R], method=..., B=..., seed=...) -> list[frozenset]

Internally every detector works from a NullContext (observed gamma + two permutation sets),
so experiments can build the context once and run all detectors on the same trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import stats

from .estimate import net_influence
from .null import NullContext, build_context

EPS = 1e-12
ALPHA_BONJOUR = 0.05


@dataclass
class Result:
    coalitions: list            # list[frozenset[int]] in original agent indices
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------- edge z-scores

def edge_moments(g_A):
    return g_A.mean(0), np.maximum(g_A.std(0), EPS)


def edge_z(ctx: NullContext):
    """z for observed and for every Set-B dataset, standardized by Set-A per-edge moments."""
    mu, sd = edge_moments(ctx.g_A)
    zr = (ctx.g_real - mu) / sd
    zA = (ctx.g_A - mu) / sd
    zB = (ctx.g_B - mu) / sd
    for z in (zr, zA, zB):
        np.einsum("...ii->...i", z)[:] = 0.0
    return zr, zA, zB


# ---------------------------------------------------------------- 1. Bonjour et al.

def detect_bonjour(ctx: NullContext, alpha=ALPHA_BONJOUR) -> Result:
    """Flag pair (i, j) if Gamma(i;j) >= alpha and Gamma(j;i) >= alpha. Report collusion only
    if exactly one pair is flagged (as in the paper)."""
    G = net_influence(ctx.g_real)
    ok = np.nan_to_num(G, nan=-np.inf) >= alpha
    both = ok & ok.T
    pairs = [frozenset((int(i), int(j))) for i, j in zip(*np.nonzero(np.triu(both, 1)))]
    coal = pairs if len(pairs) == 1 else []
    return Result(coal, {"flagged_pairs": pairs, "Gamma": G})


def pair_min_gamma(g):
    """m[.., i, j] = min(Gamma(i;j), Gamma(j;i)); a pair passes the both-directions rule at
    threshold t iff m >= t. Diagonal = -inf."""
    G = np.nan_to_num(net_influence(g), nan=-np.inf)
    return np.minimum(G, np.swapaxes(G, -1, -2))


def detect_bonjour_calibrated(ctx: NullContext, level=0.95) -> Result:
    """Same statistic and both-directions rule as Bonjour, but the threshold is the `level`
    quantile over Set B of max over unordered pairs of min(Gamma(i;j), Gamma(j;i)).
    Like the paper, report collusion only if exactly one pair passes."""
    n = ctx.n
    iu, ju = np.triu_indices(n, 1)
    m = pair_min_gamma(ctx.g_real)[iu, ju]
    thr = float(np.quantile(pair_min_gamma(ctx.g_B)[:, iu, ju].max(1), level))
    pairs = [frozenset((int(iu[k]), int(ju[k]))) for k in np.flatnonzero(m > thr)]
    coal = pairs if len(pairs) == 1 else []
    return Result(coal, {"flagged_pairs": pairs, "threshold": thr, "pair_min": m})


# ---------------------------------------------------------------- 2. per-edge BH

def _bh(p, q):
    m = len(p)
    o = np.argsort(p)
    passed = p[o] <= q * np.arange(1, m + 1) / m
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    sig = np.zeros(m, dtype=bool)
    sig[o[:k]] = True
    return sig


def _components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    groups = {}
    for i in {v for e in edges for v in e}:
        groups.setdefault(find(i), set()).add(i)
    return [frozenset(g) for g in groups.values()]


def gamma_tail(g_obs, g_null):
    """Upper-tail p-value of g_obs under a gamma fitted by moments to g_null [B, ...]."""
    m, v = g_null.mean(0), g_null.var(0)
    ok = (m > 0) & (v > 0)
    k = np.where(ok, m * m / np.where(ok, v, 1), 1.0)
    th = np.where(ok, v / np.where(ok, m, 1), 1.0)
    return np.where(ok, stats.gamma.sf(g_obs, k, scale=th), 1.0)


def edge_pvalues(ctx: NullContext, method="gamma"):
    """[n, n] one-sided per-edge p-values against the Set-A permutation null (diagonal = 1).

    "perm" : p = (1 + #{null >= obs}) / (B + 1). Floor 1/(B+1) makes BH unable to reject
             anything once q / #edges < 1/(B+1) (n >= 10 at B = 200).
    "gamma": upper tail of a gamma fitted by moments to the edge's permutation null. In the
             history-free case 2 N ln2 * gamma ~ chi2_4 (G-test), i.e. a gamma with shape 2, so
             the fit has the right family; with states it adapts to sparse cells.
    """
    if method == "perm":
        p = (1 + (ctx.g_A >= ctx.g_real).sum(0)) / (len(ctx.g_A) + 1)
    else:
        p = gamma_tail(ctx.g_real, ctx.g_A)
    p = np.array(p, dtype=np.float64)
    np.fill_diagonal(p, 1.0)
    return p


def detect_edge_bh(ctx: NullContext, q=0.05) -> Result:
    """Standard multiple-testing baseline (Benjamini & Hochberg 1995; not in Bonjour et al.):
    BH at level q over all unordered edges, coalitions = connected components."""
    n = ctx.n
    iu, ju = np.triu_indices(n, 1)
    p = edge_pvalues(ctx)[iu, ju]
    sig = _bh(p, q)
    edges = list(zip(iu[sig].tolist(), ju[sig].tolist()))
    return Result(_components(n, edges), {"p": p, "sig_edges": edges})


# ---------------------------------------------------------------- 3. node score

def detect_node_score(ctx: NullContext, level=0.95) -> Result:
    """Per-agent baseline inspired by Rose et al. (2026)'s per-agent detectors (Peak Suspicion /
    Deception Split); this is NOT their method and not in Bonjour et al.
    score_i = sum_j z_ij; calibrate max_i score_i on Set B; flag all nodes above threshold
    and report them as ONE set."""
    zr, _, zB = edge_z(ctx)
    s = zr.sum(1)
    thr = np.quantile(zB.sum(2).max(1), level)
    flagged = frozenset(np.flatnonzero(s > thr).tolist())
    return Result([flagged] if flagged else [], {"node_score": s, "threshold": thr})


# ---------------------------------------------------------------- 4. subgraph scan

@lru_cache(maxsize=4)
def subset_masks(n):
    """All subsets of size >= 2 as int bitmasks (uint32)."""
    m = np.arange(1, 1 << n, dtype=np.uint32)
    pop = np.zeros_like(m)
    for b in range(n):
        pop += (m >> b) & 1
    return m[pop >= 2]


def _pair_membership(masks, n):
    """[M, n(n-1)/2] float32: 1 if both endpoints of pair (i<j) are in the subset."""
    X = ((masks[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float32)
    iu, ju = np.triu_indices(n, 1)
    return X[:, iu] * X[:, ju]


def _membership_chunks(n, target_bytes=32 << 20):
    """Yield pair-membership chunks for all subsets of n agents (~target_bytes each).
    Not cached: keeps memory per worker small (the full n = 20 matrix is ~800 MB)."""
    masks = subset_masks(n)
    step = max(1024, target_bytes // (4 * max(1, n * (n - 1) // 2)))
    for s in range(0, len(masks), step):
        yield _pair_membership(masks[s:s + step], n)


def subset_scores(ctx: NullContext):
    """Return masks, S_real [M], and max_C S(C) for each Set-B dataset [B].

    T(C) = w_C . v (v = edge z's, w_C = pair-membership row) is linear in v, so its Set-A
    mean and variance are exact quadratic forms in the Set-A mean / covariance of v:
    mean_A T(C) = w_C . mu,  var_A T(C) = w_C' Sigma w_C   (same as the sample moments of T).
    """
    n = ctx.n
    zr, zA, zB = edge_z(ctx)
    iu, ju = np.triu_indices(n, 1)
    vr = zr[iu, ju]
    vA = zA[:, iu, ju]                              # [B, pairs]
    vB = zB[:, iu, ju].T.astype(np.float32)         # [pairs, B]
    mu_e = vA.mean(0)
    cov_e = np.atleast_2d(np.cov(vA, rowvar=False, ddof=0)).astype(np.float32)
    masks = subset_masks(n)
    S_real = np.empty(len(masks))
    maxB = np.full(vB.shape[1], -np.inf, dtype=np.float32)
    a = 0
    for Xp in _membership_chunks(n):
        b = a + len(Xp)
        mu = Xp @ mu_e.astype(np.float32)
        var = np.einsum("mp,mp->m", Xp @ cov_e, Xp)
        inv_sd = 1.0 / np.sqrt(np.maximum(var, 1e-18)).astype(np.float32)
        S_real[a:b] = (Xp.astype(np.float64) @ vr - mu) * inv_sd
        SB = Xp @ vB
        SB -= mu[:, None]
        SB *= inv_sd[:, None]
        np.maximum(maxB, SB.max(0), out=maxB)
        a = b
    return masks, S_real, maxB.astype(np.float64)


def _mask_to_set(mask, labels):
    return frozenset(int(labels[b]) for b in range(len(labels)) if (int(mask) >> b) & 1)


def _grow(ctx: NullContext, C, q=0.05):
    """Add every node outside C with a significant edge into C (per-edge permutation p-values,
    BH over all candidate edges v-c, v not in C, c in C). Repeat until nothing is added.
    C and the return value are local indices of ctx."""
    p = edge_pvalues(ctx)
    C = set(C)
    while True:
        out = [v for v in range(ctx.n) if v not in C]
        if not out:
            return C
        cand = [(v, c) for v in out for c in sorted(C)]
        sig = _bh(np.array([p[v, c] for v, c in cand]), q)
        add = {cand[k][0] for k in np.flatnonzero(sig)}
        if not add:
            return C
        C |= add


def detect_subgraph(ctx: NullContext, level=0.95, max_rounds=None, grow=False, cache=None) -> Result:
    """Exact scan over all subsets |C| >= 2; peel detected coalitions and recalibrate
    (step-down max-T). grow=True: after each detection, extend it with _grow (scan+grow).
    `cache` (dict) shares per-stage scan scores between calls on the same root context."""
    cache = {} if cache is None else cache
    labels = np.arange(ctx.n)
    found, trace = [], []
    cur = ctx
    while cur.n >= 2 and (max_rounds is None or len(found) < max_rounds):
        key = tuple(labels.tolist())
        if key not in cache:
            masks, S, maxB = subset_scores(cur)
            k = int(np.argmax(S))
            cache[key] = (int(masks[k]), float(S[k]), float(np.quantile(maxB, level)))
        mask, s_max, thr = cache[key]
        local = [b for b in range(cur.n) if (mask >> b) & 1]
        trace.append({"S_max": s_max, "threshold": thr, "argmax": frozenset(labels[local].tolist())})
        if s_max <= thr:
            break
        if grow:
            local = sorted(_grow(cur, local))
            trace[-1]["grown"] = frozenset(labels[local].tolist())
        C = frozenset(labels[local].tolist())
        found.append(C)
        keep = [i for i in range(cur.n) if i not in set(local)]
        if len(keep) < 2:
            break
        cur, kidx = cur.restrict(keep)
        labels = labels[kidx]
    return Result(found, {"trace": trace})


# ---------------------------------------------------------------- 5. triple scans (Step 6)

def _triple_scan(real, A, B, n, level):
    """Max-z scan over triples: z_t = (x_t - mean_A) / sd_A per triple; threshold = `level`
    quantile over Set B of max_t z_t (family-wise error over all triples)."""
    from .triples import triple_index
    mu, sd = A.mean(0), np.maximum(A.std(0), EPS)
    z = (real - mu) / sd
    thr = float(np.quantile(((B - mu) / sd).max(1), level))
    tri = triple_index(n)
    k = int(np.argmax(z))
    best = frozenset(tri[k].tolist())
    above = [frozenset(tri[t].tolist()) for t in np.flatnonzero(z > thr)]
    return Result([best] if z[k] > thr else [], {"z": z, "threshold": thr, "z_max": float(z[k]),
                                                  "argmax": best, "above": above})


def detect_tc_scan(ctx: NullContext, level=0.95) -> Result:
    """Total-correlation scan over triples (TC captures pairwise + higher-order dependence)."""
    return _triple_scan(ctx.tc_real, ctx.tc_A, ctx.tc_B, ctx.n, level)


def detect_synergy_scan(ctx: NullContext, level=0.95) -> Result:
    """Scan over triples of E = third-order (beyond-pairwise) part of TC."""
    return _triple_scan(ctx.e_real, ctx.e_A, ctx.e_B, ctx.n, level)


TRIPLE_DETECTORS = {"tc_scan": detect_tc_scan, "synergy_scan": detect_synergy_scan}


# ---------------------------------------------------------------- unified interface

DETECTORS = {
    "bonjour": detect_bonjour,
    "bonjour_cal": detect_bonjour_calibrated,
    "edge_bh": detect_edge_bh,
    "node_score": detect_node_score,
    "subgraph": detect_subgraph,
    "subgraph_grow": lambda ctx, **kw: detect_subgraph(ctx, grow=True, **kw),
}

# names for result tables / write-up
DETECTOR_LABELS = {
    "bonjour": "Bonjour et al. (fixed alpha = 0.05)",
    "bonjour_cal": "Bonjour, permutation-calibrated threshold",
    "edge_bh": "Per-edge BH (Benjamini & Hochberg 1995)",
    "node_score": "Node score (per-agent, inspired by Rose et al.)",
    "subgraph": "Subset scan (ours)",
    "subgraph_grow": "Subset scan + grow (ours)",
    "tc_scan": "Total-correlation triple scan (ours)",
    "synergy_scan": "Third-order (synergy) triple scan (ours)",
}


def run_all(ctx: NullContext, methods=None):
    cache = {}
    out = {}
    for m in methods or DETECTORS:
        kw = {"cache": cache} if m.startswith("subgraph") else {}
        out[m] = DETECTORS[m](ctx, **kw)
    return out


def detect(actions, states=None, method="subgraph", B=200, seed=0):
    if states is None:
        states = np.zeros(len(actions), dtype=np.int64)
    ctx = build_context(actions, states, B=B, seed=seed)
    return DETECTORS[method](ctx).coalitions
