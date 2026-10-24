import math
import random
from collections import Counter, defaultdict


def cmi(x, y, z):
    """Plug-in I(X;Y|Z) in nats."""
    n = len(x)
    cxyz, cxz, cyz, cz = Counter(zip(x, y, z)), Counter(zip(x, z)), Counter(zip(y, z)), Counter(z)
    return sum(c / n * math.log(c * cz[k[2]] / (cxz[(k[0], k[2])] * cyz[(k[1], k[2])])) for k, c in cxyz.items())


def same_moves(x, y, z):
    """Number of rows where both sellers took the same action (z unused; the shuffle holds it fixed)."""
    return sum(a == b for a, b in zip(x, y))


def within_state_test(x, y, z, perms, rng, statistic=cmi):
    """Shuffle y among rows sharing the same z. Returns (statistic, p-value, null mean)."""
    real = statistic(x, y, z)
    groups = defaultdict(list)
    for i, zz in enumerate(z):
        groups[zz].append(i)
    null = []
    for _ in range(perms):
        yp = list(y)
        for idxs in groups.values():
            vals = [y[i] for i in idxs]
            rng.shuffle(vals)
            for i, v in zip(idxs, vals):
                yp[i] = v
        null.append(statistic(x, yp, z))
    return real, (1 + sum(v >= real for v in null)) / (1 + perms), sum(null) / perms


def state_fill(z):
    counts = Counter(z)
    sizes = sorted(counts.values())
    return {
        "states_visited": len(counts),
        "rows": len(z),
        "median_rows_per_state": sizes[len(sizes) // 2],
        "rows_in_singleton_states": sum(c for c in sizes if c == 1),
        "rows_in_states_ge5": sum(c for c in sizes if c >= 5),
    }
