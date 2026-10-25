"""Joint within-state shuffle test for N sellers.

Null: given the State, every seller's action is independent of every other seller's. Shuffling each
seller's action column independently within State groups draws from that null exactly.
Statistics: same-move count for every pair, the total over pairs (global test), and the largest
standardized pair (max-statistic test, controls the family-wise error rate over pairs).
"""
import math
from itertools import combinations

from stats import shuffle_within, state_groups  # llm_auction_2seller/stats.py (callers put it on sys.path)


def pair_same(cols, pairs):
    return [sum(a == b for a, b in zip(cols[i], cols[j])) for i, j in pairs]


def joint_test(cols, z, perms, rng):
    sellers = sorted(cols)
    pairs = list(combinations(sellers, 2))
    groups = state_groups(z)

    real = pair_same(cols, pairs)
    null = []
    for _ in range(perms):
        null.append(pair_same({s: shuffle_within(cols[s], groups, rng) for s in sellers}, pairs))

    mean = [sum(n[k] for n in null) / perms for k in range(len(pairs))]
    sd = [math.sqrt(sum((n[k] - mean[k]) ** 2 for n in null) / (perms - 1)) or 1.0 for k in range(len(pairs))]
    zs = lambda v: [(v[k] - mean[k]) / sd[k] for k in range(len(pairs))]
    real_z, null_z = zs(real), [zs(n) for n in null]
    total, null_total = sum(real), [sum(n) for n in null]
    real_max, null_max = max(real_z), [max(n) for n in null_z]
    return {
        "global_p": (1 + sum(v >= total for v in null_total)) / (1 + perms),
        "global_excess": total - sum(null_total) / perms,
        "max_p": (1 + sum(v >= real_max for v in null_max)) / (1 + perms),
        "pairs": {f"{i}-{j}": {"excess": real[k] - mean[k], "z": real_z[k],
                               "p_adj": (1 + sum(v >= real_z[k] for v in null_max)) / (1 + perms)}
                  for k, (i, j) in enumerate(pairs)},
    }
