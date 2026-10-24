"""Rejection rates of the within-state shuffle test across disjoint seed batches, per condition and batch size."""
import json
import math
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from market import SELLER_COST, seller_rows
from stats import cmi, same_moves, within_state_test

PERMS, ALPHA, BATCH_SIZES = 2000, 0.05, (5, 10, 20)


STATISTICS = {"cmi": cmi, "same_moves": same_moves}


def test_batch(args):
    logs, seed, statistic = args
    x, y, z = seller_rows(logs)
    return within_state_test(x, y, z, PERMS, random.Random(seed), STATISTICS[statistic])[1]


def main(run_dir, statistic):
    games = defaultdict(dict)
    for line in open(Path(run_dir) / "games.jsonl"):
        g = json.loads(line)
        games[g["condition"]][g["seed"]] = g["log"]
    jobs, keys = [], []
    for cond, by_seed in games.items():
        seeds = sorted(by_seed)
        for size in BATCH_SIZES:
            for b in range(len(seeds) // size):
                jobs.append(([by_seed[s] for s in seeds[b * size:(b + 1) * size]], len(jobs), statistic))
                keys.append((cond, size))
    with Pool() as pool:
        ps = pool.map(test_batch, jobs)
    res = defaultdict(list)
    for k, p in zip(keys, ps):
        res[k].append(p)

    print(f"{'condition':16s} {'games':>5s} {'profit/rd':>9s} " + " ".join(f"{f'reject@{s} seeds':>22s}" for s in BATCH_SIZES))
    for cond, by_seed in games.items():
        rows = [r for log in by_seed.values() for r in log]
        profit = sum(r["price"] - SELLER_COST for r in rows if r["price"] is not None) / len(rows)
        cells = []
        for s in BATCH_SIZES:
            p = res[(cond, s)]
            rate = sum(v <= ALPHA for v in p) / len(p)
            cells.append(f"{rate:.2f} ±{math.sqrt(rate * (1 - rate) / len(p)):.2f} (n={len(p)})")
        print(f"{cond:16s} {len(by_seed):5d} {profit:9.2f} " + " ".join(f"{c:>22s}" for c in cells))
    print("\np-values at 20 seeds:")
    for cond in games:
        print(f"  {cond:16s} " + " ".join(f"{p:.3f}" for p in res[(cond, 20)]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cmi")
