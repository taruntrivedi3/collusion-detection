"""Outcomes, pooled joint test and batch rejection rates for an N-seller run directory.
Usage: analyze_n.py <run dir> [batch sizes, e.g. 5,10,20]"""
import json
import math
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_auction_2seller"))  # market.py, stats.py, llm_pilot.py

from market import SELLER_COST
from market_n import seller_columns
from stats import state_fill
from stats_n import joint_test

PERMS, ALPHA = 1000, 0.05


def load(run_dir):
    games = defaultdict(dict)
    for line in open(Path(run_dir) / "games.jsonl"):
        g = json.loads(line)
        games[g["condition"]][g["seed"]] = g["log"]
    return games


def sellers_of(logs):
    return sorted(s for s in logs[0][0]["actions"] if s.startswith("S"))


def test_batch(args):
    logs, seed = args
    return joint_test(*seller_columns(logs, sellers_of(logs)), PERMS, random.Random(seed))


def main(run_dir, sizes):
    games = load(run_dir)
    if (Path(run_dir) / "calls.jsonl").exists():
        calls = Counter()
        for line in open(Path(run_dir) / "calls.jsonl"):
            e = json.loads(line)
            calls["fallback" if "fallback" in e else "error" if "error" in e else "ok" if e.get("parsed") else "unparsed"] += 1
        print("calls:", dict(calls))

    jobs, keys = [], []
    for cond, by_seed in games.items():
        seeds = sorted(by_seed)
        jobs.append(([by_seed[s] for s in seeds], len(jobs)))
        keys.append((cond, "all"))
        for size in sizes:
            for b in range(len(seeds) // size):
                jobs.append(([by_seed[s] for s in seeds[b * size:(b + 1) * size]], len(jobs)))
                keys.append((cond, size))
    with Pool() as pool:
        out = pool.map(test_batch, jobs)
    res = defaultdict(list)
    for k, r in zip(keys, out):
        res[k].append(r)

    for cond, by_seed in games.items():
        logs = list(by_seed.values())
        rows = [r for log in logs for r in log]
        sellers = sellers_of(logs)
        acts = Counter(a for r in rows for s in sellers for a in [r["actions"][s]])
        n_act = sum(acts.values())
        traded = [r for r in rows if r["price"] is not None]
        profit = sum(r["price"] - SELLER_COST for r in traded) / len(rows)
        fill = state_fill(seller_columns(logs, sellers)[1])
        print(f"\n== {cond}: {len(logs)} games, {len(rows)} rounds, trade rate {len(traded) / len(rows):.2f}, "
              f"profit/round {profit:.2f}, mean best ask {sum(min(r['asks'].values()) for r in rows) / len(rows):.1f}")
        print("   actions: " + ", ".join(f"{a} {acts[a] / n_act:.2f}" for a in ("tighten", "hold", "widen")))
        print(f"   state fill: {fill}")
        p = res[(cond, "all")][0]
        print(f"   pooled: global p={p['global_p']:.4f} excess={p['global_excess']:+.1f}  max-pair p={p['max_p']:.4f}")
        for pair, v in sorted(p["pairs"].items(), key=lambda kv: -kv[1]["z"])[:4]:
            print(f"     {pair}: excess {v['excess']:+.1f} z {v['z']:+.2f} p_adj {v['p_adj']:.3f}")
        for size in sizes:
            rs = res[(cond, size)]
            if rs:
                g = sum(r["global_p"] <= ALPHA for r in rs) / len(rs)
                m = sum(r["max_p"] <= ALPHA for r in rs) / len(rs)
                se = lambda q: math.sqrt(q * (1 - q) / len(rs))
                print(f"   batches of {size:3d} games (n={len(rs):2d}): reject global {g:.2f}±{se(g):.2f}, "
                      f"max-pair {m:.2f}±{se(m):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [5, 10, 20])
