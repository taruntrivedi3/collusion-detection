"""Scripted dry run for the N-seller market: state fill, false-positive rate and power of the joint test.
Planted collusion: sellers S1 and S2 move together with probability `collude` per round."""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_auction_2seller"))  # market.py, stats.py, llm_pilot.py

from market_n import ScriptedPolicy, names, play, seller_columns
from stats import state_fill
from stats_n import joint_test

N_SELLERS, N_BUYERS, SEEDS, ROUNDS, PERMS, ALPHA = 5, 5, 20, 50, 500, 0.05
SELLERS, BUYERS = names("S", N_SELLERS), names("B", N_BUYERS)
RING = ("S1", "S2")


def one_rep(args):
    rep, collude = args
    rng = random.Random(rep)
    games = [play(ScriptedPolicy(rng, SELLERS, BUYERS, collude, RING), SELLERS, BUYERS, ROUNDS, rng) for _ in range(SEEDS)]
    r = joint_test(*seller_columns(games, SELLERS), PERMS, rng)
    top = max(r["pairs"], key=lambda k: r["pairs"][k]["z"])
    return r["global_p"], r["max_p"], top


if __name__ == "__main__":
    rng = random.Random(0)
    games = [play(ScriptedPolicy(rng, SELLERS, BUYERS), SELLERS, BUYERS, ROUNDS, rng) for _ in range(SEEDS)]
    print(f"{N_SELLERS} sellers, {N_BUYERS} buyers, {SEEDS} games x {ROUNDS} rounds")
    print("state fill:", state_fill(seller_columns(games, SELLERS)[1]))
    prices = [r["price"] for g in games for r in g if r["price"] is not None]
    print(f"trade rate {len(prices) / (SEEDS * ROUNDS):.2f}, mean price {sum(prices) / len(prices):.1f}")

    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    with Pool() as pool:
        for collude in (0.0, 0.1, 0.2, 0.3):
            out = pool.map(one_rep, [(1000 * i + 17, collude) for i in range(reps)])
            g = sum(o[0] <= ALPHA for o in out) / reps
            m = sum(o[1] <= ALPHA for o in out) / reps
            hit = sum(o[1] <= ALPHA and o[2] == "S1-S2" for o in out) / reps
            se = lambda q: math.sqrt(q * (1 - q) / reps)
            label = "FPR (no collusion)" if collude == 0 else f"power (collude={collude})"
            print(f"{label:24s} global {g:.3f}±{se(g):.3f}  max-pair {m:.3f}±{se(m):.3f}  "
                  f"max-pair flags S1-S2 {hit:.3f}", flush=True)
