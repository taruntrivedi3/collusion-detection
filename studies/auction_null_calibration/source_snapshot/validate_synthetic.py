"""Known-dependence examples and an empirical calibration check under iid play."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from information import scan


def null_trial(index):
    rng = np.random.default_rng(76000 + index)
    actions = rng.integers(3, size=(29, 5))
    messages = rng.integers(4, size=(29, 5))
    states = np.repeat([0, 1, 2], [9, 10, 10])
    result = scan({"direction": actions, "level": actions}, states,
                  permutations=199, seed=96000 + index, messages=messages)
    return result["omnibus_surrogate_p"]


def examples(permutations):
    rows = []
    for rounds in (29, 300):
        for case in ("independent", "pair", "triangle", "xor", "two_pairs", "fixed_policy", "public_signal_unconditioned", "public_signal_conditioned"):
            rng = np.random.default_rng(38000 + rounds)
            x = rng.integers(2, size=(rounds, 5))
            states = np.zeros(rounds, dtype=int)
            if case == "pair":
                x[:, 1] = x[:, 0]
            elif case in ("triangle", "public_signal_unconditioned", "public_signal_conditioned"):
                x[:, 1] = x[:, 2] = x[:, 0]
                if case == "public_signal_conditioned":
                    states = x[:, 0].copy()
            elif case == "xor":
                x[:, 2] = x[:, 0] ^ x[:, 1]
            elif case == "two_pairs":
                x[:, 1], x[:, 3] = x[:, 0], x[:, 2]
            elif case == "fixed_policy":
                x[:, :3] = 1
            result = scan({"direction": x}, states, permutations=permutations, seed=28000 + rounds)
            top = result["candidates"][0]
            row = {"case": case, "rounds": rounds, "surrogate_p": result["omnibus_surrogate_p"],
                   "winner_method": top["method"], "winner_members": top["members"],
                   "bonjour_pairs_above_005": result["details"]["direction"]["bonjour_pairs_above_005"]}
            for method in ("subset", "tc", "third_order"):
                best = next(c for c in result["candidates"] if c["method"] == method)
                row[method] = {"members": best["members"], "p_max_adjusted": best["p_max_adjusted"]}
            rows.append(row)
            print(f"synthetic {case} R={rounds}: p={row['surrogate_p']:.4f}, {top['method']} {top['members']}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = examples(1000)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pvalues = list(pool.map(null_trial, range(args.null_trials)))
    flags = sum(p <= .05 for p in pvalues)
    interval = binomtest(flags, len(pvalues)).proportion_ci(confidence_level=.95, method="exact")
    evidence = {"examples": rows, "iid_null_calibration": {
        "trials": len(pvalues), "flags_at_005": flags, "rate": flags / len(pvalues),
        "exact_95_interval": [interval.low, interval.high], "permutations_per_bank": 199,
        "rounds_per_trial": 29, "agents": 5, "states": [9, 10, 10], "pvalues": pvalues},
        "limitations": "Calibration trials have independent actions conditional on state. They do not validate exchangeability in real market trajectories. Examples are individual seeded demonstrations, not power estimates."}
    (args.output / "synthetic_validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"iid_null_flags": flags, "trials": len(pvalues), "exact_95_interval": [interval.low, interval.high]}), flush=True)
