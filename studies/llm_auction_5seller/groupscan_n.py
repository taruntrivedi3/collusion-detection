"""GroupScan (pairwise CMI edges, within-state shuffle, scan over seller groups) on LLM auction runs.
Works on 2- and 5-seller run directories: the recorded state is the shuffle state.

  python groupscan_n.py <run dir> [batch sizes, e.g. 5,10,20]    rejection rates per condition
  python groupscan_n.py --dry-run [replications]                  scripted false alarms and power (5 sellers)
"""
import json
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / "llm_auction_2seller")]
from collusion.detect import detect_bonjour, detect_subgraph, subset_scores  # noqa: E402
from collusion.null import build_context  # noqa: E402

M, LEVEL = 1000, 0.95
ACTION = {"tighten": 0, "hold": 1, "widen": 2}


def arrays(logs):
    """Pooled seller actions [rounds, sellers] and state ids [rounds]; a state is the recorded state dict."""
    sellers = sorted(s for s in logs[0][0]["actions"] if s.startswith("S"))
    rows = [r for log in logs for r in log]
    actions = np.array([[ACTION[r["actions"][s]] for s in sellers] for r in rows], dtype=np.int64)
    keys = [json.dumps(r["state"], sort_keys=True) for r in rows]
    states = np.unique(keys, return_inverse=True)[1].astype(np.int64)
    return actions, states, sellers


def scan(args):
    logs, seed = args
    actions, states, sellers = arrays(logs)
    ctx = build_context(actions, states, B=M, seed=seed)
    r = detect_subgraph(ctx, level=LEVEL)
    first = r.info["trace"][0]
    max_b = subset_scores(ctx)[2]  # first-stage Set-B maxima: p-value of the top group
    out = {"groups": ["".join(sellers[i][1:] for i in sorted(c)) for c in r.coalitions],
           "S_max": first["S_max"], "threshold": first["threshold"],
           "p": (1 + int((max_b >= first["S_max"]).sum())) / (1 + M)}
    if len(sellers) >= 3:  # Bonjour's net influence needs a third player; its fixed cutoff needs no shuffles
        name = lambda cs: ["".join(sellers[i][1:] for i in sorted(c)) for c in cs]
        out["bonjour"] = name(detect_bonjour(ctx).coalitions)
        out["bonjour_nostate"] = name(detect_bonjour(build_context(actions, np.zeros_like(states), B=1)).coalitions)
    return out


def label(groups):
    return " + ".join("{" + ",".join(g) + "}" for g in groups) or "none"


def main(run_dir, sizes):
    games = defaultdict(dict)
    for line in open(Path(run_dir) / "games.jsonl"):
        g = json.loads(line)
        games[g["condition"]][g["seed"]] = g["log"]
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
        out = pool.map(scan, jobs)
    res = defaultdict(list)
    for k, r in zip(keys, out):
        res[k].append(r)

    print(f"GroupScan and Bonjour, M = {M} shuffles per set, level {LEVEL}")
    for cond in games:
        p = res[(cond, "all")][0]
        print(f"\n== {cond}: {len(games[cond])} games")
        print(f"   pooled: S_max {p['S_max']:.2f} vs threshold {p['threshold']:.2f} (p = {p['p']:.3f}) -> {label(p['groups'])}" +
              (f"; Bonjour -> {label(p['bonjour'])}, without state -> {label(p['bonjour_nostate'])}" if "bonjour" in p else ""))
        for size in sizes:
            rs = res[(cond, size)]
            if rs:
                hits = sum(bool(r["groups"]) for r in rs)
                found = Counter(label(r["groups"]) for r in rs if r["groups"])
                print(f"   batches of {size:3d} games: {hits}/{len(rs)} flagged  " +
                      ", ".join(f"{g} x{c}" for g, c in found.most_common()))
                for key in ("bonjour", "bonjour_nostate"):
                    if key in rs[0]:
                        print(f"      {key}: {sum(bool(r[key]) for r in rs)}/{len(rs)} flagged  " +
                              ", ".join(f"{g} x{c}" for g, c in Counter(label(r[key]) for r in rs if r[key]).most_common()))


def dry_rep(args):
    rep, collude = args
    from market_n import ScriptedPolicy, names, play
    sellers, buyers = names("S", 5), names("B", 5)
    rng = random.Random(rep)
    logs = [play(ScriptedPolicy(rng, sellers, buyers, collude, ("S1", "S2")), sellers, buyers, 50, rng) for _ in range(20)]
    r = scan((logs, rep))
    return r["groups"], r["bonjour"], r["bonjour_nostate"]


def dry_run(reps):
    print(f"Scripted 5-seller market, 20 games x 50 hours, {reps} replications, M = {M}")
    with Pool() as pool:
        for collude in (0.0, 0.1):
            out = pool.map(dry_rep, [(1000 * i + 17, collude) for i in range(reps)])
            what = "false alarms (no collusion)" if collude == 0 else f"S1-S2 planted in {collude:.0%} of rounds"
            for k, name in enumerate(("GroupScan", "Bonjour", "Bonjour, no state")):
                flagged = sum(bool(o[k]) for o in out) / reps
                exact = sum(o[k] == ["12"] for o in out) / reps
                print(f"{what:34s} {name:18s} flagged {flagged:.3f}  exactly {{1,2}} {exact:.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--dry-run":
        sys.path.insert(0, str(HERE))
        dry_run(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        main(sys.argv[1], [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [5, 10, 20])
