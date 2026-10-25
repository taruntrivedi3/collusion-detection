"""Build viewer/data.js: a few simulated RPS games with every detector's verdict, plus sample
LLM auction games with condition-level test results. Open viewer/index.html afterwards.

  python viewer/export.py            (~1 min)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUCTION = ROOT / "studies" / "llm_auction"
sys.path[:0] = [str(ROOT), str(AUCTION)]
from collusion.detect import (detect_bonjour, detect_edge_bh, detect_node_score, detect_subgraph,  # noqa: E402
                              detect_synergy_scan, detect_tc_scan, edge_z)
from collusion.estimate import encode_states  # noqa: E402
from collusion.game import Coalition, LeaderReactive, Population, simulate  # noqa: E402
from collusion.null import build_context  # noqa: E402

N, B = 8, 300
DETECTORS = [("bonjour", "Bonjour et al. (pairs only, fixed cutoff)", detect_bonjour),
             ("edge_bh", "Pair test", detect_edge_bh),
             ("node_score", "Player score", detect_node_score),
             ("subgraph", "Group scan (ours)", detect_subgraph),
             ("tc_scan", "Triple scan: total correlation (ours)", detect_tc_scan),
             ("synergy_scan", "Triple scan: third-order part (ours)", detect_synergy_scan)]

# (id, title, explanation, population, rounds, detector state, seed)
SCENARIOS = [
    ("triangle", "Triangle: 3 players share a secret signal",
     "Players 1, 4 and 6 play the same move in 30% of rounds. Inside the group, Bonjour's net-influence "
     "score cancels out, so it reports nothing. Every shuffle-calibrated method finds this strong a triangle.",
     Population(N, [Coalition("triangle", [1, 4, 6], 0.3)]), 300, "none", 1),
    ("clique5", "5-player clique among 10, weak signal, short game",
     "Players 0, 3, 4, 7 and 9 coordinate in only 25% of 100 rounds. Each pair is weakly linked, so the pair "
     "test and player score miss or split the group; scoring the group as a whole adds the evidence up. "
     "In 40 games like this, exact recovery was 80% for the group scan vs 40% (pair test) and 57% (player score).",
     Population(10, [Coalition("clique", [0, 3, 4, 7, 9], 0.25)]), 100, "none", 7),
    ("hub", "Hub: one leader, two helpers",
     "Helpers 3 and 6 each feed leader 1 a point; the helpers never coordinate with each other. This is the "
     "group scan's weak spot: it reports one leader-helper pair, while the pair test joins both edges. "
     "In 16 games like this, the group scan split the hub in 9.",
     Population(N, [Coalition("hub", [1, 3, 6], 0.2)]), 300, "none", 0),
    ("xor", "xor: only visible three at a time",
     "Player 5 plays (move of 0 + move of 2) mod 3 in 60% of rounds. Every pair looks independent, so all "
     "pairwise methods are blind; the triple scans find it.",
     Population(N, [Coalition("xor", [0, 2, 5], 0.6)]), 1000, "none", 5),
    ("reactive_none", "No collusion, but honest players react to history (history ignored)",
     "Half the players counter last round's most common move. The detector shuffles moves across all rounds, "
     "so this honest reaction looks like coordination: a false alarm.",
     Population(N, honest={i: LeaderReactive(0.6) for i in range(0, N, 2)}), 300, "none", 5),
    ("reactive_top", "Same game, history accounted for",
     "Identical moves, but the detector shuffles only within rounds that share the same leading move. "
     "The honest reaction is explained away and nothing is flagged.",
     Population(N, honest={i: LeaderReactive(0.6) for i in range(0, N, 2)}), 300, "top", 5),
]


def rps_game(sid, title, text, pop, rounds, state_kind, seed):
    actions = simulate(pop, rounds, np.random.default_rng(seed))
    states = encode_states(actions, state_kind)
    ctx = build_context(actions, states, B=B, seed=seed, triples=True)
    z = edge_z(ctx)[0]
    results = []
    for key, label, f in DETECTORS:
        r = f(ctx)
        results.append({"key": key, "label": label, "found": [sorted(map(int, c)) for c in r.coalitions]})
    sig = detect_edge_bh(ctx).info["sig_edges"]
    return {"id": sid, "title": title, "text": text, "n": pop.n, "rounds": rounds, "state": state_kind,
            "moves": ["".join("RPS"[m] for m in actions[:, i]) for i in range(pop.n)],
            "planted": [sorted(c.members) for c in pop.coalitions],
            "coalitions": [{"kind": c.kind, "members": list(c.members)} for c in pop.coalitions],
            "honest_reactive": sorted(pop.honest),
            "z": np.round(z, 1).tolist(), "sig_edges": [list(map(int, e)) for e in sig],
            "results": results}


def auction():
    from build_report import CONDS, LABEL, load, outcomes, pooled_z
    from market import SELLER_COST
    games = load()
    oc, pz = outcomes(games), pooled_z(games)
    conds = []
    for c in CONDS:
        sample = []
        for seed in sorted(games[c])[:12]:
            log = games[c][seed]
            profit = sum(r["price"] - SELLER_COST for r in log if r["price"] is not None) / len(log)
            sample.append({"seed": seed, "profit_per_hour": round(profit, 2),
                           "same_moves": sum(r["actions"]["S1"] == r["actions"]["S2"] for r in log),
                           "log": log})
        conds.append({"id": c, "label": LABEL[c], "trade_rate": round(oc[c][0], 3),
                      "mean_price": round(oc[c][1], 2), "seller_profit_per_hour": round(oc[c][2], 2),
                      "z_cmi": round(pz[(c, "cmi")][0], 1), "z_same": round(pz[(c, "same")][0], 1),
                      "games": sample})
    return {"conditions": conds, "seller_cost": SELLER_COST}


if __name__ == "__main__":
    data = {"rps": [rps_game(*s) for s in SCENARIOS], "auction": auction()}
    for g in data["rps"]:
        print(f'{g["id"]:14s}', {r["key"]: r["found"] for r in g["results"]})
    out = Path(__file__).with_name("data.js")
    out.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
