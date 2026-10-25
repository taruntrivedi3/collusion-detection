"""Summary of the four 20-game pilots run before rates_v1 (runs/pilot_v1 ... pilot_v4).

Per condition: seller profit per hour, trade rate, the within-state shuffle test with the same-move and CMI
statistics, and the raise-following direction test (does a seller raise more after the other seller raised
than after it lowered, within the same sold-by / ask-level stratum; 95% CI by bootstrapping games).
Usage: python3 pilots.py [run dirs ...]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from market import SELLER_COST, seller_rows
from stats import cmi, same_moves, within_state_test

PERMS, BOOT = 5000, 2000
HERE = Path(__file__).resolve().parent


def raise_counts(logs):
    """Per game: {stratum: [#raise after other raised, #other raised, #raise after other lowered, #other lowered]}."""
    out = []
    for g in logs:
        c = defaultdict(lambda: [0, 0, 0, 0])
        for r in g:
            st = r["state"]
            for me, other in (("S1", "s2_move"), ("S2", "s1_move")):
                stratum = ("me" if st["sold"] == me else "none" if st["sold"] == "none" else "other", st["ask_level"])
                raised = r["actions"][me] == "widen"
                if st[other] == "widen":
                    c[stratum][0] += raised; c[stratum][1] += 1
                elif st[other] == "tighten":
                    c[stratum][2] += raised; c[stratum][3] += 1
        out.append(c)
    return out


def follow_gap(per_game):
    """Within-stratum P(raise | other raised) - P(raise | other lowered), weighted by other-raised rows."""
    t = defaultdict(lambda: [0, 0, 0, 0])
    for c in per_game:
        for k, v in c.items():
            t[k] = [a + b for a, b in zip(t[k], v)]
    usable = {k: v for k, v in t.items() if v[1] and v[3]}
    w = sum(v[1] for v in usable.values())
    if not w:
        return float("nan")
    return sum(v[1] * (v[0] / v[1] - v[2] / v[3]) for v in usable.values()) / w


def direction_test(logs, rng):
    per_game = raise_counts(logs)
    boots = sorted(b for b in (follow_gap([rng.choice(per_game) for _ in per_game]) for _ in range(BOOT)) if b == b)
    return follow_gap(per_game), boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def main(run_dirs):
    print(f"{'run':10s} {'condition':17s} {'games':>5s} {'profit/h':>8s} {'trades':>6s} {'same-move p':>11s} "
          f"{'CMI p':>6s}  direction D [95% CI]")
    for run_dir in run_dirs:
        games = defaultdict(list)
        for line in open(Path(run_dir) / "games.jsonl"):
            g = json.loads(line)
            games[g["condition"]].append(g["log"])
        for cond, logs in games.items():
            rng = random.Random(0)
            x, y, z = seller_rows(logs)
            p_same = within_state_test(x, y, z, PERMS, rng, same_moves)[1]
            p_cmi = within_state_test(x, y, z, PERMS, rng, cmi)[1]
            d, lo, hi = direction_test(logs, rng)
            rows = [r for g in logs for r in g]
            trades = [r for r in rows if r["price"] is not None]
            profit = sum(r["price"] - SELLER_COST for r in trades) / len(rows)
            print(f"{Path(run_dir).name:10s} {cond:17s} {len(logs):5d} {profit:8.2f} {len(trades) / len(rows):6.2f} "
                  f"{p_same:11.4f} {p_cmi:6.4f}  {d:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(str(p) for p in (HERE / "runs").glob("pilot_v*")))
