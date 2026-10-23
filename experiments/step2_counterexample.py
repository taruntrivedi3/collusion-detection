"""Step 2: minimal counterexample. 5 agents, history-free, round-robin payoff (payoffs do not
enter detection). All detectors use state = none.

  triangle {0,1,2} + outsiders {3,4}
  hub      primary 0, helpers 1 and 2 + outsiders {3,4}
  pair     {0,1} + outsiders {2,3,4}       (control: case the baseline is built for)
  none     5 uniform agents                (false-positive control)

Part A: one trajectory per structure: print gamma, Gamma, z matrices and every detector's output.
Part B: recovery over REPS trajectories per (structure, CP, rounds).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import Coalition, Population, simulate  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import DETECTORS, edge_z, run_all  # noqa: E402
from collusion.estimate import net_influence  # noqa: E402
from collusion.metrics import score  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "step2"
OUT.mkdir(parents=True, exist_ok=True)
N, B, REPS = 5, 200, 100


def population(kind, cp):
    if kind == "triangle":
        return Population(N, [Coalition("triangle", [0, 1, 2], cp)])
    if kind == "hub":
        return Population(N, [Coalition("hub", [0, 1, 2], cp)])
    if kind == "pair":
        return Population(N, [Coalition("pair", [0, 1], cp)])
    return Population(N)


def fmt(M):
    return pd.DataFrame(M).round(3).to_string()


def part_a(cp=0.5, rounds=1000, seed=7):
    lines = []
    for k, kind in enumerate(("triangle", "hub", "pair")):
        rng = np.random.default_rng(seed + k)
        pop = population(kind, cp)
        a = simulate(pop, rounds, rng)
        ctx = build_context(a, np.zeros(rounds, dtype=np.int64), B=B, seed=seed + k)
        zr, _, _ = edge_z(ctx)
        res = run_all(ctx)
        lines += [f"=== {kind}  truth={sorted(map(sorted, pop.true_sets))}  CP={cp} rounds={rounds} "
                  f"seed={seed + k} B={B}",
                  "gamma (bits):", fmt(ctx.g_real),
                  "Gamma(i;j) = gamma(i;j) - max_k gamma(k;j)  [row i, col j]:", fmt(net_influence(ctx.g_real)),
                  "z (edge z-score vs within-state permutation null):", fmt(zr)]
        for m, r in res.items():
            extra = ""
            if m == "bonjour_cal":
                extra = f"  threshold={r.info['threshold']:.4f}"
            if "flagged_pairs" in r.info:
                extra += f"  flagged_pairs={[sorted(p) for p in r.info['flagged_pairs']]}"
            if m == "subgraph":
                extra = "  " + "; ".join(f"S={t['S_max']:.2f} thr={t['threshold']:.2f} argmax={sorted(t['argmax'])}"
                                         for t in r.info["trace"])
            if m == "node_score":
                extra = f"  scores={np.round(r.info['node_score'], 1).tolist()} thr={r.info['threshold']:.1f}"
            lines.append(f"{m:12s} -> {[sorted(c) for c in r.coalitions]}{extra}")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "matrices.txt").write_text(text)
    print(text)


def part_b():
    rows = []
    grid = [(kind, cp, r) for kind in ("triangle", "hub", "pair") for cp in (0.2, 0.3, 0.5)
            for r in (300, 1000)] + [("none", 0.0, r) for r in (300, 1000)]
    for c, (kind, cp, rounds) in enumerate(grid):
        pop = population(kind, cp)
        for rep in range(REPS):
            seed = 1_000_000 * (c + 1) + rep
            rng = np.random.default_rng(seed)
            a = simulate(pop, rounds, rng)
            ctx = build_context(a, np.zeros(rounds, dtype=np.int64), B=B, seed=seed)
            for m, r in run_all(ctx).items():
                rows.append(dict(structure=kind, cp=cp, rounds=rounds, rep=rep, seed=seed, B=B,
                                 detector=m, found=str([sorted(x) for x in r.coalitions]),
                                 **score(r.coalitions, pop.true_sets)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recovery_runs.csv", index=False)
    summ = (df.groupby(["structure", "cp", "rounds", "detector"], sort=False)
              [["detected", "exact", "best_jacc"]].mean().mul(100).round(0).unstack("detector"))
    summ = summ.reindex(columns=list(DETECTORS), level=1)
    summ.to_csv(OUT / "recovery_summary.csv")
    with pd.option_context("display.width", 250):
        for met in ("exact", "detected"):
            print(f"\n% runs, metric = {met}"); print(summ[met].to_string())


if __name__ == "__main__":
    part_a()
    part_b()
