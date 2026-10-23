"""Step 4 final numbers at B = 1000 (fresh trajectories, seeds tagged 'final').

  planted : every cell in results/step4/flagged_cells.csv and scan_loses_cells.csv, plus the
            Fig C cells (clique4 / clique5, n = 20, 100 rounds, all CP)       100 reps
  null    : n in {5, 10, 20} x rounds {100, 300, 1000}                          500 reps

Usage: python step4_final.py run [workers]   -> results/step4_final/cells/*.csv (resumable)
       python step4_final.py summary         -> tables with 95% Wilson CIs
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import simulate  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import DETECTORS, run_all  # noqa: E402
from collusion.metrics import score, wilson  # noqa: E402
from common import WORKERS, load_cells, run_cells  # noqa: E402
from step4_planted import OUT as DEV, SHORT, STRUCTS, population  # noqa: E402

OUT = DEV.parent / "step4_final"
OUT.mkdir(parents=True, exist_ok=True)
B = 1000
DET_ORDER = list(DETECTORS)
FINAL_TAG = 1


def cells():
    keys = ["structure", "n", "cp", "rounds"]
    c = pd.concat([pd.read_csv(DEV / "flagged_cells.csv")[keys], pd.read_csv(DEV / "scan_loses_cells.csv")[keys]])
    figc = [(s, 20, cp, 100) for s in ("clique4", "clique5") for cp in (0.05, 0.1, 0.2, 0.3, 0.5)]
    c = pd.concat([c, pd.DataFrame(figc, columns=keys)]).drop_duplicates()
    planted = [(s, int(n), float(cp), int(r), 100) for s, n, cp, r in c.itertuples(index=False)]
    null = [("none", n, 0.0, r, 500) for n in (5, 10, 20) for r in (100, 300, 1000)]
    return planted + null


def run_one(task):
    kind, n, cp, rounds, rep = task
    seed = int(np.random.SeedSequence([n, STRUCTS.index(kind), round(cp * 100), rounds, rep, FINAL_TAG])
               .generate_state(1)[0])
    rng = np.random.default_rng(seed)
    pop = population(kind, n, cp)
    a = simulate(pop, rounds, rng)
    ctx = build_context(a, np.zeros(rounds, dtype=np.int64), B=B, seed=seed)
    return [dict(n=n, structure=kind, cp=cp, rounds=rounds, rep=rep, seed=seed, B=B, detector=m,
                 found=str([sorted(x) for x in r.coalitions]), n_planted=len(pop.true_sets),
                 **score(r.coalitions, pop.true_sets)) for m, r in run_all(ctx).items()]


def run(workers=WORKERS):
    cl = [(f"{k}_n{n}_cp{cp}_r{r}", [(k, n, cp, r, rep) for rep in range(reps)])
          for k, n, cp, r, reps in cells()]
    run_cells(cl, run_one, OUT / "cells", workers)


def rate_table(df, value, keys):
    """Mean rate (%) with 95% Wilson CI per keys x detector. For `recovered`, trials are
    planted coalitions (reps x n_planted)."""
    d = df.assign(k=df[value].astype(float) * (df.n_planted if value == "recovered" else 1),
                  t=(df.n_planted if value == "recovered" else 1))
    g = d.groupby(keys + ["detector"])[["k", "t"]].sum().reset_index()
    lo, hi = wilson(g.k, g.t)
    g["pct"], g["lo"], g["hi"] = 100 * g.k / g.t, 100 * lo, 100 * hi
    g["cell"] = g.apply(lambda r: f"{r.pct:.0f} [{r.lo:.0f}, {r.hi:.0f}]", axis=1)
    return g


def summary():
    df = load_cells(OUT / "cells")
    keys = ["structure", "n", "cp", "rounds"]
    pl, nu = df[df.structure != "none"], df[df.structure == "none"]
    rec = rate_table(pl, "recovered", keys)
    fpr = rate_table(nu, "detected", ["n", "rounds"])
    rec.to_csv(OUT / "recovered_ci.csv", index=False)
    fpr.to_csv(OUT / "null_fpr_ci.csv", index=False)
    wide = rec.pivot_table(index=keys, columns="detector", values="cell", aggfunc="first")[DET_ORDER]
    pw = rec.pivot_table(index=keys, columns="detector", values="pct")[DET_ORDER]
    wide["scan - best(node, BH)"] = (pw.subgraph - pw[["node_score", "edge_bh"]].max(axis=1)).round(0)
    fw = fpr.pivot_table(index=["n", "rounds"], columns="detector", values="cell", aggfunc="first")[DET_ORDER]
    fc = nu.groupby(["n", "rounds", "detector"]).false_coal.mean().unstack()[DET_ORDER].round(3)
    wide.to_csv(OUT / "recovered_ci_wide.csv")
    fw.to_csv(OUT / "null_fpr_ci_wide.csv")
    with pd.option_context("display.width", 300, "display.max_rows", 500, "display.max_colwidth", 20):
        print(f"B = {B}. % recovered exactly [95% Wilson CI], planted cells (100 reps):")
        print(wide.rename(columns=SHORT).to_string())
        print("\nNull: % runs with any detection [95% Wilson CI] (500 reps):")
        print(fw.rename(columns=SHORT).to_string())
        print("\nNull: mean false coalitions per run:")
        print(fc.rename(columns=SHORT).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS)
    else:
        summary()
