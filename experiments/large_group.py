"""Very large colluding group: an 18-player clique among 20 players (B = 1000, 100 reps per cell).

Same setup as the Figure C cells in step4_final.py (history-free, state = none), with clique size 18.
Seeds: SS[n, size, round(100 * CP), rounds, rep, 18]; colluders at random seats.

Usage: python large_group.py run [workers]   -> results/large_group/cells/*.csv (resumable)
       python large_group.py summary         -> exact recovery and detection with 95% Wilson CIs
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import Coalition, Population, random_seats, simulate  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import DETECTORS, run_all  # noqa: E402
from collusion.metrics import score, wilson  # noqa: E402
from common import WORKERS, load_cells, run_cells  # noqa: E402
from step4_planted import SHORT  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "large_group"
B, REPS, N, SIZE = 1000, 100, 20, 18
CPS, ROUNDS = (0.05, 0.1, 0.2, 0.3, 0.5), (100, 300)


def run_one(task):
    cp, rounds, rep = task
    seed = int(np.random.SeedSequence([N, SIZE, round(cp * 100), rounds, rep, 18]).generate_state(1)[0])
    pop = random_seats(Population(N, [Coalition("clique", list(range(SIZE)), cp)]), np.random.default_rng([seed, 1]))
    a = simulate(pop, rounds, np.random.default_rng(seed))
    ctx = build_context(a, np.zeros(rounds, dtype=np.int64), B=B, seed=seed)
    return [dict(n=N, size=SIZE, cp=cp, rounds=rounds, rep=rep, seed=seed, B=B, detector=m,
                 found=str([sorted(x) for x in r.coalitions]), n_planted=1, **score(r.coalitions, pop.true_sets))
            for m, r in run_all(ctx).items()]


def summary():
    df = load_cells(OUT / "cells")

    def ci(g, col):
        k, t = g[col].astype(float).sum(), len(g)
        lo, hi = wilson(k, t)
        return f"{100 * k / t:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"

    order = [d for d in DETECTORS if d in set(df.detector)]
    with pd.option_context("display.width", 250):
        for col, name in (("recovered", "exact recovery of the 18-player group"), ("detected", "any report"),
                          ("best_jacc", "best overlap (Jaccard) with the planted group, mean")):
            if col == "best_jacc":
                t = (df.groupby(["cp", "rounds", "detector"])[col].mean() * 100).round(0).unstack("detector")[order]
            else:
                t = df.groupby(["cp", "rounds", "detector"]).apply(lambda g: ci(g, col), include_groups=False).unstack("detector")[order]
            t = t.rename(columns=SHORT)
            t.to_csv(OUT / f"summary_{col}.csv")
            print(f"\n% {name}" + ("" if col == "best_jacc" else " [95% Wilson CI]") + ", n = 20, 100 games per cell"); print(t.to_string())


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS
        cells = [(f"clique{SIZE}_n{N}_cp{cp}_r{r}", [(cp, r, rep) for rep in range(REPS)]) for cp in CPS for r in ROUNDS]
        run_cells(cells, run_one, OUT / "cells", workers)
