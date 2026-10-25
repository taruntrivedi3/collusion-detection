"""Step 4: planted-structure experiments, history-free, all detectors use state = none.

Usage: python step4_planted.py N [N ...]  -> results/step4/runs_n{N}.csv (one row per run x detector)
       python step4_planted.py --summary  -> summary tables from all runs_n*.csv

Grid (per n):
  core     two_pairs, triangle             CP {0.1, 0.2, 0.3, 0.5}       rounds {100, 300, 1000}
  cliques  clique4, clique5 (n >= 10)      CP {0.05, 0.1, 0.2, 0.3, 0.5} rounds {100, 300, 1000}
  reduced  pair, hub, xor                  CP {0.2, 0.5}                 rounds {300, 1000}
  null     none                                                          rounds {100, 300, 1000}
Colluders sit at random seats (seed [seed, 1]).
Reps: 100 for n <= 10, 50 for n = 20. Permutations B = 200 per set (dev setting).
Edge-BH uses gamma-fit permutation p-values (see detect.edge_pvalues).
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import Coalition, Population, random_seats, simulate  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import DETECTOR_LABELS, DETECTORS, run_all  # noqa: E402
from collusion.metrics import score  # noqa: E402
from common import WORKERS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "step4"
OUT.mkdir(parents=True, exist_ok=True)
B = 200
STRUCTS = ["pair", "two_pairs", "hub", "triangle", "clique4", "clique5", "xor", "none"]


def population(kind, n, cp):
    C = {
        "pair": [("pair", [0, 1])],
        "two_pairs": [("pair", [0, 1]), ("pair", [2, 3])],
        "hub": [("hub", [0, 1, 2])],
        "triangle": [("triangle", [0, 1, 2])],
        "clique4": [("clique", [0, 1, 2, 3])],
        "clique5": [("clique", [0, 1, 2, 3, 4])],
        "xor": [("xor", [0, 1, 2])],
        "none": [],
    }[kind]
    return Population(n, [Coalition(k, m, cp) for k, m in C])


def grid(n):
    cells = []
    for kind in ("two_pairs", "triangle"):
        cells += [(kind, cp, r) for cp in (0.1, 0.2, 0.3, 0.5) for r in (100, 300, 1000)]
    for kind in ("clique4", "clique5"):
        if kind == "clique5" and n < 10:
            continue
        cells += [(kind, cp, r) for cp in (0.05, 0.1, 0.2, 0.3, 0.5) for r in (100, 300, 1000)]
    for kind in ("pair", "hub", "xor"):
        cells += [(kind, cp, r) for cp in (0.2, 0.5) for r in (300, 1000)]
    cells += [("none", 0.0, r) for r in (100, 300, 1000)]
    return cells


def run_one(task):
    n, kind, cp, rounds, rep = task
    seed = int(np.random.SeedSequence([n, STRUCTS.index(kind), round(cp * 100), rounds, rep])
               .generate_state(1)[0])
    rng = np.random.default_rng(seed)
    pop = random_seats(population(kind, n, cp), np.random.default_rng([seed, 1]))
    a = simulate(pop, rounds, rng)
    ctx = build_context(a, np.zeros(rounds, dtype=np.int64), B=B, seed=seed)
    rows = []
    for m, r in run_all(ctx).items():
        rows.append(dict(n=n, structure=kind, cp=cp, rounds=rounds, rep=rep, seed=seed, B=B,
                         detector=m, found=str([sorted(x) for x in r.coalitions]),
                         **score(r.coalitions, pop.true_sets)))
    return rows


def run_n(n, workers=WORKERS):
    reps = 100 if n <= 10 else 50
    tasks = [(n, *cell, rep) for cell in grid(n) for rep in range(reps)]
    with Pool(workers) as pool:
        rows = [row for rs in pool.imap_unordered(run_one, tasks, chunksize=4) for row in rs]
    df = pd.DataFrame(rows).sort_values(["structure", "cp", "rounds", "rep", "detector"])
    df.to_csv(OUT / f"runs_n{n}.csv", index=False)
    print(f"n={n}: {len(tasks)} runs -> {OUT / f'runs_n{n}.csv'}", flush=True)


DET_ORDER = list(DETECTORS)
SHORT = {"bonjour": "Bonj", "bonjour_cal": "Bonj-cal", "edge_bh": "edge-BH",
         "node_score": "node", "subgraph": "scan"}


def summarize():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("runs_n*.csv"))])
    keys = ["structure", "n", "cp", "rounds"]
    planted = df[df.structure != "none"]
    cell = (planted.groupby(keys + ["detector"])[["recovered", "exact_first", "false_coal", "best_jacc"]]
            .mean().reset_index())
    rec = cell.pivot_table(index=keys, columns="detector", values="recovered").mul(100)[DET_ORDER]
    fc = cell.pivot_table(index=keys, columns="detector", values="false_coal")[DET_ORDER]
    rec.round(1).to_csv(OUT / "cell_recovered_pct.csv")
    fc.round(3).to_csv(OUT / "cell_false_coalitions.csv")
    # structure x n summary (mean over that structure's CP x rounds cells)
    sxn = rec.groupby(level=["structure", "n"]).mean().round(0)
    sxn["cells"] = rec.groupby(level=["structure", "n"]).size()
    sxn.to_csv(OUT / "summary_structure_by_n.csv")
    # null: false-positive rate and false coalitions per run
    nul = df[df.structure == "none"].groupby(["n", "rounds", "detector"])[["detected", "false_coal"]].mean()
    nul = nul.assign(detected=nul.detected * 100).unstack("detector")
    nul = nul.reindex(columns=DET_ORDER, level=1)
    nul.round(2).to_csv(OUT / "null_fpr.csv")
    # flagged cells: scan beats both node-score and edge-BH by > 10 points (recovered)
    flags = []
    for scan in ("subgraph",):
        d = rec[(rec[scan] - rec[["node_score", "edge_bh"]].max(axis=1)) > 10]
        flags.append(d.assign(scan=scan, margin=d[scan] - d[["node_score", "edge_bh"]].max(axis=1)))
    flags = pd.concat(flags).round(1)
    flags.to_csv(OUT / "flagged_cells.csv")
    # reverse: a baseline beats the scan by > 10
    rev = rec[(rec[["node_score", "edge_bh"]].max(axis=1) - rec["subgraph"]) > 10].round(1)
    rev.to_csv(OUT / "scan_loses_cells.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print("Detector key:", "; ".join(f"{SHORT[k]} = {v}" for k, v in DETECTOR_LABELS.items() if k in SHORT))
        print("\n% planted coalitions recovered exactly, mean over CP x rounds cells:")
        print(sxn.rename(columns=SHORT).to_string())
        print("\nNull (no collusion): % runs with any detection | mean false coalitions per run")
        print(nul.rename(columns=SHORT, level=1).to_string())
        print("\nCells where scan beats max(node, edge-BH) by > 10 points (recovered %):")
        print(flags.rename(columns=SHORT).to_string())
        print("\nCells where max(node, edge-BH) beats scan by > 10 points:")
        print(rev.rename(columns=SHORT).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        summarize()
    else:
        for n in map(int, sys.argv[1:]):
            run_n(n)
