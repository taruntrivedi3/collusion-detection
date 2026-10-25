"""Step 5: false positives and power under history dependence (n = 10, B = 1000).

Part A (no colluders): honest mix {uniform, leader, margin, mixed} x state {none, top, order, full}
        x rounds {300, 1000, 3000}, REPS_A reps. Metric: % runs with any detection.
Part B (planted triangle / clique4, CP 0.3, among 'mixed' honest agents): same states x rounds,
        REPS_B reps. Metric: recovery.
Part C (profitable collusion, among 'margin' honest agents): a 4-player 'steer' coalition at
        CP {0.1, 0.2, 0.3}, and a control of 4 honest Exploiters reacting independently at the
        same rate. Metrics: payoff per round, recovery (steer), any detection (exploit).
Colluders and honest types sit at random seats (seed [seed, 1]).
All six detectors use the cell's state definition. Each trajectory is shared across states.

Usage: python step5_run.py A|B|C [workers] [n]   -> results/step5/part{A,B,C}_n{n}/*.csv (resumable)
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import MarginReactive, payoff_round_robin, random_seats, simulate  # noqa: E402
from collusion.estimate import encode_states  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import run_all  # noqa: E402
from collusion.metrics import score  # noqa: E402
from common import WORKERS, load_cells, run_cells  # noqa: E402
from step5_common import MIXES, STATE_DEFS, group, population  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "step5"
OUT.mkdir(parents=True, exist_ok=True)
B, REPS_A, REPS_B, CP = 1000, 200, 200, 0.3
ROUNDS = (300, 1000, 3000)
PLANTED = (None, "triangle", "clique4", "steer", "exploit")
CPS_C = (0.1, 0.2, 0.3)


def one(task):
    """One trajectory, all state definitions."""
    n, mix, planted, rounds, rep, *c = task               # c = [cp] for Part C
    cp = c[0] if c else CP
    key = [n, MIXES.index(mix), PLANTED.index(planted), rounds, rep, 5] + [round(100 * cp)] * bool(c)
    seed = int(np.random.SeedSequence(key).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    pop = random_seats(population(mix, n, planted, cp), np.random.default_rng([seed, 1]))
    a = simulate(pop, rounds, rng)
    pay = payoff_round_robin(a)
    g = group(pop)
    reactive = [i for i, ag in pop.honest.items() if isinstance(ag, MarginReactive)]
    pays = dict(group_pay=pay[:, g].mean() if g else np.nan,
                reactive_pay=pay[:, reactive].mean() if reactive else np.nan)
    rows = []
    for k, state in enumerate(STATE_DEFS):
        s = encode_states(a, state)
        ctx = build_context(a, s, B=B, seed=seed + k + 1)
        for m, r in run_all(ctx).items():
            rows.append(dict(n=n, mix=mix, structure=planted or "none", cp=cp if planted else 0.0, **pays,
                             state=state, n_states=int(s.max() + 1), rounds=rounds, rep=rep,
                             seed=seed, B=B, detector=m, n_planted=len(pop.true_sets),
                             found=str([sorted(x) for x in r.coalitions]),
                             **score(r.coalitions, pop.true_sets)))
    return rows


def cells(part, n):
    if part == "A":
        return [(f"A_n{n}_{mix}_r{r}", [(n, mix, None, r, rep) for rep in range(REPS_A)])
                for mix in MIXES for r in ROUNDS]
    if part == "B":
        return [(f"B_n{n}_{p}_r{r}", [(n, "mixed", p, r, rep) for rep in range(REPS_B)])
                for p in ("triangle", "clique4") for r in ROUNDS]
    return [(f"C_n{n}_{p}_cp{round(100 * cp)}_r{r}", [(n, "margin", p, r, rep, cp) for rep in range(REPS_B)])
            for p in ("steer", "exploit") for cp in CPS_C for r in ROUNDS]


def load(part, n=10):
    return load_cells(OUT / f"part{part}_n{n}")


if __name__ == "__main__" and sys.argv[1] == "summary":
    pass
elif __name__ == "__main__":
    part = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run_cells(cells(part, n), one, OUT / f"part{part}_n{n}", workers)


def partB_false_and_clean(Bp):
    """Mean false coalitions per run, and % runs with the planted coalition recovered AND no
    false coalition ('clean recovery'), for the edge-based set detectors."""
    d = Bp[Bp.detector.isin(["edge_bh", "subgraph", "node_score"])]
    keys = ["structure", "rounds", "state", "detector"]
    fc = d.groupby(keys).false_coal.mean().unstack("detector").round(2)
    d2 = d.assign(clean=(d.recovered == 1) & (d.false_coal == 0))
    clean = (d2.groupby(keys).clean.mean() * 100).unstack("detector").round(0)
    return fc, clean


def summary(n=10):
    from collusion.metrics import wilson
    short = {"bonjour": "Bonj", "bonjour_cal": "Bonj-cal", "edge_bh": "edge-BH", "node_score": "node",
             "subgraph": "scan"}

    def ci(g, col, w=None):
        k = (g[col].astype(float) * (g[w] if w else 1)).sum()
        t = g[w].sum() if w else len(g)
        lo, hi = wilson(k, t)
        return f"{100 * k / t:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"

    order = list(short)
    A, Bp = load("A", n), load("B", n)
    fa = A.groupby(["mix", "rounds", "state", "detector"]).apply(lambda g: ci(g, "detected"), include_groups=False)
    fa = fa.unstack("detector")[order].rename(columns=short)
    pw = Bp.groupby(["structure", "rounds", "state", "detector"]).apply(lambda g: ci(g, "recovered", "n_planted"),
                                                                         include_groups=False)
    pw = pw.unstack("detector")[order].rename(columns=short)
    fa.to_csv(OUT / f"partA_false_alarm_ci_n{n}.csv"); pw.to_csv(OUT / f"partB_recovery_ci_n{n}.csv")
    fc, clean = partB_false_and_clean(Bp)
    fc.to_csv(OUT / f"partB_false_coalitions_n{n}.csv"); clean.to_csv(OUT / f"partB_exact_and_clean_n{n}.csv")
    C = load("C", n)
    one_row = C[(C.state == "none") & (C.detector == "subgraph")]
    pay = one_row.groupby(["structure", "cp", "rounds"])[["group_pay", "reactive_pay"]].mean().round(3)
    cs = C.assign(hit=np.where(C.structure == "steer", C.recovered, C.detected))
    cpw = cs.groupby(["structure", "cp", "rounds", "state", "detector"]).apply(
        lambda g: ci(g, "hit"), include_groups=False).unstack("detector")[order].rename(columns=short)
    pay.to_csv(OUT / f"partC_payoff_n{n}.csv"); cpw.to_csv(OUT / f"partC_detection_ci_n{n}.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print("Part A: % runs with any detection, no colluders [95% Wilson CI]"); print(fa.to_string())
        print("\nPart B: % planted coalitions recovered exactly [95% Wilson CI]"); print(pw.to_string())
        print("\nPart C: mean payoff per agent per round (group = steer coalition or Exploiters)"); print(pay.to_string())
        print("\nPart C: steer = % recovered exactly; exploit = % runs with any detection [95% Wilson CI]")
        print(cpw.to_string())


if __name__ == "__main__" and sys.argv[1] == "summary":
    summary()
