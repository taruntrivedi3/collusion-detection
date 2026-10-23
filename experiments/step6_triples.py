"""Step 6: triple-level total-correlation (TC) and third-order (E) scans vs the edge subset scan.

No state, B = 200 (dev). Structures: xor CP {0.3, 0.5, 1.0}, triangle CP {0.2, 0.3, 0.5},
pair CP 0.3, none. Rounds {300, 1000, 3000}; n {5, 10, 20}. Reps 100 (n <= 10), 50 (n = 20).
Triple stats are computed on the same permuted datasets as gamma.

Usage: python step6_triples.py N [N ...] [--workers W]  -> results/step6/cells/*.csv (resumable)
       python step6_triples.py --null-final [W]   -> results/step6_null_final/cells (B = 1000, 500 reps)
       python step6_triples.py --summary | --null-final-summary
       python step6_triples.py --summary
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORKERS, load_cells, run_cells  # noqa: E402
from collusion.game import Coalition, Population, simulate  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import TRIPLE_DETECTORS, detect_subgraph  # noqa: E402
from collusion.metrics import score, wilson  # noqa: E402
from collusion.triples import triple_index  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "step6"
OUT.mkdir(parents=True, exist_ok=True)
B = 200
KINDS = ["xor", "triangle", "pair", "none"]
DETS = ["subgraph", "tc_scan", "synergy_scan"]


def population(kind, n, cp):
    if kind == "none":
        return Population(n)
    members = [0, 1] if kind == "pair" else [0, 1, 2]
    return Population(n, [Coalition(kind, members, cp)])


def grid():
    cells = [("xor", cp) for cp in (0.3, 0.5, 1.0)] + [("triangle", cp) for cp in (0.2, 0.3, 0.5)]
    cells += [("pair", 0.3), ("none", 0.0)]
    return [(k, cp, r) for k, cp in cells for r in (300, 1000, 3000)]


def run_one(task):
    n, kind, cp, rounds, rep, *final = task           # final = [B, tag] for the B = 1000 null rerun
    Bt, tag = (final[0], final[1]) if final else (B, 6)
    seed = int(np.random.SeedSequence([n, KINDS.index(kind), round(cp * 100), rounds, rep, tag])
               .generate_state(1)[0])
    pop = population(kind, n, cp)
    a = simulate(pop, rounds, np.random.default_rng(seed))
    ctx = build_context(a, np.zeros(rounds, dtype=np.int64), B=Bt, seed=seed, triples=True)
    res = {"subgraph": detect_subgraph(ctx)}
    res.update({m: f(ctx) for m, f in TRIPLE_DETECTORS.items()})
    t012 = 0                                        # triple (0, 1, 2) is index 0
    k2 = 2 * rounds * np.log(2)                     # bits -> G-statistic scale
    null_diag = {
        "null_tc_mean_G": float((ctx.tc_A * k2).mean()), "null_tc_var_G": float((ctx.tc_A * k2).var(0).mean()),
        "null_e_mean_G": float((ctx.e_A * k2).mean()), "null_e_var_G": float((ctx.e_A * k2).var(0).mean()),
    }
    planted = {"tc_012": float(ctx.tc_real[t012]), "e_012": float(ctx.e_real[t012]),
               "z_tc_012": float(res["tc_scan"].info["z"][t012]), "z_e_012": float(res["synergy_scan"].info["z"][t012])}
    rows = []
    for m in DETS:
        r = res[m]
        found = r.coalitions[:1]                    # first detection (scan) / argmax triple
        pair_in = bool(found) and frozenset({0, 1}) <= found[0]
        rows.append(dict(n=n, structure=kind, cp=cp, rounds=rounds, rep=rep, seed=seed, B=Bt, detector=m,
                         found=str([sorted(x) for x in r.coalitions]), contains_pair01=pair_in,
                         n_planted=len(pop.true_sets), **score(found, pop.true_sets), **planted, **null_diag))
    return rows


def run_n(n, workers=WORKERS):
    reps = 100 if n <= 10 else 50
    cl = [(f"n{n}_{k}_cp{cp}_r{r}", [(n, k, cp, r, rep) for rep in range(reps)]) for k, cp, r in grid()]
    run_cells(cl, run_one, OUT / "cells", workers)


NULL_FINAL = OUT.parent / "step6_null_final"


def run_null_final(workers=WORKERS, reps=500, Bf=1000, tag=61):
    """Null cells only, B = 1000, 500 reps, fresh seeds (tag 61)."""
    cl = [(f"n{n}_none_r{r}", [(n, "none", 0.0, r, rep, Bf, tag) for rep in range(reps)])
          for n in (5, 10, 20) for r in (300, 1000, 3000)]
    run_cells(cl, run_one, NULL_FINAL / "cells", workers)


def summarize_null_final():
    df = load_cells(NULL_FINAL / "cells")
    g = df.groupby(["n", "rounds", "detector"]).detected.agg(["sum", "count"]).reset_index()
    lo, hi = wilson(g["sum"], g["count"])
    g["cell"] = [f"{100 * k / t:.1f} [{100 * a:.1f}, {100 * b:.1f}]" for k, t, a, b in zip(g["sum"], g["count"], lo, hi)]
    t = g.pivot_table(index=["n", "rounds"], columns="detector", values="cell", aggfunc="first")[DETS]
    t.to_csv(NULL_FINAL / "null_fpr_ci.csv")
    one = df[df.detector == "tc_scan"]
    nd = one.groupby(["n", "rounds"])[["null_tc_mean_G", "null_tc_var_G", "null_e_mean_G", "null_e_var_G"]].mean()
    nd["tc_df_from_mean"], nd["tc_df_from_var"] = nd.null_tc_mean_G, nd.null_tc_var_G / 2
    nd["e_df_from_mean"], nd["e_df_from_var"] = nd.null_e_mean_G, nd.null_e_var_G / 2
    nd.round(2).to_csv(NULL_FINAL / "null_vs_chi2_df.csv")
    with pd.option_context("display.width", 250):
        print("B = 1000, 500 reps: % runs with any detection [95% Wilson CI]"); print(t.to_string())
        print("\nImplied chi2 degrees of freedom (mean of G = df; var of G = 2 df). Theory: TC 20, E 8")
        print(nd[["tc_df_from_mean", "tc_df_from_var", "e_df_from_mean", "e_df_from_var"]].round(2).to_string())


def summarize():
    df = load_cells(OUT / "cells")
    keys = ["structure", "cp", "n", "rounds"]

    def ci(g, col):
        k, t = g[col].astype(float).sum(), len(g)
        lo, hi = wilson(k, t)
        return f"{100 * k / t:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"

    tabs = {}
    for name, sub, col in (("exact recovery of planted triple (%)", df[df.structure.isin(["xor", "triangle"])], "recovered"),
                           ("pair: first detection contains the pair {0,1} (%)", df[df.structure == "pair"], "contains_pair01"),
                           ("none: any detection = false alarm (%)", df[df.structure == "none"], "detected")):
        t = sub.groupby(keys + ["detector"]).apply(lambda g: ci(g, col), include_groups=False).unstack("detector")[DETS]
        tabs[name] = t
        t.to_csv(OUT / f"summary_{col}_{name.split(':')[0].split()[0]}.csv")
    one = df[df.detector == "tc_scan"]
    planted = one[one.structure.isin(["xor", "triangle"])].groupby(["structure", "cp", "rounds"])[
        ["tc_012", "e_012", "z_tc_012", "z_e_012"]].median().round(3)
    planted.to_csv(OUT / "planted_triple_values.csv")
    nd = one[one.structure == "none"].groupby(["n", "rounds"])[
        ["null_tc_mean_G", "null_tc_var_G", "null_e_mean_G", "null_e_var_G"]].mean().round(2)
    nd.to_csv(OUT / "null_vs_chi2.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        for name, t in tabs.items():
            print(f"\n{name}  [95% Wilson CI]"); print(t.to_string())
        print("\nPlanted triple (0,1,2): median TC, E (bits) and their z vs own null (pooled over n):")
        print(planted.to_string())
        print("\nNull (none) on G scale 2N ln2 x stat: chi2 theory TC mean 20 / var 40, E mean 8 / var 16:")
        print(nd.to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        summarize()
    elif sys.argv[1:2] == ["--null-final"]:
        run_null_final(int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS)
    elif sys.argv[1:] == ["--null-final-summary"]:
        summarize_null_final()
    else:
        args = sys.argv[1:]
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else WORKERS
        ns = [int(a) for i, a in enumerate(args) if a != "--workers" and (i == 0 or args[i - 1] != "--workers")]
        for n in ns:
            run_n(n, w)
