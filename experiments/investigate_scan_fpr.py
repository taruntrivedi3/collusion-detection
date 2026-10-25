"""Investigate the scan's 8% [6, 11] false-alarm rate at n = 20, 300 rounds (Step 4 final).

Stage-1 scan p-value per run: p = (1 + #{max_B >= S_real_max}) / (B + 1); false alarm iff
S_real_max > 95th percentile of max_B (the pipeline rule). Variants:
  V0  original cell: same seeds as step4_final (FINAL_TAG = 1), B = 1000, 500 reps
  V1  fresh seeds, pipeline as-is (simulation seed = permutation seed), B = 200, 1000 reps
  V2  fresh seeds, independent permutation seed, B = 200, 1000 reps
  V3  fresh seeds, "real" input = a within-state permutation of the simulated data, B = 200
Usage: python investigate_scan_fpr.py run | summary
"""
from common import WORKERS, run_cells  # first: sets single-threaded BLAS before numpy is imported

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import Population, simulate  # noqa: E402
from collusion.null import build_context, permute_within_states  # noqa: E402
from collusion.detect import detect_edge_bh, subset_scores  # noqa: E402
from collusion.metrics import wilson  # noqa: E402
from step4_final import FINAL_TAG  # noqa: E402
from step4_planted import STRUCTS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "investigate_scan_fpr"
N, R = 20, 300


def one(task):
    variant, rep = task
    if variant == "V0":
        seed = int(np.random.SeedSequence([N, STRUCTS.index("none"), 0, R, rep, FINAL_TAG]).generate_state(1)[0])
        B, pseed = 1000, seed
    else:
        seed = int(np.random.SeedSequence([N, R, rep, 777]).generate_state(1)[0])
        B = 200
        pseed = seed if variant == "V1" else seed + 1
    a = simulate(Population(N), R, np.random.default_rng(seed))
    s = np.zeros(R, dtype=np.int64)
    if variant == "V3":
        a = next(permute_within_states(a, s, 1, np.random.default_rng(seed + 2)))[0]
    ctx = build_context(a, s, B=B, seed=pseed)
    _, S, maxB = subset_scores(ctx)
    smax = float(S.max())
    thr = float(np.quantile(maxB, 0.95))
    return [dict(variant=variant, rep=rep, seed=seed, perm_seed=pseed, B=B, S_max=smax, threshold=thr,
                 p_scan=(1 + (maxB >= smax).sum()) / (B + 1), scan_fa=smax > thr,
                 bh_fa=bool(detect_edge_bh(ctx).coalitions))]


CELLS = [("V0", 500), ("V1", 1000), ("V2", 1000), ("V3", 1000)]


def summary():
    df = pd.concat([pd.read_csv(OUT / f"{v}.csv") for v, _ in CELLS], ignore_index=True)
    rows = []
    for v, g in df.groupby("variant"):
        for col in ("scan_fa", "bh_fa"):
            k, t = int(g[col].sum()), len(g)
            lo, hi = wilson(k, t)
            rows.append(dict(variant=v, detector=col[:-3], n=t, fa_pct=100 * k / t, lo=100 * lo, hi=100 * hi))
        # uniformity of stage-1 scan p-values
        h = np.histogram(g.p_scan, bins=10, range=(0, 1))[0]
        rows[-2]["p_deciles"] = h.tolist()
        rows[-2]["frac_p_le_0.05"] = float((g.p_scan <= 0.05).mean())
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    # Level of the pipeline rule S > np.quantile(maxB, 0.95) (linear interpolation) under an exact
    # null: h = 0.95 (B - 1); the cut lies between order statistics floor(h) and floor(h) + 1, at
    # fraction frac(h). Level ≈ (B - floor(h) - 1 + (1 - frac(h))) / (B + 1):
    # 10.95 / 201 = 5.45% at B = 200; 50.95 / 1001 = 5.09% at B = 1000.
    lines = []
    for B in (200, 1000):
        h = 0.95 * (B - 1)
        lvl = (B - np.floor(h) - 1 + (1 - (h - np.floor(h)))) / (B + 1)
        lines.append(f"level of the rule at B = {B}: {100 * lvl:.2f}%")
    fresh = df[df.variant.isin(["V1", "V2", "V3"])]
    for col in ("scan_fa", "bh_fa"):
        k, t = int(fresh[col].sum()), len(fresh)
        lo, hi = wilson(k, t)
        lines.append(f"V1-V3 pooled {col[:-3]}: {k}/{t} = {100 * k / t:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]")
    for v, g in df.groupby("variant"):
        h = np.histogram(g.p_scan, bins=10, range=(0, 1))[0]
        e = len(g) / 10
        lines.append(f"{v}: chi2 uniformity of scan p-value deciles = {((h - e) ** 2 / e).sum():.1f} "
                     f"(df 9, 95% critical value 16.9)")
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_cells([(v, [(v, r) for r in range(k)]) for v, k in CELLS], one, OUT, WORKERS)
    else:
        summary()
