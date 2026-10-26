"""Worked example for Appendix E: one weak 3-clique among 6 players, 300 rounds, CP 0.1, B = 1000.

The pair test reports nothing; GroupScan reports {0, 1, 2}. Players are at fixed seats 0-2 so the
numbers are easy to read.

Usage: python worked_example.py   -> prints the numbers and writes results/worked_example.txt
"""
from common import WORKERS  # noqa: F401  first: sets single-threaded BLAS before numpy is imported

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import Coalition, Population, simulate  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import _mask_to_set, edge_pvalues, edge_z, run_all, subset_scores  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "worked_example.txt"
N, ROUNDS, CP, B, SEED = 6, 300, 0.1, 1000, 0


def main():
    pop = Population(N, [Coalition("clique", [0, 1, 2], CP)])
    a = simulate(pop, ROUNDS, np.random.default_rng(SEED))
    ctx = build_context(a, np.zeros(ROUNDS, dtype=np.int64), B=B, seed=SEED)
    res = run_all(ctx, ["edge_bh", "subgraph"])
    z, p = edge_z(ctx)[0], edge_pvalues(ctx)
    iu, ju = np.triu_indices(N, 1)
    masks, S, maxB = subset_scores(ctx)
    score = {_mask_to_set(m, range(N)): s for m, s in zip(masks, S)}

    lines = [f"{N} players, {ROUNDS} rounds, clique {{0, 1, 2}} at CP {CP}, {B} shuffles, seed {SEED}", "",
             "pair   z      p"]
    lines += [f"{i}-{j}   {z[i, j]:5.2f}  {p[i, j]:.3f}" for i, j in zip(iu, ju)]
    ps = np.sort(p[iu, ju])
    lines += ["", f"BH: k-th smallest p vs 0.05 k / {len(iu)}"]
    lines += [f"  k = {k}: {ps[k - 1]:.4f} vs {0.05 * k / len(iu):.4f}" for k in range(1, 5)]
    lines += ["", f"{len(masks)} groups; top 6 by S(C):"]
    lines += [f"  {sorted(c)}  {s:.2f}" for c, s in sorted(score.items(), key=lambda t: -t[1])[:6]]
    lines += ["", f"threshold (95th percentile of max S over Set B): {np.quantile(maxB, 0.95):.2f}"]
    lines += [f"stage {k + 1}: best {sorted(t['argmax'])} S = {t['S_max']:.2f}, threshold {t['threshold']:.2f}"
              for k, t in enumerate(res["subgraph"].info["trace"])]
    lines += ["", f"pair test reports: {[sorted(c) for c in res['edge_bh'].coalitions]}",
              f"GroupScan reports: {[sorted(c) for c in res['subgraph'].coalitions]}"]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    OUT.write_text(text)


if __name__ == "__main__":
    main()
