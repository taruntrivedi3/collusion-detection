"""Step 5 pre-check: validate the two shortcuts under each state definition on null data.

(a) gamma-fit edge p-values: fit on Set A, evaluate on Set B. Set B datasets are fresh draws
    from the same within-state permutation null, so P(p < x) should equal x if the gamma fit
    matches that null's tail. Reported as ratio P(p < x) / x over all edges x Set-B datasets.
(b) closed-form subset moments: (b1) identity check vs the Set-A sample moments of T(C);
    (b2) out-of-sample: Set-A closed-form mean / sd vs the empirical mean / sd of T(C) on Set B.
Output: results/step5/validate_shortcuts.csv
"""
from common import WORKERS  # first: sets single-threaded BLAS before numpy is imported

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import simulate  # noqa: E402
from collusion.estimate import encode_states  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import _membership_chunks, edge_z, gamma_tail  # noqa: E402
from step5_common import MIXES, STATE_DEFS, population  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "step5"
OUT.mkdir(parents=True, exist_ok=True)
N, B, RUNS = 10, 1000, 4
XS = (0.05, 0.01, 0.001)


def one(task):
    mix, state, rounds, run = task
    seed = int(np.random.SeedSequence([MIXES.index(mix), STATE_DEFS.index(state), rounds, run, 55])
               .generate_state(1)[0])
    rng = np.random.default_rng(seed)
    a = simulate(population(mix, N), rounds, rng)
    s = encode_states(a, state)
    ctx = build_context(a, s, B=B, seed=seed)
    iu, ju = np.triu_indices(N, 1)
    # (a) gamma fit (Set A) applied to Set-B null draws
    p = gamma_tail(ctx.g_B[:, iu, ju], ctx.g_A[:, iu, ju])
    row = dict(mix=mix, state=state, rounds=rounds, run=run, seed=seed, n_states=int(s.max() + 1))
    for x in XS:
        row[f"ratio_p<{x}"] = (p < x).mean() / x
    # (b) subset moments
    _, zA, zB = edge_z(ctx)
    vA, vB = zA[:, iu, ju], zB[:, iu, ju]
    Xp = np.concatenate(list(_membership_chunks(N))).astype(np.float64)
    mu_e, cov_e = vA.mean(0), np.cov(vA, rowvar=False, ddof=0)
    mu_cf = Xp @ mu_e
    sd_cf = np.sqrt(np.einsum("mp,mp->m", Xp @ cov_e, Xp))
    TA, TB = Xp @ vA.T, Xp @ vB.T
    row["b1_max_rel_err_sd"] = np.max(np.abs(sd_cf - TA.std(1)) / TA.std(1))
    row["b1_max_abs_err_mean"] = np.max(np.abs(mu_cf - TA.mean(1)))
    row["b2_sd_ratio_median"] = np.median(sd_cf / TB.std(1))
    row["b2_sd_ratio_max_dev"] = np.max(np.abs(sd_cf / TB.std(1) - 1))
    row["b2_mean_shift_in_sd_max"] = np.max(np.abs(mu_cf - TB.mean(1)) / TB.std(1))
    return row


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else WORKERS
    tasks = [(m, s, r, k) for m in MIXES for s in STATE_DEFS for r in (300, 1000, 3000) for k in range(RUNS)]
    with Pool(workers) as pool:
        rows = pool.map(one, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "validate_shortcuts.csv", index=False)
    cols = [c for c in df.columns if c.startswith(("ratio", "b1", "b2"))]
    agg = df.groupby(["state", "rounds"])[cols].agg(["mean", "max"]).round(3)
    with pd.option_context("display.width", 300):
        print(df.groupby(["state", "rounds"]).n_states.mean().to_string())
        print(agg.to_string())
