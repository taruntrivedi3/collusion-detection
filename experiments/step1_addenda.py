"""Step 1 addenda.
  (1) Bonjour CDA at CP = 0.1, 0.2 (1000 rounds) with the 27-state (previous action profile)
      estimator vs no state; paper reports ~8% and 24.3%.
  (1b) python step1_addenda.py by_rounds -> cda_low_cp_by_rounds.csv
  (2) False-positive rate on 3 uniform agents (no collusion): % of runs flagging any pair, for
      fixed alpha = 0.05 and for the calibrated variant (sanity check, should be ~5%).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import Population, simulate  # noqa: E402
from collusion.estimate import encode_states, gamma_batch, net_influence  # noqa: E402
from collusion.null import build_context  # noqa: E402
from collusion.detect import detect_bonjour, detect_bonjour_calibrated  # noqa: E402
from step1_reproduce import OUT, bonjour_exact_pair, sims  # noqa: E402

ITERS = 1000
STATES = {"none": "none", "27-state": "profile"}


def low_cp_cda():
    rows = []
    for cp in (0.1, 0.2):
        a = sims(cp, 1000, ITERS, seed=30_000 + int(cp * 10))
        for label, kind in STATES.items():
            G = np.stack([net_influence(gamma_batch(x, encode_states(x, kind))) for x in a])
            gab = np.minimum(G[:, 0, 1], G[:, 1, 0])
            rows.append(dict(cp=cp, rounds=1000, state=label, cda=100 * bonjour_exact_pair(G).mean(),
                             mean_min_Gamma_AB=gab.mean(), sd_min_Gamma_AB=gab.std()))
    return pd.DataFrame(rows)


def null_fpr(B=200):
    rows = []
    rng = np.random.default_rng(31_000)
    for r in (100, 300, 1000):
        a = np.stack([simulate(Population(3), r, rng) for _ in range(ITERS)])
        for label, kind in STATES.items():
            fix = cal = 0
            for it, x in enumerate(a):
                ctx = build_context(x, encode_states(x, kind), B=B, seed=it)
                fix += bool(detect_bonjour(ctx).info["flagged_pairs"])
                cal += bool(detect_bonjour_calibrated(ctx).info["flagged_pairs"])
            rows.append(dict(rounds=r, state=label, iters=ITERS, B=B,
                             fpr_fixed_alpha=100 * fix / ITERS, fpr_calibrated=100 * cal / ITERS))
    return pd.DataFrame(rows)


def low_cp_by_rounds():
    """(1b) CDA at CP 0.1 / 0.2 for every round count 100..1000 (paper: CP 0.1 '≈8% on average')."""
    rows = []
    for cp in (0.1, 0.2):
        for r in range(100, 1001, 100):
            a = sims(cp, r, 1000, seed=32_000 + r + int(cp * 10))
            for lab, k in STATES.items():
                G = np.stack([net_influence(gamma_batch(x, encode_states(x, k))) for x in a])
                rows.append(dict(cp=cp, rounds=r, state=lab, cda=100 * bonjour_exact_pair(G).mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__" and sys.argv[1:] == ["by_rounds"]:
    low_cp_by_rounds().to_csv(OUT / "cda_low_cp_by_rounds.csv", index=False)
elif __name__ == "__main__":
    c = low_cp_cda(); c.to_csv(OUT / "cda_low_cp.csv", index=False)
    print(c.round(4).to_string(index=False))
    f = null_fpr(); f.to_csv(OUT / "null_fpr_bonjour.csv", index=False)
    print(f.to_string(index=False))
