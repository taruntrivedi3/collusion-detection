"""Step 1 diagnostic: how does the conditioning state change Bonjour CDA and Gamma noise?
The paper never states which state its RPS detector conditions on."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.estimate import encode_states, gamma_batch, net_influence  # noqa: E402
from step1_reproduce import OUT, bonjour_exact_pair, sims  # noqa: E402

rows = []
for kind in ("none", "full", "profile"):
    for cp in (0.3, 0.4):
        for r in (100, 300, 500, 1000):
            a = sims(cp, r, 1000, seed=20_000 + r + int(cp * 10))
            G = np.stack([net_influence(gamma_batch(x, encode_states(x, kind))) for x in a])
            rows.append(dict(state=kind, cp=cp, rounds=r, cda=100 * bonjour_exact_pair(G).mean(),
                             mean_AB=G[:, 0, 1].mean(), sd_AB=G[:, 0, 1].std(), sd_AC=G[:, 0, 2].std()))
df = pd.DataFrame(rows)
df.to_csv(OUT / "cda_state_sensitivity.csv", index=False)
print(df.round(3).to_string(index=False))
