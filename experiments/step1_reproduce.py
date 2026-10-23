"""Step 1: reproduce Bonjour et al. 3-player RPS (A primary, B helper, C uniform).

Outputs (results/step1/):
  gamma_trace.csv / .png   Gamma for all 6 ordered pairs vs rounds, one trajectory per CP (cf. Fig 2)
  gamma_vs_cp.csv / .png   mean Gamma at 1000 rounds vs CP over 200 runs (cf. Fig 3)
  cda.csv                  collusion detection accuracy over 1000 iterations (cf. Table 2)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collusion.game import Coalition, Population, simulate  # noqa: E402
from collusion.estimate import gamma_batch, net_influence  # noqa: E402
from collusion.detect import ALPHA_BONJOUR  # noqa: E402
from collusion.plotting import INK2, SERIES, style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "step1"
OUT.mkdir(parents=True, exist_ok=True)
A, B_, C = 0, 1, 2
NAMES = "ABC"
PAIRS = [(A, B_), (B_, A), (A, C), (C, A), (B_, C), (C, B_)]
CPS = np.round(np.arange(0.1, 1.01, 0.1), 1)


def pop(cp):
    return Population(3, [Coalition("pair", [A, B_], cp)])


def sims(cp, rounds, iters, seed):
    rng = np.random.default_rng(seed)
    return np.stack([simulate(pop(cp), rounds, rng) for _ in range(iters)])


def bonjour_exact_pair(G, alpha=ALPHA_BONJOUR):
    """[it, 3, 3] Gamma -> bool: exactly one unordered pair passes both ways, and it is {A, B}."""
    ok = np.nan_to_num(G, nan=-np.inf) >= alpha
    both = ok & ok.transpose(0, 2, 1)
    iu = np.triu_indices(3, 1)
    n_pass = both[:, iu[0], iu[1]].sum(1)
    return (n_pass == 1) & both[:, A, B_]


def trace():
    rows = []
    grid = np.arange(50, 1001, 50)
    for k, cp in enumerate(CPS):
        a = sims(cp, 1000, 1, seed=100 + k)[0]
        for r in grid:
            G = net_influence(gamma_batch(a[:r], np.zeros(r, dtype=np.int64)))
            for i, j in PAIRS:
                rows.append(dict(cp=cp, rounds=r, pair=f"G({NAMES[i]};{NAMES[j]})", Gamma=G[i, j]))
    return pd.DataFrame(rows)


def vs_cp(iters=200):
    rows = []
    for k, cp in enumerate(CPS):
        a = sims(cp, 1000, iters, seed=200 + k)
        G = net_influence(gamma_batch(a, np.zeros(1000, dtype=np.int64)))
        for i, j in PAIRS:
            rows.append(dict(cp=cp, pair=f"G({NAMES[i]};{NAMES[j]})",
                             mean=G[:, i, j].mean(), sd=G[:, i, j].std()))
    return pd.DataFrame(rows)


def cda(iters=1000):
    rows = []
    cases = [(cp, list(range(100, 1001, 100))) for cp in (0.1, 0.2, 0.3, 0.4)]
    cases += [(1.0, [20, 40, 60, 80, 100]), (0.2, [10000])]
    for k, (cp, grid) in enumerate(cases):
        for r in grid:
            seed = 10_000 + 1000 * k + r
            a = sims(cp, r, iters, seed)
            G = net_influence(gamma_batch(a, np.zeros(r, dtype=np.int64)))
            rows.append(dict(cp=cp, rounds=r, iters=iters, seed=seed,
                             cda=100 * bonjour_exact_pair(G).mean()))
    return pd.DataFrame(rows)


def plot_trace(df):
    import matplotlib.pyplot as plt
    style()
    fig, axs = plt.subplots(2, 5, figsize=(15, 5.5), sharex=True)
    for ax, cp in zip(axs.flat, CPS):
        d = df[df.cp == cp]
        for c, (name, g) in zip(SERIES, d.groupby("pair", sort=False)):
            ax.plot(g.rounds, g.Gamma, color=c, label=name)
        ax.axhline(ALPHA_BONJOUR, color=INK2, lw=1, ls="--")
        ax.set_title(f"CP = {cp}")
    for ax in axs[1]:
        ax.set_xlabel("rounds")
    for ax in axs[:, 0]:
        ax.set_ylabel("net influence Γ (bits)")
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Net influence Γ for all ordered pairs vs rounds (one trajectory per CP; dashed = α)")
    fig.tight_layout()
    fig.savefig(OUT / "gamma_trace.png", dpi=130)


def plot_vs_cp(df):
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, (name, g) in zip(SERIES, df.groupby("pair", sort=False)):
        ax.errorbar(g.cp, g["mean"], yerr=g.sd, color=c, marker="o", ms=4, capsize=2, label=name)
    ax.axhline(np.log2(3), color=INK2, lw=1, ls=":")
    ax.text(0.1, np.log2(3) + 0.04, "log2 3", color=INK2, fontsize=8)
    ax.set_xlabel("collusion probability CP")
    ax.set_ylabel("net influence Γ at 1000 rounds (bits)")
    ax.set_title("Mean ± sd of Γ over 200 runs (cf. paper Fig 3)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "gamma_vs_cp.png", dpi=130)


if __name__ == "__main__":
    t = trace(); t.to_csv(OUT / "gamma_trace.csv", index=False); plot_trace(t)
    v = vs_cp(); v.to_csv(OUT / "gamma_vs_cp.csv", index=False); plot_vs_cp(v)
    c = cda(); c.to_csv(OUT / "cda.csv", index=False)
    print(v[v.pair.isin(["G(A;B)", "G(B;A)", "G(A;C)"])].pivot(index="cp", columns="pair", values="mean").round(3))
    print(c.pivot_table(index="rounds", columns="cp", values="cda").round(1))
