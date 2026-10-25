"""Write-up figures. Each figure saves a PNG and the CSV behind it to results/figures/.

  python make_figures.py A B C D E      (any subset)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collusion.plotting import GRID, INK2, SERIES, style  # noqa: E402
from collusion.metrics import wilson  # noqa: E402
from collusion.detect import DETECTOR_LABELS, DETECTORS, edge_z  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])
DET = list(DETECTORS)
COLOR = dict(zip(DET, SERIES))                      # fixed slot per detector, never re-ordered
SHORT = {"bonjour": "Bonjour (fixed cutoff)", "bonjour_cal": "Bonjour (calibrated)", "edge_bh": "Pair test",
         "node_score": "Player score", "subgraph": "Group scan"}
PLOT = ["bonjour", "edge_bh", "node_score", "subgraph"]   # detectors shown in the write-up


def _load(sub):
    return pd.concat([pd.read_csv(f) for f in sorted((ROOT / "results" / sub).glob("*.csv"))], ignore_index=True)


def _rate(df, value, keys):
    w = df.n_planted if value == "recovered" else 1
    d = df.assign(k=df[value].astype(float) * w, t=w)
    g = d.groupby(keys)[["k", "t"]].sum().reset_index()
    lo, hi = wilson(g.k, g.t)
    return g.assign(pct=100 * g.k / g.t, lo=100 * lo, hi=100 * hi)


def _errline(ax, x, g, det, **kw):
    ax.errorbar(x, g.pct, yerr=[(g.pct - g.lo).clip(lower=0), (g.hi - g.pct).clip(lower=0)], color=COLOR[det], marker="o", ms=5,
                capsize=2, lw=2, label=SHORT[det], **kw)


# ---------------------------------------------------------------- A: cancellation
def fig_a():
    from collusion.game import Coalition, Population, simulate
    from collusion.null import build_context
    from collusion.estimate import net_influence
    style()
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8))
    rows = []
    for r, (kind, seed) in enumerate((("triangle", 7), ("hub", 8))):
        pop = Population(5, [Coalition(kind, [0, 1, 2], 0.5)])
        a = simulate(pop, 1000, np.random.default_rng(seed))
        ctx = build_context(a, np.zeros(1000, dtype=np.int64), B=1000, seed=seed)
        G = net_influence(ctx.g_real)
        z = edge_z(ctx)[0]
        np.fill_diagonal(z, np.nan)
        for name, M, ax, norm, fmt in (
            ("Net influence Γ(i;j) (bits)", G, axs[r, 0], TwoSlopeNorm(0, -0.45, 0.45), "{:+.3f}"),
            ("Edge z-score", z, axs[r, 1], SymLogNorm(2, vmin=-250, vmax=250), "{:.1f}"),
        ):
            ax.imshow(M, cmap=DIVERGING, norm=norm)
            for i in range(5):
                for j in range(5):
                    if i != j:
                        ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7.5,
                                color="white" if abs(norm(M[i, j]) - 0.5) > 0.35 else "#0b0b0b")
                        rows.append(dict(structure=kind, matrix=name, i=i, j=j, value=M[i, j]))
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.set_xlabel("agent j"); ax.set_ylabel("agent i")
            ax.grid(False)
            who = "triangle {0,1,2}" if kind == "triangle" else "hub: primary 0, helpers 1, 2"
            ax.set_title(f"{name}\n{who}; agents 3, 4 independent", fontsize=9)
    fig.suptitle("Inside a coalition, Γ cancels to about 0 while edge z-scores stay large\n"
                 "(5 agents, CP = 0.5, 1000 rounds; the paper flags a pair only if Γ ≥ 0.05 both ways)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figA_cancellation.png", dpi=150)
    pd.DataFrame(rows).to_csv(FIG / "figA_cancellation.csv", index=False)


# ---------------------------------------------------------------- B: dilution vs n
def fig_b():
    df = pd.concat([pd.read_csv(ROOT / f"results/step4/runs_n{n}.csv") for n in (5, 10, 20)])
    df = df[df.structure.isin(["triangle", "clique4"]) & df.detector.isin(["node_score", "subgraph"])]
    df = df.assign(n_planted=1)
    g = _rate(df, "recovered", ["structure", "detector", "n"])
    g.to_csv(FIG / "figB_dilution.csv", index=False)
    style()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, s in zip(axs, ("triangle", "clique4")):
        for det in ("node_score", "subgraph"):
            d = g[(g.structure == s) & (g.detector == det)]
            _errline(ax, d.n, d, det)
        ax.set_xticks([5, 10, 20]); ax.set_xlabel("number of agents n")
        ax.set_title({"triangle": "3-agent coalition", "clique4": "4-agent coalition"}[s])
    axs[0].set_ylabel("% coalitions recovered exactly")
    axs[0].set_ylim(0, 100)
    axs[0].legend(loc="lower left")
    fig.suptitle("Player score loses power as agents are added; the group scan holds\n"
                 "(pooled over CP × rounds grid; B = 200; 100 reps per cell at n ≤ 10, 50 at n = 20; 95% CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figB_dilution.png", dpi=150)


# ---------------------------------------------------------------- C: weak broad cliques
def fig_c():
    df = _load("step4_final/cells")
    df = df[df.structure.isin(["clique4", "clique5"]) & (df.n == 20) & (df.rounds == 100)]
    g = _rate(df, "recovered", ["structure", "detector", "cp"])
    g.to_csv(FIG / "figC_weak_cliques.csv", index=False)
    style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s in zip(axs, ("clique4", "clique5")):
        for det in PLOT:
            d = g[(g.structure == s) & (g.detector == det)]
            _errline(ax, d.cp, d, det)
        ax.set_xlabel("collusion probability CP"); ax.set_title(f"{s[-1]}-agent coalition among 20 agents")
    axs[0].set_ylabel("% coalitions recovered exactly")
    axs[0].set_ylim(0, 100)
    axs[1].legend(fontsize=8)
    fig.suptitle("Weak, broad coalitions in short games (100 rounds): the group scan recovers more\n"
                 "(B = 1000, 100 reps per point, 95% Wilson CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figC_weak_cliques.png", dpi=150)


# ---------------------------------------------------------------- D: null false alarms
def fig_d():
    df = _load("step4_final/cells")
    df = df[df.structure == "none"]
    g = _rate(df, "detected", ["n", "detector", "rounds"])
    g.to_csv(FIG / "figD_null_false_alarms.csv", index=False)
    style()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, n in zip(axs, (5, 10, 20)):
        for det in PLOT[:1] + ["bonjour_cal"] + PLOT[1:]:
            d = g[(g.n == n) & (g.detector == det)]
            _errline(ax, d.rounds, d, det)
        ax.axhline(5, color=INK2, lw=1, ls="--")
        ax.set_xscale("log"); ax.set_xticks([100, 300, 1000]); ax.set_xticklabels(["100", "300", "1000"])
        ax.set_xlabel("rounds"); ax.set_title(f"n = {n} independent agents")
    axs[0].set_ylabel("% runs with any detection")
    axs[0].legend(fontsize=8)
    fig.suptitle("False alarms with no collusion: the fixed threshold drifts with n and rounds; "
                 "calibrated detectors stay near 5% (dashed)\n(B = 1000, 500 reps per point, 95% Wilson CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figD_null_false_alarms.png", dpi=150)


# ---------------------------------------------------------------- E: state definition
def fig_e():
    df = _load("step5/partA_n10")
    g = _rate(df, "detected", ["mix", "rounds", "state", "detector"])
    g.to_csv(FIG / "figE_state_false_alarms.csv", index=False)
    style()
    mixes = ["uniform", "leader", "margin", "mixed"]
    states = ["none", "top", "order", "full"]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    off = np.linspace(-0.25, 0.25, len(PLOT))
    for ax, mix in zip(axs, mixes):
        d0 = g[(g.mix == mix) & (g.rounds == 1000)]
        for k, det in enumerate(PLOT):
            d = d0[d0.detector == det].set_index("state").loc[states]
            ax.errorbar(np.arange(4) + off[k], d.pct, yerr=[(d.pct - d.lo).clip(lower=0), (d.hi - d.pct).clip(lower=0)], fmt="o",
                        ms=5, capsize=2, color=COLOR[det], label=SHORT[det])
        ax.axhline(5, color=INK2, lw=1, ls="--")
        ax.set_xticks(range(4)); ax.set_xticklabels(states)
        ax.set_xlabel("state the detector conditions on")
        ax.set_title({"uniform": "all honest agents uniform", "leader": "half leader-reactive",
                      "margin": "half margin-reactive", "mixed": "leader + margin + uniform"}[mix], fontsize=9)
    axs[0].set_ylabel("% runs with any detection")
    axs[0].legend(fontsize=7.5)
    fig.suptitle("False alarms among honest history-reacting agents depend on the state definition\n"
                 "(n = 10, 1000 rounds, no colluders, B = 1000, 200 reps, 95% Wilson CI; dashed = 5%)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figE_state_false_alarms.png", dpi=150)


# ---------------------------------------------------------------- T: structure x n table
def table_t():
    """Structure x n summary with pooled 95% Wilson CIs, from the full Step 4 grid (B = 200)."""
    df = pd.concat([pd.read_csv(ROOT / f"results/step4/runs_n{n}.csv") for n in (5, 10, 20)], ignore_index=True)
    df = df[df.structure != "none"]
    df = df.assign(n_planted=df.structure.map(lambda s: 2 if s == "two_pairs" else 1))
    g = _rate(df, "recovered", ["structure", "n", "detector"])
    g["cell"] = g.apply(lambda r: f"{r.pct:.0f} [{r.lo:.0f}, {r.hi:.0f}]", axis=1)
    t = g.pivot_table(index=["structure", "n"], columns="detector", values="cell", aggfunc="first")[PLOT]
    t = t.rename(columns=SHORT)
    t.to_csv(FIG / "table_structure_by_n_ci.csv")
    print(t.to_string())


if __name__ == "__main__":
    for k in sys.argv[1:]:
        {"A": fig_a, "B": fig_b, "C": fig_c, "D": fig_d, "E": fig_e, "T": table_t}[k]()
        print("fig", k, "done")
