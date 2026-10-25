"""Write-up figures. Each figure saves a PNG and the CSV behind it to results/figures/.

  python make_figures.py A B C D E      (any subset)
  python make_figures.py T              structure x n table -> results/figures/table_structure_by_n_ci.csv
  python make_figures.py R              print-size Figures 1-3 for report/report.md -> report/figures/
  python make_figures.py F              report Table 2 (fixed setting) -> results/figures/table_fixed_setting_cp02_r300.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collusion.plotting import INK2, SERIES, style  # noqa: E402
from collusion.metrics import wilson  # noqa: E402
from collusion.detect import DETECTORS, edge_z  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])
DET = list(DETECTORS)
COLOR = dict(zip(DET, SERIES))                      # fixed slot per detector, never re-ordered
SHORT = {"bonjour": "Bonjour (fixed cutoff)", "bonjour_cal": "Bonjour (calibrated)", "edge_bh": "Pair test",
         "node_score": "Player score", "subgraph": "GroupScan"}
PLOT = ["bonjour", "edge_bh", "node_score", "subgraph"]   # detectors shown in the write-up
MARKER = {"bonjour": "o", "bonjour_cal": "v", "edge_bh": "s", "node_score": "^", "subgraph": "D"}
REPORT_FIG = ROOT / "report" / "figures"


def _load(sub):
    return pd.concat([pd.read_csv(f) for f in sorted((ROOT / "results" / sub).glob("*.csv"))], ignore_index=True)


def _rate(df, value, keys):
    w = df.n_planted if value == "recovered" else 1
    d = df.assign(k=df[value].astype(float) * w, t=w)
    g = d.groupby(keys)[["k", "t"]].sum().reset_index()
    lo, hi = wilson(g.k, g.t)
    return g.assign(pct=100 * g.k / g.t, lo=100 * lo, hi=100 * hi)


def _errline(ax, x, g, det, **kw):
    ax.errorbar(x, g.pct, yerr=[(g.pct - g.lo).clip(lower=0), (g.hi - g.pct).clip(lower=0)], color=COLOR[det], marker=MARKER[det], ms=5,
                capsize=2, lw=2, label=SHORT[det], **kw)


# ---------------------------------------------------------------- A: cancellation
def fig_a(kinds=(("triangle", 7), ("hub", 8)), figsize=(9.5, 8), out=FIG / "figA_cancellation.png", title=True):
    from collusion.game import Coalition, Population, simulate
    from collusion.null import build_context
    from collusion.estimate import net_influence
    style()
    fig, axs = plt.subplots(len(kinds), 2, figsize=figsize, squeeze=False)
    rows = []
    for r, (kind, seed) in enumerate(kinds):
        pop = Population(5, [Coalition(kind, [0, 1, 2], 0.5)])
        a = simulate(pop, 1000, np.random.default_rng(seed))
        ctx = build_context(a, np.zeros(1000, dtype=np.int64), B=1000, seed=seed)
        G = net_influence(ctx.g_real)
        z = edge_z(ctx)[0]
        np.fill_diagonal(z, np.nan)
        for name, M, ax, norm, fmt in (
            ("Published score: net influence (bits)", G, axs[r, 0], TwoSlopeNorm(0, -0.45, 0.45),
             "{:+.3f}" if title else "{:+.2f}"),
            ("Link strength (z-score vs shuffles)", z, axs[r, 1], SymLogNorm(2, vmin=-250, vmax=250), "{:.1f}"),
        ):
            ax.imshow(M, cmap=DIVERGING, norm=norm)
            for i in range(5):
                for j in range(5):
                    if i != j:
                        ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7.5 if title else 9,
                                color="white" if abs(norm(M[i, j]) - 0.5) > 0.35 else "#0b0b0b")
                        rows.append(dict(structure=kind, matrix=name, i=i, j=j, value=M[i, j]))
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            if not title and kind == "triangle":            # report figure: outline the colluding block
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, lw=2, ec="#0b0b0b"))
            ax.set_xlabel("player j"); ax.set_ylabel("player i")
            ax.grid(False)
            who = "triangle {0,1,2}" if kind == "triangle" else "hub: leader 0, helpers 1, 2"
            ax.set_title(f"{name}\n{who}; players 3, 4 honest", fontsize=9)
    if title:
        fig.suptitle("Inside a group, the published score cancels to about 0 while the links stay strong\n"
                     "(5 players, collusion probability 0.5, 1000 rounds; the rule flags a pair only if the score ≥ 0.05 both ways)",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200 if not title else 150)
    if title:
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
        ax.set_xticks([5, 10, 20]); ax.set_xlabel("number of players")
        ax.set_title({"triangle": "Triangle (3 colluders)", "clique4": "4-clique (4 colluders)"}[s])
    axs[0].set_ylabel("% recovered exactly")
    axs[0].set_ylim(0, 100)
    axs[0].legend(loc="lower left")
    fig.suptitle("The player score loses power as players are added; GroupScan holds\n"
                 "(averaged over collusion probabilities and rounds; 200 shuffles; 100 games per setting at 5–10 players, 50 at 20; 95% CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figB_dilution.png", dpi=150)


# ---------------------------------------------------------------- C: weak broad cliques
def fig_c(figsize=(10, 4), out=FIG / "figC_weak_cliques.png", title=True):
    df = _load("step4_final/cells")
    df = df[df.structure.isin(["clique4", "clique5"]) & (df.n == 20) & (df.rounds == 100)]
    g = _rate(df, "recovered", ["structure", "detector", "cp"])
    g.to_csv(FIG / "figC_weak_cliques.csv", index=False)
    style()
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, s in zip(axs, ("clique4", "clique5")):
        for det in PLOT:
            d = g[(g.structure == s) & (g.detector == det)]
            _errline(ax, d.cp, d, det)
        ax.set_xlabel("collusion probability"); ax.set_title(f"{s[-1]}-player group among 20 players")
    axs[0].set_ylabel("% recovered exactly")
    axs[0].set_ylim(0, 100)
    axs[1].legend(fontsize=8)
    if title:
        fig.suptitle("Weak, broad groups in short games (100 rounds): GroupScan recovers more\n"
                     "(1000 shuffles, 100 games per point, 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200 if not title else 150)


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
        ax.set_xlabel("rounds"); ax.set_title(f"{n} random players")
    axs[0].set_ylabel("% of games with any report")
    axs[0].legend(fontsize=8)
    fig.suptitle("False alarms with no collusion: the fixed cutoff drifts with players and rounds; "
                 "calibrated detectors stay near 5% (dashed)\n(1000 shuffles, 500 games per point, 95% CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "figD_null_false_alarms.png", dpi=150)


# ---------------------------------------------------------------- E: state definition
def fig_e(grid=(1, 4), figsize=(14, 4), out=FIG / "figE_state_false_alarms.png", title=True):
    df = _load("step5/partA_n10")
    g = _rate(df, "detected", ["mix", "rounds", "state", "detector"])
    g.to_csv(FIG / "figE_state_false_alarms.csv", index=False)
    style()
    mixes = ["uniform", "leader", "margin", "mixed"]
    states = ["none", "top", "order", "full"]
    fig, axs = plt.subplots(*grid, figsize=figsize, sharey=True, squeeze=False)
    axs = axs.ravel()
    off = np.linspace(-0.25, 0.25, len(PLOT))
    for i, (ax, mix) in enumerate(zip(axs, mixes)):
        d0 = g[(g.mix == mix) & (g.rounds == 1000)]
        for k, det in enumerate(PLOT):
            d = d0[d0.detector == det].set_index("state").loc[states]
            # report figure: grey out Bonjour, which sits at 0 only because its cutoff never fires here
            color = "#b5b4af" if (not title and det == "bonjour") else COLOR[det]
            ax.errorbar(np.arange(4) + off[k], d.pct, yerr=[(d.pct - d.lo).clip(lower=0), (d.hi - d.pct).clip(lower=0)],
                        fmt=MARKER[det], ms=5, capsize=2, color=color, label=SHORT[det])
        ax.axhline(5, color=INK2, lw=1, ls="--")
        ax.set_xticks(range(4)); ax.set_xticklabels(states)
        if i >= len(axs) - grid[1] and (title or grid[0] > 1):
            ax.set_xlabel("history state the detector conditions on")
        ax.set_title({"uniform": "all honest players random", "leader": "half leader-reactive",
                      "margin": "half margin-reactive", "mixed": "leader + margin + random"}[mix],
                     fontsize=9 if title else 8)
    for i in range(0, len(axs), grid[1]):
        axs[i].set_ylabel("% of games with any report")
    if title:
        axs[0].legend(fontsize=7.5)
        fig.suptitle("False alarms among honest players who react to history depend on the state\n"
                     "(10 players, 1000 rounds, no colluders, 1000 shuffles, 200 games per point, 95% CI; dashed = 5%)",
                     fontsize=10)
        fig.tight_layout()
    else:
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0))
        if grid[0] == 1:
            fig.supxlabel("history state the detector conditions on", fontsize=9, y=0.135)
        fig.tight_layout(rect=(0, 0.17 if grid[0] == 1 else 0.07, 1, 1))
    fig.savefig(out, dpi=200 if not title else 150)


# ---------------------------------------------------------------- T: structure x n table
def _recovery_table(df, out_name):
    """Exact recovery 'pct [lo, hi]' per structure x n x detector from Step 4 runs -> CSV + stdout."""
    df = df.assign(n_planted=df.structure.map(lambda s: 2 if s == "two_pairs" else 1))
    g = _rate(df, "recovered", ["structure", "n", "detector"])
    g["cell"] = g.apply(lambda r: f"{r.pct:.0f} [{r.lo:.0f}, {r.hi:.0f}]", axis=1)
    t = g.pivot_table(index=["structure", "n"], columns="detector", values="cell", aggfunc="first")[PLOT]
    t = t.rename(columns=SHORT)
    t.to_csv(FIG / out_name)
    print(t.to_string())


def _step4_runs(ns):
    return pd.concat([pd.read_csv(ROOT / f"results/step4/runs_n{n}.csv") for n in ns], ignore_index=True)


def table_t():
    """Structure x n summary with pooled 95% Wilson CIs, from the full Step 4 grid (B = 200)."""
    df = _step4_runs((5, 10, 20))
    _recovery_table(df[df.structure != "none"], "table_structure_by_n_ci.csv")


def table_fixed():
    """Report Table 2: exact recovery at the one setting every group was run at (CP 0.2, 300 rounds, B = 200)."""
    df = _step4_runs((5, 20))
    _recovery_table(df[(df.cp == 0.2) & (df.rounds == 300) & (df.structure != "none")], "table_fixed_setting_cp02_r300.csv")


def report_figures():
    """Figures 1-3 of report/report.md at print width (7 in), no suptitles (the captions carry them)."""
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    fig_a(kinds=(("triangle", 7),), figsize=(7, 3.3), out=REPORT_FIG / "fig1_cancellation.png", title=False)
    fig_c(figsize=(7, 2.5), out=REPORT_FIG / "fig2_weak_cliques.png", title=False)
    fig_e(grid=(1, 4), figsize=(7.2, 2.7), out=REPORT_FIG / "fig3_state_false_alarms.png", title=False)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        {"A": fig_a, "B": fig_b, "C": fig_c, "D": fig_d, "E": fig_e, "T": table_t, "F": table_fixed,
         "R": report_figures}[k]()
        print("fig", k, "done")
