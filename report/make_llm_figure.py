"""Report Figure 4: share of batches flagged vs games per batch, 2-seller and 5-seller LLM auctions.

  python report/make_llm_figure.py      (from the repo root; recomputes the 2-seller batch tests)

2 sellers: same-move test, recomputed with studies/llm_auction_2seller/build_report.py's batch_rates.
5 sellers: global test, read from studies/llm_auction_5seller/runs/n5_rates_v1/analysis.txt
(written by analyze_n.py).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "studies" / "llm_auction_2seller")]
sys.argv = sys.argv[:1]                                    # build_report reads argv[1] as its output path

import matplotlib.pyplot as plt  # noqa: E402
from build_report import CONDS, GRID, INK, INK2, LABEL, MARKER, batch_rates, load  # noqa: E402

from collusion.metrics import wilson  # noqa: E402

SIZES = (5, 10, 20)
# Conditions get colours no detector uses in Figures 2-3 (palette slots 6 and 7, plus grey for the
# no-comms, no-pressure control), so blue never means two different things in the report.
COLOR = {"comms_off": "#8a8984", "pressure_off": "#6250d6", "comms_pressure": "#008300"}
OUT = Path(__file__).with_name("figures") / "fig4_llm_detection.png"


def five_seller_rates():
    """{(cond, size): (rate, n_batches)} for the global test, parsed from analyze_n.py output."""
    text = (ROOT / "studies" / "llm_auction_5seller" / "runs" / "n5_rates_v1" / "analysis.txt").read_text()
    out = {}
    for block in text.split("\n== ")[1:]:
        cond = block.split(":", 1)[0].strip("= ")
        for size, n, g in re.findall(r"batches of\s+(\d+) games \(n=\s*(\d+)\): reject global ([\d.]+)", block):
            out[(cond, int(size))] = (float(g), int(n))
    return out


def main():
    two = batch_rates(load())["same_moves"]
    panels = [("2 sellers (30-hour games)", two),
              ("5 sellers (50-hour games), all 10 pairs", five_seller_rates())]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (title, rates) in zip(axes, panels):
        ax.axhline(5, color=INK2, lw=1, ls=(0, (4, 3)))
        for k, cond in enumerate(CONDS):
            xs = [s + (k - 1) * 0.35 for s in SIZES]          # small offset so intervals don't overlap
            ys = [100 * rates[(cond, s)][0] for s in SIZES]
            ci = [tuple(100 * v for v in wilson(round(r * n), n)) for r, n in (rates[(cond, s)] for s in SIZES)]
            err = [[y - lo for y, (lo, _) in zip(ys, ci)], [hi - y for y, (_, hi) in zip(ys, ci)]]
            ax.errorbar(xs, ys, yerr=err, color=COLOR[cond], lw=2, marker=MARKER[cond], ms=6, capsize=2,
                        markeredgecolor="white", markeredgewidth=1, label=LABEL[cond], zorder=3)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xticks(SIZES)
        ax.set_xlim(3.5, 21.5)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("games per batch")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% of batches flagged (p ≤ 0.05)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=220)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
