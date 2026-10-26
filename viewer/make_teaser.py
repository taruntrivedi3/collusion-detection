"""Make the README teaser from viewer/data.js: teaser.gif (the triangle game scrolling by, "Who's colluding?")
and reveal.png (the same moves with the colluders picked out, and the detected link graph).

  python viewer/make_teaser.py       (needs matplotlib and Pillow)
"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
COLOR = {"R": "#2a78d6", "P": "#eb6834", "S": "#1baf7a"}  # same as index.html
INK, INK2, MUTED, GRID, SURFACE, CRIT = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#d03b3b"
WINDOW, FRAMES, START = 30, 60, 0
DPI = 100


def load():
    s = (HERE / "data.js").read_text()
    data = json.loads(s[s.index("=") + 1:].strip().rstrip(";"))
    return next(g for g in data["rps"] if g["id"] == "triangle")


def draw_grid(ax, g, start, members=()):
    n = g["n"]
    ax.set_xlim(-2.2, WINDOW + 0.2)
    ax.set_ylim(n + 0.9, -0.2)
    ax.axis("off")
    for i in range(n):
        fade = 0.22 if members and i not in members else 1.0
        bold = i in members
        ax.text(-0.4, i + 0.5, f"P{i}", ha="right", va="center", fontsize=12, color=INK,
                fontweight="bold" if bold else "normal", alpha=max(fade, 0.45))
        for r in range(WINDOW):
            m = g["moves"][i][start + r]
            ax.add_patch(FancyBboxPatch((r + 0.06, i + 0.06), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.15",
                                        fc=COLOR[m], ec="none", alpha=fade))
            ax.text(r + 0.5, i + 0.53, m, ha="center", va="center", fontsize=8.5, color="white",
                    fontweight="bold", alpha=fade)
    for r in range(0, WINDOW, 10):
        ax.text(r, n + 0.55, f"{start + r + 1}", fontsize=9, color=INK2, va="center")
    ax.text(WINDOW, n + 0.55, "round →", fontsize=9, color=MUTED, ha="right", va="center")


def legend(fig, y):
    x = 0.5 - 0.17
    for k, name in (("R", "Rock"), ("P", "Paper"), ("S", "Scissors")):
        fig.patches.append(FancyBboxPatch((x, y - 0.012), 0.016, 0.024, transform=fig.transFigure, fc=COLOR[k], ec="none",
                                          boxstyle="round,pad=0,rounding_size=0.003"))
        fig.text(x + 0.024, y, name, fontsize=10, color=INK2, va="center")
        x += 0.12


def frame(g, start):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=DPI, facecolor=SURFACE)
    fig.text(0.5, 0.93, "Who's colluding?", ha="center", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.5, 0.855, f"{g['n']} players, rock-paper-scissors, {g['rounds']} rounds. Some share a secret signal.",
             ha="center", fontsize=11, color=INK2)
    ax = fig.add_axes([0.03, 0.1, 0.94, 0.72])
    draw_grid(ax, g, start)
    legend(fig, 0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def reveal(g):
    members = sorted({m for c in g["planted"] for m in c})
    found = next(r["found"] for r in g["results"] if r["key"] == "subgraph")
    bonjour = next(r["found"] for r in g["results"] if r["key"] == "bonjour")
    fig = plt.figure(figsize=(10.5, 4.2), dpi=DPI * 1.5, facecolor=SURFACE)
    names = ", ".join(f"P{i}" for i in members[:-1]) + f" and P{members[-1]}"
    fig.text(0.5, 0.93, f"{names}: a triangle", ha="center", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.5, 0.86, "They play the same move in 30% of rounds. Each looks random on its own.",
             ha="center", fontsize=11, color=INK2)
    ax = fig.add_axes([0.01, 0.1, 0.6, 0.72])
    draw_grid(ax, g, START, members)
    legend(fig, 0.05)

    import numpy as np
    n, z = g["n"], np.array(g["z"])
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = np.c_[np.cos(ang), np.sin(ang)]
    net = fig.add_axes([0.64, 0.1, 0.34, 0.72])
    net.set_xlim(-1.35, 1.35)
    net.set_ylim(-1.35, 1.35)
    net.set_aspect("equal")
    net.axis("off")
    sig = {tuple(sorted(e)) for e in g["sig_edges"]}
    for a in range(n):
        for b in range(a + 1, n):
            on = (a, b) in sig
            net.plot(*pos[[a, b]].T, color=CRIT if on else GRID, lw=3.5 if on else 0.8, zorder=2 if on else 1,
                     solid_capstyle="round")
            if on:
                mx, my = pos[[a, b]].mean(0) * 1.0
                net.text(mx, my, f"z = {z[a, b]:.0f}", fontsize=9, color=INK, ha="center", va="center", zorder=4,
                         bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    for i in range(n):
        hit = i in members
        net.scatter(*pos[i], s=620, color=INK if hit else SURFACE, ec=INK if hit else MUTED, lw=1.5, zorder=3)
        net.text(*pos[i], f"P{i}", ha="center", va="center", fontsize=10, zorder=5,
                 color=SURFACE if hit else INK2, fontweight="bold" if hit else "normal")
    net.text(0, -1.33, f"GroupScan: {', '.join('{' + ', '.join(f'P{i}' for i in c) + '}' for c in found) or 'nothing'}"
             f"    Bonjour (pairwise): {'nothing' if not bonjour else bonjour}",
             ha="center", fontsize=9, color=INK2)
    fig.savefig(HERE / "reveal.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    g = load()
    frames = [frame(g, START + k) for k in range(FRAMES)]
    pal = frames[0].quantize(colors=64)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(HERE / "teaser.gif", save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=True)
    reveal(g)
    print("wrote", HERE / "teaser.gif", HERE / "reveal.png")


if __name__ == "__main__":
    main()
