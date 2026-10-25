"""Resumable cell runner: one CSV per cell; cells already on disk are skipped.

A crash loses at most the cell in progress. Rerun the same command to resume.
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

WORKERS = int(os.environ.get("WORKERS", 12))   # ~0.5 GB per worker at n = 20, B = 1000


def run_cells(cells, run_one, out_dir, workers=WORKERS):
    """cells: list of (key, [task, ...]); run_one(task) -> list of row dicts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(k, t) for k, t in cells if not (out_dir / f"{k}.csv").exists()]
    print(f"{len(cells) - len(todo)} cells done, {len(todo)} to run -> {out_dir}", flush=True)
    with Pool(workers) as pool:
        for i, (key, tasks) in enumerate(todo, 1):
            rows = [r for rs in pool.imap_unordered(run_one, tasks) for r in rs]
            tmp = out_dir / f"{key}.csv.tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.rename(out_dir / f"{key}.csv")
            print(f"[{i}/{len(todo)}] {key}", flush=True)


def load_cells(out_dir):
    files = sorted(Path(out_dir).glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def ci(g, col, w=None):
    """'pct [lo, hi]' (95% Wilson) for the share of rows of g where `col` holds, optionally
    weighting each row by column `w` (e.g. planted coalitions per run)."""
    from collusion.metrics import wilson
    k = (g[col].astype(float) * (g[w] if w else 1)).sum()
    t = g[w].sum() if w else len(g)
    lo, hi = wilson(k, t)
    return f"{100 * k / t:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"
