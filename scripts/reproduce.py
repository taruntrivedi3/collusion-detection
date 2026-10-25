"""Rerun the recorded-market calibration study in a fresh directory, leaving the published
results untouched.

  python scripts/reproduce.py --output /tmp/auction-check                 # build + validation checks
  python scripts/reproduce.py --mode full --workers 8 --output /tmp/auction-run
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Prepared inputs the auction runner reads instead of re-parsing the upstream archive.
AUCTION_INPUTS = ["cases.json", "pool.npz", "pool_provenance.json", "state_audit.json", "parse_audit.json"]


def run(command, cwd, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new directory (must not exist)")
    parser.add_argument("--mode", choices=["validate", "full"], default="validate")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source, out = ROOT / "studies" / "auction_null_calibration", args.output.resolve()
    if out.exists():
        parser.error("output directory already exists")
    out.mkdir(parents=True)
    shutil.copytree(source / "code", out / "code")
    shutil.copytree(source / "source_snapshot", out / "source_snapshot")
    for p in [*source.glob("*.json"), *source.glob("*.md")]:
        shutil.copyfile(p, out / p.name)
    (out / "results").mkdir()
    for name in ["INTERPRETATION.md", *AUCTION_INPUTS]:
        if (source / "results" / name).exists():
            shutil.copyfile(source / "results" / name, out / "results" / name)

    tmp = out / "tmp"
    (tmp / "matplotlib").mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), TMPDIR=str(tmp), MPLBACKEND="Agg",
               MPLCONFIGDIR=str(tmp / "matplotlib"), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    py = sys.executable
    run(["bash", out / "code/build.sh"], out, env)
    run([py, out / "code/validate.py"], out, env)
    if args.mode == "full":
        run([py, "-u", out / "code/run.py", "all", "--workers", args.workers], out, env)
        for script in ("matched_pool.py", "score_transfer.py"):
            run([py, "-u", out / "code" / script], out, env)
        run([py, out / "code/analyze.py"], out, env)
    print(f"Done ({args.mode}): {out}")


if __name__ == "__main__":
    main()
