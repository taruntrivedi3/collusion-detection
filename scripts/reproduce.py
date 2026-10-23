"""Run unchanged study code in a new output directory using the current Python."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from verify_package import ROOT, digest, verify


STUDIES = {"rps": "rps_null_calibration", "auction": "auction_null_calibration"}
AUCTION_INPUTS = [
    "cases.json", "pool.npz", "pool_provenance.json", "state_audit.json",
    "parse_audit.json",
]


def run(command, output, environment):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=output, env=environment, check=True)


def prepare(source, output):
    # Refuse an existing target, so stale outputs cannot masquerade as a rerun.
    output.mkdir(parents=True, exist_ok=False)
    for name in ["code", "source_snapshot", "provenance"]:
        shutil.copytree(source / name, output / name)
    for path in source.iterdir():
        if path.is_file() and path.suffix in [".md", ".json"]:
            shutil.copyfile(path, output / path.name)
    (output / "results").mkdir()
    for name in ["INTERPRETATION.md", *AUCTION_INPUTS]:
        path = source / "results" / name
        if path.exists():
            shutil.copyfile(path, output / "results" / name)
    (output / "tmp" / "matplotlib").mkdir(parents=True)


def portable_audit_manifest(study, output):
    if study == "rps":
        manifest_name, key = "experiment_manifest.json", "libnullstudy.so"
    else:
        manifest_name, key = "design_manifest.json", "code/mi.so"
    path = output / manifest_name
    document = json.loads(path.read_text())
    original_hash = document["files"][key]
    library = output / ("code/" + key if study == "rps" else key)
    rebuilt_hash = digest(library)
    shutil.copyfile(path, output / "provenance" / ("original_" + manifest_name))
    document["files"][key] = rebuilt_hash
    path.write_text(json.dumps(document, indent=2) + "\n")
    return {"manifest": manifest_name, "only_updated_entry": key,
            "original_sha256": original_hash, "rebuilt_sha256": rebuilt_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing directories are refused")
    parser.add_argument("--mode", choices=["validate", "full"], default="validate")
    parser.add_argument("--workers", type=int, default=8,
                        help="Primary-run workers; original auction supplements use 12")
    parser.add_argument("--audit", action="store_true",
                        help="Full replay audit; auction also downloads the public source")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.audit and args.mode != "full":
        parser.error("--audit requires --mode full")
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    verify()
    source = ROOT / "studies" / STUDIES[args.study]
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("Choose an output directory outside the published project")
    if output.exists():
        parser.error("Output directory already exists; choose a new path")
    prepare(source, output)
    temporary = str(output / "tmp")
    environment = dict(os.environ, TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                       PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg",
                       MPLCONFIGDIR=str(output / "tmp" / "matplotlib"),
                       NUMBA_CACHE_DIR=str(output / "tmp" / "numba"))
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "study": args.study, "mode": args.mode, "audit_requested": args.audit,
               "python": sys.version, "platform": platform.platform(),
               "original_config_sha256": digest(source / "config.json"),
               "package_manifest_sha256": digest(ROOT / "package_manifest.json"),
               "scientific_source_modified": False}
    receipt_path = output / "reproduction.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    run(["bash", output / "code/build.sh"], output, environment)
    receipt["rebuilt_library"] = portable_audit_manifest(args.study, output)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    run([sys.executable, output / "code/validate.py"], output, environment)
    if args.study == "rps":
        run([sys.executable, "-m", "pytest", "-q", "--capture=sys", "-p", "no:cacheprovider",
             output / "source_snapshot/tests"], output / "source_snapshot", environment)
    if args.mode == "full":
        runner = "run_study.py" if args.study == "rps" else "run.py"
        run([sys.executable, "-u", output / "code" / runner,
             "all", "--workers", args.workers], output, environment)
        if args.study == "auction":
            for script in ["matched_pool.py", "score_transfer.py"]:
                run([sys.executable, "-u", output / "code" / script], output, environment)
        run([sys.executable, output / "code/analyze.py"], output, environment)
        if args.audit:
            if args.study == "auction":
                run([sys.executable, output / "source_snapshot/fetch_data.py",
                     "--destination", output], output, environment)
                fetched = json.loads((output / "source_manifest.json").read_text())
                original = json.loads((source / "source_manifest.json").read_text())
                if fetched["commit"] != original["commit"] or fetched["files"] != original["files"]:
                    raise ValueError("Fetched source differs from the original study")
            run([sys.executable, output / "code/audit.py"], output, environment)
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["validation_sha256"] = digest(output / "results/validation.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Completed {args.mode}: {output}")


if __name__ == "__main__":
    main()
