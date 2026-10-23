#!/usr/bin/env bash
set -euo pipefail
study_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TMPDIR="$study_root/tmp" TMP="$study_root/tmp" TEMP="$study_root/tmp"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$study_root/tmp/matplotlib" MPLBACKEND=Agg
study_python=/home/trivedit/.conda/envs/andromeda_env/bin/python
bash "$study_root/code/build.sh"
"$study_python" "$study_root/code/validate.py"
"$study_python" -u "$study_root/code/run_study.py" all --workers 16
"$study_python" "$study_root/code/analyze.py"
"$study_python" "$study_root/code/audit.py"
