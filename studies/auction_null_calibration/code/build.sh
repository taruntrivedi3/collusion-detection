#!/usr/bin/env bash
set -euo pipefail
study_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TMPDIR="$study_root/tmp" TMP="$study_root/tmp" TEMP="$study_root/tmp"
g++ -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared "$study_root/code/mi.cpp" -o "$study_root/code/mi.so"
