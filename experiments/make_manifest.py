"""Write results/MANIFEST.tsv: every tracked file with size, SHA-256 and mtime (ISO-8601 with
explicit UTC offset). This file is the pre-sprint freeze record.
Read-only on everything except MANIFEST.tsv itself."""
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
OUT = ROOT / "results" / "MANIFEST.tsv"

rows = []
for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
    for f in sorted(files):
        p = Path(dirpath) / f
        if p == OUT:
            continue
        st = p.stat()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        mt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        rows.append(f"{p.relative_to(ROOT)}\t{st.st_size}\t{h}\t{mt}")
OUT.write_text("path\tbytes\tsha256\tmtime_utc\n" + "\n".join(rows) + "\n")
print(f"{len(rows)} files -> {OUT}")
