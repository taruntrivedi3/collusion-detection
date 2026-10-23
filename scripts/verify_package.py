"""Verify the distribution and unchanged original study artifacts; no network."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    manifest = json.loads((root / "package_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Path outside package: {relative}")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Package checksum mismatch: {relative}")
    original = json.loads((root / "study_provenance.json").read_text())
    for relative, record in original["files"].items():
        if manifest["files"].get(relative) != record["sha256"]:
            raise ValueError(f"Original study artifact changed: {relative}")
    print(f"Verified {len(manifest['files'])} package files and "
          f"{len(original['files'])} unchanged study artifacts.", flush=True)


if __name__ == "__main__":
    verify()
