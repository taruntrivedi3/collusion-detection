"""Fetch a pinned public dataset. This script never imports the auction runner."""

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests


COMMIT = "2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff"
REPOSITORY = "andyjliu/llm-agent-collusion"


def verify(destination: Path) -> None:
    evidence = json.loads((destination / "source_manifest.json").read_text())
    response = requests.get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1", timeout=60)
    response.raise_for_status()
    tree = response.json()
    if tree["truncated"]:
        raise ValueError("Incomplete upstream Git tree")
    expected = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "blob"}
    if set(expected) != {row["path"] for row in evidence["files"]}:
        raise ValueError("Archive/Git tree file set mismatch")
    for row in evidence["files"]:
        raw = (destination / "upstream" / row["path"]).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if git_blob != expected[row["path"]] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Source checksum mismatch: {row['path']}")
    result = {"commit": COMMIT, "verified_git_blobs": len(expected),
              "verified_sha256_files": len(expected), "git_tree_sha": tree["sha"],
              "archive_sha256": evidence["archive_sha256"]}
    (destination / "source_integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def fetch(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"upstream-{COMMIT}.tar.gz"
    url = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{COMMIT}"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        partial.replace(archive)
    source = destination / "upstream"
    source.mkdir(exist_ok=True)
    manifest = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            path = source.joinpath(*relative.parts[1:])
            path.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(member) as incoming, path.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            raw = path.read_bytes()
            manifest.append({"path": str(path.relative_to(source)), "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest()})
    evidence = {"repository": REPOSITORY, "commit": COMMIT, "archive_url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": manifest}
    (destination / "source_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"source": str(source), "files": len(manifest),
                      "bytes": sum(row["bytes"] for row in manifest), "commit": COMMIT}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        fetch(args.destination)
    verify(args.destination)
