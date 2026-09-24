#!/usr/bin/env python3
"""Verify the downloaded release, retained v1 evidence and study fingerprints.

Uses only Python's standard library. This verifies recorded content integrity
and completion; it does not independently rerun predictions or prove claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes its directory: {relative}")
    return path


def verify(root: Path) -> dict:
    failures: list[str] = []
    counts = dict(release_files=0, retained_v1_files=0, study_hashes=0, completed_studies=0)

    def check_file(path: Path, expected: str, size: int | None = None) -> None:
        label = str(path.relative_to(root))
        if not path.is_file():
            failures.append(f"Missing file: {label}")
        elif size is not None and path.stat().st_size != size:
            failures.append(f"Size mismatch: {label}")
        elif digest(path) != expected:
            failures.append(f"SHA-256 mismatch: {label}")

    manifest = json.loads((root / "FILE_MANIFEST.json").read_text())
    if manifest.get("self_excluded") is not True:
        failures.append("Release manifest must declare its own exclusion")
    paths = [entry["path"] for entry in manifest["files"]]
    if len(set(paths)) != len(paths):
        failures.append("Duplicate paths in release manifest")
    if "FILE_MANIFEST.json" in paths:
        failures.append("Release manifest cannot recursively hash itself")
    for entry in manifest["files"]:
        check_file(safe_path(root, entry["path"]), entry["sha256"], entry["size"])
        counts["release_files"] += 1

    # The old README/release identity live in provenance. Every old scientific
    # code, cache, table, trace, note and test remains at the same relative path.
    prior = json.loads((root / "provenance/v1/FILE_MANIFEST.json").read_text())
    for entry in prior["files"]:
        name = entry["path"]
        if name in {"README.md", "RELEASE.json"}:
            name = f"provenance/v1/{name}"
        check_file(safe_path(root, name), entry["sha256"], entry["size"])
        counts["retained_v1_files"] += 1

    registry = json.loads((root / "results/extensions/study_registry.json").read_text())
    for study in registry["studies"]:
        directory = safe_path(root, study["directory"])
        run = json.loads((directory / "run_manifest.json").read_text())
        receipt = json.loads((directory / "resource_receipt.json").read_text())
        if run.get("status") != "complete" or receipt.get("status") != "complete":
            failures.append(f"Study not complete: {study['name']}")
        if run.get("worlds_completed") != study["worlds"]:
            failures.append(f"World count mismatch: {study['name']}")
        for key in ["source_sha256", "input_sha256"]:
            for name, expected in run.get(key, {}).items():
                check_file(safe_path(root, name), expected)
                counts["study_hashes"] += 1
        for name, expected in run.get("output_sha256", {}).items():
            check_file(safe_path(directory, name), expected)
            counts["study_hashes"] += 1
        artifacts = directory / "artifact_manifest.json"
        if artifacts.is_file():
            recorded = json.loads(artifacts.read_text())
            if all(isinstance(value, str) for value in recorded.values()):
                for name, expected in recorded.items():
                    check_file(safe_path(directory, name), expected)
                    counts["study_hashes"] += 1
        counts["completed_studies"] += 1

    # Report paths only, never credential-like content.
    pattern = re.compile(rb"hf_[A-Za-z0-9]{20,}")
    text_suffixes = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".py", ".toml", ".log", ".csv"}
    for name in paths:
        path = safe_path(root, name)
        if path.is_file() and path.suffix in text_suffixes and pattern.search(path.read_bytes()):
            failures.append(f"Credential-like string in text artifact: {name}")
    return dict(status="passed" if not failures else "failed", **counts, failures=failures,
                scope="recorded hashes/completion/provenance; scientific review is separate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, help="Optional JSON report; prefer a path outside the release")
    args = parser.parse_args()
    result = verify(args.root.resolve())
    rendered = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n")
    print(rendered)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
