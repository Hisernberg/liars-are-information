"""Run manifests: config hash, git commit, seeds, timestamps, GPU model.

Requirement 9 of the spec: *every* run writes one of these next to its outputs,
so any parquet in ``results/`` can be traced back to the exact configuration and
code state that produced it.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Any, length: int = 12) -> str:
    """Stable short hash of a configuration object (dict, dataclass, or model)."""
    if hasattr(config, "model_dump"):
        payload = config.model_dump(mode="json")
    elif hasattr(config, "__dataclass_fields__"):
        payload = asdict(config)
    else:
        payload = config
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:length]


def stable_seed(key: Any, modulus: int = 100_000) -> int:
    """Deterministic integer seed from any config key.

    ``hash()`` cannot be used for this. Python randomises string hashing per
    process unless ``PYTHONHASHSEED`` is pinned, so a sweep seeded with
    ``hash(key)`` draws *different swarms on every run* while reporting the same
    configuration. That defeats the point of logging seeds, and it is exactly
    what the Phase E consistency pass caught: 7,485 cells belonging to methods
    the gate cannot touch had changed between two runs of the same sweep.
    """
    return int(hashlib.sha256(canonical_json(key).encode()).hexdigest()[:12], 16) % modulus


def text_hash(text: str, length: int = 12) -> str:
    """Stable short hash of a string (used for prompt-template hashes)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def git_commit(repo_root: Path | None = None) -> str:
    """Current git commit, suffixed ``-dirty`` if the tree has changes.

    Returns ``"unknown"`` when the tree is not a git repository or has no
    commits yet, rather than raising: a manifest that records ``unknown`` is
    more useful than a run that refuses to start.
    """
    root = repo_root or Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if commit.returncode != 0:
            return "unknown"
        sha = commit.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return f"{sha}-dirty" if status.stdout.strip() else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def gpu_info() -> list[dict[str, Any]]:
    """Describe visible GPUs via ``nvidia-smi``; empty list if none."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return []
        gpus = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpus.append(
                    {"name": parts[0], "memory_total_mib": int(parts[1]), "driver": parts[2]}
                )
        return gpus
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


@dataclass
class Manifest:
    """Provenance record for one run."""

    run_id: str
    phase: str
    config_hash: str
    config: dict[str, Any]
    seeds: dict[str, int]
    git_commit: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    hostname: str = field(default_factory=socket.gethostname)
    user: str = field(default_factory=lambda: _safe_user())
    gpus: list[dict[str, Any]] = field(default_factory=gpu_info)
    package_versions: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        phase: str,
        config: Any,
        seeds: dict[str, int],
        repo_root: Path | None = None,
        notes: dict[str, Any] | None = None,
    ) -> Manifest:
        if hasattr(config, "model_dump"):
            cfg_dict = config.model_dump(mode="json")
        elif hasattr(config, "__dataclass_fields__"):
            cfg_dict = asdict(config)
        else:
            cfg_dict = dict(config)
        chash = config_hash(cfg_dict)
        started = datetime.now(UTC).isoformat()
        return cls(
            run_id=(
                f"{phase}-{chash}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
                f"-{uuid.uuid4().hex[:4]}"
            ),
            phase=phase,
            config_hash=chash,
            config=cfg_dict,
            seeds=dict(seeds),
            git_commit=git_commit(repo_root),
            started_at=started,
            package_versions=_package_versions(),
            notes=notes or {},
        )

    def finish(self, status: str = "ok", **notes: Any) -> Manifest:
        self.finished_at = datetime.now(UTC).isoformat()
        self.status = status
        self.notes.update(notes)
        return self

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return path


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "pyarrow", "scipy", "sympy", "datasets", "transformers", "vllm"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out
