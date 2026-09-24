"""Figure provenance: which script wrote a figure, from what, at which commit.

The manuscript spent this entire regeneration compiling seven figures left over
from the L4 era, because the generators defaulted to ``figures/`` while
``paper/figures/`` was what LaTeX read. Nothing detected it: the files existed,
the references resolved, and a PDF from a voided hardware regime looks exactly
like a current one.

A stamp fixes the class of bug rather than the instance. Every generator records
its output here, and :func:`aip.analysis.provenance.load` lets a test assert that
each figure the manuscript cites was produced by a script in this repository from
parquets in this regime -- so a stale file fails the build instead of being
printed.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "MANIFEST.json"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment
        return "unknown"


def manifest_path(figure: Path) -> Path:
    return figure.parent / MANIFEST_NAME


def load(directory: Path) -> dict[str, dict[str, object]]:
    """Return the figure manifest for a directory, or an empty mapping."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def stamp(figure: Path, *, script: str, sources: list[Path] | None = None) -> Path:
    """Record that ``script`` wrote ``figure`` from ``sources``.

    The manifest is keyed by file name and rewritten in place, so regenerating
    one figure updates only its own entry and a figure that is never regenerated
    keeps an entry naming the commit that last produced it.
    """
    path = manifest_path(figure)
    entries = load(figure.parent)
    entries[figure.name] = {
        "script": script,
        "sources": sorted(str(s) for s in (sources or [])),
        "git_commit": _git_commit(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(dict(sorted(entries.items())), indent=2) + "\n")
    return path
