#!/usr/bin/env python3
"""Emit paper/numbers.tex — one LaTeX macro per reported quantity.

The manuscript may not contain a hand-typed number. Run this after any change to
the result parquets; ``tests/test_paper.py`` fails the build if a ``.tex`` file
outside ``numbers.tex`` contains a bare decimal, or if the manuscript uses a
macro this file does not define.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.analysis.paper_numbers import collect  # noqa: E402
from aip.harness.manifest import git_commit  # noqa: E402

#: Refuse to shrink the macro file below this fraction of its current size.
MIN_RETENTION = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("paper/numbers.tex"))
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="permit emitting far fewer macros than the existing file (deliberate reruns)",
    )
    args = parser.parse_args()

    nums = collect()

    # FAIL LOUD. This file is the paper's provenance chain: every reported number
    # is a macro emitted here, and tests/test_paper.py fails the build on a bare
    # decimal anywhere else. When the source parquets are missing, `collect()`
    # quietly returns the handful of numbers it can still compute and the file is
    # rewritten with a fraction of its macros -- which is exactly how a committed
    # 84-macro numbers.tex became a 7-macro one earlier in this project. An
    # unattended run must never do that silently.
    existing = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    n_existing = existing.count("\\newcommand")
    if n_existing and len(nums) < n_existing * MIN_RETENTION and not args.allow_shrink:
        print(
            f"REFUSING to write {args.out}: would emit {len(nums)} macros over an "
            f"existing {n_existing}. The result parquets are probably missing or "
            f"incomplete. Re-run the phases that produce them, or pass "
            f"--allow-shrink if the shrink is intended.",
            file=sys.stderr,
        )
        return 1

    lines = [
        "% GENERATED FILE — do not edit by hand.",
        "% Produced by scripts/make_numbers.py from the result parquets.",
        "% Every number in the manuscript must be one of these macros; a bare",
        "% decimal anywhere else is a build failure (tests/test_paper.py).",
        f"% generated: {datetime.now(UTC).isoformat()}",
        f"% git: {git_commit(Path.cwd())}",
        "",
    ]
    by_claim: dict[str, list] = {}
    for n in nums:
        by_claim.setdefault(n.claim or "(context)", []).append(n)
    for claim in sorted(by_claim):
        lines.append(f"% ---- {claim} " + "-" * max(0, 60 - len(claim)))
        for n in by_claim[claim]:
            suffix = f"  % {n.note}" if n.note else ""
            lines.append(f"\\newcommand{{\\{n.macro}}}{{{n.rendered()}}}{suffix}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} with {len(nums)} macros")
    missing = [n.macro for n in nums if n.rendered().startswith(r"\textsc")]
    if missing:
        print(f"  WARNING: {len(missing)} macros are n/a: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
