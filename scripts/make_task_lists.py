#!/usr/bin/env python3
"""Freeze the study's task selection into configs/task_lists.json.

Run once; commit the result. Every phase that touches tasks reads it, so the
adversarial runs provably operate on the same questions as the honest cache.

The six benchmarks are not drawn at one size, and two of them are *extensions*
of lists this study has already frozen:

    gsm8k    500  = the frozen 100 + the disjoint 400 of the ext calibration set,
                    both carried verbatim, nothing redrawn
    math500  200  = the frozen 100 preserved + 100 drawn from the complement
    mmlu     200  = the frozen 100 preserved + 100 drawn from the complement
    medqa    200  new
    boolq    200  new, label-balanced
    arc      200  new, restricted to the four-option items

Preserving is explicit rather than implied by the seed. Only MMLU's sampler
nests; MATH-500's uniform draw does not, and re-drawing it at 200 under the same
seed would have retained just 33 of the frozen 100.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.harness.logging import configure_logging  # noqa: E402
from aip.tasks.task_lists import DEFAULT_PATH, build_task_lists, write_task_lists  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20250825)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("configs/task_lists.json"),
        help="list whose ids must be preserved (the frozen 100s)",
    )
    parser.add_argument(
        "--gsm8k-ext",
        type=Path,
        default=Path("configs/task_lists_gsm8k_ext.json"),
        help="the disjoint GSM8K 400 that completes the 500",
    )
    args = parser.parse_args()

    configure_logging(level="INFO")

    specs = {"gsm8k": 500, "math500": 200, "mmlu": 200, "medqa": 200, "boolq": 200, "arc": 200}

    preserve: dict[str, list[str]] = {}
    if args.previous.exists():
        prev = json.loads(args.previous.read_text(encoding="utf-8"))["benchmarks"]
        for name in ("gsm8k", "math500", "mmlu"):
            if name in prev:
                preserve[name] = list(prev[name]["task_ids"])
    if args.gsm8k_ext.exists():
        ext = json.loads(args.gsm8k_ext.read_text(encoding="utf-8"))["benchmarks"]["gsm8k"]
        preserve["gsm8k"] = preserve.get("gsm8k", []) + list(ext["task_ids"])

    payload = build_task_lists(specs, seed=args.seed, preserve=preserve)
    write_task_lists(payload, args.out)

    print(f"\nFrozen task lists -> {args.out}  (seed {args.seed})")
    for name, rec in payload["benchmarks"].items():
        extra = f", {rec['n_subjects']} subjects" if "n_subjects" in rec else ""
        print(
            f"  {name:<10} n={rec['n']:>3} (preserved {rec['n_preserved']:>3})"
            f"{extra}  first={rec['task_ids'][0]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
