#!/usr/bin/env python3
"""Diff two numbers.tex files macro by macro against the measurement floors.

X3 changes the cache under every downstream surface, so the question is not
"did the numbers move" -- they did -- but which ones moved by more than the
study can resolve. The floors are the study's own:
read from results/measurement_floor.json rather than hardcoded, so adopting a
new floor cannot leave this comparison testing against the old one.

Verdicts: `unchanged`, `within floor`, `BEYOND FLOOR`, `not floor-testable`
(counts, token totals, ratios -- the floor is a difference between proportions
and says nothing about them), `added`, `removed`, and `non-numeric`.

    python scripts/x3_macro_diff.py OLD.tex NEW.tex --out results/x3_macro_diff.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: Values never contain a brace; a trailing "  % note" comment is common.
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
# A leading + is a sign, not text. numbers.tex writes signed correlations as
# "+0.372", and matching without the sign filed two genuine BEYOND FLOOR moves
# (numRealSepMathfiveFseven, numCompAccVsAccMean) as "non-numeric change".
NUMERIC = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

def _floors() -> tuple[float, float]:
    """GSM8K and non-GSM8K floors, from the generated artefact.

    These were literals (0.0684 / 0.1003). When the floor was recomputed on
    clean resample rows -- 0.066 GSM8K, 0.099 elsewhere -- a literal here would
    have kept silently testing against the superseded value.
    """
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "results/measurement_floor.json").read_text())
    per = raw["per_benchmark"]
    gsm = max(v for k, v in per.items() if k.startswith("gsm8k"))
    other = max(v for k, v in per.items() if not k.startswith("gsm8k"))
    return float(gsm), float(other)


GSM8K_FLOOR, DEFAULT_FLOOR = _floors()


def parse(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _num(value: str) -> float:
    """Parse a macro value, tolerating a leading + and LaTeX escapes."""
    return float(str(value).replace("\\", "").replace("%", "")
                 .replace("$", "").strip().lstrip("+"))


def _is_proportion(value: str) -> bool:
    """True when a macro's value can be read as a proportion in [0, 1.5]."""
    try:
        return 0.0 <= abs(float(str(value).replace("\\", "").replace("%", "")
                                .replace("$", "").replace("+", "").strip())) <= 1.5
    except ValueError:
        return False


def floor_for(name: str) -> float:
    """GSM8K macros get the tighter floor; everything else the worst case.

    Matching on the name is crude, but the alternative -- a hand-maintained map
    from 231 macros to benchmarks -- would rot the first time a macro is added,
    and erring toward the LOOSER floor would under-report movement.
    """
    return GSM8K_FLOOR if "Gsm" in name or "gsm" in name else DEFAULT_FLOOR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/x3_macro_diff.md"))
    args = ap.parse_args()

    old, new = parse(args.old), parse(args.new)
    rows = []
    for name in sorted(set(old) | set(new)):
        o, n = old.get(name), new.get(name)
        if o is None:
            rows.append((name, "--", n, "", "added"))
            continue
        if n is None:
            rows.append((name, o, "--", "", "removed"))
            continue
        if o == n:
            rows.append((name, o, n, "", "unchanged"))
            continue
        if not (NUMERIC.match(o) and NUMERIC.match(n)):
            rows.append((name, o, n, "", "non-numeric change"))
            continue
        delta = abs(_num(n) - _num(o))
        # The floor is a minimum detectable difference between two PROPORTIONS.
        # Applying it to a token count, a a wall-clock second or a ratio like
        # numCostInferenceRatio (5073 -> 6598) asks whether 1525 exceeds 0.0994,
        # which is true and meaningless -- it made a count-heavy diff report 21
        # 'BEYOND FLOOR' moves when 6 proportions had actually moved.
        if not (_is_proportion(o) and _is_proportion(n)):
            rows.append((name, o, n, f"{delta:.4f}", "not floor-testable"))
            continue
        fl = floor_for(name)
        rows.append((name, o, n, f"{delta:.4f}",
                     "BEYOND FLOOR" if delta > fl else "within floor"))

    order = {"BEYOND FLOOR": 0, "not floor-testable": 1, "non-numeric change": 2,
             "within floor": 3, "added": 4, "removed": 5, "unchanged": 6}
    rows.sort(key=lambda r: (order[r[4]], -float(r[3] or 0), r[0]))

    counts = {k: sum(1 for r in rows if r[4] == k) for k in order}
    lines = [
        "# X3 macro diff", "",
        f"`{args.old}` -> `{args.new}`. Floors: {GSM8K_FLOOR} for GSM8K macros, "
        f"{DEFAULT_FLOOR} elsewhere.", "",
        "| verdict | count |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in counts.items() if v],
        "", "| macro | old | new | abs delta | verdict |", "|---|---|---|---|---|",
        *[f"| `{a}` | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows],
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, v in counts.items():
        if v:
            print(f"{k:20s} {v}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
