#!/usr/bin/env python3
"""X3, adversarial half: repair ``data/cache_adversarial`` after the honest cache.

Two separate reasons a row here is stale, and both are repaired:

1. **Truncation.** 741 of 12,000 adversarial generations hit the old cap, 502 of
   them on MATH-500 -- the same defect, in the same proportions, as the honest
   cache.
2. **Context drift, for ``rushing`` only.** That attack's prompt embeds the
   honest models' broadcast answers for the task, read live from
   ``data/cache``. X3 changed 944 of those answers, so every rushing row whose
   task had a repaired honest broadcast was generated against a context that no
   longer exists, whether or not it truncated.

Leaving (2) out would be the worse failure of the two: the honest side of Phase
D would be repaired while the adversarial side still answers the pre-repair
swarm.

    python scripts/x3_repair_adversarial.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from phase_d_adversarial import ADVERSARY_MODELS, generate_cell  # noqa: E402

from aip.attacks import INFERENCE_ATTACKS  # noqa: E402
from aip.harness import cache as cache_mod  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.models.inference import ModelRunner  # noqa: E402
from aip.models.registry import load_registry  # noqa: E402
from aip.tasks import get_benchmark  # noqa: E402
from aip.tasks.task_lists import load_fixed_tasks  # noqa: E402

log = configure_logging(level="INFO")

ADV = ROOT / "data/cache_adversarial"
HONEST = ROOT / "data/cache"
#: The tag written before the repair started; `git show` against it gives the
#: pre-repair honest answers without keeping a second copy of the cache around.
PRE_TAG = "pre-x3-cache"
SEED = 20250825


def finish_reason(extra_json: str | None) -> str | None:
    return json.loads(extra_json).get("finish_reason") if extra_json else None


def honest_answers_before(benchmark: str) -> dict[str, dict[str, str]]:
    """Per honest model, the extracted answer each task had before X3."""
    out: dict[str, dict[str, str]] = {}
    listing = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{PRE_TAG}:data/cache/{benchmark}"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    for name in listing:
        if not name.endswith(".parquet"):
            continue
        blob = subprocess.run(
            ["git", "show", f"{PRE_TAG}:data/cache/{benchmark}/{name}"],
            cwd=ROOT, capture_output=True, check=True).stdout
        frame = pd.read_parquet(BytesIO(blob))
        out[name[: -len(".parquet")]] = {
            str(t): ("" if pd.isna(v) else str(v))
            for t, v in zip(frame.task_id, frame.extracted_answer, strict=True)
        }
    return out


def drifted_tasks(benchmark: str) -> set[str]:
    """Tasks whose honest broadcast set changed when X3 repaired the cache."""
    before = honest_answers_before(benchmark)
    drifted: set[str] = set()
    for model, prior in before.items():
        path = cache_mod.cache_path(benchmark, model, root=HONEST)
        if not path.exists():
            continue
        now = pd.read_parquet(path)
        for task, value in zip(now.task_id, now.extracted_answer, strict=True):
            value = "" if pd.isna(value) else str(value)
            if prior.get(str(task), value) != value:
                drifted.add(str(task))
    return drifted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(ADVERSARY_MODELS))
    ap.add_argument("--attacks", nargs="*", default=list(INFERENCE_ATTACKS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/x3_repair/x3_adversarial_report.json")
    args = ap.parse_args()

    registry = load_registry("configs/models.yaml")
    benchmarks = sorted({p.parent.name for p in ADV.glob("*/*/*.parquet")})

    # `rushing` is the only attack that reads the honest cache, so it is the only
    # one with a context-drift set. Computed once per benchmark, not per model.
    drift_by_benchmark = {b: drifted_tasks(b) for b in benchmarks} \
        if "rushing" in args.attacks else {}

    plan: list[dict[str, Any]] = []
    for model in args.models:
        for attack in args.attacks:
            for bench in benchmarks:
                path = cache_mod.cache_path(bench, model, root=ADV / attack)
                if not path.exists():
                    continue
                frame = pd.read_parquet(path)
                fr = frame["extra_json"].map(finish_reason)
                ids = set(frame.loc[fr == "length", "task_id"].astype(str))
                stale = set()
                if attack == "rushing":
                    stale = drift_by_benchmark.get(bench, set()) & set(
                        frame.task_id.astype(str))
                if ids or stale:
                    plan.append({"model": model, "attack": attack, "benchmark": bench,
                                 "truncated": sorted(ids), "context_stale": sorted(stale - ids),
                                 "ids": sorted(ids | stale)})

    n_trunc = sum(len(p["truncated"]) for p in plan)
    n_stale = sum(len(p["context_stale"]) for p in plan)
    print(f"X3 adversarial plan: {n_trunc} truncated + {n_stale} context-stale "
          f"= {n_trunc + n_stale} rows across {len(plan)} cells")
    for p in plan:
        print(f"  {p['attack']:24s} {p['benchmark']:9s} {p['model']:16s} "
              f"trunc={len(p['truncated']):4d} stale={len(p['context_stale']):4d}")
    if args.dry_run or not plan:
        return 0

    manifest = Manifest.create(phase="x3_repair_adversarial",
                               config={"cells": len(plan), "truncated": n_trunc,
                                       "context_stale": n_stale,
                                       "registry_hash": registry.hash()},
                               seeds={"generation": SEED})
    report: list[dict[str, Any]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        cells = [p for p in plan if p["model"] == model]
        if not cells:
            continue
        entry = registry.get(model)
        with ModelRunner(entry) as runner:
            for cell in cells:
                bench = cell["benchmark"]
                wanted = set(cell["ids"])
                items = [i for i in load_fixed_tasks(bench) if i.task_id in wanted]
                started = time.perf_counter()
                log.info("x3adv.cell.start", model=model, attack=cell["attack"],
                         benchmark=bench, n=len(items))
                preds, _ = generate_cell(
                    runner, bench, cell["attack"], items, SEED,
                    max_tokens=None,
                    confidence_tokens=entry.generation.confidence_max_tokens)
                still = sum(1 for p in preds
                            if (p.extra or {}).get("finish_reason") == "length")
                cache_mod.write_predictions(
                    preds, bench, model, root=ADV / cell["attack"],
                    manifest={"attack": cell["attack"], "hf_id": entry.hf_id,
                              "prompt_template_hash": get_benchmark(bench).prompt_template_hash(),
                              "run_id": manifest.run_id,
                              "x3_truncated_rows": len(cell["truncated"]),
                              "x3_context_stale_rows": len(cell["context_stale"]),
                              "x3_budget": entry.generation.max_tokens})
                row = {"model": model, "attack": cell["attack"], "benchmark": bench,
                       "n_truncated": len(cell["truncated"]),
                       "n_context_stale": len(cell["context_stale"]),
                       "still_pinned": still,
                       "seconds": round(time.perf_counter() - started, 1)}
                report.append(row)
                log.info("x3adv.cell.done", **row)
                out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest.finish(status="ok", n_cells=len(report)).write(
        Path("results/manifests") / f"{manifest.run_id}.json")
    print(f"\nX3 adversarial: {sum(r['n_truncated'] + r['n_context_stale'] for r in report)} "
          f"rows rewritten; still cap-pinned {sum(r['still_pinned'] for r in report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
