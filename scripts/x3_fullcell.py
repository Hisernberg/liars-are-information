#!/usr/bin/env python3
"""X3: regenerate whole cells, for the models where a subset repair is not a repair.

`x3_repair.py` rewrites only the rows that hit the token cap, and its control
rows test the assumption that makes that legitimate: regenerating a subset
changes the batch, and the result is a repair rather than a fresh sample only if
the engine is invariant to that. Six of nine models passed across 192 control
rows. Three did not:

    olmo3_32b_think   37 / 40 control rows differed, 1 changed the answer
    qwen38_27b        19 / 40                       not measured (evicted)
    llama32_3b        14 / 24                       2 of 6 changed the answer

For those models the repaired cell is a mixture: most rows from the original
batch of 200 or 500, the repaired rows from a batch of 9 to 59. This script
removes the mixture by regenerating every row of the cell in one batch, so the
cell is one internally consistent sample again.

It is deliberately a separate script. It rewrites rows that are not defective,
which `x3_repair.py` promises never to do, and that promise is worth keeping
where it holds.

    python scripts/x3_fullcell.py --models llama32_3b --dry-run
    python scripts/x3_fullcell.py --models llama32_3b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from phase_a_cache import run_cell  # noqa: E402

from aip.harness import cache as cache_mod  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.models.inference import ModelRunner  # noqa: E402
from aip.models.registry import load_registry  # noqa: E402
from aip.tasks import get_benchmark  # noqa: E402
from aip.tasks.task_lists import load_fixed_tasks  # noqa: E402

log = configure_logging(level="INFO")
CONFIG = ROOT / "configs/experiments/phase_a_cache.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    ap.add_argument("--cache-root", type=Path, default=ROOT / "data/cache")
    ap.add_argument("--out", type=Path, default=ROOT / "results/x3_fullcell")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    registry = load_registry(ROOT / "configs/models.yaml")
    root = args.cache_root
    task_list_path = cfg["paths"]["task_lists"]
    benches = args.benchmarks or [b["name"] for b in cfg["benchmarks"]]
    args.out.mkdir(parents=True, exist_ok=True)

    plan = []
    for model in args.models:
        for b in benches:
            path = cache_mod.cache_path(b, model, root=root)
            if path.exists():
                plan.append({"model": model, "benchmark": b, "n": len(pd.read_parquet(path))})
    total = sum(p["n"] for p in plan)
    print(f"X3 full-cell plan: {total} rows across {len(plan)} cells")
    for p in plan:
        print(f"  {p['benchmark']:9s} {p['model']:22s} rows={p['n']:4d}")
    if args.dry_run:
        return 0

    known = frozenset(cfg.get("known_nonreproducing", []) or [])
    report: list[dict[str, Any]] = []
    for model in args.models:
        cells = [p for p in plan if p["model"] == model]
        if not cells:
            continue
        entry = registry.get(model)
        with ModelRunner(entry, deterministic=True) as runner:
            for cell in cells:
                bench = get_benchmark(cell["benchmark"])
                items = load_fixed_tasks(bench.name, path=task_list_path)
                started = time.perf_counter()
                log.info("x3fc.cell.start", model=model, benchmark=bench.name,
                         n=len(items), budget=entry.generation.max_tokens)
                result = run_cell(runner, bench, items, entry, cfg,
                                  seed=int(cfg["seeds"]["generation"]),
                                  known_nonreproducing=known)

                before = pd.read_parquet(
                    cache_mod.cache_path(bench.name, model, root=root)).set_index("task_id")
                changed = answer_changed = 0
                for p in result.predictions:
                    if p.task_id not in before.index:
                        continue
                    if p.raw_completion != before.at[p.task_id, "raw_completion"]:
                        changed += 1
                        was = before.at[p.task_id, "extracted_answer"]
                        was = None if pd.isna(was) else str(was)
                        now = None if p.extracted_answer is None else str(p.extracted_answer)
                        if was != now:
                            answer_changed += 1

                acc_before = float(before["is_correct"].mean())
                manifest = Manifest.begin(
                    run_id=f"x3_fullcell_{model}_{bench.name}",
                    config={"model": model, "benchmark": bench.name,
                            "budget": entry.generation.max_tokens},
                )
                cache_mod.write_predictions(
                    list(result.predictions), bench.name, model, root=root,
                    manifest={
                        "prompt_template_hash": bench.prompt_template_hash(),
                        "hf_id": entry.hf_id, "dtype": entry.dtype,
                        "max_model_len": entry.max_model_len,
                        "seed": int(cfg["seeds"]["generation"]),
                        "batch_invariant": True, "sample_index": 0, "temperature": 0.0,
                        "run_id": manifest.run_id,
                        "x3_fullcell": True,
                        "x3_rows_rewritten": len(result.predictions),
                    },
                )
                after = pd.read_parquet(cache_mod.cache_path(bench.name, model, root=root))
                row = {
                    "model": model, "benchmark": bench.name,
                    "n_rows": len(result.predictions),
                    "n_text_changed": changed, "n_answer_changed": answer_changed,
                    "acc_before": round(acc_before, 4),
                    "acc_after": round(float(after["is_correct"].mean()), 4),
                    "seconds": round(time.perf_counter() - started, 1),
                }
                row["acc_delta"] = round(row["acc_after"] - row["acc_before"], 4)
                report.append(row)
                log.info("x3fc.cell.done", **row)
                (args.out / "x3_fullcell_report.json").write_text(json.dumps(report, indent=2))

    print(f"X3 full-cell: {sum(r['n_rows'] for r in report)} rows rewritten; "
          f"{sum(r['n_answer_changed'] for r in report)} answers changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
