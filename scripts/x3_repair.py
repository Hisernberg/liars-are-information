#!/usr/bin/env python3
"""X3: regenerate the cache rows whose generation hit the token cap.

Design principle 1 says no phase after A re-runs inference. This is not a later
phase -- it is a repair of Phase A's own output, and it is deliberately the
narrowest one that fixes the defect:

* only rows with ``finish_reason == "length"`` are regenerated;
* same prompts, same task list, same models, same seed, temperature 0;
* the only thing that changes is the completion budget, which the registry now
  carries at 3072 for every model (X3 amendment);
* naturally-finished rows are not written at all, so they stay byte-identical.

**Control rows.** Regenerating a subset changes the batch, and the claim that
this is a repair rather than a fresh sample rests on batch invariance. So each
cell also regenerates a deterministic sample of rows that finished naturally and
compares them byte-for-byte against the cache. Those rows are never written --
they exist to make the batch-invariance claim falsifiable, and a mismatch is
reported rather than absorbed.

    python scripts/x3_repair.py --models phi4_mini_reasoning --dry-run
    python scripts/x3_repair.py --models phi4_mini_reasoning
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

#: Naturally-finished rows re-generated per cell purely as a batch-invariance
#: control. Enough to catch a systematic break, few enough to be free.
N_CONTROL = 8


def finish_reason(extra_json: str | None) -> str | None:
    if not extra_json:
        return None
    return json.loads(extra_json).get("finish_reason")


def split_cell(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (task ids to repair, task ids used as byte-identity controls)."""
    fr = frame["extra_json"].map(finish_reason)
    repair = sorted(frame.loc[fr == "length", "task_id"].astype(str))
    finished = sorted(frame.loc[fr == "stop", "task_id"].astype(str))
    if not repair or not finished:
        return repair, []
    # Evenly spaced through the sorted ids rather than the first N: the first N
    # would all come from one end of the task list.
    step = max(1, len(finished) // N_CONTROL)
    return repair, finished[::step][:N_CONTROL]


def main() -> int:
    global N_CONTROL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--cache-root", default="data/cache")
    ap.add_argument("--out", default="results/x3_repair")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no inference")
    # data/cache_t07 is the same defect in a cache generated at T=0.7,
    # sample_index 1. Its rows must be rewritten with those values or the
    # upsert key (task_id, model, sample_index) lands on the wrong row.
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the cell temperature (data/cache_t07 is 0.7)")
    ap.add_argument("--sample-index", type=int, default=0)
    # Measures drift without repairing anything: three models do not reproduce
    # their finished rows byte-identically under batch recomposition, and the
    # question that matters is whether that reaches the extracted answer.
    ap.add_argument("--n-control", type=int, default=None,
                    help="override N_CONTROL. The default 8 detects a systematic "
                         "break; a flip RATE with a usable confidence bound needs "
                         "a couple of hundred rows")
    ap.add_argument("--controls-only", action="store_true",
                    help="regenerate ONLY the control rows, write nothing. A harsher "
                         "condition than the repair: the batch shrinks to 8, and drift "
                         "is batch-composition-driven, so this bounds drift rather "
                         "than measuring it")
    # Two candidate mechanisms for the drift, both visible in the engine config
    # and neither covered by VLLM_BATCH_INVARIANT, which makes ATTENTION
    # invariant to batch composition and nothing else:
    #   * CUDA graphs are captured at bucketed batch sizes (1, 2, 4, 8, 16, 24,
    #     32, 40, ...), so a batch of 8, 24 and 59 execute three different
    #     captured graphs;
    #   * prefix caching is on, so which prefixes are already cached depends on
    #     what else is in the batch.
    # --enforce-eager removes the first. If drift survives it, the graphs are
    # not the cause.
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs (drift diagnostic)")
    ap.add_argument("--no-write", action="store_true",
                    help="run the repair batch exactly as the repair did but write "
                         "nothing. This is the batch-matched probe -- same prompts, "
                         "same batch size -- so its drift figure is the repair's own")
    args = ap.parse_args()
    if args.n_control is not None:
        N_CONTROL = args.n_control

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = Path(args.cache_root)
    registry = load_registry(cfg["paths"]["registry"])
    reg_hash = registry.hash()
    known = frozenset(cfg.get("reproducibility", {}).get("known_nonreproducing") or [])
    task_list_path = cfg["paths"]["task_lists"]
    benches = args.benchmarks or [b["name"] for b in cfg["benchmarks"]]

    plan: list[dict[str, Any]] = []
    for model in args.models:
        for bench in benches:
            path = cache_mod.cache_path(bench, model, root=root)
            if not path.exists():
                continue
            repair, control = split_cell(pd.read_parquet(path))
            if repair:
                plan.append({"model": model, "benchmark": bench,
                             "n_repair": len(repair), "n_control": len(control),
                             "repair": repair, "control": control})

    total = sum(p["n_repair"] for p in plan)
    print(f"X3 plan: {total} rows to repair across {len(plan)} cells")
    for p in plan:
        print(f"  {p['benchmark']:9s} {p['model']:22s} repair={p['n_repair']:4d} "
              f"control={p['n_control']}")
    if args.dry_run or not plan:
        return 0

    manifest = Manifest.create(phase="x3_repair",
                               config={"cells": len(plan), "n_repair": total,
                                       "registry_hash": reg_hash},
                               seeds={"generation": int(cfg["seeds"]["generation"])})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []

    for model in args.models:
        cells = [p for p in plan if p["model"] == model]
        if not cells:
            continue
        entry = registry.get(model)
        budget = entry.generation.max_tokens
        with ModelRunner(entry, deterministic=True,
                         enforce_eager=args.enforce_eager) as runner:
            for cell in cells:
                bench = get_benchmark(cell["benchmark"])
                items = load_fixed_tasks(bench.name, path=task_list_path)
                wanted = (set(cell["control"]) if args.controls_only
                          else set(cell["repair"]) | set(cell["control"]))
                items = [it for it in items if it.task_id in wanted]
                started = time.perf_counter()
                log.info("x3.cell.start", model=model, benchmark=bench.name,
                         n=len(items), budget=budget)
                result = run_cell(runner, bench, items, entry, cfg,
                                  seed=int(cfg["seeds"]["generation"]),
                                  known_nonreproducing=known,
                                  temperature=args.temperature,
                                  sample_index=args.sample_index)

                by_id = {p.task_id: p for p in result.predictions}
                before = pd.read_parquet(cache_mod.cache_path(bench.name, model, root=root))
                before = before.set_index("task_id")

                # Control: naturally-finished rows must come back byte-identical.
                # Text drift and ANSWER drift are counted separately. Two models
                # reproduce the text imperfectly under batch recomposition, and
                # whether that reaches the extracted answer is the question that
                # decides whether the drift matters at all -- a reworded
                # derivation ending in the same letter is not the same defect as
                # a different letter.
                drift, answer_drift = [], []
                for task in cell["control"]:
                    if task not in by_id:
                        continue
                    if by_id[task].raw_completion == before.at[task, "raw_completion"]:
                        continue
                    drift.append(task)
                    was = before.at[task, "extracted_answer"]
                    was = None if pd.isna(was) else str(was)
                    now = by_id[task].extracted_answer
                    now = None if now is None else str(now)
                    if was != now:
                        answer_drift.append(task)

                repaired = [] if (args.controls_only or args.no_write) else [
                    by_id[t] for t in cell["repair"] if t in by_id]
                still_pinned = sum(
                    1 for p in repaired if (p.extra or {}).get("finish_reason") == "length"
                )
                if repaired:
                    cache_mod.write_predictions(
                        repaired, bench.name, model, root=root,
                        manifest={
                            "prompt_template_hash": bench.prompt_template_hash(),
                            "hf_id": entry.hf_id, "dtype": entry.dtype,
                            "max_model_len": entry.max_model_len,
                            "seed": int(cfg["seeds"]["generation"]),
                            "registry_hash": reg_hash,
                            "batch_invariant": True,
                            "sample_index": args.sample_index,
                            "temperature": float(args.temperature or 0.0),
                            "run_id": manifest.run_id,
                            "x3_repaired_rows": len(repaired),
                            "x3_budget": budget,
                            "x3_control_drift": len(drift),
                            "x3_control_answer_drift": len(answer_drift),
                        },
                    )
                row = {"model": model, "benchmark": bench.name, "budget": budget,
                       "n_repair": len(repaired), "still_pinned": still_pinned,
                       "n_control": len(cell["control"]), "control_drift": drift,
                       "control_answer_drift": answer_drift,
                       "seconds": round(time.perf_counter() - started, 1)}
                report.append(row)
                log.info("x3.cell.done",
                         **{k: v for k, v in row.items()
                            if k not in ("control_drift", "control_answer_drift")},
                         control_drift=len(drift), control_answer_drift=len(answer_drift))
                (out / "x3_repair_report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8")

    manifest.finish(status="ok", n_cells=len(report)).write(
        Path("results/manifests") / f"{manifest.run_id}.json")
    pinned = sum(r["still_pinned"] for r in report)
    drifted = sum(len(r["control_drift"]) for r in report)
    print(f"\nX3 repaired {sum(r['n_repair'] for r in report)} rows; "
          f"still cap-pinned {pinned}; control drift {drifted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
