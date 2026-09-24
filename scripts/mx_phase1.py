#!/usr/bin/env python3
"""MX Phase 1: harness and probe run. 20 problems, all arms, NO fault.

Proves four things before any faulted run is made, per `docs/mx_prereg.md` §8:

1. per-stage outputs are logged;
2. gate decisions are logged;
3. liar firings are logged (zero here, by construction -- the field exists and
   reads zero, which is what a no-fault run must show);
4. determinism -- the same seed reproduces identical outputs.

It also evaluates the §5 negative-control invariant early, at small N, so a
wiring fault is found on 20 problems rather than on 200.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aip.aggregation.aip import AIPAggregator, InversionThresholds  # noqa: E402
from aip.aggregation.base import ParityConfig  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.models.inference import ModelRunner  # noqa: E402
from aip.models.registry import load_registry  # noqa: E402
from aip.pipeline.arms import build_arms  # noqa: E402
from aip.pipeline.runner import (  # noqa: E402
    GenerationCache,
    aggregate_stage,
    build_stage_prompt,
    context_key,
    fit_stage,
    record_generation,  # noqa: E402
)
from aip.pipeline.stages import STAGE_ORDER  # noqa: E402
from aip.tasks import roster  # noqa: E402

BENCHMARK = "medqa"
SAME_MODEL = "ministral3_14b"
CROSS_FAMILY = ("ministral3_14b", "phi4_mini_reasoning", "llama32_3b")
ANCHOR_MODEL = "ministral3_14b"

#: Amendment A2 gave phi4_mini_reasoning an MX-local 3072-token budget here,
#: deliberately leaving `configs/models.yaml` alone so the frozen roster the
#: main study's manifests reference stayed byte-identical. X3 then amended the
#: registry itself to 3072 for every entry, so the override became a duplicate
#: of what the registry says and was removed. MX now takes its budgets from the
#: registry like everything else; A2 is superseded, not reversed.


def load_items(n: int):
    """The frozen MedQA list, truncated -- never a fresh subsample.

    `load_fixed_tasks` raises if a recorded task id is absent from the dataset,
    which is what stops MX silently running on different questions from the
    cached study results it is compared against.
    """
    from aip.tasks import get_benchmark
    from aip.tasks.task_lists import load_fixed_tasks

    bench = get_benchmark(BENCHMARK)
    return bench, load_fixed_tasks(BENCHMARK, limit=n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20250825)
    ap.add_argument("--out", type=Path, default=Path("results/mx"))
    ap.add_argument("--cache", type=Path, default=Path("data/mx_cache/phase1.parquet"))
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the per-model generation budget. Default None "
                         "means each model gets its own registry max_tokens -- "
                         "phi4_mini_reasoning needs 2048 on MedQA (main-cache mean "
                         "1299) and a single global cap truncated it at 512, "
                         "manufacturing answers it never committed to.")
    ap.add_argument("--window", type=int, default=20,
                    help="AIP sliding window W. Phase 2 and Phase 3 run windowed "
                         "at W=20, so Phase 1 does too -- a wiring proof on the "
                         "pooled gate would prove a path the campaign never takes. "
                         "The two modes even store decisions in different places "
                         "(_stats vs _task_decisions).")
    ap.add_argument("--determinism-sample", type=int, default=24,
                    help="contexts to regenerate with the same seed and compare, "
                         "for the Phase 1 determinism gate. 0 disables.")
    args = ap.parse_args()

    bench, items = load_items(args.n)
    labels = list(roster.label_space(BENCHMARK) or [])
    arms = build_arms(SAME_MODEL, CROSS_FAMILY, ANCHOR_MODEL)
    models_needed = sorted({m for a in arms.values() for m in a.replicas} | {ANCHOR_MODEL})
    registry = load_registry(ROOT / "configs" / "models.yaml")
    cache = GenerationCache(args.cache).load()
    print(f"cache: {len(cache)} generations on disk")

    # ---------------------------------------------------------------- generate
    # Stage 1 depends only on the task, so it is generated once per (model,
    # sample). Stages 2 and 3 depend on the previous stage's AGGREGATE, which is
    # arm-specific, so they are generated per arm after that arm's stage output
    # is known. Phase 1 walks arms sequentially; Phase 3 will batch.
    pending: dict[str, list[tuple[str, str, int]]] = {m: [] for m in models_needed}
    fields = {it.task_id: bench._format_fields(it) for it in items}
    gold = {it.task_id: bench.normalize(it.gold_answer) for it in items}

    def want(model: str, stage_name: str, task_id: str, prior: dict, sample: int) -> str:
        text = build_stage_prompt(stage_name, fields[task_id], prior)
        key = context_key(stage_name, model, text, sample)
        if cache.get(key) is None:
            pending[model].append((key, text, sample))
            cache.put(key, stage=stage_name, model=model, task_id=task_id,
                      sample_index=sample, prompt=text, answer=None, raw="")
        return key

    # queue stage-1 work: k samples per model plus the anchor
    keys_s1: dict[tuple[str, str, int], str] = {}
    for it in items:
        for m in models_needed:
            for s in range(3):
                keys_s1[(it.task_id, m, s)] = want(m, "solver", it.task_id, {}, s)
            keys_s1[(it.task_id, m, 99)] = want(m, "solver", it.task_id, {}, 99)

    _run_pending(registry, pending, cache, bench, args.max_tokens, args.seed)
    cache.save()

    # ------------------------------------------------------------- aggregate 1
    thresholds = InversionThresholds.load()
    parity = ParityConfig(0.1)
    rows: list[dict] = []
    stage_out: dict[tuple[str, str], str | None] = {}

    aggs = {}
    for name, arm in arms.items():
        if arm.rule == "aip":
            for st in STAGE_ORDER:
                aggs[(name, st)] = AIPAggregator(BENCHMARK, "gated", thresholds,
                                                 parity, labels,
                                                 window=args.window or None)

    for st_i, st in enumerate(STAGE_ORDER):
        if st_i > 0:
            pending = {m: [] for m in models_needed}
            for name, arm in arms.items():
                for it in items:
                    prior = {p: stage_out[(name, f"{p}:{it.task_id}")] or "?"
                             for p in STAGE_ORDER[:st_i]}
                    for slot, m in enumerate(arm.replicas):
                        want(m, st, it.task_id, prior, slot)
                    want(arm.anchor_model, st, it.task_id, prior, 99)
            _run_pending(registry, pending, cache, bench, args.max_tokens, args.seed)
            cache.save()

        for name, arm in arms.items():
            agg = aggs.get((name, st))
            # Collect the whole stage before aggregating any of it. AIPAggregator
            # declares needs_fit, and an unfitted gate has no calibrated
            # honest-coincidence ceiling: it logs no decisions and returns a
            # constant. Within a stage `prior` depends only on EARLIER stages,
            # which are already complete, so reading every task first changes
            # nothing about what each task sees.
            task_answers: dict[str, list[str | None]] = {}
            anchors: dict[str, str | None] = {}
            for it in items:
                prior = {p: stage_out[(name, f"{p}:{it.task_id}")] or "?"
                         for p in STAGE_ORDER[:st_i]}
                text = build_stage_prompt(st, fields[it.task_id], prior)
                answers = []
                for slot, m in enumerate(arm.replicas):
                    r = cache.get(context_key(st, m, text, slot))
                    answers.append(None if r is None else r["answer"])
                ar = cache.get(context_key(st, arm.anchor_model, text, 99))
                task_answers[it.task_id] = answers
                anchors[it.task_id] = None if ar is None else ar["answer"]

            fit_stage(agg, arm, task_answers, anchors)

            for it in items:
                answers = task_answers[it.task_id]
                anchor = anchors[it.task_id]
                d = aggregate_stage(arm, it.task_id, answers, anchor, labels, agg)
                stage_out[(name, f"{st}:{it.task_id}")] = d.output
                rows.append({
                    "arm": name, "stage": st, "task_id": it.task_id,
                    "output": d.output, "anchor": anchor, "gold": gold[it.task_id],
                    "correct": d.output == gold[it.task_id],
                    "replica_answers": list(d.replica_answers),
                    "replica_models": list(d.replica_models),
                    "decisions": list(d.decisions),
                    "liar_slot": d.liar_slot, "liar_variant": d.liar_variant,
                    "liar_fired": int(d.liar_slot is not None),
                    "honest_inverts": d.honest_inverts(),
                    "rule": arm.rule,
                })

    det = _determinism_pass(registry, cache, bench, args) if args.determinism_sample else None

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    if det is not None:
        det.to_parquet(args.out / "phase1_determinism.parquet", index=False)
    frame.to_parquet(args.out / "phase1_stages.parquet", index=False)
    Manifest.create(phase="mx_phase1",
                    config={"n": args.n, "benchmark": BENCHMARK, "arms": sorted(arms),
                            "same_model": SAME_MODEL, "cross_family": list(CROSS_FAMILY)},
                    seeds={"analysis": args.seed}).write(
        args.out / "phase1_stages.manifest.json")
    _report(frame, cache)
    return 0


def _determinism_pass(registry, cache, bench, args) -> pd.DataFrame:
    """Regenerate a sample of contexts with the same seed and compare.

    The generation cache is keyed by prompt hash, so a second run of the whole
    script would simply hit the cache and prove nothing. This regenerates into a
    throwaway dict and diffs against what is stored -- which is the only way the
    determinism gate means anything.
    """
    keys = sorted(cache.rows)[: args.determinism_sample]
    by_model: dict[str, list[str]] = {}
    for k in keys:
        by_model.setdefault(cache.rows[k]["model"], []).append(k)
    out = []
    for model, ks in by_model.items():
        entry = registry.models[model]
        print(f"  [determinism] {model}: regenerating {len(ks)}")
        with ModelRunner(entry) as r:
            msgs = [[{"role": "system", "content": bench.system_prompt},
                     {"role": "user", "content": cache.rows[k]["prompt"]}] for k in ks]
            toks = [r.encode_messages(m) for m in msgs]
            outs = r.generate(toks, max_tokens=args.max_tokens,
                              temperature=0.7, seed=args.seed)
            for k, o in zip(ks, outs, strict=True):
                fresh: dict = {}
                record_generation(fresh, o, bench)
                stored = cache.rows[k]
                out.append({
                    "key": k, "model": model,
                    "answer_stored": stored["answer"], "answer_fresh": fresh["answer"],
                    "answer_match": stored["answer"] == fresh["answer"],
                    "text_match": stored["raw"] == fresh["raw"],
                })
    return pd.DataFrame(out)


def _run_pending(registry, pending, cache, bench, max_tokens, seed) -> None:
    """Generate every queued context, one resident model at a time."""
    for model, jobs in pending.items():
        if not jobs:
            continue
        entry = registry.models[model]
        print(f"  [{model}] {len(jobs)} generations")
        with ModelRunner(entry) as r:
            msgs = [[{"role": "system", "content": bench.system_prompt},
                     {"role": "user", "content": text}] for _, text, _ in jobs]
            toks = [r.encode_messages(m) for m in msgs]
            outs = r.generate(toks, max_tokens=max_tokens, temperature=0.7,
                              seed=seed)
            for (key, _text, _s), o in zip(jobs, outs, strict=True):
                record_generation(cache.rows[key], o, bench)


def _report(frame: pd.DataFrame, cache: GenerationCache) -> None:
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("MX PHASE 1 — harness probe, no fault")
    print("=" * 90)
    print(f"generations cached: {len(cache)}")
    print(f"stage rows: {len(frame)}  ({frame.task_id.nunique()} tasks x "
          f"{frame.arm.nunique()} arms x {frame.stage.nunique()} stages)")
    print("\nfinal-stage accuracy by arm:")
    fin = frame[frame.stage == STAGE_ORDER[-1]]
    print(fin.groupby("arm").correct.mean().round(3).to_string())
    print("\nper-stage accuracy:")
    print(frame.pivot_table(index="stage", columns="arm", values="correct")
          .reindex(STAGE_ORDER).round(3).to_string())
    print(f"\nliar firings (must be 0): {int(frame.liar_fired.sum())}")
    print(f"honest-channel INVERTs (false inversions): "
          f"{int(frame.honest_inverts.sum())}")
    logged = frame[frame.rule == "aip"]
    have = int((logged.decisions.map(len) > 0).sum())
    print(f"gate decisions logged: {have}/{len(logged)} AIP stage-rows")
    print(f"extraction failures (None output): {int(frame.output.isna().sum())}")


if __name__ == "__main__":
    raise SystemExit(main())
