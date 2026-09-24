#!/usr/bin/env python3
"""Phase A: cache-once inference over the benchmark x model grid.

Everything downstream reads what this writes. No later phase may re-run
inference (design principle 1).

Two generation passes per task, deliberately separate:

1. the answer, greedy, with token logprobs -- yielding the *unfakeable*
   confidence surface (mean logprob over the answer span only);
2. a self-reported 0-100 confidence, which is the surface the
   ``falsified_confidence`` attack corrupts.

Models are visited in the outer loop and released before the next is loaded, so
exactly one is resident at a time.

    python scripts/phase_a_cache.py --config configs/experiments/phase_a_cache.yaml --dry-run
    python scripts/phase_a_cache.py --config configs/experiments/phase_a_cache.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.analysis.metrics import rank_auc  # noqa: E402
from aip.harness import cache as cache_mod  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest, text_hash  # noqa: E402
from aip.models.confidence import span_confidence  # noqa: E402
from aip.models.inference import GenerationResult, ModelRunner  # noqa: E402
from aip.models.registry import ModelEntry, load_registry  # noqa: E402
from aip.tasks import get_benchmark, parse_self_reported_confidence  # noqa: E402
from aip.tasks.base import Benchmark  # noqa: E402
from aip.tasks.task_lists import load_fixed_tasks  # noqa: E402
from aip.types import Prediction, TaskItem  # noqa: E402

log = configure_logging(level="INFO")


@dataclass
class CellResult:
    """Outcome of one (model, benchmark) cell, plus its quality-gate inputs."""

    model: str
    benchmark: str
    predictions: list[Prediction] = field(default_factory=list)
    answer_results: list[GenerationResult] = field(default_factory=list)
    items: list[TaskItem] = field(default_factory=list)
    generate_seconds: float = 0.0
    confidence_seconds: float = 0.0
    determinism_ok: bool | None = None
    determinism_detail: str = ""
    error: str | None = None
    sample_index: int = 0
    temperature: float = 0.0
    wall_seconds: float = 0.0

    # -- gate f: accuracy headroom ---------------------------------------
    @property
    def headroom_flag(self) -> str:
        """Ceiling/floor flag. Reported, never acted on -- accuracy is data."""
        if not self.predictions:
            return "empty"
        if self.accuracy > 0.95:
            return "CEILING"
        if self.accuracy < 0.20:
            return "FLOOR"
        return "ok"

    # -- gate b revisited: AUC -------------------------------------------
    def confidence_auc(self) -> float | None:
        correct = [
            p.logprob_confidence
            for p in self.predictions
            if p.is_correct and p.logprob_confidence is not None
        ]
        wrong = [
            p.logprob_confidence
            for p in self.predictions
            if not p.is_correct and p.logprob_confidence is not None
        ]
        return rank_auc(correct, wrong)

    # -- gate a ----------------------------------------------------------
    @property
    def extraction_failures(self) -> list[str]:
        return [p.task_id for p in self.predictions if p.extracted_answer is None]

    @property
    def extraction_rate(self) -> float:
        if not self.predictions:
            return 0.0
        return 1.0 - len(self.extraction_failures) / len(self.predictions)

    @property
    def truncated(self) -> list[str]:
        return [
            p.task_id
            for p, r in zip(self.predictions, self.answer_results, strict=False)
            if r.truncated
        ]

    # -- gate b ----------------------------------------------------------
    def confidence_by_correctness(self) -> tuple[float | None, float | None]:
        correct = [
            p.logprob_confidence
            for p in self.predictions
            if p.is_correct and p.logprob_confidence is not None
        ]
        wrong = [
            p.logprob_confidence
            for p in self.predictions
            if not p.is_correct and p.logprob_confidence is not None
        ]
        return (
            statistics.fmean(correct) if correct else None,
            statistics.fmean(wrong) if wrong else None,
        )

    # -- gate c ----------------------------------------------------------
    @property
    def self_report_failures(self) -> list[str]:
        return [p.task_id for p in self.predictions if p.self_reported_confidence is None]

    @property
    def accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(1 for p in self.predictions if p.is_correct) / len(self.predictions)

    @property
    def tasks_per_second(self) -> float:
        total = self.generate_seconds + self.confidence_seconds
        return len(self.predictions) / total if total > 0 else 0.0


def completed_for_sample(
    benchmark: str,
    model: str,
    root: Path,
    sample_index: int,
    require_confidence: bool = True,
) -> set[str]:
    """Task ids already cached *for this sample slot*.

    Resume must be sample-aware: the T>0 second pass writes sample_index=1 rows
    alongside the greedy sample_index=0 rows, and a slot-blind check would treat
    the greedy pass as having already completed the sampled one.
    """
    import pandas as pd

    path = cache_mod.cache_path(benchmark, model, root)
    if not path.exists():
        return set()
    frame = cache_mod.enforce_schema(pd.read_parquet(path))
    frame = frame[frame["sample_index"] == sample_index]
    if require_confidence:
        frame = frame[frame["self_reported_confidence"].notna()]
    return set(frame["task_id"].dropna().astype(str).tolist())


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_cell(
    runner: ModelRunner,
    benchmark: Benchmark,
    items: list[TaskItem],
    entry: ModelEntry,
    cfg: dict[str, Any],
    seed: int,
    check_determinism: bool = False,
    known_nonreproducing: frozenset[str] = frozenset(),
    temperature: float | None = None,
    sample_index: int = 0,
    elicit_confidence: bool | None = None,
) -> CellResult:
    """Generate answers and confidences for one (model, benchmark) cell."""
    temp = entry.generation.temperature if temperature is None else temperature
    result = CellResult(
        model=entry.name,
        benchmark=benchmark.name,
        items=items,
        sample_index=sample_index,
        temperature=temp,
    )
    inference_cfg = cfg.get("inference", {})
    wall_start = time.perf_counter()

    # -- pass 1: answers, with logprobs ----------------------------------
    prompts = [runner.encode_messages(benchmark.build_messages(it)) for it in items]
    started = time.perf_counter()
    # None => use the model's own registry budget. A single global max_tokens
    # would truncate reasoning models mid-thought (Phi-4-mini lost 3/5 GSM8K
    # answers to a 640-token cap meant for non-reasoning models).
    answers = runner.generate(
        prompts,
        max_tokens=inference_cfg.get("max_tokens_override"),
        temperature=temp,
        seed=seed if temp > 0 else None,
    )
    result.generate_seconds = time.perf_counter() - started
    result.answer_results = answers

    extractions = [benchmark.extract_answer(gen.text) for gen in answers]

    # -- pass 2: self-reported confidence --------------------------------
    self_reports: list[float | None] = [None] * len(items)
    want_confidence = (
        inference_cfg.get("elicit_self_reported_confidence", True)
        if elicit_confidence is None
        else elicit_confidence
    )
    if want_confidence:
        conf_prompts = [
            runner.encode_messages(benchmark.build_confidence_messages(it, ex.value))
            for it, ex in zip(items, extractions, strict=True)
        ]
        started = time.perf_counter()
        conf_out = runner.generate(
            conf_prompts,
            max_tokens=inference_cfg.get("confidence_max_tokens_override")
            or entry.generation.confidence_max_tokens,
        )
        result.confidence_seconds = time.perf_counter() - started
        self_reports = [parse_self_reported_confidence(g.text) for g in conf_out]
        result.confidence_texts = [g.text for g in conf_out]  # type: ignore[attr-defined]

    # -- assemble predictions --------------------------------------------
    # A reasoning model's answer is the tail of a sampled chain of thought, so a
    # T>0 resample varies the whole path, not just the committed answer. That is
    # a different quantity from answer-level resampling and Phase 2 must not pool
    # the two -- it is exactly the contrast behind "temperature is a weak
    # decorrelator for reasoning models". Derived from the frozen roster tier.
    resample_kind = "path_level" if entry.tier == "reasoning" else "answer_level"

    for item, gen, extraction, reported in zip(
        items, answers, extractions, self_reports, strict=True
    ):
        confidence, n_answer_tokens = span_confidence(gen.tokens, extraction)
        correct = benchmark.score(extraction.value, item.gold_answer)
        result.predictions.append(
            Prediction(
                task_id=item.task_id,
                benchmark=benchmark.name,
                model=entry.name,
                raw_completion=gen.text,
                extracted_answer=extraction.value,
                gold_answer=item.gold_answer,
                is_correct=correct,
                logprob_confidence=confidence,
                self_reported_confidence=reported,
                n_answer_tokens=n_answer_tokens,
                n_completion_tokens=gen.n_tokens,
                seed=seed,
                temperature=float(temp),
                sample_index=sample_index,
                nonreproducing=item.task_id in known_nonreproducing,
                resample_kind=resample_kind,
                extra={
                    "finish_reason": gen.finish_reason,
                    "truncated": gen.truncated,
                    "answer_span": [extraction.start, extraction.end],
                },
            )
        )

    result.wall_seconds = time.perf_counter() - wall_start

    # -- gate d: determinism ---------------------------------------------
    if check_determinism and items:
        # Re-run the *whole batch*, not a single prompt: vLLM's numerics depend
        # on batch composition, so comparing a batch-of-1 rerun against a
        # batch-of-N original would flag benign batching effects as
        # nondeterminism.
        repeat = runner.generate(prompts, max_tokens=inference_cfg.get("max_tokens_override"))
        diffs = [items[i].task_id for i in range(len(prompts)) if repeat[i].text != answers[i].text]
        # An id already recorded in reproducibility.known_nonreproducing is an
        # ACCEPTED caveat, not a failure: it has been measured, its cause is
        # documented, and downstream phases read the cache rather than
        # regenerating it. A NEW id is a genuine gate failure. Without this
        # split the list is decorative -- it was written into every manifest but
        # never consulted, so a known caveat still failed the gate.
        result.determinism_ok, result.determinism_detail = classify_determinism(
            diffs, len(prompts), known_nonreproducing
        )
    return result


def classify_determinism(
    diffs: list[str], n_prompts: int, known_nonreproducing: frozenset[str]
) -> tuple[bool, str]:
    """Split re-run differences into accepted caveats and genuine failures.

    An id already recorded in ``reproducibility.known_nonreproducing`` is an
    ACCEPTED caveat: it has been measured, its cause is documented, and every
    downstream phase reads the cache rather than regenerating it. A NEW id is a
    real gate failure. Without this split the list is decorative -- it was
    written into every manifest but never consulted, so a documented caveat
    still failed the gate.

    Kept as a free function precisely so it can be tested: the nondeterminism it
    exists for is intermittent, so a live run cannot be relied on to exercise it.
    """
    expected = [t for t in diffs if t in known_nonreproducing]
    unexpected = [t for t in diffs if t not in known_nonreproducing]
    if not diffs:
        return True, f"all {n_prompts} byte-identical"
    if not unexpected:
        return True, (
            f"{len(expected)}/{n_prompts} differ, all previously recorded "
            f"in known_nonreproducing: {expected}"
        )
    detail = f"{len(unexpected)}/{n_prompts} NEWLY differ: {unexpected}"
    if expected:
        detail += f" (plus {len(expected)} already recorded: {expected})"
    return False, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/phase_a_cache.yaml")
    )
    parser.add_argument("--dry-run", action="store_true", help="few tasks, separate cache root")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="override sampling temperature (second-sample pass)",
    )
    parser.add_argument(
        "--sample-index", type=int, default=0, help="sample slot; the T>0 second pass uses 1"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=None, help="override the prediction cache namespace"
    )
    parser.add_argument(
        "--no-self-report", action="store_true", help="skip the self-reported confidence pass"
    )
    parser.add_argument("--tag", default=None, help="label for the report filename")
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="override the per-benchmark task count from the config",
    )
    parser.add_argument(
        "--task-lists",
        type=Path,
        default=None,
        help="override the frozen task list (e.g. the gsm8k_ext calibration set)",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="disable vLLM batch-invariant kernels (faster, but the cache stops being bit-reproducible)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    runtime = cfg.get("runtime", {})
    seeds = cfg.get("seeds", {})

    registry = load_registry(paths.get("registry", "configs/models.yaml"))
    reproducibility = cfg.get("reproducibility", {}) or {}
    task_list_path = args.task_lists or Path(paths.get("task_lists", "configs/task_lists.json"))
    task_list_hash = (
        text_hash(task_list_path.read_text(encoding="utf-8")) if task_list_path.exists() else None
    )
    model_names = args.models or cfg.get("models") or list(registry.enabled_names)
    bench_cfgs = cfg.get("benchmarks", [])
    if args.benchmarks:
        bench_cfgs = [b for b in bench_cfgs if b["name"] in args.benchmarks]

    cache_root = args.cache_root or Path(
        "data/cache_dryrun" if args.dry_run else paths.get("prediction_cache", "data/cache")
    )
    n_override = int(runtime.get("dry_run_tasks", 5)) if args.dry_run else None

    manifest = Manifest.create(
        phase="phase_a_dryrun" if args.dry_run else "phase_a",
        config=cfg,
        seeds={k: int(v) for k, v in seeds.items()},
        notes={
            "dry_run": args.dry_run,
            "models": list(model_names),
            "sample_index": args.sample_index,
            "temperature_override": args.temperature,
            "batch_invariant": not args.no_deterministic,
            "task_list_hash": task_list_hash,
            # Gate d caveat, accepted as a known limitation: these tasks did not
            # reproduce byte-identically across identical re-runs even with
            # batch-invariant kernels. Everything downstream reads the cached
            # values, so results stay internally consistent; what is not
            # guaranteed is regenerating a bit-identical cache from scratch.
            "known_nonreproducing_task_ids": reproducibility.get("known_nonreproducing", []),
            "reproducibility_note": reproducibility.get("note", ""),
        },
    )

    results: list[CellResult] = []
    load_times: dict[str, float] = {}

    for name in model_names:
        entry = registry.get(name)
        try:
            with ModelRunner(
                entry,
                enforce_eager=args.enforce_eager,
                deterministic=not args.no_deterministic,
            ) as runner:
                load_times[name] = runner.load_seconds
                for bench_cfg in bench_cfgs:
                    benchmark = get_benchmark(bench_cfg["name"])
                    n_tasks = n_override or args.n_tasks or int(bench_cfg.get("n_tasks", 100))
                    # The frozen list in configs/task_lists.json is the authority:
                    # Phase D must run on exactly these questions or its
                    # adversarial broadcasts cannot be joined to this cache.
                    items = load_fixed_tasks(benchmark.name, path=task_list_path, limit=n_tasks)

                    if runtime.get("resume", True) and not args.dry_run:
                        done = completed_for_sample(
                            benchmark.name,
                            name,
                            root=cache_root,
                            sample_index=args.sample_index,
                            require_confidence=not args.no_self_report,
                        )
                        items = [it for it in items if it.task_id not in done]
                        if not items:
                            log.info("cell.skipped", model=name, benchmark=benchmark.name)
                            continue

                    log.info("cell.start", model=name, benchmark=benchmark.name, n_tasks=len(items))
                    cell = run_cell(
                        runner,
                        benchmark,
                        items,
                        entry,
                        cfg,
                        seed=int(seeds.get("generation", 0)),
                        check_determinism=args.dry_run,
                        known_nonreproducing=frozenset(
                            reproducibility.get("known_nonreproducing") or []
                        ),
                        temperature=args.temperature,
                        sample_index=args.sample_index,
                        elicit_confidence=False if args.no_self_report else None,
                    )
                    results.append(cell)
                    cache_mod.write_predictions(
                        cell.predictions,
                        benchmark.name,
                        name,
                        root=cache_root,
                        manifest={
                            "prompt_template_hash": benchmark.prompt_template_hash(),
                            "hf_id": entry.hf_id,
                            "dtype": entry.dtype,
                            "max_model_len": entry.max_model_len,
                            "seed": int(seeds.get("generation", 0)),
                            "registry_hash": registry.hash(),
                            "batch_invariant": not args.no_deterministic,
                            "sample_index": args.sample_index,
                            "temperature": cell.temperature,
                            "task_list_hash": task_list_hash,
                            "run_id": manifest.run_id,
                            "dry_run": args.dry_run,
                        },
                    )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            log.error("model.failed", model=name, error=str(exc)[:800], exc_info=True)
            results.append(CellResult(model=name, benchmark="*", error=str(exc)[:2000]))

    tag = args.tag or ("dryrun" if args.dry_run else f"full_s{args.sample_index}")
    report_path = Path(paths.get("results", "results")) / f"phase_a_{tag}_report.json"
    manifest_path = Path(paths.get("results", "results")) / "manifests" / f"{manifest.run_id}.json"
    manifest.finish(status="ok", n_cells=len(results)).write(manifest_path)

    gates_ok = True
    if args.dry_run:
        gates_ok = print_dry_run_report(results, load_times, registry, bench_cfgs, report_path)
    else:
        print_full_report(results, load_times, report_path)

    failures = [r for r in results if r.error]
    if failures:
        return 1
    # A dry run exists to gate the full run. Reporting a red gate and then
    # exiting 0 would let an orchestrator walk straight from a 40%-extraction
    # dry run into a 1500-task cache job, which is the one thing the gate is
    # there to prevent.
    return 0 if gates_ok else 2


#: Minimum members in BOTH the correct and incorrect classes before gate b's
#: comparison of mean confidences means anything. Below this the cell is
#: INCONCLUSIVE, not failing: rejecting a model on a one-sample comparison is
#: worse than admitting the dry run cannot answer the question.
MIN_CLASS_FOR_GATE_B = 2


def print_dry_run_report(
    results: list[CellResult],
    load_times: dict[str, float],
    registry: Any,
    bench_cfgs: list[dict],
    report_path: Path,
) -> bool:
    """Sample outputs plus the five quality gates. Returns whether they passed.

    Gates a, b and d are pass/fail and are returned; c and e are reported for
    judgement (a self-report parse failure is data about the model, and
    throughput is a projection, not a threshold).
    """
    ok = [r for r in results if not r.error]
    bar = "=" * 100
    # A gate with no data is not a pass. Vacuous truth here would let a run in
    # which every model crashed print five green gates.
    have_data = bool(ok)

    # -- sample outputs ---------------------------------------------------
    for cell in ok:
        print(f"\n{bar}\nCELL  {cell.model}  x  {cell.benchmark}\n{bar}")
        for i, pred in enumerate(cell.predictions[:2]):
            print(f"\n--- sample {i + 1}: {pred.task_id} " + "-" * 60)
            print("RAW COMPLETION (verbatim, untrimmed):")
            print(pred.raw_completion)
            print("-" * 76)
            print(f"  extracted        : {pred.extracted_answer!r}")
            print(f"  gold             : {pred.gold_answer!r}")
            print(f"  correct          : {pred.is_correct}")
            lp = pred.logprob_confidence
            sr = pred.self_reported_confidence
            lp_txt = "None" if lp is None else f"{lp:.4f}"
            sr_txt = "None" if sr is None else f"{sr:.3f}"
            print(
                f"  logprob conf     : {lp_txt}"
                f"   (over {pred.n_answer_tokens} answer tokens of {pred.n_completion_tokens})"
            )
            print(f"  self-reported    : {sr_txt}   parsed={'yes' if sr is not None else 'NO'}")
            print(f"  finish_reason    : {pred.extra.get('finish_reason')}")

    # -- gates ------------------------------------------------------------
    print(f"\n\n{bar}\nQUALITY GATES\n{bar}")

    print("\n[a] EXTRACTION RATE  (target: >=95% gsm8k/mmlu, >=90% math500)")
    print(f"{'cell':<40}{'n':>4}{'extracted':>11}{'rate':>8}{'trunc':>7}  verdict")
    gate_a = have_data
    for cell in ok:
        target = 0.90 if cell.benchmark == "math500" else 0.95
        passed = cell.extraction_rate >= target
        gate_a &= passed
        n_ok = len(cell.predictions) - len(cell.extraction_failures)
        print(
            f"{cell.model + ' x ' + cell.benchmark:<40}{len(cell.predictions):>4}"
            f"{n_ok:>11}{cell.extraction_rate:>8.0%}{len(cell.truncated):>7}"
            f"  {'PASS' if passed else 'FAIL'} (target {target:.0%})"
        )
    for cell in ok:
        if cell.extraction_failures:
            print(f"    {cell.model} x {cell.benchmark} failed: {cell.extraction_failures}")

    print("\n[b] CONFIDENCE SANITY  (mean answer-span confidence; correct must exceed incorrect)")
    print(f"{'cell':<40}{'correct':>10}{'incorrect':>11}{'delta':>9}  verdict")
    gate_b = have_data
    for cell in ok:
        c, w = cell.confidence_by_correctness()
        n_wrong = sum(1 for p in cell.predictions if not p.is_correct)
        n_right = len(cell.predictions) - n_wrong
        # A class with one or two members cannot support a comparison of means.
        # This is rule R2 applied to the dry-run gate: olmo2_7b failed gate b on
        # a single wrong answer (0.978 correct vs 1.0 wrong, 4 right / 1 wrong),
        # which says nothing about its confidence surface. Reporting that as FAIL
        # rejects a model on a sample of one; INCONCLUSIVE is the honest verdict,
        # and gate f's full-run AUC is where the question is actually answered.
        if c is None or w is None or min(n_wrong, n_right) < MIN_CLASS_FOR_GATE_B:
            print(
                f"{cell.model + ' x ' + cell.benchmark:<40}"
                f"{'n/a' if c is None else round(c, 4):>10}"
                f"{'n/a' if w is None else round(w, 4):>11}{'--':>9}  "
                f"INCONCLUSIVE (n_correct={n_right}, n_wrong={n_wrong}; "
                f"need >={MIN_CLASS_FOR_GATE_B} of each)"
            )
            continue
        passed = c > w
        gate_b &= passed
        print(
            f"{cell.model + ' x ' + cell.benchmark:<40}{c:>10.4f}{w:>11.4f}{c - w:>9.4f}"
            f"  {'PASS' if passed else 'FAIL -- possible span bug'}"
        )

    print("\n[c] SELF-REPORTED CONFIDENCE PARSE FAILURES")
    print(f"{'cell':<40}{'n':>4}{'parsed':>8}{'failed':>8}  failing task_ids")
    for cell in ok:
        fails = cell.self_report_failures
        print(
            f"{cell.model + ' x ' + cell.benchmark:<40}{len(cell.predictions):>4}"
            f"{len(cell.predictions) - len(fails):>8}{len(fails):>8}  {fails if fails else ''}"
        )

    print("\n[d] DETERMINISM AT TEMPERATURE 0  (cache validity depends on this)")
    gate_d = have_data
    for cell in ok:
        if cell.determinism_ok is None:
            continue
        gate_d &= cell.determinism_ok
        print(
            f"{cell.model + ' x ' + cell.benchmark:<40}"
            f"{'PASS' if cell.determinism_ok else 'FAIL'}  {cell.determinism_detail}"
        )

    print("\n[e] THROUGHPUT AND FULL-PHASE-A PROJECTION")
    print(f"{'cell':<40}{'tasks':>6}{'gen s':>8}{'conf s':>8}{'tasks/s':>9}")
    per_model_rate: dict[str, list[float]] = {}
    for cell in ok:
        print(
            f"{cell.model + ' x ' + cell.benchmark:<40}{len(cell.predictions):>6}"
            f"{cell.generate_seconds:>8.1f}{cell.confidence_seconds:>8.1f}"
            f"{cell.tasks_per_second:>9.2f}"
        )
        per_model_rate.setdefault(cell.model, []).append(cell.tasks_per_second)

    n_full = sum(int(b.get("n_tasks", 100)) for b in bench_cfgs)
    total_seconds = 0.0
    print(f"\n{'model':<24}{'load s':>9}{'mean tasks/s':>14}{'projected s':>13}{'projected':>12}")
    for model, rates in per_model_rate.items():
        rate = statistics.fmean(rates)
        secs = (n_full / rate if rate > 0 else 0.0) + load_times.get(model, 0.0)
        total_seconds += secs
        print(
            f"{model:<24}{load_times.get(model, 0.0):>9.1f}{rate:>14.2f}"
            f"{secs:>13.0f}{_hms(secs):>12}"
        )
    print(
        f"\n  FULL PHASE A PROJECTION: {_hms(total_seconds)} "
        f"({n_full} tasks x {len(per_model_rate)} models, both passes)"
    )
    print("  Conservative: measured at batch=5, and larger batches amortise better.")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\n{bar}\nMODEL FAILURES\n{bar}")
        for cell in errors:
            print(f"\n{cell.model}:\n{cell.error}")

    print(f"\n{bar}")
    if not have_data:
        print("NO CELL PRODUCED DATA -- every gate below is a failure, not a pass.")
    print(f"GATE a (extraction) : {'PASS' if gate_a else 'FAIL'}")
    print(f"GATE b (confidence) : {'PASS' if gate_b else 'FAIL'}")
    print("GATE c (self-report): see table above")
    print(f"GATE d (determinism): {'PASS' if gate_d else 'FAIL'}")
    print("GATE e (throughput) : see projection above")
    print(bar)
    gates_ok = bool(gate_a and gate_b and gate_d and not errors)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "model": c.model,
                        "benchmark": c.benchmark,
                        "n": len(c.predictions),
                        "accuracy": c.accuracy,
                        "extraction_rate": c.extraction_rate,
                        "extraction_failures": c.extraction_failures,
                        "truncated": c.truncated,
                        "confidence_correct_incorrect": c.confidence_by_correctness(),
                        "self_report_failures": c.self_report_failures,
                        "determinism_ok": c.determinism_ok,
                        "generate_seconds": c.generate_seconds,
                        "confidence_seconds": c.confidence_seconds,
                        "tasks_per_second": c.tasks_per_second,
                        "error": c.error,
                    }
                    for c in results
                ],
                "load_times": load_times,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nMachine-readable report: {report_path}")
    if not gates_ok:
        print("GATES FAILED -- not eligible for a full run.")
    return gates_ok


def print_full_report(
    results: list[CellResult], load_times: dict[str, float], report_path: Path
) -> None:
    """Summary table for a full Phase A run, plus gates f and b-revisited."""
    ok = [r for r in results if not r.error]
    bar = "=" * 118

    print(f"\n{bar}\nPHASE A SUMMARY\n{bar}")
    # `head` is swarm headroom, 1 - accuracy: the fraction of tasks on which
    # there is anything for a swarm to disagree about. It sits next to accuracy
    # because it, not accuracy, decides which cells can carry a headline figure.
    # A cell at acc 0.98 has 0.02 headroom, so ten agents aggregating it are
    # almost always aggregating unanimous correctness and every method scores the
    # same -- the comparison is vacuous however good the accuracy looks.
    header = (
        f"{'cell':<34}{'n':>4}{'acc':>7}{'head':>7}{'extract':>9}{'conf|ok':>9}"
        f"{'conf|wrong':>11}{'AUC':>7}{'srep':>6}{'wall':>9}  flag"
    )
    print(header)
    print("-" * len(header))
    for cell in ok:
        c, w = cell.confidence_by_correctness()
        auc = cell.confidence_auc()
        n_sr = len(cell.predictions) - len(cell.self_report_failures)
        print(
            f"{cell.model + ' x ' + cell.benchmark:<34}{len(cell.predictions):>4}"
            f"{cell.accuracy:>7.2f}{1.0 - cell.accuracy:>7.2f}"
            f"{cell.extraction_rate:>9.0%}"
            f"{'--' if c is None else f'{c:.4f}':>9}"
            f"{'--' if w is None else f'{w:.4f}':>11}"
            f"{'--' if auc is None else f'{auc:.3f}':>7}"
            f"{n_sr:>6}{_hms(cell.wall_seconds):>9}"
            f"  {cell.headroom_flag}"
        )

    print("\n[f] ACCURACY HEADROOM  (report only -- no prompt or model tuning)")
    ceiling = [c for c in ok if c.headroom_flag == "CEILING"]
    floor = [c for c in ok if c.headroom_flag == "FLOOR"]
    if not ceiling and not floor:
        print("  no cell above 0.95 or below 0.20; every cell leaves errors for a swarm to correct")
    for cell in ceiling:
        print(
            f"  CEILING  {cell.model} x {cell.benchmark}: acc={cell.accuracy:.2f} > 0.95 -- "
            f"only {sum(1 for p in cell.predictions if not p.is_correct)} wrong answers, so "
            "aggregation has almost nothing to correct here"
        )
    for cell in floor:
        print(
            f"  FLOOR    {cell.model} x {cell.benchmark}: acc={cell.accuracy:.2f} < 0.20 -- "
            "too little signal for aggregation to pool"
        )

    print("\n[b] CONFIDENCE AUC, WITH POWER  (P(correct scores above incorrect); 0.5 = no signal)")
    print(f"{'cell':<34}{'n correct':>11}{'n wrong':>9}{'AUC':>8}")
    for cell in ok:
        auc = cell.confidence_auc()
        n_c = sum(1 for p in cell.predictions if p.is_correct)
        n_w = len(cell.predictions) - n_c
        print(
            f"{cell.model + ' x ' + cell.benchmark:<34}{n_c:>11}{n_w:>9}"
            f"{'undefined' if auc is None else f'{auc:.3f}':>8}"
        )

    total = sum(c.wall_seconds for c in ok) + sum(load_times.values())
    print(f"\nTotal wall clock: {_hms(total)}  ({len(ok)} cells)")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\n{bar}\nFAILURES\n{bar}")
        for cell in errors:
            print(f"\n{cell.model}:\n{cell.error}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "model": c.model,
                        "benchmark": c.benchmark,
                        "sample_index": c.sample_index,
                        "temperature": c.temperature,
                        "n": len(c.predictions),
                        "accuracy": c.accuracy,
                        # 1 - accuracy: the fraction of tasks on which there is
                        # anything for a swarm to disagree about, and therefore
                        # the number that decides which cells can carry a
                        # headline figure.
                        "swarm_headroom": None if c.accuracy is None else 1.0 - c.accuracy,
                        "extraction_rate": c.extraction_rate,
                        "extraction_failures": c.extraction_failures,
                        "truncated": c.truncated,
                        "confidence_correct_incorrect": c.confidence_by_correctness(),
                        "confidence_auc": c.confidence_auc(),
                        "headroom_flag": c.headroom_flag,
                        "self_report_failures": c.self_report_failures,
                        "wall_seconds": c.wall_seconds,
                        "error": c.error,
                    }
                    for c in results
                ],
                "load_times": load_times,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Machine-readable report: {report_path}")


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


if __name__ == "__main__":
    raise SystemExit(main())
