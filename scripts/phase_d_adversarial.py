#!/usr/bin/env python3
"""Phase D: generate real-LLM adversarial broadcasts.

Only the four inference attacks run here. Honest Phase A caches are read-only;
adversarial outputs go to ``data/cache_adversarial/{attack}/{benchmark}/{model}``.

    python scripts/phase_d_adversarial.py --project            # budget only, no GPU
    python scripts/phase_d_adversarial.py --pilot              # 5 tasks, verbatim
    python scripts/phase_d_adversarial.py                      # full generation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.attacks import INFERENCE_ATTACKS, get_attack  # noqa: E402
from aip.harness import cache as cache_mod  # noqa: E402
from aip.harness.logging import configure_logging  # noqa: E402
from aip.harness.manifest import Manifest  # noqa: E402
from aip.models.confidence import span_confidence  # noqa: E402
from aip.models.inference import ModelRunner  # noqa: E402
from aip.models.registry import load_registry  # noqa: E402
from aip.tasks import (  # noqa: E402
    get_benchmark,
    parse_self_reported_confidence,
    roster,  # noqa: E402
)
from aip.tasks.task_lists import load_fixed_tasks  # noqa: E402
from aip.types import Prediction  # noqa: E402

log = configure_logging(level="INFO")

BENCHMARKS = roster.benchmarks()  # from configs/task_lists.json
#: Phase 4's adversarial generators, taken from the FROZEN roster rather than
#: named literally. The previous literal still said `ministral_8b`, a model
#: retired and disabled when the roster moved to Ministral 3 14B -- so Phase 4
#: was generating attacks with a checkpoint that is not part of this hardware
#: regime's population at all, which RULE 1 forbids outright.
#:
#: The weak/fast tier leads because compliance, not capability, is what makes a
#: good adversarial generator: llama32_3b followed the always_wrong injection
#: 100% of the time on the L4 regime where Ministral complied only 29%.
ADVERSARY_MODELS = [
    n
    for n in load_registry("configs/models.yaml").enabled_names
    if load_registry("configs/models.yaml").get(n).tier in ("weak_fast", "mid")
]
HONEST_CACHE = Path("data/cache")
ADV_CACHE = Path("data/cache_adversarial")
BUDGET_SECONDS = 3 * 3600


def measured_rates() -> dict[tuple[str, str], float]:
    """Seconds per task per (model, benchmark), from the Phase A full run."""
    import json

    report = json.load(open("results/phase_a_full_s0_report.json"))
    return {
        (c["model"], c["benchmark"]): c["wall_seconds"] / max(c["n"], 1)
        for c in report["cells"]
        if not c.get("error")
    }


def load_times() -> dict[str, float]:
    import json

    return json.load(open("results/phase_a_full_s0_report.json"))["load_times"]


def tasks_per_benchmark(benchmarks: list[str], override: int | None) -> dict[str, int]:
    """How many tasks each benchmark contributes.

    Not one number: GSM8K carries 500 and the rest 200. The previous single
    scalar defaulted to 100 and silently truncated every adversarial cache below
    the honest one, which made phase_d_sweep discard every attack.
    """
    from aip.tasks.task_lists import read_task_lists

    frozen = read_task_lists()["benchmarks"]
    return {b: override if override is not None else int(frozen[b]["n"]) for b in benchmarks}


def project(
    models: list[str], benchmarks: list[str], n_tasks: dict[str, int]
) -> tuple[float, pd.DataFrame]:
    rates = measured_rates()
    loads = load_times()
    rows = []
    for model in models:
        for attack_name in INFERENCE_ATTACKS:
            attack = get_attack(attack_name)
            for benchmark in benchmarks:
                per_task = rates.get((model, benchmark), 1.0)
                # Adversarial generation is one pass, not two; Phase A's measured
                # rate includes the self-report pass, which we also run here.
                seconds = per_task * n_tasks[benchmark] * attack.token_multiplier
                rows.append(
                    {
                        "model": model,
                        "attack": attack_name,
                        "benchmark": benchmark,
                        "n_tasks": n_tasks[benchmark],
                        "seconds": seconds,
                        "minutes": seconds / 60,
                    }
                )
    frame = pd.DataFrame(rows)
    total = frame.seconds.sum() + sum(loads.get(m, 60.0) for m in models)
    return total, frame


def print_projection(
    models: list[str], benchmarks: list[str], n_tasks: dict[str, int]
) -> float:
    total, frame = project(models, benchmarks, n_tasks)
    print(f"\n{'=' * 78}\nPHASE D INFERENCE BUDGET PROJECTION\n{'=' * 78}")
    print(f"cap: {BUDGET_SECONDS / 3600:.1f} GPU-hours   basis: measured Phase A per-task rates\n")
    print(f"{'model':<16}{'attack':<24}{'benchmark':<10}{'tasks':>6}{'minutes':>9}")
    for r in frame.itertuples():
        print(f"{r.model:<16}{r.attack:<24}{r.benchmark:<10}{r.n_tasks:>6}{r.minutes:>9.1f}")
    print(f"\n{'subtotal by model':<50}")
    for m, g in frame.groupby("model"):
        print(f"  {m:<24}{g.minutes.sum():>8.1f} min")
    print(f"\n  model loads{'':<13}{sum(load_times().get(m, 60.0) for m in models) / 60:>8.1f} min")
    print(f"\n  TOTAL{'':<19}{total / 60:>8.1f} min  ({total / 3600:.2f} GPU-hours)")
    print(
        f"  CAP  {'':<19}{BUDGET_SECONDS / 60:>8.1f} min  ({BUDGET_SECONDS / 3600:.2f} GPU-hours)"
    )
    print(f"  -> {'WITHIN BUDGET' if total <= BUDGET_SECONDS else 'OVER BUDGET, MUST CUT'}\n")
    return total


def honest_broadcasts_for(
    benchmark: str, task_ids: list[str]
) -> dict[str, list[tuple[str, str | None]]]:
    """Cached honest answers per task, for the rushing adversary to condition on."""
    out: dict[str, list[tuple[str, str | None]]] = {t: [] for t in task_ids}
    for path in sorted((HONEST_CACHE / benchmark).glob("*.parquet")):
        model = path.stem
        frame = pd.read_parquet(path).set_index("task_id")
        for task_id in task_ids:
            if task_id in frame.index:
                value = frame.loc[task_id, "extracted_answer"]
                out[task_id].append((model, None if pd.isna(value) else str(value)))
    return out


def generate_cell(
    runner: ModelRunner,
    benchmark_name: str,
    attack_name: str,
    items,
    seed: int,
    max_tokens: int | None,
    confidence_tokens: int,
) -> tuple[list[Prediction], list[Any]]:
    benchmark = get_benchmark(benchmark_name)
    attack = get_attack(attack_name)
    context_by_task: dict[str, dict] = {}
    if attack_name == "rushing":
        peers = honest_broadcasts_for(benchmark_name, [i.task_id for i in items])
        context_by_task = {i.task_id: {"honest_broadcasts": peers[i.task_id]} for i in items}

    prompts = [
        runner.encode_messages(
            attack.build_messages(benchmark, item, context_by_task.get(item.task_id, {}))
        )
        for item in items
    ]
    answers = runner.generate(prompts, max_tokens=max_tokens)
    extractions = [benchmark.extract_answer(g.text) for g in answers]

    conf_prompts = [
        runner.encode_messages(benchmark.build_confidence_messages(item, ex.value))
        for item, ex in zip(items, extractions, strict=True)
    ]
    conf_out = runner.generate(conf_prompts, max_tokens=confidence_tokens)
    self_reports = [parse_self_reported_confidence(g.text) for g in conf_out]

    predictions = []
    for item, gen, ex, sr in zip(items, answers, extractions, self_reports, strict=True):
        confidence, n_answer_tokens = span_confidence(gen.tokens, ex)
        predictions.append(
            Prediction(
                task_id=item.task_id,
                benchmark=benchmark_name,
                model=runner.entry.name,
                raw_completion=gen.text,
                extracted_answer=ex.value,
                gold_answer=item.gold_answer,
                # For an adversary, "is_correct" records whether the attack FAILED
                # to be wrong. It is diagnostic, never a training signal.
                is_correct=benchmark.score(ex.value, item.gold_answer),
                logprob_confidence=confidence,
                self_reported_confidence=sr,
                n_answer_tokens=n_answer_tokens,
                n_completion_tokens=gen.n_tokens,
                seed=seed,
                temperature=float(runner.entry.generation.temperature),
                extra={"attack": attack_name, "finish_reason": gen.finish_reason},
            )
        )
    return predictions, answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="store_true", help="print the budget and exit")
    parser.add_argument("--pilot", action="store_true", help="5-task pilot, verbatim output")
    parser.add_argument("--models", nargs="*", default=ADVERSARY_MODELS)
    parser.add_argument("--attacks", nargs="*", default=INFERENCE_ATTACKS)
    parser.add_argument("--benchmarks", nargs="*", default=BENCHMARKS)
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help=(
            "tasks per benchmark; default None means the WHOLE frozen list. "
            "It used to default to 100, which silently truncated every "
            "adversarial cache: the honest cache carries 500 GSM8K tasks and 200 "
            "elsewhere, and phase_d_sweep requires the honest task ids to be a "
            "subset of the adversarial ones. A 100-task cache therefore failed "
            "that check for every attack and every benchmark, and the sweep "
            "silently produced a parquet containing only the noise control."
        ),
    )
    parser.add_argument("--seed", type=int, default=20250825)
    parser.add_argument("--out", type=Path, default=ADV_CACHE)
    args = parser.parse_args()

    n_tasks = 5 if args.pilot else args.n_tasks
    models = args.models
    benchmarks = args.benchmarks
    if args.pilot:
        models, benchmarks = ["llama32_3b"], ["gsm8k"]

    n_tasks = tasks_per_benchmark(benchmarks, n_tasks)
    total = print_projection(models, benchmarks, n_tasks)
    if args.project:
        return 0
    if total > BUDGET_SECONDS:
        print(
            "Refusing to start: projection exceeds the cap. Cut adversary models "
            "(phi4_mini_reasoning first) or MATH-500 cells."
        )
        return 1

    registry = load_registry("configs/models.yaml")
    root = args.out if not args.pilot else Path("data/cache_adversarial_pilot")
    manifest = Manifest.create(
        phase="phase_d_pilot" if args.pilot else "phase_d",
        config={
            "models": models,
            "attacks": args.attacks,
            "benchmarks": benchmarks,
            "n_tasks": n_tasks,
            "projected_seconds": total,
        },
        seeds={"generation": args.seed},
        notes={"honest_caches_read_only": True, "cache_root": str(root)},
    )

    started = time.perf_counter()
    for model_name in models:
        entry = registry.get(model_name)
        with ModelRunner(entry) as runner:
            for attack_name in args.attacks:
                for benchmark_name in benchmarks:
                    items = load_fixed_tasks(
                        benchmark_name, limit=n_tasks[benchmark_name]
                    )
                    key_root = root / attack_name
                    done = cache_mod.completed_task_ids(
                        benchmark_name, model_name, root=key_root, require_confidence=False
                    )
                    todo = [i for i in items if i.task_id not in done]
                    if not todo:
                        log.info(
                            "cell.skipped",
                            attack=attack_name,
                            benchmark=benchmark_name,
                            model=model_name,
                        )
                        continue
                    log.info(
                        "cell.start",
                        attack=attack_name,
                        benchmark=benchmark_name,
                        model=model_name,
                        n=len(todo),
                    )
                    preds, _ = generate_cell(
                        runner,
                        benchmark_name,
                        attack_name,
                        todo,
                        args.seed,
                        max_tokens=None,
                        confidence_tokens=entry.generation.confidence_max_tokens,
                    )
                    cache_mod.write_predictions(
                        preds,
                        benchmark_name,
                        model_name,
                        root=key_root,
                        manifest={
                            "attack": attack_name,
                            "hf_id": entry.hf_id,
                            "prompt_template_hash": get_benchmark(
                                benchmark_name
                            ).prompt_template_hash(),
                            "run_id": manifest.run_id,
                            "pilot": args.pilot,
                        },
                    )
    elapsed = time.perf_counter() - started
    manifest.finish(status="ok", elapsed_seconds=elapsed).write(
        Path("results/manifests") / f"{manifest.run_id}.json"
    )
    print(
        f"\nGeneration complete in {elapsed / 60:.1f} min "
        f"(projected {total / 60:.1f} min). Cache: {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
