#!/usr/bin/env python3
"""Bounded, CPU-only integrity and transfer audit of the honest greedy caches.

Writes aggregated diagnostics only. It never edits thresholds or loads models.
Bootstrap units are task IDs, preserving dependence among pairs and receivers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import resource
import shutil
import signal
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

START = time.monotonic()
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "task_id", "benchmark", "model", "extracted_answer", "gold_answer",
    "is_correct", "temperature", "sample_index", "nonreproducing",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_hash(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def missing_answer(values: pd.Series) -> np.ndarray:
    return (values.isna() | values.fillna("").astype(str).str.strip().eq("")).to_numpy(bool)


def normalized_answers(values: pd.Series) -> np.ndarray:
    result = values.fillna("").astype(str).to_numpy(dtype=object)
    result[missing_answer(values)] = None
    return result


def phi(x: np.ndarray, y: np.ndarray) -> float:
    """Phi is undefined if either binary error indicator is constant."""
    x, y = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool)
    n11 = float(np.count_nonzero(x & y))
    n10 = float(np.count_nonzero(x & ~y))
    n01 = float(np.count_nonzero(~x & y))
    n00 = float(np.count_nonzero(~x & ~y))
    denominator = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return (n11 * n00 - n10 * n01) / math.sqrt(denominator) if denominator else float("nan")


def coincidence_counts(answer: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    """Task-wise sufficient counts; answer and correctness have shape model × task."""
    n_models, n_tasks = answer.shape
    counts = {k: np.zeros(n_tasks, dtype=np.int64) for k in (
        "receiver_num", "receiver_den", "gold_num", "gold_den", "gold_present_den",
    )}
    present = answer != None  # noqa: E711
    for a, b in itertools.combinations(range(n_models), 2):
        both_present = present[a] & present[b]
        same = both_present & (answer[a] == answer[b])
        both_wrong = (~correct[a]) & (~correct[b])
        counts["gold_num"] += same & both_wrong
        counts["gold_den"] += both_wrong
        counts["gold_present_den"] += both_wrong & both_present
        for receiver in range(n_models):
            if receiver in (a, b):
                continue
            eligible = (
                both_present & present[receiver]
                & (answer[a] != answer[receiver]) & (answer[b] != answer[receiver])
            )
            counts["receiver_den"] += eligible
            counts["receiver_num"] += eligible & same
    return counts


def ratio(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def interval(draws: np.ndarray) -> tuple[float, float, int]:
    good = draws[np.isfinite(draws)]
    if not len(good):
        return float("nan"), float("nan"), 0
    low, high = np.quantile(good, [0.025, 0.975])
    return float(low), float(high), int(len(good))


def bootstrap_coincidence(counts, n_draws: int, seed: int) -> dict:
    n_tasks = len(counts["gold_num"])
    rng = np.random.default_rng(seed)
    pairs = {
        "receiver": ("receiver_num", "receiver_den"),
        "gold": ("gold_num", "gold_den"),
        "gold_present": ("gold_num", "gold_present_den"),
    }
    draws = {key: np.full(n_draws, np.nan) for key in pairs}
    for draw in range(n_draws):
        idx = rng.integers(0, n_tasks, n_tasks)
        for key, (num, den) in pairs.items():
            draws[key][draw] = ratio(counts[num][idx].sum(), counts[den][idx].sum())
    result = {}
    for key, (num, den) in pairs.items():
        low, high, valid = interval(draws[key])
        result.update({
            f"q_{key}": ratio(counts[num].sum(), counts[den].sum()),
            f"q_{key}_ci_low": low, f"q_{key}_ci_high": high,
            f"q_{key}_bootstrap_valid": valid,
            f"{key}_numerator": int(counts[num].sum()),
            f"{key}_denominator": int(counts[den].sum()),
            f"{key}_tasks_with_denominator": int(np.count_nonzero(counts[den])),
        })
    low, high, valid = interval(draws["receiver"] - draws["gold"])
    result.update({
        "q_receiver_minus_gold": result["q_receiver"] - result["q_gold"],
        "q_receiver_minus_gold_ci_low": low,
        "q_receiver_minus_gold_ci_high": high,
        "q_receiver_minus_gold_bootstrap_valid": valid,
    })
    return result


def check_estimators() -> None:
    """Small analytical edge cases, independent of the benchmark outputs."""
    x = np.array([False, False, True, True])
    assert phi(x, x) == 1.0 and phi(x, ~x) == -1.0
    assert math.isnan(phi(x, np.zeros(4, dtype=bool)))
    answer = np.array([["A", "A"], ["A", "B"], ["B", "C"]], dtype=object)
    correct = np.array([[True, False], [True, False], [False, False]])
    counts = coincidence_counts(answer, correct)
    assert counts["receiver_num"].tolist() == [1, 0]
    assert counts["receiver_den"].tolist() == [1, 3]
    assert counts["gold_num"].tolist() == [0, 0]
    assert counts["gold_den"].tolist() == [0, 3]
    assert bootstrap_coincidence(counts, 20, 1)["q_receiver"] == 0.25
    answer[2, 1] = None
    assert coincidence_counts(answer, correct)["receiver_den"].tolist() == [1, 0]


def file_audit(path: Path, expected_ids: set[str]) -> tuple[dict, pd.DataFrame]:
    table = pq.read_table(path, columns=COLUMNS, use_threads=False)
    frame = table.to_pandas(use_threads=False)
    meta_path = path.with_suffix(".manifest.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    expected_digest = next((meta[k] for k in ("parquet_sha256", "file_sha256", "sha256") if k in meta), None)
    if expected_digest is None and isinstance(meta.get("checksums"), dict):
        expected_digest = meta["checksums"].get("sha256")
    actual_digest = sha256(path)
    task_ids = set(frame["task_id"].dropna().astype(str))
    correct = frame["is_correct"].dropna()
    row = {
        "benchmark": path.parent.name, "model": path.stem,
        "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
        "rows": len(frame), "unique_tasks": len(task_ids),
        "duplicate_task_rows": int(frame["task_id"].duplicated().sum()),
        "duplicate_key_rows": int(frame.duplicated(["task_id", "sample_index"]).sum()),
        "missing_task_ids": int(frame["task_id"].isna().sum()),
        "missing_answers": int(missing_answer(frame["extracted_answer"]).sum()),
        "missing_gold": int(missing_answer(frame["gold_answer"]).sum()),
        "missing_correctness": int(frame["is_correct"].isna().sum()),
        "correct": int(correct.sum()), "n_scored": len(correct),
        "accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "nonreproducing_rows": int(frame["nonreproducing"].fillna(False).sum()),
        "nongreedy_rows": int((frame["temperature"].fillna(-1) != 0).sum()),
        "nonzero_sample_rows": int((frame["sample_index"].fillna(-1) != 0).sum()),
        "benchmark_name_mismatches": int((frame["benchmark"] != path.parent.name).sum()),
        "model_name_mismatches": int((frame["model"] != path.stem).sum()),
        "task_list_missing": len(expected_ids - task_ids),
        "task_list_extra": len(task_ids - expected_ids),
        "task_ids_sha256": ids_hash(task_ids),
        "parquet_sha256": actual_digest,
        "manifest_present": bool(meta),
        "manifest_sha256": sha256(meta_path) if meta_path.exists() else None,
        "manifest_declared_parquet_sha256": expected_digest,
        "manifest_digest_status": (
            "not_recorded" if expected_digest is None
            else "match" if str(expected_digest).lower() == actual_digest else "mismatch"
        ),
        "manifest_rows_match": meta.get("n_rows") == len(frame) if "n_rows" in meta else None,
        "manifest_tasks_match": meta.get("n_tasks") == len(task_ids) if "n_tasks" in meta else None,
        "manifest_task_list_hash": meta.get("task_list_hash"),
    }
    return row, frame


def task_list_ids(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    return {b: set(v["task_ids"]) for b, v in json.loads(path.read_text())["benchmarks"].items()}


def save_figures(out: Path, files: pd.DataFrame, pairs: pd.DataFrame, coincidence: pd.DataFrame):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "savefig.dpi": 160})
    matrix = files.pivot(index="model", columns="benchmark", values="accuracy")
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    picture = ax.imshow(matrix.to_numpy() * 100, vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j] * 100
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value > 65 else "black")
    ax.set_title("Honest greedy cache accuracy (%) · fixed task lists")
    fig.colorbar(picture, ax=ax, label="Accuracy (%)", shrink=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"accuracy_heatmap.{ext}")
    plt.close(fig)

    benchmarks = sorted(pairs["benchmark"].unique())
    models = sorted(files["model"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad("#d9d9d9")
    for ax, benchmark in zip(axes.flat, benchmarks, strict=False):
        mat = np.full((len(models), len(models)), np.nan)
        for row in pairs.loc[pairs.benchmark == benchmark].itertuples():
            i, j = models.index(row.model_i), models.index(row.model_j)
            mat[i, j] = mat[j, i] = row.phi
        picture = ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap)
        ax.set_title(benchmark)
        ax.set_xticks(range(len(models)), models, rotation=75, ha="right", fontsize=7)
        ax.set_yticks(range(len(models)), models, fontsize=7)
    fig.suptitle("Aligned-task error phi · diagonal is identical-cache replay; gray = undefined")
    fig.colorbar(picture, ax=axes, label="Error phi", shrink=0.65)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"error_phi_heatmaps.{ext}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(coincidence))
    for name, offset, color in (("receiver", -0.12, "#1f77b4"), ("gold", 0.12, "#d95f02")):
        y = coincidence[f"q_{name}"].to_numpy()
        lower = coincidence[f"q_{name}_ci_low"].to_numpy()
        upper = coincidence[f"q_{name}_ci_high"].to_numpy()
        ax.vlines(x + offset, lower, upper, color=color, linewidth=2)
        ax.scatter(x + offset, y, color=color, label=f"{name}-conditioned q", zorder=3)
    ax.scatter(x, coincidence["legacy_receiver_ceiling"], marker="x", color="black", s=65, label="Legacy receiver ceiling (transfer reference)")
    ax.set_xticks(x, coincidence["benchmark"])
    ax.set_ylim(-0.04, 1.05)
    ax.set_ylabel("Pooled coincidence probability")
    ax.set_title("Honest caches: conditioning matters · task bootstrap 95% intervals")
    ax.legend(loc="lower right", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"coincidence_transfer.{ext}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache")
    parser.add_argument("--out", type=Path, default=ROOT / "results/dgx/cache_audit")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 20 <= args.resamples <= 200:
        parser.error("resamples must be between 20 and 200")
    if not 1 <= args.timeout_seconds <= 600:
        parser.error("timeout must be between 1 and 600 seconds")
    signal.alarm(args.timeout_seconds)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min([4 * 1024**3] + [v for v in (old_soft, old_hard) if v >= 0])
    resource.setrlimit(resource.RLIMIT_AS, (cap, old_hard))
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
    if available < 16 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("Insufficient free host resources for the conservative audit envelope")
    args.out.mkdir(parents=True, exist_ok=True)
    check_estimators()
    thresholds_path = ROOT / "configs/inversion_thresholds.yaml"
    thresholds = yaml.safe_load(thresholds_path.read_text())
    expected = task_list_ids(ROOT / "configs/task_lists.json")
    legacy_lists = {
        "legacy_l4": task_list_ids(ROOT / "configs/task_lists_v1_l4.json"),
        "legacy_gsm8k_extension": task_list_ids(ROOT / "configs/task_lists_gsm8k_ext.json"),
    }
    paths = sorted(args.cache.glob("*/*.parquet"))
    if not paths:
        raise RuntimeError("No honest greedy cache parquets found")
    files, frames, pairs, alignments, coincidence, overlaps = [], {}, [], [], [], []
    for path in paths:
        row, frame = file_audit(path, expected.get(path.parent.name, set()))
        files.append(row)
        frames.setdefault(path.parent.name, {})[path.stem] = frame
    for benchmark, by_model in sorted(frames.items()):
        models = sorted(by_model)
        if any(frame.task_id.isna().any() or frame.task_id.duplicated().any() for frame in by_model.values()):
            raise RuntimeError(f"{benchmark}: ambiguous task IDs; refusing a silent deduplication")
        indexed = {m: f.set_index("task_id") for m, f in by_model.items()}
        task_sets = [set(f.index) for f in indexed.values()]
        shared, union = set.intersection(*task_sets), set.union(*task_sets)
        gold = pd.concat([f["gold_answer"].rename(m) for m, f in indexed.items()], axis=1)
        inconsistent_gold = set(gold.index[gold.nunique(axis=1, dropna=True) > 1])
        invalid = set()
        for frame in indexed.values():
            invalid.update(frame.index[frame["is_correct"].isna().to_numpy() | missing_answer(frame["gold_answer"])])
        analysis_ids = sorted(shared - inconsistent_gold - invalid)
        if not analysis_ids:
            raise RuntimeError(f"{benchmark}: no gold-consistent shared tasks")
        alignment = {
            "benchmark": benchmark, "models": len(models), "union_tasks": len(union),
            "shared_tasks": len(shared), "analysis_tasks": len(analysis_ids),
            "inconsistent_gold_task_ids_count": len(inconsistent_gold),
            "excluded_missing_gold_or_correctness_task_ids_count": len(shared & invalid),
            "all_models_same_task_ids": all(s == shared for s in task_sets),
            "shared_task_ids_sha256": ids_hash(shared),
            "analysis_task_ids_sha256": ids_hash(analysis_ids),
        }
        alignments.append(alignment)
        for label, lists in legacy_lists.items():
            historical_ids = lists.get(benchmark)
            overlaps.append({
                "benchmark": benchmark, "legacy_list": label,
                "listed_legacy_tasks": len(historical_ids) if historical_ids is not None else None,
                "current_shared_tasks": len(shared),
                "overlap_tasks": len(shared & historical_ids) if historical_ids is not None else None,
                "fraction_legacy_tasks_overlapping": ratio(len(shared & historical_ids), len(historical_ids)) if historical_ids is not None else None,
                "fraction_current_tasks_overlapping": ratio(len(shared & historical_ids), len(shared)) if historical_ids is not None else None,
                "evidence_status": "observed_task_id_overlap" if historical_ids is not None else "no_benchmark_in_legacy_list",
            })
        answer = np.array([normalized_answers(indexed[m].loc[analysis_ids, "extracted_answer"]) for m in models], dtype=object)
        correct = np.array([indexed[m].loc[analysis_ids, "is_correct"].to_numpy(bool) for m in models])
        for i, j in itertools.combinations_with_replacement(range(len(models)), 2):
            error_i, error_j = ~correct[i], ~correct[j]
            value = phi(error_i, error_j)
            pairs.append({
                "benchmark": benchmark, "model_i": models[i], "model_j": models[j],
                "kind": "same_cache_replay" if i == j else "cross_model",
                "n_tasks": len(analysis_ids), "phi": value,
                "phi_status": "estimated" if math.isfinite(value) else "undefined_constant_errors",
                "n_errors_i": int(error_i.sum()), "n_errors_j": int(error_j.sum()),
                "n_both_wrong": int((error_i & error_j).sum()),
                "co_error_rate": float((error_i & error_j).mean()),
                "answer_agreement_rate": float(((answer[i] == answer[j]) & (answer[i] != None) & (answer[j] != None)).mean()),  # noqa: E711
            })
        counts = coincidence_counts(answer, correct)
        benchmark_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{benchmark}".encode()).digest()[:8], "big")
        measure = bootstrap_coincidence(counts, args.resamples, benchmark_seed)
        answer_class = thresholds["benchmark_classes"][benchmark]
        legacy = thresholds["classes"][answer_class]
        overlap_count = len(shared & legacy_lists["legacy_l4"].get(benchmark, set()))
        measure.update({
            "benchmark": benchmark, "n_tasks": len(analysis_ids), "n_models": len(models),
            "answer_class": answer_class, "bootstrap_resamples": args.resamples,
            "bootstrap_seed": benchmark_seed, "bootstrap_unit": "task_id",
            "legacy_receiver_ceiling": legacy["ceiling"],
            "legacy_receiver_q_recorded": legacy.get("receiver_conditioned_q"),
            "legacy_gold_q_recorded": legacy.get("gold_conditioned_q_phase_b2"),
            "legacy_threshold_version": thresholds["version"],
            "legacy_calibration_model_count": thresholds.get("n_models"),
            "legacy_model_ids_in_current_roster": len(set(models) & set(thresholds.get("models", []))),
            "current_receiver_point_exceeds_legacy_ceiling": bool(measure["q_receiver"] > legacy["ceiling"]),
            "current_receiver_ci_low_exceeds_legacy_ceiling": bool(measure["q_receiver_ci_low"] > legacy["ceiling"]),
            "legacy_l4_task_id_overlap": overlap_count,
            "independent_threshold_validation": False,
            "interpretation": "transfer_diagnostic_only; no threshold modification",
        })
        coincidence.append(measure)
        print(json.dumps({"benchmark": benchmark, "tasks": len(analysis_ids), "q_receiver": round(measure["q_receiver"], 4), "q_gold": round(measure["q_gold"], 4), "legacy_overlap": overlap_count}), flush=True)

    file_df, pair_df, coincidence_df = pd.DataFrame(files), pd.DataFrame(pairs), pd.DataFrame(coincidence)
    for filename, frame in (
        ("cache_cells.csv", file_df), ("task_alignment.csv", pd.DataFrame(alignments)),
        ("pairwise_error_phi.csv", pair_df), ("coincidence_transfer.csv", coincidence_df),
        ("calibration_task_overlap.csv", pd.DataFrame(overlaps)),
    ):
        frame.to_csv(args.out / filename, index=False)
    save_figures(args.out, file_df, pair_df, coincidence_df)
    cross = pair_df.loc[pair_df.kind == "cross_model"]
    summary = {
        "cache_files": len(files), "models": int(file_df.model.nunique()),
        "benchmarks": int(file_df.benchmark.nunique()), "prediction_rows": int(file_df.rows.sum()),
        "missing_answers": int(file_df.missing_answers.sum()),
        "duplicate_task_rows": int(file_df.duplicate_task_rows.sum()),
        "gold_inconsistent_tasks": sum(x["inconsistent_gold_task_ids_count"] for x in alignments),
        "nonreproducing_rows": int(file_df.nonreproducing_rows.sum()),
        "parquet_manifest_digest_status_counts": dict(Counter(file_df.manifest_digest_status)),
        "manifest_row_count_mismatches": int((file_df.manifest_rows_match == False).sum()),  # noqa: E712
        "manifest_task_count_mismatches": int((file_df.manifest_tasks_match == False).sum()),  # noqa: E712
        "cross_model_pairs": len(cross), "undefined_cross_model_phi": int(cross.phi.isna().sum()),
        "accuracy_min": float(file_df.accuracy.min()), "accuracy_max": float(file_df.accuracy.max()),
        "receiver_ci_low_exceeds_legacy_ceiling": [r["benchmark"] for r in coincidence if r["current_receiver_ci_low_exceeds_legacy_ceiling"]],
        "receiver_point_only_exceeds_legacy_ceiling": [r["benchmark"] for r in coincidence if r["current_receiver_point_exceeds_legacy_ceiling"] and not r["current_receiver_ci_low_exceeds_legacy_ceiling"]],
        "self_checks": "passed",
    }
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "existing honest greedy caches; no new model inference",
        "script_sha256": sha256(Path(__file__)),
        "configuration_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [thresholds_path] + sorted((ROOT / "configs").glob("task_lists*.json"))},
        "source_parquet_hashes": {row["path"]: row["parquet_sha256"] for row in files},
        "seed": args.seed, "resamples": args.resamples,
        "bootstrap": "within-benchmark task resampling; all model/receiver observations for each task stay together; percentile 95% intervals",
        "scope": "descriptive cache audit and legacy-threshold transfer diagnostic; not an independent calibration study",
        "missingness": "accuracy uses stored non-null correctness; gold q includes all jointly wrong events with missing answers never agreeing (repository convention); receiver q skips triples with any missing answer; gold_present is a complete-answer sensitivity",
        "alignment": "all-model task-ID intersection, excluding inconsistent gold and missing gold/correctness; duplicates are fatal",
        "honest_only": True, "thresholds_modified": False,
        "same_model_caveat": "reusing one deterministic cached answer for multiple agents forces identical errors; replay phi=1 when variable and undefined when constant; this does not estimate correlation between fresh stochastic generations",
        "task_overlap_caveat": "legacy preserved task lists overlap current IDs on the same benchmarks; exact threshold calibration membership is inferred from YAML provenance and the preserved lists, not an archived per-estimate membership ledger; disjointness and independent validation cannot be claimed",
    }
    write_json(args.out / "audit_results.json", {"summary": summary, "provenance": provenance, "cache_cells": files, "task_alignment": alignments, "coincidence_transfer": coincidence, "calibration_task_overlap": overlaps})
    table = "\n".join(
        f"| {r['benchmark']} | {r['n_tasks']} | {r['q_receiver']:.3f} [{r['q_receiver_ci_low']:.3f}, {r['q_receiver_ci_high']:.3f}] | {r['q_gold']:.3f} [{r['q_gold_ci_low']:.3f}, {r['q_gold_ci_high']:.3f}] | {r['legacy_receiver_ceiling']:.3f} | {r['legacy_l4_task_id_overlap']} |"
        for r in coincidence
    )
    report = f"""# Honest-cache audit on DGX

This run analyzed {summary['cache_files']} existing greedy caches, covering {summary['models']} models × {summary['benchmarks']} benchmarks and {summary['prediction_rows']:,} predictions. It performed no model inference and changed no calibration thresholds.

## Integrity and coverage

- Duplicate task rows: {summary['duplicate_task_rows']}; inconsistent gold labels across model caches: {summary['gold_inconsistent_tasks']} task IDs.
- Missing extracted answers: {summary['missing_answers']}; source-marked nonreproducing rows: {summary['nonreproducing_rows']}.
- Recorded cache accuracies range from {summary['accuracy_min']:.1%} to {summary['accuracy_max']:.1%}. Accuracy uses the existing scoring field; this audit does not rerun semantic scoring or independently validate benchmark labels.
- Manifest row-count mismatches: {summary['manifest_row_count_mismatches']}; task-count mismatches: {summary['manifest_task_count_mismatches']}.
- Parquet SHA-256 comparison status: {summary['parquet_manifest_digest_status_counts']}. A newly computed checksum fingerprints these inputs; where the source manifest lacks a digest, it does not prove historical byte identity.
- Every per-cell count, task-list difference, manifest check, and SHA-256 appears in `cache_cells.csv`. `task_alignment.csv` records the shared task counts and gold checks. No raw prompts, completions, gold answers, or task IDs are exported.

## Error dependence and repeated agents

`pairwise_error_phi.csv` contains {summary['cross_model_pairs']} aligned cross-model pairs; {summary['undefined_cross_model_phi']} have undefined phi because at least one error vector is constant. Undefined correlations remain null/blank and appear gray in the plots. All-model task-ID intersections define a common comparison set for each benchmark.

The diagonal explicitly represents replay of the same cached model. Copying one deterministic answer into several agent slots produces identical errors and phi = 1 whenever there is variation. It supplies no estimate of correlation between fresh independent generations. Treating such slots as independent observations would overstate the effective sample size. There are no new sampled model outputs in this audit.

## Honest coincidence and legacy threshold transfer

| Benchmark | Tasks | Receiver-conditioned q [95% CI] | Gold-conditioned q [95% CI] | Legacy receiver ceiling | Legacy L4 task overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

The lower interval bound exceeds the legacy class ceiling on: {', '.join(summary['receiver_ci_low_exceeds_legacy_ceiling']) or 'none'}. The point alone exceeds the ceiling, with an interval crossing the reference, on: {', '.join(summary['receiver_point_only_exceeds_legacy_ceiling']) or 'none'}. These are diagnostics of ceiling transfer to this honest population and task mixture, not observations of gate decisions.

Receiver-conditioned q is P(two peers agree | both dissent from the receiver), pooled over distinct honest receivers and peer pairs. Gold-conditioned q is P(two models agree | both are wrong), pooled over distinct model pairs. The two events have different denominators. Gold conditioning uses labels strictly for evaluation. Missing answers never coincide; receiver triples with missing answers are omitted. `q_gold_present` in the CSV provides the complete-answer sensitivity.

Intervals use {args.resamples} task bootstrap draws per benchmark with deterministic seeds. Each sampled task carries all pairs and receivers together, preserving their dependence; receiver-pair events are not resampled independently. Counts of events, contributing tasks, valid bootstrap draws, and the paired receiver-minus-gold interval are saved. Two hundred draws provide a bounded diagnostic interval, not publication-grade tail precision; intervals condition on this fixed model roster and cache realization.

The legacy YAML reports version {thresholds['version']} and {thresholds.get('n_models')} calibration models. Its ceilings are displayed only as transfer references. The current nine-model population differs, and the preserved L4 task lists overlap the current benchmark IDs as shown above. The separate GSM8K extension overlap is recorded in `calibration_task_overlap.csv`. Exact historical calibration membership is not backed by a per-estimate archived membership ledger, so overlap is inferred from the recorded provenance and preserved lists. These measurements are not independent validation and do not replace calibration. A pooled q exceeding a reference ceiling is not itself a measured per-channel gate false-positive rate.

BoolQ has two answer labels: when two present answers both disagree with a present receiver, those two answers coincide. Its receiver-conditioned q is structurally degenerate, so it cannot distinguish honest and adversarial coordination.

## Reproduction and actual resource use

Run from the project root:

```bash
../.venv/bin/python scripts/dgx_cache_audit.py --resamples 200 --seed 20260912
```

The script enforces at most two allowed CPU cores, one numerical/Arrow thread, nice ≥10, no visible CUDA device, a 4 GiB address-space ceiling, and a 600-second alarm. It uses only the honest cache columns needed for this audit. `resource_usage.json` records actual wall/CPU time, peak RSS, affinity, and library versions. Analytical self-checks passed for perfect/inverse/undefined phi, conditioning counts, and missing-answer handling. `artifact_manifest.json` fingerprints the outputs; source and configuration hashes are saved in `audit_results.json`.
"""
    (args.out / "REPORT.md").write_text(report)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    resource_record = {
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(), "architecture": platform.machine(),
        "python": platform.python_version(), "wall_seconds": time.monotonic() - START,
        "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
        "peak_rss_mib": usage.ru_maxrss / 1024,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "thread_limits": {k: os.environ[k] for k in THREAD_VARIABLES},
        "arrow_cpu_threads": pa.cpu_count(), "arrow_io_threads": pa.io_thread_count(),
        "cuda_visible_devices": "", "gpu_compute_used": False,
        "address_space_limit_bytes": cap, "wall_time_alarm_seconds": args.timeout_seconds,
        "packages": {k: importlib.metadata.version(k) for k in ("numpy", "pandas", "pyarrow", "matplotlib", "PyYAML")},
    }
    with (args.out / "REPORT.md").open("a") as handle:
        handle.write(f"\nObserved run: {resource_record['wall_seconds']:.3f} seconds wall time; {resource_record['user_cpu_seconds'] + resource_record['system_cpu_seconds']:.3f} CPU seconds; {resource_record['peak_rss_mib']:.1f} MiB peak RSS; CPU affinity {resource_record['cpu_affinity']}; nice {resource_record['nice']}; GPU compute disabled. Timing is sampled immediately before final metadata and checksum writes.\n")
    resource_record["output_data_bytes_excluding_resource_and_manifest"] = sum(
        p.stat().st_size for p in args.out.iterdir()
        if p.is_file() and p.name not in {"resource_usage.json", "artifact_manifest.json"}
    )
    write_json(args.out / "resource_usage.json", resource_record)
    artifacts = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(args.out.iterdir()) if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(args.out / "artifact_manifest.json", {"artifacts": artifacts, "self_hash_excluded": True})
    signal.alarm(0)
    print(json.dumps({"summary": summary, "resources": resource_record}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
