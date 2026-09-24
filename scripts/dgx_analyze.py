#!/usr/bin/env python3
"""Additional conditional uncertainty, gate behavior, and scoring analyses."""
# ruff: noqa: E402
import json
import os
import resource
import sys
import time
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
os.nice(10)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import pandas as pd
from dgx_low_resource import paired_interval, stable_seed


def main():
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    source = root / "results/dgx/standard"
    out = root / "results/dgx/analysis"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    task = pd.read_parquet(source / "per_task.parquet")
    selected = pd.read_parquet(source / "selected_attack_predictions.parquet")
    channels = pd.read_parquet(source / "channel_diagnostics.parquet")
    channels = channels[channels.peer != channels.receiver].copy()
    axes = ["study", "benchmark", "f", "p_obs", "composition", "coherence_p", "seed", "method", "peer_is_byzantine"]
    counts = channels.groupby(axes + ["decision"]).size().unstack(fill_value=0)
    for key in ["trust", "discard", "invert"]:
        if key not in counts:
            counts[key] = 0
    counts["n_peer_channels"] = counts[["trust", "discard", "invert"]].sum(axis=1)
    counts["inversion_rate"] = counts["invert"] / counts.n_peer_channels
    counts["trust_rate"] = counts["trust"] / counts.n_peer_channels
    counts.reset_index().to_csv(out / "gate_diagnostics.csv", index=False)
    rows = []
    for (b, f, target), data in selected.groupby(["benchmark", "f", "target_method"]):
        table = data.groupby(["task_id", "method"]).accuracy.mean().unstack()
        for baseline in ["self_only", "aip_trust_only"]:
            point, low, high = paired_interval(table[target] - table[baseline],
                stable_seed("adaptive_ci", b, f, target, baseline), 1000)
            rows.append(dict(benchmark=b, f=f, target_method=target, baseline=baseline,
                delta=point, ci_low=low, ci_high=high, n_test_tasks=len(table),
                scope="conditional on validation-selected p per seed; fixed validation/history; pointwise"))
    pd.DataFrame(rows).to_csv(out / "adaptive_comparisons.csv", index=False)
    test = task[task.split == "test"].copy()
    test["semantic_gain"] = test.accuracy - test.exact_match_accuracy
    scoring = test.groupby(["benchmark", "study", "method"]).agg(
        accuracy=("accuracy", "mean"), exact_match_accuracy=("exact_match_accuracy", "mean"),
        semantic_gain=("semantic_gain", "mean"))
    scoring.reset_index().to_csv(out / "scoring_sensitivity.csv", index=False)
    fig, axes_plot = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    methods = {"self_only": ("Self only", "C0"), "majority": ("Majority", "C1"),
        "aip_gated": ("AIP gated", "C2"), "aip_trust_only": ("Trust only", "C3"),
        "dawid_skene_full": ("Full DS", "C4")}
    for ax, benchmark in zip(axes_plot, ["mmlu", "boolq", "math500"], strict=True):
        data = test[(test.study == "mechanism") & (test.p_obs == 1.) & (test.benchmark == benchmark)]
        for method, (label, color) in methods.items():
            cells = data[data.method == method]
            if cells.empty:
                continue
            estimates = []
            for fraction, group in cells.groupby("f"):
                values = group.groupby("task_id").accuracy.mean()
                point, lo, hi = paired_interval(values, stable_seed("figure", benchmark, method, fraction), 1000)
                estimates.append((fraction, point, lo, hi))
            fractions, points, lows, highs = zip(*estimates, strict=True)
            ax.plot(fractions, points, marker="o", label=label, color=color)
            ax.fill_between(fractions, lows, highs, alpha=.08, color=color)
        ax.set(title=benchmark, xlabel="Byzantine fraction", ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
    axes_plot[0].set_ylabel("Test accuracy")
    handles, labels = axes_plot[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("DGX cache replay · 3 seeds · task-bootstrap 95% intervals")
    fig.tight_layout(rect=(0, .08, 1, 1))
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"mechanism.{extension}", dpi=170)
    plt.close(fig)
    controls = test[(test.study != "adaptive") & (test.p_obs == 1.) & test.f.isin([0., .5])]
    controls.groupby(["benchmark", "f", "composition", "method"]).accuracy.mean().reset_index().to_csv(
        out / "composition_comparison.csv", index=False)
    pairs = pd.read_csv(source / "paired_comparisons.csv")
    chosen = pairs[(pairs.study == "mechanism") & (pairs.p_obs == 1.) & (pairs.f == .5)
        & pairs.baseline.isin(["self_only", "majority", "aip_trust_only", "dawid_skene_full"])]
    lines = ["# Additional DGX analyses", "",
        "Paired contrasts below compare gated AIP with each control at f=0.5 and full visibility.",
        "Confidence intervals resample 80 test tasks after averaging three swarm seeds. Intervals are pointwise and conditional on the fixed cached population and legacy calibration.",
        "", "| Benchmark | Comparator | AIP minus comparator | 95% CI |",
        "|---|---|---:|---|"]
    for row in chosen.itertuples():
        lines.append(f"| {row.benchmark} | {row.baseline} | {row.delta:+.3f} | [{row.ci_low:+.3f}, {row.ci_high:+.3f}] |")
    lines += ["", "## Interpretation boundaries", "",
        "- gate_diagnostics.csv counts receiver/peer decisions. Honest inversion rates and malicious trust rates are descriptive; channels and tasks are correlated, so these counts do not prove a familywise error bound.",
        "- adaptive_comparisons.csv compares each target to controls under that target's validation-selected stationary attack. The same test tasks are paired. It conditions on the chosen attack and does not bootstrap the entire attack-selection procedure.",
        "- scoring_sensitivity.csv quantifies semantic versus exact-string scoring on identical predictions. Differences are expected for MATH-500. The repository scorer is an approximation and has not been externally adjudicated.",
        "- composition_comparison.csv contrasts a mixed roster with repeated Qwen outputs. Competence and diversity both differ, so a difference cannot be attributed to diversity alone. Repeated outputs are not independent model generations.",
        "- BoolQ's gate cannot invert at its ceiling of one; any robust claim must include its failure cases. Open-answer inversion need not recover the correct candidate or beat an honest receiver.",
        "- Stronger inference would require independently calibrated thresholds, larger untouched task sets, multiple independently generated caches, and model-driven/interactive attacks.",
        ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    receipt = dict(elapsed_seconds=time.monotonic()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)), computational_threads=1,
        inference_run=False, source_worlds=manifest["worlds_completed"], status="complete")
    (out / "resource_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
