#!/usr/bin/env python3
"""Self-driving orchestrator for the H100 regeneration.

Runs the phases in order with no agent attached. Per phase it runs the command,
checks the exit code, appends one line to RUNLOG.md, commits on success, and on
failure writes STATUS.json and exits non-zero. Phase 7 (paper prose) is
deliberately excluded: writing is not a batch job.

    python scripts/run_all.py --dry-run     # print the ordered plan, run nothing
    python scripts/run_all.py --smoke       # tiny inputs, prove the wiring
    python scripts/run_all.py               # the real run
    python scripts/run_all.py --from phase3 # resume

Design notes that matter for an unattended run:

* **Idempotence is the contract.** Every phase is expected to skip completed work
  on a re-run, so a resume after a crash costs only the unfinished part. Phases
  that cannot yet honour that say so in `resumable=False` rather than pretending.
* **A phase that needs judgement does not get judgement.** It logs loudly and
  continues; it never silently drops a result. The one thing this orchestrator
  will not do is decide that a surprising number is uninteresting.
* **Exit codes are the whole interface.** A phase that reports a red gate and
  exits 0 would let the chain walk straight past it, which is the failure mode
  the gate exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_BIN = str(ROOT / ".venv" / "bin" / "python")
RUNLOG = ROOT / "RUNLOG.md"
# Run state lives with the other run outputs. It was at the repository
# root, which put an orchestrator scratch file on the artifact's landing
# page next to the README.
STATUS = ROOT / "results" / "STATUS.json"
LOGDIR = ROOT / "results" / "logs"
STAMPS = ROOT / "results" / ".stamps"
"""Completion stamps written by THIS orchestrator.

Artifact presence is not a completion test: results/claims.md and
paper/numbers.tex both survive from the L4 era, so keying "done" off them
would have skipped Phase 6 entirely on the first run. A stamp is written
only after a phase's own steps exit 0 here."""


@dataclass(frozen=True)
class Phase:
    name: str
    title: str
    steps: list[list[str]]
    smoke: list[list[str]] = field(default_factory=list)
    #: Optional extra check: even with a stamp, re-run if this is missing.
    done_marker: Path | None = None
    gpu: bool = False
    resumable: bool = True
    #: Set when a phase cannot run unattended yet; the orchestrator stops here
    #: with an explanation rather than failing obscurely mid-run.
    blocked_reason: str | None = None


def phases() -> list[Phase]:
    p = lambda *a: [PY_BIN, *a]  # noqa: E731
    return [
        Phase(
            name="phase1_tail",
            title="Phase 1 tail: olmo3 cache, T=0.7 second pass, gate report",
            steps=[
                ["bash", str(ROOT / "scripts" / "run_phase1.sh"), "olmo3_32b_think"],
                ["bash", str(ROOT / "scripts" / "run_phase1_t07.sh")],
                p(str(ROOT / "scripts" / "phase1_gate_report.py")),
            ],
            smoke=[p(str(ROOT / "scripts" / "phase1_gate_report.py"))],
            done_marker=ROOT / "results" / "phase_1_gate_report.parquet",
            gpu=True,
        ),
        Phase(
            name="phase2",
            title="Phase 2: correlation, multiclass law, honest-q v3 recalibration",
            steps=[
                p(str(ROOT / "scripts" / "phase_b_correlation.py")),
                p(str(ROOT / "scripts" / "phase_b2_multiclass.py")),
                p(str(ROOT / "scripts" / "calibrate_receiver_coherence.py")),
            ],
            smoke=[p(str(ROOT / "scripts" / "phase_b_correlation.py"), "--help")],
            done_marker=ROOT / "results" / "correlation" / "pairwise_estimates.parquet",
        ),
        Phase(
            name="phase3",
            title="Phase 3: aggregation sweep, N-scaling, parity audit, minimax regret",
            steps=[
                p(str(ROOT / "scripts" / "phase_c_aggregate.py")),
                p(str(ROOT / "scripts" / "phase_c_report.py")),
                p(str(ROOT / "scripts" / "phase_c_figures.py")),
            ],
            smoke=[p(str(ROOT / "scripts" / "phase_c_aggregate.py"), "--help")],
            done_marker=ROOT / "results" / "aggregation" / "sweep.parquet",
        ),
        Phase(
            name="phase4",
            title="Phase 4: real-LLM adversaries, pilot gate then full sweep",
            steps=[
                p(str(ROOT / "scripts" / "phase_d_adversarial.py")),
                p(str(ROOT / "scripts" / "phase_d_sweep.py")),
                # The adaptive bandit is the seventh attack class, not an extra:
                # it is the deterrence experiment, and its bandit_log.parquet is
                # one of the claims ledger's inputs.
                p(str(ROOT / "scripts" / "phase_d_bandit.py")),
                p(str(ROOT / "scripts" / "phase_d_figures.py")),
            ],
            smoke=[p(str(ROOT / "scripts" / "phase_d_adversarial.py"), "--help")],
            done_marker=ROOT / "results" / "adversarial" / "adversarial_sweep.parquet",
            gpu=True,
        ),
        Phase(
            name="phase5",
            title="Phase 5: sleeper (E1) and Sybil (E2)",
            steps=[p(str(ROOT / "scripts" / "phase_e_sleeper_sybil.py"))],
            smoke=[p(str(ROOT / "scripts" / "phase_e_sleeper_sybil.py"), "--smoke")],
            done_marker=ROOT / "results" / "adversarial" / "sleeper_e1.parquet",
        ),
        Phase(
            name="phase6",
            title="Phase 6: ledger v2, numbers.tex, claims map, CI checks",
            steps=[
                # ORDER MATTERS. phase_e_claims reads thirteen parquets and four
                # of their producers were missing from this list, so it crashed
                # on a parity_audit written by phase_c_report with a different
                # column name (`parity_effect`) than the one phase_e_tables
                # writes (`effect`). The mechanism experiments and the tables
                # must run BEFORE the ledger that cites them.
                p(str(ROOT / "scripts" / "phase_e_burst.py")),
                p(str(ROOT / "scripts" / "phase_e_decomposition.py")),
                p(str(ROOT / "scripts" / "phase_e_tables.py")),
                p(str(ROOT / "scripts" / "phase_e_figures.py")),
                p(str(ROOT / "scripts" / "phase_e_claims.py")),
                # The pending-claim audit runs before numbers.tex: it is the
                # step that fails loudly if a claim the manuscript cites can no
                # longer be re-derived from the parquets.
                p(str(ROOT / "scripts" / "task6_audit_pending.py")),
                # Task 9: all three read the cache and load no model.
                p(str(ROOT / "scripts" / "phase_e3_cost.py")),
                p(str(ROOT / "scripts" / "phase_composition.py")),
                p(str(ROOT / "scripts" / "phase_confidence_auc.py")),
                p(str(ROOT / "scripts" / "make_numbers.py")),
                # Every figure the manuscript cites is regenerated here, into
                # the one directory LaTeX reads. Seven of them spent this whole
                # regeneration as L4-era leftovers because a second figures/
                # directory existed and nothing rebuilt paper/figures/.
                p(str(ROOT / "scripts" / "phase_c_figures.py")),
                p(str(ROOT / "scripts" / "phase_d_figures.py")),
                p(str(ROOT / "scripts" / "fig_evasion_band.py")),
                p(str(ROOT / "scripts" / "fig_sybil.py")),
                p("-m", "pytest", "tests/test_paper.py", "-q"),
            ],
            smoke=[p("-m", "pytest", "tests/test_paper.py", "-q")],
            done_marker=ROOT / "results" / "claims.md",
        ),
    ]


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() or "-"


def log_row(phase: str, step: str, result: str, commit: str = "-") -> None:
    with RUNLOG.open("a", encoding="utf-8") as f:
        f.write(f"| {now()} | {phase} | {step} | {result} | {commit} |\n")


def write_status(**kwargs: object) -> None:
    STATUS.write_text(json.dumps({"updated_at": now(), **kwargs}, indent=2), encoding="utf-8")


def commit(message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=ROOT, capture_output=True)
    return git_head()


#: Free space a GPU phase needs before it may start. The largest checkpoint in
#: the roster is ~65 GB and the drivers evict as they go, so this is headroom for
#: one model plus working space, not for the whole roster.
MIN_FREE_GB_FOR_GPU_PHASE = 80


def free_gb() -> float:
    import shutil

    return shutil.disk_usage(ROOT).free / 1e9


def ensure_disk(min_gb: float) -> bool:
    """Reclaim before a GPU phase, and refuse to start it blind.

    Disk is the failure mode that does not announce itself: a run that fills the
    filesystem dies with "No space left on device" deep inside an engine
    subprocess, minutes or hours in, and leaves partial blobs behind that make
    the next attempt worse. Cheap reclamation first, then a hard stop rather than
    starting a phase that cannot finish.
    """
    if free_gb() >= min_gb:
        return True
    log_row("run_all", "disk", f"{free_gb():.0f}GB free, reclaiming toward {min_gb}GB")
    subprocess.run([PY_BIN, str(ROOT / "scripts" / "hf_evict.py"), "--drop-consolidated"],
                   cwd=ROOT, capture_output=True)
    if free_gb() >= min_gb:
        log_row("run_all", "disk", f"reclaimed to {free_gb():.0f}GB")
        return True
    return False


#: Any of these already running means another driver owns the GPU.
GPU_OWNER_PATTERNS = ("scripts/run_phase1.sh", "scripts/phase_a_cache.py")


def wait_for_gpu(poll_seconds: int = 60, timeout_hours: float = 6.0) -> bool:
    """Block until no other driver owns the GPU.

    Launching straight into a GPU phase is wrong whenever a hand-started driver
    is still finishing -- two vLLM engines on one card contend for VRAM and at
    least one dies. This orchestrator is meant to be started at any time,
    including while a previous run is mid-model, so it waits rather than racing.
    Returns False if it waits past the timeout, which is a hard stop rather than
    a licence to proceed anyway.
    """
    deadline = time.time() + timeout_hours * 3600
    announced = False
    while time.time() < deadline:
        owners = []
        for pattern in GPU_OWNER_PATTERNS:
            r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
            owners += [ln for ln in r.stdout.split() if ln]
        if not owners:
            return True
        if not announced:
            print(f"    waiting: another driver owns the GPU (pids {','.join(owners)})")
            log_row("run_all", "wait", f"GPU held by pids {','.join(owners)}")
            announced = True
        time.sleep(poll_seconds)
    return False


def run_phase(ph: Phase, smoke: bool, log_path: Path) -> tuple[bool, str]:
    steps = ph.smoke if smoke else ph.steps
    if not steps:
        return True, "no steps"
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        for step in steps:
            log.write(f"\n=== {now()} :: {' '.join(step)} ===\n")
            log.flush()
            rc = subprocess.run(step, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                return False, f"exit {rc} from: {' '.join(step[-2:])}"
    return True, "ok"


def write_final_report_skeleton(plan: list[Phase], skipped: list[str]) -> None:
    """Numbers-only skeleton. Prose is Phase 7 and waits for an agent session."""
    gate = ROOT / "results" / "phase_1_gate_report.md"
    lines = [
        "# FINAL REPORT (skeleton)",
        "",
        f"Generated {now()} by scripts/run_all.py. **Numbers only** -- the prose,",
        "the adversarial self-review and the venue formatting are Phase 7, which is",
        "deliberately excluded from unattended running.",
        "",
        "## Phases",
        "",
        "| phase | title | state |",
        "|---|---|---|",
    ]
    for ph in plan:
        state = "SKIPPED (needs agent)" if ph.name in skipped else (
            "complete" if (STAMPS / f"{ph.name}.done").exists() else "not run")
        lines.append(f"| {ph.name} | {ph.title} | {state} |")
    lines += ["", "## Artifacts", ""]
    for rel in ("results/phase_1_gate_report.md", "results/correlation",
                "results/aggregation", "results/adversarial", "results/claims.md"):
        pth = ROOT / rel
        lines.append(f"- `{rel}` -- {'present' if pth.exists() else 'MISSING'}")
    if gate.exists():
        lines += ["", "## Phase 1 gate report", "", gate.read_text(encoding="utf-8")]
    (ROOT / "results" / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--smoke", action="store_true", help="tiny inputs, prove the wiring")
    ap.add_argument("--from", dest="start", default=None, help="resume from this phase")
    args = ap.parse_args()

    plan = phases()
    if args.start:
        names = [p.name for p in plan]
        if args.start not in names:
            print(f"unknown phase {args.start!r}; known: {names}", file=sys.stderr)
            return 2
        plan = plan[names.index(args.start):]

    if args.dry_run:
        print(f"ORDERED PLAN ({len(plan)} phases; phase7 excluded -- prose is agent work)\n")
        for i, ph in enumerate(plan, 1):
            done = (STAMPS / f"{ph.name}.done").exists()
            tag = "GPU" if ph.gpu else "cpu"
            state = ("NEEDS AGENT" if ph.blocked_reason
                     else "done" if done else "pending")
            print(f"{i}. {ph.name:<12} [{tag}] {state:<8} {ph.title}")
            for s in ph.steps:
                print(f"      - {' '.join(Path(x).name if '/' in x else x for x in s[1:])}")
            if ph.blocked_reason:
                print(f"      ! {ph.blocked_reason}")
        return 0

    LOGDIR.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    for ph in plan:
        if ph.blocked_reason:
            # Skipped LOUDLY, not treated as a failure and not allowed to block
            # the phases behind it: Phase 6's mechanical outputs are wanted
            # unattended even when Phase 5 needs an agent to build first.
            skipped.append(ph.name)
            log_row(ph.name, "SKIPPED (needs agent)", ph.blocked_reason.split(".")[0])
            print(f"SKIP {ph.name}: {ph.blocked_reason.split('.')[0]}")
            continue
        stamp = STAMPS / f"{ph.name}.done"
        if not args.smoke and stamp.exists():
            log_row(ph.name, "skip", "already complete (stamp)")
            print(f"skip {ph.name} (already complete)")
            continue

        log_path = LOGDIR / f"{ph.name}{'_smoke' if args.smoke else ''}.log"
        write_status(state="RUNNING", phase=ph.name, log=str(log_path))
        print(f"[{now()}] {ph.name}: {ph.title}")
        if ph.gpu and not args.smoke and not ensure_disk(MIN_FREE_GB_FOR_GPU_PHASE):
            reason = (f"only {free_gb():.0f}GB free, need {MIN_FREE_GB_FOR_GPU_PHASE}GB "
                      "for a GPU phase after reclamation")
            log_row(ph.name, "FAILED", reason)
            write_status(state="STOPPED", phase=ph.name, reason=reason, log=str(log_path))
            print(f"STOPPED at {ph.name}: {reason}")
            return 1
        if ph.gpu and not args.smoke and not wait_for_gpu():
            reason = "timed out waiting for another driver to release the GPU"
            log_row(ph.name, "FAILED", reason)
            write_status(state="STOPPED", phase=ph.name, reason=reason, log=str(log_path))
            print(f"STOPPED at {ph.name}: {reason}")
            return 1
        started = time.time()
        ok, detail = run_phase(ph, args.smoke, log_path)
        mins = (time.time() - started) / 60

        if not ok:
            log_row(ph.name, "FAILED", f"{detail} after {mins:.1f}m")
            write_status(state="STOPPED", phase=ph.name, reason=detail, log=str(log_path))
            print(f"STOPPED at {ph.name}: {detail}  (log: {log_path})")
            return 1

        if not args.smoke:
            STAMPS.mkdir(parents=True, exist_ok=True)
            (STAMPS / f"{ph.name}.done").write_text(now() + "\n", encoding="utf-8")
        sha = "-" if args.smoke else commit(f"auto: {ph.name} complete")
        log_row(ph.name, "smoke" if args.smoke else "complete", f"{detail} in {mins:.1f}m", sha)
        print(f"    ok in {mins:.1f}m  {sha}")

    state = "COMPLETE" if not skipped else "COMPLETE_WITH_SKIPS"
    write_status(state=state, phases=[p.name for p in plan], skipped=skipped,
                 note=("Phase 7 (paper prose) is agent work and was never in this chain."
                       + (f" Needs an agent session: {', '.join(skipped)}." if skipped else "")))
    log_row("run_all", state, f"{len(plan) - len(skipped)} ran, {len(skipped)} skipped",
            git_head())
    write_final_report_skeleton(plan, skipped)
    print(f"{state}: {len(plan) - len(skipped)} ran, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
