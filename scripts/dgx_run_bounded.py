#!/usr/bin/env python3
"""Run cache study in a bounded child; save success/failure resource receipt."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--rss-limit-mib", type=int, default=2048)
    a = p.parse_args()
    if a.out.exists():
        p.error("Output directory already exists; use a new run directory.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg")
    cpus = sorted(os.sched_getaffinity(0))[:2]
    script = Path(__file__).with_name("dgx_low_resource.py")
    command = [sys.executable, "-u", str(script), "--profile", a.profile, "--out", str(a.out)]
    log = a.out.parent / (a.out.name + ".log")
    started = time.monotonic()
    peak, cpu_seconds, reason = 0, 0, None
    def setup():
        os.sched_setaffinity(0, cpus)
        os.nice(10)
    with log.open("w") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, preexec_fn=setup)
        proc = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                peak = max(peak, proc.memory_info().rss)
                timing = proc.cpu_times()
                cpu_seconds = timing.user + timing.system
            except psutil.NoSuchProcess:
                pass
            elapsed = time.monotonic() - started
            if peak > a.rss_limit_mib * 1024**2:
                reason = "rss_limit"
            if elapsed > a.timeout:
                reason = "wall_timeout"
            if reason:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
            time.sleep(.2)
    manifest = a.out / "run_manifest.json"
    status = "complete" if child.returncode == 0 and manifest.exists() else "failed"
    receipt = dict(status=status, exit_code=child.returncode, stop_reason=reason,
        elapsed_seconds=time.monotonic()-started, peak_sampled_rss_mib=peak/1024**2,
        sampled_cpu_seconds=cpu_seconds, cpu_affinity=cpus, nice=10, computational_threads=1,
        gpu_inference=False, wall_limit_seconds=a.timeout, rss_limit_mib=a.rss_limit_mib,
        command=command, log_file=log.name)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "resource_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
