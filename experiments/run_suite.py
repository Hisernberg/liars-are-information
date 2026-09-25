#!/usr/bin/env python3
"""Run one named study of the v3 evaluation suite.

    python experiments/run_suite.py main      # E1  f-sweep, coherent liars, 6 benchmarks
    python experiments/run_suite.py zoo       # E2  attack zoo incl. RACE-targeted attacks
    python experiments/run_suite.py llm       # E3  replayed real-LLM deceivers
    python experiments/run_suite.py swarm     # E4  composition, clones and swarm size
    python experiments/run_suite.py history   # E5  sample complexity (history length)

Every study writes ``results/<study>/per_task.parquet`` (one row per world x
method x split x task, accuracy averaged over honest receivers), ``worlds.json``
and ``channel_diagnostics.parquet``, plus a ``run_manifest.json`` with the
input hashes, wall time and peak memory.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from lai.data import ATTACK_PROMPTS, BENCHMARKS  # noqa: E402
from lai.sim import CORE_METHODS, World, run_worlds  # noqa: E402

SEEDS5 = (0, 1, 2, 3, 4)
SEEDS3 = (0, 1, 2)
F_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def study_main() -> list[World]:
    return [
        World(b, f, "coherent", 1.0, p, s, study="main")
        for b, f, p, s in itertools.product(BENCHMARKS, F_GRID, (1.0, 0.5), SEEDS5)
    ]


ZOO = (
    [("independent", 1.0), ("uninformative", 1.0), ("echo", 1.0), ("attractor", 1.0), ("sleeper", 1.0)]
    + [("gate_aware", p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    + [("partial", e) for e in (0.5, 0.8)]
    + [("camouflage", th) for th in (0.7, 0.9)]
)


def study_zoo() -> list[World]:
    return [
        World(b, f, a, prm, 1.0, s, study="zoo")
        for b, f, (a, prm), s in itertools.product(BENCHMARKS, (0.3, 0.5, 0.7), ZOO, SEEDS3)
    ]


def study_llm() -> list[World]:
    return [
        World(b, f, f"llm:{a}", 1.0, 1.0, s, study="llm")
        for b, f, a, s in itertools.product(BENCHMARKS, (0.1, 0.3, 0.5, 0.7), ATTACK_PROMPTS, SEEDS5)
    ]


def study_swarm() -> list[World]:
    worlds = []
    for b, f, s in itertools.product(("mmlu", "medqa", "boolq", "math500"), (0.3, 0.5, 0.7), SEEDS3):
        for comp in ("frozen", "frozen+weak", "weak", "hom_qwen38_27b", "hom_llama32_3b"):
            worlds.append(World(b, f, "coherent", 1.0, 1.0, s, composition=comp, study="swarm"))
        for n in (5, 20, 40):
            worlds.append(World(b, f, "coherent", 1.0, 1.0, s, n_agents=n, study="swarm"))
            worlds.append(World(b, f, "gate_aware", 0.25, 1.0, s, n_agents=n, study="swarm"))
    return worlds


def study_history() -> list[World]:
    return [
        World(b, f, a, prm, 1.0, s, history_n=h, study="history")
        for b, f, (a, prm), h, s in itertools.product(
            ("mmlu", "medqa", "boolq", "math500"), (0.3, 0.5, 0.7),
            (("coherent", 1.0), ("gate_aware", 0.25)), (5, 10, 20, 40, 80), SEEDS3,
        )
    ]


def study_ext() -> list[World]:
    """Extensions: information budget (two known-channel oracles) and RACE-D."""
    zoo = [World(b, f, a, prm, 1.0, s, study="ext")
           for b, f, (a, prm), s in itertools.product(BENCHMARKS, (0.3, 0.5, 0.7), ZOO, SEEDS3)]
    llm = [World(b, f, f"llm:{a}", 1.0, 1.0, s, study="ext")
           for b, f, a, s in itertools.product(BENCHMARKS, (0.3, 0.5, 0.7), ATTACK_PROMPTS, SEEDS3)]
    return zoo + llm


STUDIES = {"main": study_main, "zoo": study_zoo, "llm": study_llm, "swarm": study_swarm, "history": study_history,
           "ext": study_ext}
EXT_METHODS = ("self", "majority", "aip_gated", "race", "race_onecoin", "race_d", "oracle_channel", "oracle_channel_honest")
# The expensive AIP variants are kept wherever they are the comparison of record.
LIGHT = tuple(m for m in CORE_METHODS if m not in ("aip_naive", "sac", "confidence"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="smoke: only the first N worlds")
    args = parser.parse_args()
    worlds = STUDIES[args.study]()
    if args.limit:
        worlds = worlds[: args.limit]
    methods = CORE_METHODS if args.study in ("main", "llm") else EXT_METHODS if args.study == "ext" else LIGHT
    out = args.out or ROOT / "results" / args.study
    started = time.monotonic()
    frame, _, _ = run_worlds(worlds, methods, out, processes=args.processes)
    inputs = sorted((ROOT / "data" / "cache").glob("*/*.parquet")) + [ROOT / "configs/inversion_thresholds.yaml"]
    manifest = {
        "study": args.study,
        "worlds": len(worlds),
        "methods": list(methods),
        "rows": len(frame),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "peak_rss_mib_parent": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "peak_rss_mib_children": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
        "processes": args.processes,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu_inference": False,
        "inputs_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest(),
        "code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted((ROOT / "src/lai").glob("*.py")))).hexdigest(),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
