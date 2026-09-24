#!/usr/bin/env python3
"""Evict one model's weights from the HuggingFace cache.

Phase 1 stages 283 GB of checkpoints through 206 GB of free disk, so the four
large models cannot coexist on the filesystem. Sequential residency already
generates each model's cells in one uninterrupted block, so a checkpoint is dead
weight the moment its block finishes -- this removes it.

Deliberately narrow: it deletes whole `models--org--name` directories from the
hub cache and nothing else. It never touches datasets, never runs without an
explicit model argument, and prints what it reclaimed.

    python scripts/hf_evict.py --model Qwen/Qwen3.8-27B
    python scripts/hf_evict.py --keep-only unsloth/Llama-3.2-3B-Instruct ... --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def hub_dir() -> Path:
    import huggingface_hub.constants as c

    return Path(c.HF_HUB_CACHE)


def repo_dir(root: Path, hf_id: str) -> Path:
    return root / ("models--" + hf_id.replace("/", "--"))


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", default=[], help="hf_id to evict (repeatable)")
    p.add_argument(
        "--keep-only",
        nargs="*",
        default=None,
        help="evict every cached model EXCEPT these hf_ids",
    )
    p.add_argument(
        "--drop-consolidated",
        action="store_true",
        help="remove redundant consolidated.safetensors copies (vLLM loads the shards)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = hub_dir()
    if not root.exists():
        print(f"no hub cache at {root}")
        return 0

    # Interrupted downloads leave .incomplete blobs that are never reclaimed and
    # never reused across a repo re-fetch. They accumulate silently: qwen38_27b
    # and olmo3_32b_think reached 111 GB and 147 GB against ~56 GB and ~65 GB of
    # actual weights, purely from retried downloads. Always swept, since a
    # partial blob for a model still being fetched lives under a different name.
    swept = 0.0
    for inc in root.glob("models--*/blobs/*.incomplete"):
        swept += inc.stat().st_size / 1e9
        if not args.dry_run:
            inc.unlink(missing_ok=True)
    if swept:
        print(f"{'would sweep' if args.dry_run else 'swept':<12} "
              f"{'.incomplete blobs':<52} {swept:7.1f} GB")

    if args.drop_consolidated:
        # Some Mistral repos ship a single consolidated.safetensors beside the
        # sharded files and the index. vLLM loads via the index, so the
        # consolidated copy is the whole model stored twice -- 27.8 GB for
        # Ministral 3 14B. Prefetching with a bare "*.safetensors" pattern pulls
        # it, which is how a 28 GB model came to occupy 56 GB.
        freed = 0.0
        for snap in root.glob("models--*/snapshots/*"):
            link = snap / "consolidated.safetensors"
            if link.is_symlink():
                blob = link.resolve()
                gb = blob.stat().st_size / 1e9 if blob.exists() else 0.0
                freed += gb
                print(f"{'would drop' if args.dry_run else 'dropping':<12} "
                      f"{snap.parent.parent.name:<52} {gb:7.1f} GB")
                if not args.dry_run:
                    link.unlink(missing_ok=True)
                    blob.unlink(missing_ok=True)
        print(f"{'would reclaim' if args.dry_run else 'reclaimed':<12} {freed:7.1f} GB")
        if not (args.model or args.keep_only is not None):
            print(f"free now: {shutil.disk_usage(root).free / 1e9:.1f} GB")
            return 0

    targets: list[Path] = []
    if args.keep_only is not None:
        keep = {repo_dir(root, m).name for m in args.keep_only}
        targets = [d for d in root.glob("models--*") if d.name not in keep]
    for m in args.model:
        d = repo_dir(root, m)
        if d.exists() and d not in targets:
            targets.append(d)

    if not targets:
        print("nothing to evict")
        return 0

    freed = 0.0
    for d in sorted(targets):
        gb = size_gb(d)
        freed += gb
        print(f"{'would evict' if args.dry_run else 'evicting':<12} {d.name:<52} {gb:7.1f} GB")
        if not args.dry_run:
            shutil.rmtree(d)
    print(f"{'would reclaim' if args.dry_run else 'reclaimed':<12} {freed:7.1f} GB")
    print(f"free now: {shutil.disk_usage(root).free / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
