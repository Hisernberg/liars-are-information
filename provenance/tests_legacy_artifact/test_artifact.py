"""Integrity guards for the artifact's data files.

These exist because of one incident. A `.gitattributes` declaring
`*.parquet filter=lfs` caused git-lfs to replace every parquet in the working
tree with a 131-byte pointer. Nothing reported a missing filter: pandas raised
`ArrowInvalid: Parquet magic bytes not found in footer`, which reads as file
corruption, and every analysis script in the project failed at once. The bytes
were recoverable from `.git/lfs/objects`, but a check is cheaper than the
diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

#: Parquets the study cannot be re-derived without.
LOAD_BEARING = [
    "data/cache/gsm8k/qwen38_27b.parquet",
    "results/aggregation/sweep.parquet",
    "results/adversarial/adversarial_sweep.parquet",
    "results/adversarial/gate_aware_mitigations.parquet",
    "results/correlation/pairwise_estimates.parquet",
]


def all_parquets() -> list[Path]:
    return sorted(
        list((ROOT / "data").rglob("*.parquet")) + list((ROOT / "results").rglob("*.parquet"))
    )


def test_the_tree_has_parquets_at_all() -> None:
    assert len(all_parquets()) > 100, "the cache is missing; nothing downstream can run"


def test_no_parquet_is_an_lfs_pointer() -> None:
    """A pointer where data should be is the failure this file exists for."""
    pointers = [
        str(p.relative_to(ROOT))
        for p in all_parquets()
        if p.stat().st_size < 1024 and p.read_bytes().startswith(LFS_MAGIC)
    ]
    assert not pointers, (
        "these parquets are git-lfs pointers, not data. Run `git lfs checkout` to "
        f"restore them, and do not reintroduce an lfs filter for *.parquet: {pointers}"
    )


def test_gitattributes_declares_no_lfs_filter() -> None:
    path = ROOT / ".gitattributes"
    assert path.exists()
    offenders = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#") and "filter=lfs" in line
    ]
    assert not offenders, (
        "a git-lfs filter is declared again. It converts the working tree to "
        f"pointers and breaks every reader: {offenders}"
    )


@pytest.mark.parametrize("rel", LOAD_BEARING)
def test_load_bearing_parquet_opens(rel: str) -> None:
    path = ROOT / rel
    assert path.exists(), f"missing {rel}"
    frame = pd.read_parquet(path)
    assert not frame.empty, f"{rel} is empty"
