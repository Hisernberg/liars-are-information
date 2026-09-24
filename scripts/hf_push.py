#!/usr/bin/env python3
"""Publish the artifact to a Hugging Face Hub dataset repository.

This repository is a *dataset* on the Hub, not a model: it produces no weights.
What it produces is the inference cache (the irreplaceable part, roughly ten
GPU-hours), the derived result parquets, the claims ledger, the manuscript and
the code that turns one into the other.

An earlier attempt to push failed because the token supplied was not a Hugging
Face token at all. Token shape is therefore checked before anything is uploaded,
with a message that says what is wrong rather than surfacing a 401 from three
frames deep.

    python scripts/hf_push.py --repo-id <user>/liars-are-information --dry-run
    python scripts/hf_push.py --repo-id <user>/liars-are-information

The token is read from ``--token``, then ``$HF_TOKEN``, then the local Hub login
(``huggingface-cli login``), in that order.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Everything under these patterns stays local. Scratch caches, logs and the
#: virtualenv are not part of the artifact; excluding them here rather than in
#: .gitignore keeps them available to a rerun on this machine.
IGNORE = [
    ".git/*", ".venv/*", "**/__pycache__/*", "*.pyc",
    ".pytest_cache/*", ".ruff_cache/*",
    "data/cache_dryrun/*",            # smoke-test output, 6 files, not results
    "data/cache_adversarial_pilot/*",  # empty pilot directory
    "results/.stamps/*",               # local resume state
    "paper/*.aux", "paper/*.bbl", "paper/*.blg",
    "paper/main.log", "paper/*.out",  # build files; main.pdf IS shipped
    "*.egg-info/*",
]

#: Files whose absence means the upload would be misleading rather than merely
#: incomplete. A dataset card with no ledger, or a ledger with no cache, is
#: worse than no upload.
REQUIRED = [
    "README.md",
    "LICENSE",
    ".gitattributes",
    "configs/models.yaml",
    "configs/task_lists.json",
    "configs/decision_rules.yaml",
    "results/claims.md",
    "results/pending_claims_audit.md",
    "results/measurement_floor.json",
    "paper/numbers.tex",
    "paper/figures/MANIFEST.json",
    "paper/main.pdf",
    "data/cache/gsm8k",
]


class TokenError(RuntimeError):
    """The supplied credential is not a usable Hugging Face token."""


def resolve_token(explicit: str | None) -> str:
    """Return a Hub token, or explain precisely which check failed.

    Hugging Face user access tokens start with ``hf_``. The push that failed
    before was given a credential for a different service entirely, and the
    resulting 401 said nothing about that, so the shape is checked here.
    """
    token = explicit or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TokenError(
                "huggingface_hub is not installed. Run: uv pip install -e '.[hub]'"
            ) from exc
        token = get_token()
    if not token:
        raise TokenError(
            "no Hugging Face token found. Pass --token, set $HF_TOKEN, or run "
            "`huggingface-cli login`."
        )
    token = token.strip()
    if not token.startswith("hf_"):
        raise TokenError(
            "this does not look like a Hugging Face token: they begin with "
            f"'hf_' and this one begins with {token[:3]!r}. Create one at "
            "https://huggingface.co/settings/tokens with write access to the "
            "target repository."
        )
    return token


def check_required(root: Path) -> list[str]:
    return [p for p in REQUIRED if not (root / p).exists()]


#: Verbatim copies of the source benchmark items -- questions, choices and gold
#: answers for ARC, BoolQ, GSM8K, MATH-500, MedQA and MMLU. Excluded by default
#: for two reasons. Licensing: ARC and BoolQ are CC BY-SA, which this
#: repository's MIT licence cannot cover, and republishing them under it would
#: be a misstatement. Redundancy: no offline phase reads this directory. The
#: label spaces come from `configs/inversion_thresholds.yaml`, the task
#: identities from `configs/task_lists.json`, and the gold answers travel inside
#: the cache parquets, so every analysis in the study reproduces without it.
#: Only re-running inference needs the question text, and the loaders fetch it
#: from the original sources on demand.
BENCHMARK_CACHE = ["data/benchmarks/*"]


def is_ignored(rel: str, ignore: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in ignore)


def plan(root: Path, ignore: list[str]) -> tuple[list[Path], int]:
    """Return the files that would be uploaded and their total size."""
    files, total = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if is_ignored(rel, ignore) or rel.startswith(".git/"):
            continue
        files.append(path)
        total += path.stat().st_size
    return files, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", help="e.g. your-username/liars-are-information")
    ap.add_argument("--token", help="Hub token; defaults to $HF_TOKEN then the local login")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--repo-type", choices=["dataset", "model"], default="dataset",
                    help="Hub repo type. This project is a dataset -- it produces no "
                         "weights -- but a model repo is supported because one may "
                         "already exist under that name.")
    ap.add_argument("--prune", action="store_true",
                    help="delete files on the Hub that are not in the local plan. "
                         "upload_folder only adds and overwrites, so without this a "
                         "renamed or deleted source file lingers remotely forever.")
    ap.add_argument("--message", default="Publish AIP artifact",
                    help="commit message on the Hub")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the upload plan and exit without contacting the Hub")
    ap.add_argument("--include-benchmark-cache", action="store_true",
                    help="also upload data/benchmarks/ (verbatim ARC/BoolQ/GSM8K/"
                         "MATH-500/MedQA/MMLU items). Off by default: ARC and BoolQ "
                         "are CC BY-SA and no offline phase needs the directory.")
    args = ap.parse_args()

    ignore = list(IGNORE) if args.include_benchmark_cache else IGNORE + BENCHMARK_CACHE

    missing = check_required(ROOT)
    if missing:
        print("REFUSING TO PUSH -- the artifact is incomplete:", file=sys.stderr)
        for item in missing:
            print(f"  missing {item}", file=sys.stderr)
        print("\nRun `python scripts/run_all.py` first, or fix the paths above.",
              file=sys.stderr)
        return 1

    files, total = plan(ROOT, ignore)
    by_top: dict[str, tuple[int, int]] = {}
    for path in files:
        top = path.relative_to(ROOT).parts[0]
        count, size = by_top.get(top, (0, 0))
        by_top[top] = (count + 1, size + path.stat().st_size)

    print(f"upload plan: {len(files)} files, {total / 1e6:.1f} MB")
    for top, (count, size) in sorted(by_top.items(), key=lambda kv: -kv[1][1]):
        print(f"  {top:<24} {count:>5} files  {size / 1e6:>8.2f} MB")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    if not args.repo_id:
        print("\n--repo-id is required for a real push "
              "(e.g. your-username/liars-are-information)", file=sys.stderr)
        return 1

    try:
        token = resolve_token(args.token)
    except TokenError as exc:
        print(f"\nTOKEN ERROR: {exc}", file=sys.stderr)
        return 1

    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type,
                    private=args.private, exist_ok=True)

    stale: list[str] = []
    if args.prune:
        try:
            remote = set(api.list_repo_files(args.repo_id, repo_type=args.repo_type))
        except Exception:
            remote = set()
        local = {p.relative_to(ROOT).as_posix() for p in files}
        # Anything remote that is not in the local plan goes, including paths the
        # IGNORE list covers. A file that is ignored locally is by definition not
        # part of the artifact, and leaving it on the Hub is worse than leaving
        # it out: `results/.stamps/*.done` in a clone tells run_all.py that
        # phases are already complete. Only the Hub's own dotfiles are spared.
        stale = sorted(f for f in remote - local if not f.startswith(".git"))
        if stale:
            print(f"\npruning {len(stale)} file(s) no longer in the source tree:")
            for f in stale[:20]:
                print(f"  - {f}")
            if len(stale) > 20:
                print(f"  ... and {len(stale) - 20} more")
            api.create_commit(
                repo_id=args.repo_id, repo_type=args.repo_type,
                operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
                commit_message=f"Prune {len(stale)} files removed from the source tree",
            )

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(ROOT),
        commit_message=args.message,
        ignore_patterns=ignore,
    )
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    print(f"\npushed to https://huggingface.co/{prefix}{args.repo_id}")
    if stale:
        print(f"pruned {len(stale)} stale file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
