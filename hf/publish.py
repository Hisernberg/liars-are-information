#!/usr/bin/env python3
"""Publish the v3 release to the Hugging Face Hub (additive; never deletes).

Two targets, pick one or both:

* ``--bucket Nabidnur/Liars_Are_Information-bucket`` -- uploads the release
  under a new dated prefix ``releases/<release>/`` (the bucket's convention:
  earlier releases and root objects are left untouched), then replaces the
  bucket-root ``README.md`` index only if ``--update-root-readme`` is given.
* ``--dataset-repo <user>/<name>`` -- pushes the same tree to a new dataset
  repo as one commit (the dataset card is ``hf/DATASET_CARD.md``).
* ``--space-repo <user>/<name>`` -- creates a static Space (``hf/space/``) that
  plays the explainer videos and shows the figures and headline results.

The token is read from ``HF_TOKEN`` (or ``--token-file``); it is never written
anywhere. ``--dry-run`` prints the plan and uploads nothing.

    HF_TOKEN=hf_... python hf/publish.py --bucket Nabidnur/Liars_Are_Information-bucket --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "lai-v3.1-race"
INCLUDE = ["README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "src", "tests", "experiments", "configs", "docs",
           "data/cache", "data/cache_adversarial", "data/cache_t07", "data/live_cache", "data/derived",
           "results", "figures", "media", "paper", "hf", "provenance"]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "film_parts"}
SPACE_ASSETS = ["media/*.mp4", "media/gif/*.gif", "media/*.gif", "figures/*.png"]


def files() -> list[Path]:
    out = []
    for entry in INCLUDE:
        p = ROOT / entry
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and not (EXCLUDE_PARTS & set(f.parts))]
    return sorted(out)


def manifest(paths: list[Path]) -> dict:
    return {
        "release": RELEASE,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in paths},
    }


def token(args) -> str:
    tok = os.environ.get("HF_TOKEN") or (Path(args.token_file).read_text().strip() if args.token_file else "")
    if not tok.startswith("hf_") or len(tok) < 30:
        sys.exit("HF_TOKEN missing or not shaped like a Hugging Face token (hf_...).")
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--space-repo", default=None)
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--update-root-readme", action="store_true")
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = files()
    man = manifest(paths)
    total = sum(v["bytes"] for v in man["files"].values())
    print(f"{len(paths)} files, {total / 1e6:.1f} MB -> release {args.release}")
    if args.dry_run:
        for p in paths[:25]:
            print("  ", p.relative_to(ROOT))
        print("   ...")
        if args.space_repo:
            import fnmatch
            assets = [f for f in paths if any(fnmatch.fnmatch(str(f.relative_to(ROOT)), pat) for pat in SPACE_ASSETS)]
            print(f"space: page from hf/space/ + {len(assets)} assets "
                  f"({sum(a.stat().st_size for a in assets) / 1e6:.1f} MB)")
        return
    from huggingface_hub import HfApi

    tok = token(args)
    api = HfApi(token=tok)
    who = api.whoami()
    print("authenticated as", who.get("name"))
    blob = json.dumps(man, indent=1).encode()
    if args.bucket:
        prefix = f"releases/{args.release}"
        adds = [(str(p), f"{prefix}/{p.relative_to(ROOT)}") for p in paths]
        adds.append((blob, f"{prefix}/FILE_MANIFEST.json"))
        for i in range(0, len(adds), 200):
            api.batch_bucket_files(args.bucket, add=adds[i:i + 200])
            print(f"  uploaded {min(i + 200, len(adds))}/{len(adds)}")
        if args.update_root_readme:
            api.batch_bucket_files(args.bucket, add=[(str(ROOT / "hf" / "BUCKET_ROOT_README.md"), "README.md")])
        print(f"done: https://huggingface.co/buckets/{args.bucket}/tree/{prefix}")
    if args.dataset_repo:
        api.create_repo(args.dataset_repo, repo_type="dataset", exist_ok=True)
        staging = ROOT / "hf" / "DATASET_CARD.md"
        api.upload_folder(folder_path=str(ROOT), repo_id=args.dataset_repo, repo_type="dataset",
                          allow_patterns=[str(p.relative_to(ROOT)) for p in paths],
                          commit_message=f"{args.release}: RACE release")
        api.upload_file(path_or_fileobj=str(staging), path_in_repo="README.md", repo_id=args.dataset_repo,
                        repo_type="dataset", commit_message="dataset card")
        api.upload_file(path_or_fileobj=blob, path_in_repo="FILE_MANIFEST.json", repo_id=args.dataset_repo,
                        repo_type="dataset", commit_message="file manifest")
        print(f"done: https://huggingface.co/datasets/{args.dataset_repo}")
    if args.space_repo:
        api.create_repo(args.space_repo, repo_type="space", space_sdk="static", exist_ok=True)
        # The page, then every video (MP4, played by the page), GIF and figure, each as a single commit.
        api.upload_folder(folder_path=str(ROOT / "hf" / "space"), repo_id=args.space_repo, repo_type="space",
                          ignore_patterns=["*.template.*"], commit_message=f"{args.release}: demo page")
        api.upload_folder(folder_path=str(ROOT), repo_id=args.space_repo, repo_type="space",
                          allow_patterns=SPACE_ASSETS, commit_message=f"{args.release}: videos and figures")
        print(f"done: https://huggingface.co/spaces/{args.space_repo}")


if __name__ == "__main__":
    main()
