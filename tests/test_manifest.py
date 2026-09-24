"""Run manifests: hashing, provenance capture, serialization."""

from __future__ import annotations

import json
from pathlib import Path

from aip.harness.manifest import Manifest, config_hash, git_commit, gpu_info, text_hash


class TestHashing:
    def test_config_hash_is_order_invariant(self) -> None:
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_config_hash_is_sensitive(self) -> None:
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_text_hash_stable(self) -> None:
        assert text_hash("prompt") == text_hash("prompt")
        assert text_hash("prompt") != text_hash("prompt ")


class TestProvenance:
    def test_git_commit_never_raises(self, tmp_path: Path) -> None:
        assert isinstance(git_commit(tmp_path), str)

    def test_gpu_info_returns_list(self) -> None:
        assert isinstance(gpu_info(), list)


class TestManifest:
    def test_create_and_write(self, tmp_path: Path) -> None:
        manifest = Manifest.create(
            phase="phase_a", config={"n_tasks": 5}, seeds={"task_selection": 1}
        )
        assert manifest.status == "running"
        assert manifest.run_id.startswith("phase_a-")

        manifest.finish(status="ok", n_rows=5)
        path = manifest.write(tmp_path / "manifest.json")
        payload = json.loads(path.read_text())
        assert payload["status"] == "ok"
        assert payload["notes"]["n_rows"] == 5
        assert payload["seeds"] == {"task_selection": 1}
        assert payload["finished_at"]

    def test_same_config_same_hash(self) -> None:
        a = Manifest.create(phase="p", config={"x": 1}, seeds={})
        b = Manifest.create(phase="p", config={"x": 1}, seeds={})
        assert a.config_hash == b.config_hash
        assert a.run_id != b.run_id  # timestamped
