"""Model registry: parsing, sequential-only residency, budget, and fit checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aip.models.registry import BudgetError, FitError, ModelRegistry, load_registry

#: A small fixture card. The synthetic fit-arithmetic tests below use this
#: because a 22.49 GiB budget is what makes "does a 14B model fit" a meaningful
#: question; it is deliberately NOT the card this study runs on.
SMALL_CARD_GIB = 22.49

#: The real device: NVIDIA H100 NVL, 93.10 GiB usable via
#: torch.cuda.mem_get_info (nvidia-smi board total 93.58 GiB). The shipped
#: registry must fit THIS, not the fixture card.
H100_GIB = 93.10


def _registry(**overrides) -> ModelRegistry:
    base = {
        "residency": "sequential",
        "max_total_utilization": 0.90,
        "min_slack_gib": 4.0,
        "models": {
            "a": {
                "hf_id": "org/a",
                "role": "r",
                "gpu_memory_utilization": 0.40,
                "max_model_len": 4096,
                "params_b": 8.0,
                "arch": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
            },
            "b": {
                "hf_id": "org/b",
                "role": "r",
                "gpu_memory_utilization": 0.40,
                "max_model_len": 4096,
                "params_b": 7.0,
                "arch": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
            },
        },
    }
    base.update(overrides)
    return ModelRegistry.model_validate(base)


class TestParsing:
    def test_name_injected_from_key(self) -> None:
        assert _registry().get("a").name == "a"

    def test_defaults(self) -> None:
        entry = _registry().get("a")
        assert entry.dtype == "bfloat16"
        assert entry.generation.temperature == 0.0
        assert entry.enabled

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _registry(
                models={
                    "a": {
                        "hf_id": "org/a",
                        "role": "r",
                        "gpu_memory_utilization": 0.4,
                        "max_model_len": 4096,
                        "params_b": 1.0,
                        "typo_field": 1,
                    }
                }
            )

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            _registry().get("nope")


class TestSequentialOnly:
    """Co-resident mode is removed, not merely defaulted off."""

    def test_coresident_config_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sequential-only"):
            _registry(residency="coresident")

    def test_any_other_residency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _registry(residency="parallel")

    def test_default_is_sequential(self) -> None:
        registry = ModelRegistry.model_validate(
            {
                "models": {
                    "a": {
                        "hf_id": "org/a",
                        "role": "r",
                        "gpu_memory_utilization": 0.9,
                        "max_model_len": 4096,
                        "params_b": 1.0,
                        "arch": {"n_layers": 4, "n_kv_heads": 2, "head_dim": 64},
                    }
                }
            }
        )
        assert registry.residency == "sequential"

    def test_sum_over_budget_is_fine_when_sequential(self) -> None:
        """The whole point: two 0.9 models never share the device."""
        registry = _registry(
            models={
                "a": {
                    "hf_id": "org/a",
                    "role": "r",
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 4096,
                    "params_b": 8.0,
                    "arch": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
                },
                "b": {
                    "hf_id": "org/b",
                    "role": "r",
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 4096,
                    "params_b": 8.0,
                    "arch": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
                },
            }
        )
        assert registry.validate_budget() == pytest.approx(0.90)

    def test_single_model_over_budget_rejected(self) -> None:
        with pytest.raises(BudgetError, match="above the budget"):
            _registry(
                models={
                    "a": {
                        "hf_id": "org/a",
                        "role": "r",
                        "gpu_memory_utilization": 0.99,
                        "max_model_len": 4096,
                        "params_b": 8.0,
                        "arch": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
                    }
                }
            )


class TestKVCacheMath:
    def test_kv_bytes_per_token(self) -> None:
        # 2 (K and V) * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072
        assert _registry().get("a").kv_bytes_per_token() == 131072

    def test_kv_scales_with_context(self) -> None:
        entry = _registry().get("a")
        assert entry.kv_cache_bytes(4096) == 131072 * 4096
        assert entry.kv_cache_bytes(2048) == entry.kv_cache_bytes(4096) // 2

    def test_kv_defaults_to_max_model_len(self) -> None:
        entry = _registry().get("a")
        assert entry.kv_cache_bytes() == entry.kv_cache_bytes(entry.max_model_len)

    def test_grouped_query_attention_is_cheaper(self) -> None:
        """KV cost tracks kv_heads, not attention heads -- the GQA saving."""
        gqa = _registry().get("a")
        mha = _registry(
            models={
                "a": {
                    "hf_id": "org/a",
                    "role": "r",
                    "gpu_memory_utilization": 0.4,
                    "max_model_len": 4096,
                    "params_b": 8.0,
                    "arch": {"n_layers": 32, "n_kv_heads": 32, "head_dim": 128},
                }
            }
        ).get("a")
        assert mha.kv_bytes_per_token() == 4 * gqa.kv_bytes_per_token()

    def test_missing_arch_yields_none(self) -> None:
        entry = _registry(
            models={
                "a": {
                    "hf_id": "org/a",
                    "role": "r",
                    "gpu_memory_utilization": 0.4,
                    "max_model_len": 4096,
                    "params_b": 8.0,
                }
            }
        ).get("a")
        assert entry.kv_bytes_per_token() is None
        assert entry.kv_cache_bytes() is None


class TestFitCheck:
    def _one(self, params_b: float, **arch) -> ModelRegistry:
        return _registry(
            models={
                "m": {
                    "hf_id": "org/m",
                    "role": "r",
                    "gpu_memory_utilization": 0.90,
                    "max_model_len": 4096,
                    "params_b": params_b,
                    "arch": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, **arch},
                }
            }
        )

    def test_report_counts_weights_and_kv(self) -> None:
        report = self._one(8.0).fit_report(SMALL_CARD_GIB)["m"]
        assert report["weights_gib"] == pytest.approx(14.90, abs=0.05)
        assert report["kv_gib"] == pytest.approx(0.50, abs=0.01)
        assert report["total_gib"] == pytest.approx(report["weights_gib"] + report["kv_gib"])
        assert report["slack_gib"] == pytest.approx(SMALL_CARD_GIB - report["total_gib"])

    def test_small_model_passes(self) -> None:
        assert self._one(3.2).fit_report(SMALL_CARD_GIB)["m"]["passes"]

    def test_14b_fails_on_a_small_card(self) -> None:
        """Phi-4 14B: ~27 GiB of weights against a 22.49 GiB card."""
        registry = self._one(14.66, n_layers=40, n_kv_heads=10)
        assert not registry.fit_report(SMALL_CARD_GIB)["m"]["passes"]
        with pytest.raises(FitError, match="OVER|slack"):
            registry.check_fit(SMALL_CARD_GIB)

    def test_slack_floor_is_enforced(self) -> None:
        """A model that fits the card but leaves too little headroom still fails."""
        registry = self._one(9.6)  # ~17.9 GiB weights + 0.5 GiB KV on 22.49 GiB
        assert registry.fit_report(SMALL_CARD_GIB, min_slack_gib=1.0)["m"]["passes"]
        assert not registry.fit_report(SMALL_CARD_GIB, min_slack_gib=6.0)["m"]["passes"]

    def test_kv_at_full_context_is_what_counts(self) -> None:
        """Halving max_model_len halves the KV term and raises slack."""
        long_ctx = self._one(9.0).fit_report(SMALL_CARD_GIB)["m"]
        short = _registry(
            models={
                "m": {
                    "hf_id": "org/m",
                    "role": "r",
                    "gpu_memory_utilization": 0.90,
                    "max_model_len": 2048,
                    "params_b": 9.0,
                    "arch": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
                }
            }
        ).fit_report(SMALL_CARD_GIB)["m"]
        assert short["kv_gib"] == pytest.approx(long_ctx["kv_gib"] / 2)
        assert short["slack_gib"] > long_ctx["slack_gib"]

    def test_missing_arch_fails_closed(self) -> None:
        """No arch block means the KV cache is unknown, which is not a pass."""
        registry = _registry(
            models={
                "m": {
                    "hf_id": "org/m",
                    "role": "r",
                    "gpu_memory_utilization": 0.4,
                    "max_model_len": 4096,
                    "params_b": 1.0,
                }
            }
        )
        assert not registry.fit_report(SMALL_CARD_GIB)["m"]["passes"]
        with pytest.raises(FitError, match="arch"):
            registry.check_fit(SMALL_CARD_GIB)

    def test_vllm_budget_is_tighter_than_the_card(self) -> None:
        """gpu_memory_utilization < 1 means vLLM sees less than the card has."""
        report = self._one(8.0).fit_report(SMALL_CARD_GIB)["m"]
        assert report["vllm_budget_gib"] == pytest.approx(SMALL_CARD_GIB * 0.90)
        assert report["vllm_slack_gib"] < report["slack_gib"]


class TestSwarmComposition:
    def test_homogeneous_swarms(self) -> None:
        assert _registry().homogeneous_swarms() == [("a",), ("b",)]

    def test_all_pairs(self) -> None:
        assert _registry().all_pairs() == [("a", "b")]

    def test_subset_validates(self) -> None:
        assert [e.name for e in _registry().subset(["a"])] == ["a"]

    def test_hash_is_stable_and_sensitive(self) -> None:
        assert _registry().hash() == _registry().hash()
        assert _registry().hash() != _registry(max_total_utilization=0.5).hash()


class TestShippedConfig:
    def test_loads_and_validates(self, registry_path: Path) -> None:
        registry = load_registry(registry_path)
        assert registry.residency == "sequential"
        assert len(registry.enabled_names) >= 2

    def test_every_enabled_model_fits_this_h100(self, registry_path: Path) -> None:
        load_registry(registry_path).check_fit(H100_GIB)

    def test_no_enabled_model_is_gated(self, registry_path: Path) -> None:
        registry = load_registry(registry_path)
        gated = [n for n in registry.enabled_names if registry.get(n).gated]
        assert not gated, f"enabled but gated (needs authorized HF_TOKEN): {gated}"

    def test_every_enabled_model_is_bf16(self, registry_path: Path) -> None:
        """No quantisation anywhere: it would shift Phase B's correlation structure."""
        registry = load_registry(registry_path)
        for name in registry.enabled_names:
            assert registry.get(name).dtype == "bfloat16", name

    def test_context_is_at_least_4096(self, registry_path: Path) -> None:
        """Context must not be shrunk below 4096 to force a fit."""
        registry = load_registry(registry_path)
        for name in registry.enabled_names:
            assert registry.get(name).max_model_len >= 4096, name

    def test_every_model_declares_arch(self, registry_path: Path) -> None:
        registry = load_registry(registry_path)
        for name in registry.models:
            assert registry.get(name).arch is not None, name

    def test_oversized_entries_are_disabled(self, registry_path: Path) -> None:
        """Anything that fails the fit check must not be enabled."""
        registry = load_registry(registry_path)
        report = registry.fit_report(H100_GIB, names=list(registry.models))
        for name, r in report.items():
            if not r["passes"]:
                assert not registry.get(name).enabled, f"{name} fails the fit check but is enabled"

    def test_vllm_kwargs_shape(self, registry_path: Path) -> None:
        registry = load_registry(registry_path)
        kwargs = registry.get(registry.enabled_names[0]).vllm_kwargs()
        assert kwargs["dtype"] == "bfloat16"
        assert 0 < kwargs["gpu_memory_utilization"] <= 0.90
