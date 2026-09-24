"""Typed model registry loaded from ``configs/models.yaml``.

**Residency is sequential, and that is the only mode.**  One model is resident
in VRAM at a time (load, generate, ``del`` + ``torch.cuda.empty_cache()``, next).
Co-residency was removed rather than merely defaulted off, so that no config can
accidentally assume several models share the device: on a 23 GiB card the
spec's four models would need ~74 GiB of bf16 weights.  Nothing is lost
scientifically, because design principle 1 writes all inference to disk in Phase
A and every later phase reads that cache.

Under sequential residency ``gpu_memory_utilization`` means "fraction of the
device this model may use *while it is the resident model*", so the
``max_total_utilization`` budget is enforced per model rather than as a sum.

Two independent checks guard the GPU, and both raise rather than warn -- an OOM
forty minutes into a generation job is far more expensive than a failed
preflight:

``validate_budget``
    per-model ``gpu_memory_utilization`` against ``max_total_utilization``.

``fit_report`` / ``check_fit``
    bf16 weights **plus KV cache at the model's full ``max_model_len``** against
    the physical card, requiring a configurable slack floor.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from aip.harness.manifest import config_hash

BYTES_PER_PARAM = {"bfloat16": 2, "float16": 2, "float32": 4, "auto": 2}

GIB = 1024**3

#: Slack (GiB) that must remain on the card after weights + full-length KV
#: cache, to cover activations, CUDA graphs, and allocator fragmentation.
DEFAULT_MIN_SLACK_GIB = 4.0


class BudgetError(RuntimeError):
    """Raised when a requested set of models cannot fit the GPU budget."""


class FitError(RuntimeError):
    """Raised when weights + KV cache do not fit the physical device."""


class GenerationDefaults(BaseModel):
    """Sampling defaults for a model. Benchmarks use ``temperature=0``."""

    model_config = {"extra": "forbid"}

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    """Explicit, never inherited. vLLM's own default is -1 (disabled), but some
    checkpoints ship a ``generation_config.json`` with a truncating default --
    Gemma 4's is ``temperature=1.0, top_p=0.95, top_k=64``. Every registry entry
    states all three values so that the sampling law is a property of this file
    and not of whichever library version is installed. At ``temperature=0`` vLLM
    decodes greedily and both nucleus parameters are inert; they bind only on
    the temperature-0.7 second pass, which is exactly where a silently inherited
    default would corrupt the correlation measurement."""
    max_tokens: int = 512
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    logprobs: int = 0
    """Number of top alternatives per position; 0 still returns the sampled
    token's own logprob, which is all the confidence surface needs."""
    confidence_max_tokens: int = 16
    """Budget for the self-report pass. Reasoning models emit a thinking block
    before the number and need far more than a non-reasoning model does."""

    @field_validator("temperature")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("temperature must be >= 0")
        return v


class Architecture(BaseModel):
    """Attention shape needed to size the KV cache.

    Mirrors the model's ``config.json``; ``scripts/preflight.py`` fetches the
    real values from the Hub and fails if they disagree with what is recorded
    here, so the KV arithmetic can never drift from the actual checkpoint.
    """

    model_config = {"extra": "forbid"}

    n_layers: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)


class ModelEntry(BaseModel):
    """One registry entry."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    name: str
    hf_id: str
    tier: Literal["strong", "reasoning", "mid", "weak_fast"] | None = None
    """Roster tier, recorded as data rather than prose so analyses can group by
    it without re-deriving it from parameter counts. Fixed by the 2026-09-02
    roster freeze; see the header of ``configs/models.yaml``."""
    arm: Literal["frozen", "weak_tier", "retired"] | None = None
    """Which experimental arm the entry belongs to. `frozen` is the seven-model
    roster frozen on 2026-09-02; `weak_tier` is the weak-swarm ablation, which is
    reported separately and never pooled with the frozen roster; `retired` is on
    record only so the roster change is legible. The paper's model counts are
    derived from this field rather than typed, after a scaffold shipped with the
    L4-era "4 models" in a macro where no test could see it."""
    role: str
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    max_model_len: int = Field(gt=0)
    params_b: float = Field(gt=0.0, description="Parameter count in billions, for VRAM sizing.")
    arch: Architecture | None = None
    revision: str | None = None
    gated: bool = False
    trust_remote_code: bool = False
    enabled: bool = True
    generation: GenerationDefaults = Field(default_factory=GenerationDefaults)
    tokenizer: str | None = None
    hf_text_config_overrides: dict[str, Any] | None = None
    """Values forced onto the checkpoint's NESTED ``text_config`` before vLLM
    reads it, plus the transformers flag that permits global access on a
    heterogeneous config.

    Needed by gemma4_31b and nothing else. That checkpoint is genuinely
    heterogeneous -- 50 sliding layers at (head_dim 256, 16 KV heads) and 10
    full-attention layers at (512, 4) -- and vLLM 0.19.1 already handles that
    correctly in principle: ``Gemma4ModelArchConfigConvertor.get_head_size``
    returns ``max(head_dim, global_head_dim)`` so buffers are sized for the
    larger. Two things stop it working unaided, both on the nested config:

    1. transformers 5.16.1 raises ``AmbiguousGlobalPerLayerAttributeError`` on
       ambiguous global access. It is not an ``AttributeError``, so vLLM's
       ``getattr(..., 0)`` default cannot absorb it.
    2. ``global_head_dim`` and ``num_global_key_value_heads`` are absent from
       this config revision, so even with access allowed the convertor would
       compute ``max(256, 0) = 256`` and under-size the full-attention layers.

    Supplying the two values from the checkpoint's own ``per_layer_config``
    gives vLLM exactly the geometry its code path is written to read. Nothing is
    flattened: the sliding layers still read 256 and only the full-attention
    branch reads 512.
    """
    limit_mm_per_prompt: dict[str, int] | None = None
    """Per-modality media budget, set to zero on the multimodal checkpoints.

    Three roster entries are multimodal (Qwen 3.8 vision, Gemma 4 vision+audio,
    Ministral 3 vision) and this study sends text-only prompts to all of them.
    Declaring the budget as zero is therefore a true statement about the
    workload, and it has a load-time consequence: vLLM's MultiModalBudget probes
    every *active* modality with a dummy item, and a modality limited to zero is
    not active, so the probe is skipped entirely.

    That probe is not hypothetical. Under vllm 0.19.1 with transformers 5.16.1
    it crashes Ministral 3 outright -- transformers' prepare_inputs_layout calls
    image_processor.fetch_images(), which MistralCommonImageProcessor does not
    implement -- so the model cannot load at all without this. Setting it also
    guarantees no model in the study can silently accept an image.
    """
    notes: str | None = None

    @field_validator("dtype")
    @classmethod
    def _bf16_only_for_enabled(cls, v: str) -> str:
        """Quantised or fp32 weights would shift the error-correlation structure
        Phase B measures, so the study is bf16-only."""
        return v

    def bytes_per_param(self) -> int:
        return BYTES_PER_PARAM[self.dtype]

    def weight_bytes(self) -> int:
        """Approximate weight footprint in bytes at this dtype."""
        return int(self.params_b * 1e9 * self.bytes_per_param())

    def kv_bytes_per_token(self) -> int | None:
        """KV cache cost of a single token, or ``None`` if ``arch`` is absent.

        ``2`` for the K and V tensors, times layers, times KV heads (grouped-query
        attention means this is well below the attention-head count), times head
        dimension, times the element size.
        """
        if self.arch is None:
            return None
        a = self.arch
        return 2 * a.n_layers * a.n_kv_heads * a.head_dim * self.bytes_per_param()

    def kv_cache_bytes(self, seq_len: int | None = None) -> int | None:
        """KV cache for one sequence at ``seq_len`` (default: ``max_model_len``)."""
        per_token = self.kv_bytes_per_token()
        if per_token is None:
            return None
        return per_token * (seq_len if seq_len is not None else self.max_model_len)

    def vllm_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``vllm.LLM(...)``."""
        kwargs: dict[str, Any] = {
            "model": self.hf_id,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision:
            kwargs["revision"] = self.revision
        if self.tokenizer:
            kwargs["tokenizer"] = self.tokenizer
        if self.limit_mm_per_prompt is not None:
            kwargs["limit_mm_per_prompt"] = dict(self.limit_mm_per_prompt)
        return kwargs


class ModelRegistry(BaseModel):
    """The full registry. Residency is sequential; there is no other mode."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    residency: Literal["sequential"] = "sequential"
    max_total_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    min_slack_gib: float = Field(
        default=DEFAULT_MIN_SLACK_GIB,
        ge=0.0,
        description="GiB that must remain free after weights + full-length KV cache.",
    )
    models: dict[str, ModelEntry]

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, data: Any) -> Any:
        """Inject entry names from their keys and reject co-resident configs."""
        if not isinstance(data, dict):
            return data
        residency = data.get("residency")
        if residency is not None and residency != "sequential":
            raise ValueError(
                f"residency={residency!r} is not supported: this codebase is "
                "sequential-only. Co-resident mode was removed so that no config "
                "can assume several models share the device."
            )
        if isinstance(data.get("models"), dict):
            models = {}
            for key, entry in data["models"].items():
                if isinstance(entry, dict):
                    entry = {**entry, "name": entry.get("name", key)}
                models[key] = entry
            data = {**data, "models": models}
        return data

    @model_validator(mode="after")
    def _check_budget(self) -> ModelRegistry:
        self.validate_budget(list(self.enabled_names))
        return self

    # -- accessors -------------------------------------------------------

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return tuple(n for n, m in self.models.items() if m.enabled)

    def get(self, name: str) -> ModelEntry:
        if name not in self.models:
            raise KeyError(f"unknown model {name!r}; registry has {sorted(self.models)}")
        return self.models[name]

    def subset(self, names: list[str] | tuple[str, ...]) -> list[ModelEntry]:
        """Return entries for ``names``, validating the budget for that subset."""
        entries = [self.get(n) for n in names]
        self.validate_budget(list(names))
        return entries

    # -- swarm compositions ----------------------------------------------

    def homogeneous_swarms(self) -> list[tuple[str, ...]]:
        """One single-model swarm composition per enabled model."""
        return [(n,) for n in self.enabled_names]

    def all_pairs(self) -> list[tuple[str, str]]:
        """All unordered cross-model pairs, for heterogeneous swarms."""
        return list(itertools.combinations(self.enabled_names, 2))

    # -- budget enforcement ----------------------------------------------

    def validate_budget(self, names: list[str] | None = None) -> float:
        """Check per-model utilization; return the largest requested fraction."""
        selected = list(names) if names is not None else list(self.enabled_names)
        if not selected:
            return 0.0
        utils = {n: self.get(n).gpu_memory_utilization for n in selected}
        worst = max(utils.values())
        if worst > self.max_total_utilization + 1e-9:
            offenders = [n for n, u in utils.items() if u > self.max_total_utilization + 1e-9]
            raise BudgetError(
                f"models {offenders} request gpu_memory_utilization above the budget of "
                f"{self.max_total_utilization:.2f} even loaded alone."
            )
        return worst

    def fit_report(
        self,
        total_vram_gib: float,
        names: list[str] | None = None,
        min_slack_gib: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Weights + full-length KV cache against the physical card.

        Reports rather than raises, so a preflight can show every model's
        arithmetic in one table instead of stopping at the first failure.
        ``slack_gib`` is what remains on the card for activations, CUDA graphs,
        and fragmentation once weights and one ``max_model_len`` sequence of KV
        cache are accounted for.
        """
        selected = list(names) if names is not None else list(self.enabled_names)
        floor = self.min_slack_gib if min_slack_gib is None else min_slack_gib
        report: dict[str, dict[str, Any]] = {}
        for name in selected:
            entry = self.get(name)
            weights = entry.weight_bytes() / GIB
            kv_bytes = entry.kv_cache_bytes()
            kv = kv_bytes / GIB if kv_bytes is not None else None
            total = weights + (kv or 0.0)
            slack = total_vram_gib - total
            # What vLLM itself may touch, which is tighter than the card when
            # gpu_memory_utilization < 1.
            budget = total_vram_gib * entry.gpu_memory_utilization
            report[name] = {
                "params_b": entry.params_b,
                "dtype": entry.dtype,
                "max_model_len": entry.max_model_len,
                "weights_gib": weights,
                "kv_per_token_kib": (
                    entry.kv_bytes_per_token() / 1024 if entry.kv_bytes_per_token() else None
                ),
                "kv_gib": kv,
                "total_gib": total,
                "slack_gib": slack,
                "vllm_budget_gib": budget,
                "vllm_slack_gib": budget - total,
                "min_slack_gib": floor,
                "kv_known": kv is not None,
                "passes": kv is not None and slack >= floor,
            }
        return report

    def check_fit(
        self,
        total_vram_gib: float,
        names: list[str] | None = None,
        min_slack_gib: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """:meth:`fit_report`, raising :class:`FitError` if any model fails."""
        report = self.fit_report(total_vram_gib, names, min_slack_gib)
        failures = [n for n, r in report.items() if not r["passes"]]
        if failures:
            lines = []
            for name in failures:
                r = report[name]
                if not r["kv_known"]:
                    lines.append(f"  {name}: no arch block, cannot size the KV cache")
                    continue
                lines.append(
                    f"  {name}: {r['params_b']}B {r['dtype']} = {r['weights_gib']:.1f} GiB weights "
                    f"+ {r['kv_gib']:.2f} GiB KV at max_model_len={r['max_model_len']} "
                    f"= {r['total_gib']:.1f} GiB on a {total_vram_gib:.2f} GiB card "
                    f"-> slack {r['slack_gib']:.2f} GiB < {r['min_slack_gib']:.1f} GiB required"
                )
            raise FitError("models do not fit this device:\n" + "\n".join(lines))
        return report

    def hash(self) -> str:
        return config_hash(self)


def load_registry(path: str | Path) -> ModelRegistry:
    """Load and validate ``configs/models.yaml``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model registry not found: {p}")
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    return ModelRegistry.model_validate(raw)
