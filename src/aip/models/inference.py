"""vLLM offline inference wrapper (``vllm.LLM`` + batched ``generate``).

Sequential residency is enforced structurally: :class:`ModelRunner` is a context
manager that loads exactly one model and releases it -- ``del`` plus
``gc.collect()`` plus ``torch.cuda.empty_cache()`` -- on exit.  Phase A iterates
models in the outer loop so at most one is ever resident.

Chat templates are applied through each model's own tokenizer rather than
``llm.chat`` so that the exact prompt string is available for hashing, and so
Qwen3-style thinking modes can be switched off explicitly.

Token logprobs come back aligned to *character offsets* in the completion, which
is what :mod:`aip.models.confidence` needs to restrict confidence to the answer
span rather than the whole chain of thought.

.. warning::
   vLLM starts its engine with the ``spawn`` start method, so the child
   re-imports the entry module. Any script that constructs a
   :class:`ModelRunner` **must** guard its entry point with
   ``if __name__ == "__main__":``, or the second model load dies with
   "An attempt has been made to start a new process before the current process
   has finished its bootstrapping phase".
"""

from __future__ import annotations

import gc
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from aip.harness.logging import get_logger
from aip.models.confidence import TokenLogprob
from aip.models.registry import ModelEntry

log = get_logger(__name__)

ChatMessage = dict[str, str]


@dataclass(slots=True)
class GenerationResult:
    """One completion with token-level logprobs and character offsets."""

    text: str
    tokens: list[TokenLogprob] = field(default_factory=list)
    finish_reason: str | None = None
    n_prompt_tokens: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def truncated(self) -> bool:
        """True when generation stopped on the token budget, not on an EOS.

        A truncated completion usually means the answer marker was never emitted,
        so this is tracked per cell as a distinct failure mode from a model that
        answered in the wrong format.
        """
        return self.finish_reason == "length"


def align_tokens(
    tokenizer: Any, token_ids: Sequence[int], logprobs: Sequence[float], text: str
) -> list[TokenLogprob]:
    """Attach exact character offsets to each generated token.

    Per-token detokenization is not simply ``decode([id])``: BPE and
    SentencePiece merges make a token's rendered text depend on its left
    context.  Cumulative decoding is exact, and the result is verified against
    the full completion before being returned -- if the pieces do not
    reconstruct ``text``, offsets are dropped rather than silently misreported,
    because a wrong span would corrupt the confidence surface without any
    visible error.
    """
    pieces: list[str] = []
    previous = ""
    for i in range(1, len(token_ids) + 1):
        current = tokenizer.decode(token_ids[:i], skip_special_tokens=True)
        pieces.append(current[len(previous) :])
        previous = current

    if "".join(pieces) != text:
        log.warning(
            "tokens.offset_mismatch",
            reconstructed_len=len("".join(pieces)),
            text_len=len(text),
        )
        return [
            TokenLogprob(text="", logprob=float(lp), start=0, end=0)
            for lp in logprobs[: len(token_ids)]
        ]

    tokens: list[TokenLogprob] = []
    cursor = 0
    for piece, logprob in zip(pieces, logprobs, strict=False):
        tokens.append(
            TokenLogprob(text=piece, logprob=float(logprob), start=cursor, end=cursor + len(piece))
        )
        cursor += len(piece)
    return tokens


class ModelRunner:
    """One resident vLLM model. Use as a context manager.

    ``with ModelRunner(entry) as runner: runner.generate(...)``
    """

    def __init__(
        self,
        entry: ModelEntry,
        enforce_eager: bool = False,
        deterministic: bool = True,
        attention_backend: str = "TRITON_ATTN",
        **overrides: Any,
    ) -> None:
        self.entry = entry
        self.enforce_eager = enforce_eager
        self.deterministic = deterministic
        self.attention_backend = attention_backend
        self.overrides = overrides
        self._llm: Any = None
        self._tokenizer: Any = None
        self.load_seconds: float = 0.0

    # -- lifecycle -------------------------------------------------------

    def load(self) -> ModelRunner:
        from vllm import LLM

        kwargs = self.entry.vllm_kwargs()
        overrides = self.entry.hf_text_config_overrides
        if overrides:
            # Applied as a callable, not a dict: vLLM's dict form patches the
            # OUTER config, while the convertor that needs these values reads
            # hf_text_config -- the nested one.
            values = dict(overrides)

            def _patch(config: Any) -> Any:
                text = getattr(config, "text_config", config)
                for obj in (config, text):
                    try:
                        object.__setattr__(
                            obj, "allow_global_per_layer_attribute_access", True
                        )
                    except Exception:  # noqa: BLE001 - the nested one is what matters
                        pass
                for key, value in values.items():
                    setattr(text, key, value)
                return config

            kwargs["hf_overrides"] = _patch
        if self.deterministic:
            # Without this, vLLM's kernels are not batch-invariant: two runs of
            # the *same* batch return logprobs that differ at ~1e-6, and output
            # text can diverge when engine state differs. The cache is written
            # once and read by every later phase, so bit-level reproducibility
            # is what makes a cached number citable. Costs roughly 1.5x latency.
            # batch_invariant reads this at import time in the engine child
            # process, which inherits the environment, and it requires an
            # explicitly chosen attention backend.
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            kwargs["attention_backend"] = self.attention_backend
        kwargs.update(self.overrides)
        if self.enforce_eager:
            kwargs["enforce_eager"] = True

        log.info("model.loading", model=self.entry.name, **_loggable(kwargs))
        started = time.perf_counter()
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self.load_seconds = time.perf_counter() - started
        log.info("model.loaded", model=self.entry.name, seconds=round(self.load_seconds, 1))
        return self

    def close(self) -> None:
        """Release the model and its VRAM. Safe to call twice.

        Sequential residency makes this load-bearing rather than tidy: vLLM v1
        runs the engine in a *separate process*, and if that process is not shut
        down the next model in the loop meets a card that is still occupied.
        The engine is therefore shut down explicitly before dropping the handle,
        and the freed VRAM is verified against the device afterwards.
        """
        if self._llm is None:
            return
        log.info("model.releasing", model=self.entry.name)
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        before = _gpu_used_gib(torch)

        # vLLM v1 keeps the engine in a child process; ask it to stop first.
        for owner, attr in (
            (self._llm, "shutdown"),
            (getattr(self._llm, "llm_engine", None), "shutdown"),
        ):
            if owner is None:
                continue
            fn = getattr(owner, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - best effort teardown
                    log.warning(
                        "model.shutdown_failed", model=self.entry.name, error=str(exc)[:200]
                    )

        del self._llm
        self._llm = None
        self._tokenizer = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        clean, leftover = _wait_for_engine_exit()
        after = _gpu_used_gib(torch)
        log.info(
            "model.released",
            model=self.entry.name,
            engine_exited=clean,
            device_used_gib=None if after is None else round(after, 2),
            before_gib=None if before is None else round(before, 2),
        )
        if not clean:
            # Expected: the child typically needs ~90s. Informational, because
            # the next load's memory profiling is the check that would actually
            # fail if the card were still occupied.
            log.info(
                "model.engine_exit_pending",
                model=self.entry.name,
                leftover_pids=sorted(leftover),
                note="engine child usually needs ~90s; next load will fail loudly if truly blocked",
            )

    def __enter__(self) -> ModelRunner:
        return self.load()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- prompting -------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("model is not loaded")
        return self._tokenizer

    def encode_messages(
        self, messages: list[ChatMessage], enable_thinking: bool | None = False
    ) -> list[int]:
        """Apply the chat template and return **token ids**.

        Tokenizing the template directly, rather than rendering to a string and
        re-encoding, is the only correct path.  Re-encoding double-prepends BOS
        (vLLM adds its own), and for Mistral models vLLM uses
        ``MistralCommonTokenizer``, where a rendered string's ``<s>`` becomes
        literal text rather than the control token -- both of which made
        Ministral-8B emit an immediate EOS and score 0% extraction.

        ``enable_thinking=False`` is forwarded to templates that accept it so a
        hybrid-reasoning model does not spend its whole budget in a ``<think>``
        block; templates that reject the kwarg fall back to a plain call.
        """
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
        if enable_thinking is not None:
            try:
                out = self.tokenizer.apply_chat_template(
                    messages, enable_thinking=enable_thinking, **kwargs
                )
                return _as_token_ids(out)
            except TypeError:
                pass
        return _as_token_ids(self.tokenizer.apply_chat_template(messages, **kwargs))

    def render_prompt(self, messages: list[ChatMessage]) -> str:
        """Human-readable rendering, for logs and debugging only.

        Never feed this back to :meth:`generate` -- see :meth:`encode_messages`.
        """
        return self.tokenizer.decode(self.encode_messages(messages))

    # -- generation ------------------------------------------------------

    def generate(
        self,
        prompts: list[list[int]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> list[GenerationResult]:
        """Batched greedy generation with per-token logprobs.

        ``prompts`` are token id lists from :meth:`encode_messages`.
        """
        from vllm import SamplingParams

        if self._llm is None:
            raise RuntimeError("model is not loaded")

        gen = self.entry.generation
        params = SamplingParams(
            temperature=gen.temperature if temperature is None else temperature,
            top_p=gen.top_p,
            top_k=gen.top_k,
            max_tokens=gen.max_tokens if max_tokens is None else max_tokens,
            logprobs=gen.logprobs,
            stop=stop if stop is not None else (gen.stop or None),
            seed=seed if seed is not None else gen.seed,
        )
        outputs = self._llm.generate([{"prompt_token_ids": ids} for ids in prompts], params)

        results: list[GenerationResult] = []
        for prompt, output in zip(prompts, outputs, strict=True):
            completion = output.outputs[0]
            logprobs = _sampled_logprobs(completion)
            tokens = align_tokens(
                self.tokenizer, list(completion.token_ids), logprobs, completion.text
            )
            results.append(
                GenerationResult(
                    text=completion.text,
                    tokens=tokens,
                    finish_reason=completion.finish_reason,
                    n_prompt_tokens=len(prompt),
                )
            )
        return results


def _sampled_logprobs(completion: Any) -> list[float]:
    """Extract the logprob of each *sampled* token from a vLLM output.

    ``SamplingParams(logprobs=0)`` still returns the sampled token's own entry;
    the payload is a list of ``{token_id: Logprob}`` dicts, one per position.
    """
    raw = getattr(completion, "logprobs", None)
    if not raw:
        return []
    out: list[float] = []
    for token_id, entry in zip(completion.token_ids, raw, strict=False):
        if entry is None:
            out.append(float("nan"))
            continue
        record = entry.get(token_id)
        if record is None:
            # Fall back to the highest-ranked entry at this position.
            record = min(entry.values(), key=lambda r: getattr(r, "rank", 0) or 0)
        out.append(float(getattr(record, "logprob", record)))
    return out


def _gpu_compute_pids() -> set[int] | None:
    """PIDs currently holding GPU memory, via nvidia-smi; ``None`` if unavailable.

    This is the right instrument for "did the engine child exit?".
    ``torch.cuda.mem_get_info`` cannot answer it: called from the parent it
    reports the device total minus free as seen through the *parent's* context,
    which does not track another process releasing its allocation promptly. Using
    it made every model release look like a leak while the run was in fact fine.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return None
        return {int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _wait_for_engine_exit(timeout_s: float = 10.0) -> tuple[bool, set[int]]:
    """Briefly check whether the engine child has released the GPU.

    Returns ``(exited, leftover_pids)``. This is a *courtesy* check, not a
    barrier, and the distinction was earned empirically:

    * vLLM 0.17's engine child takes roughly 90 seconds to exit after shutdown
      is requested, so any short budget reports a leak that is not one;
    * blocking here is worse than useless -- the parent has to run its own
      multiprocessing teardown for the child to be reaped, so sitting in a poll
      loop delays the very exit it is waiting for;
    * in two full Phase A runs (16 cells, four models loaded back to back at up
      to 0.90 utilization) the next model has never failed to allocate, and the
      device returns to idle afterwards.

    So we look once, record what we saw, and let the next load proceed. If the
    card were genuinely still occupied, vLLM's own memory profiling would fail
    loudly at the next load -- which is the check that actually matters.
    """
    own = os.getpid()
    deadline = time.monotonic() + timeout_s
    leftover: set[int] = set()
    while True:
        pids = _gpu_compute_pids()
        if pids is None:
            return True, set()  # cannot observe; do not cry wolf
        leftover = pids - {own}
        if not leftover or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    return not leftover, leftover


def _gpu_used_gib(torch: Any) -> float | None:
    """VRAM currently allocated on device 0, in GiB, or ``None`` without CUDA."""
    if torch is None or not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024**3)


def _as_token_ids(out: Any) -> list[int]:
    """Normalise apply_chat_template output to a flat list of ids."""
    if hasattr(out, "input_ids"):
        out = out.input_ids
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def _loggable(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Scalar vLLM kwargs, renamed to avoid colliding with structlog's own keys.

    ``vllm_kwargs()`` uses ``model`` for the HF id, which would clash with the
    ``model=<registry name>`` field every log line already carries.
    """
    renames = {"model": "hf_id"}
    return {
        renames.get(k, k): v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))
    }
