#!/usr/bin/env python3
"""GPU + Hub preflight. Run this before downloading a single weight file.

Three checks, all of which must pass for the enabled model set:

1. **Fit.** bf16 weight size plus KV cache at the model's full ``max_model_len``
   against the physical card, requiring ``min_slack_gib`` of headroom for
   activations, CUDA graphs, and allocator fragmentation.
2. **Architecture agreement.** The ``arch`` block recorded in the registry is
   re-fetched from the Hub's ``config.json`` and compared, so the KV arithmetic
   can never drift from the actual checkpoint. ``params_b`` is likewise checked
   against the Hub's safetensors parameter count.
3. **Access.** Gated status, plus an authenticated HEAD against ``config.json``
   to prove the token can actually read the repo. A 401/403 discovered here
   costs a second; discovered mid-download it costs the download.

Exit status is non-zero if any *enabled* model fails. Disabled models are
reported for the record but never block.

    python scripts/preflight.py --registry configs/models.yaml
    python scripts/preflight.py --all          # include disabled entries
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aip.harness.logging import configure_logging  # noqa: E402
from aip.models.registry import GIB, ModelRegistry, load_registry  # noqa: E402

HUB = "https://huggingface.co"


def detect_vram_gib() -> tuple[float, str]:
    """Usable VRAM of GPU 0.

    Prefers ``torch.cuda.mem_get_info``, which reports what a process can
    actually allocate, over nvidia-smi's board total -- the two differ by a few
    hundred MiB of firmware/ECC reservation, and on a card this small that gap
    is the difference between a model fitting and not.  Falls back to nvidia-smi
    when torch is absent (the preflight must run on a CPU-only checkout).
    """
    name = "unknown"
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    smi_gib = None
    if proc.returncode == 0 and proc.stdout.strip():
        name, mib = (p.strip() for p in proc.stdout.strip().splitlines()[0].split(","))
        smi_gib = int(mib) / 1024

    try:
        import torch

        if torch.cuda.is_available():
            _, total = torch.cuda.mem_get_info()
            return total / GIB, f"{name} (usable; nvidia-smi board total {smi_gib:.2f} GiB)"
    except Exception:  # noqa: BLE001 - torch optional here
        pass

    if smi_gib is None:
        raise RuntimeError(f"nvidia-smi failed: {proc.stderr.strip() or proc.returncode}")
    return smi_gib, name


def _request(url: str, token: str | None, method: str = "GET") -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read() if method == "GET" else b""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:400]
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, str(exc).encode()[:400]


def hub_config(hf_id: str, token: str | None) -> dict[str, Any] | None:
    status, body = _request(f"{HUB}/{hf_id}/resolve/main/config.json", token)
    if status != 200:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def hub_metadata(hf_id: str, token: str | None) -> dict[str, Any]:
    status, body = _request(f"{HUB}/api/models/{hf_id}?expand[]=safetensors&expand[]=gated", token)
    if status != 200:
        return {"_status": status}
    try:
        data = json.loads(body)
        data["_status"] = status
        return data
    except json.JSONDecodeError:
        return {"_status": status}


def arch_from_config(cfg: dict[str, Any]) -> dict[str, int] | None:
    """Pull (n_layers, n_kv_heads, head_dim) out of a HF config."""
    text = cfg.get("text_config", cfg)

    def pick(*keys: str) -> Any:
        for key in keys:
            if text.get(key) is not None:
                return text[key]
        return None

    n_layers = pick("num_hidden_layers", "n_layer")
    n_kv_heads = pick("num_key_value_heads", "num_attention_heads")
    n_heads = pick("num_attention_heads")
    head_dim = pick("head_dim")
    if head_dim is None and text.get("hidden_size") and n_heads:
        head_dim = text["hidden_size"] // n_heads
    if not (n_layers and n_kv_heads and head_dim):
        return None
    return {"n_layers": int(n_layers), "n_kv_heads": int(n_kv_heads), "head_dim": int(head_dim)}


def check_access(hf_id: str, token: str | None) -> tuple[bool, str]:
    status, body = _request(f"{HUB}/{hf_id}/resolve/main/config.json", token)
    if status == 200:
        return True, "ok"
    if status in (401, 403):
        return False, f"HTTP {status} (gated; token not authorized for this repo)"
    return False, f"HTTP {status}: {body.decode(errors='replace')[:120]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--all", action="store_true", help="include disabled entries")
    parser.add_argument("--vram-gib", type=float, default=None, help="override detected VRAM")
    parser.add_argument("--min-slack-gib", type=float, default=None)
    parser.add_argument("--no-network", action="store_true", help="skip Hub verification")
    args = parser.parse_args()

    configure_logging(level="WARNING")
    registry: ModelRegistry = load_registry(args.registry)
    token = os.environ.get("HF_TOKEN") or None

    if args.vram_gib is not None:
        vram, gpu_name = args.vram_gib, "(override)"
    else:
        vram, gpu_name = detect_vram_gib()

    names = list(registry.models) if args.all else list(registry.enabled_names)
    floor = args.min_slack_gib if args.min_slack_gib is not None else registry.min_slack_gib
    report = registry.fit_report(vram, names=names, min_slack_gib=floor)

    print(f"\nGPU: {gpu_name}  |  {vram:.2f} GiB  |  residency={registry.residency}")
    print(f"Slack floor: {floor:.1f} GiB   (weights + KV at full max_model_len must leave this)")
    print(f"HF_TOKEN: {'present' if token else 'ABSENT'}\n")

    header = (
        f"{'model':<22}{'params':>9}{'ctx':>6}{'wt GiB':>9}{'KV/tok':>9}"
        f"{'KV GiB':>8}{'total':>8}{'slack':>8}  fit"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        r = report[name]
        entry = registry.get(name)
        flag = "PASS" if r["passes"] else "FAIL"
        if not entry.enabled:
            flag += " (disabled)"
        kv_tok = f"{r['kv_per_token_kib']:.0f}KiB" if r["kv_per_token_kib"] else "?"
        print(
            f"{name:<22}{r['params_b']:>8.2f}B{r['max_model_len']:>6}"
            f"{r['weights_gib']:>9.2f}{kv_tok:>9}"
            f"{r['kv_gib']:>8.2f}{r['total_gib']:>8.2f}{r['slack_gib']:>8.2f}  {flag}"
        )

    # -- Hub verification -------------------------------------------------
    problems: list[str] = []
    if not args.no_network:
        print(f"\n{'model':<22}{'access':<52}{'arch match':<12}{'params match'}")
        print("-" * 100)
        for name in names:
            entry = registry.get(name)
            ok, detail = check_access(entry.hf_id, token)
            meta = hub_metadata(entry.hf_id, token)
            gated = meta.get("gated", "?")

            arch_match = "n/a"
            params_match = "n/a"
            if ok:
                cfg = hub_config(entry.hf_id, token)
                hub_arch = arch_from_config(cfg) if cfg else None
                if hub_arch and entry.arch:
                    declared = {
                        "n_layers": entry.arch.n_layers,
                        "n_kv_heads": entry.arch.n_kv_heads,
                        "head_dim": entry.arch.head_dim,
                    }
                    if hub_arch == declared:
                        arch_match = "yes"
                    else:
                        arch_match = f"NO {hub_arch} != {declared}"
                        problems.append(f"{name}: arch disagrees with Hub -> {arch_match}")
                elif entry.arch is None:
                    arch_match = "NO arch block"
                    problems.append(f"{name}: registry has no arch block")

                total = (meta.get("safetensors") or {}).get("total")
                if total:
                    declared_params = entry.params_b * 1e9
                    delta = abs(total - declared_params) / total
                    params_match = "yes" if delta < 0.01 else f"NO hub={total / 1e9:.3f}B"
                    if delta >= 0.01:
                        problems.append(
                            f"{name}: params_b={entry.params_b} but Hub reports {total / 1e9:.3f}B"
                        )
            elif entry.enabled:
                problems.append(f"{name}: not downloadable -> {detail}")

            access = f"{'OK' if ok else detail}  [gated={gated}]"
            print(f"{name:<22}{access:<52}{arch_match:<12}{params_match}")

    # -- verdict ----------------------------------------------------------
    failed_fit = [n for n in names if not report[n]["passes"] and registry.get(n).enabled]
    print()
    for name in names:
        r = report[name]
        if not r["passes"]:
            over = -r["slack_gib"] if r["slack_gib"] < 0 else None
            detail = (
                f"{over:.2f} GiB OVER the card"
                if over
                else f"slack {r['slack_gib']:.2f} GiB < {floor:.1f} GiB floor"
            )
            state = "enabled" if registry.get(name).enabled else "disabled"
            print(f"FIT FAIL [{state}] {name}: {detail}")
    for problem in problems:
        print(f"HUB ISSUE  {problem}")

    if failed_fit or any(p for p in problems if "not downloadable" in p):
        print("\nPREFLIGHT FAILED. Do not download.")
        return 1

    tight = [n for n in names if report[n]["passes"] and report[n]["slack_gib"] < floor + 1.0]
    if tight:
        print(f"\nPREFLIGHT PASSED, but marginal (<1 GiB above the floor): {tight}")
    else:
        print("\nPREFLIGHT PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
