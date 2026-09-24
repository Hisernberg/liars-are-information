# DGX resource audit

Recorded UTC: 2026-09-11T15:22:23.016042+00:00. This is a point-in-time host observation, not an exclusive resource reservation.

## Observed resources

| Item | Observation |
| --- | --- |
| Host | `aitopatom-bb03` |
| Architecture / kernel | `aarch64` / `6.17.0-1029-nvidia` |
| CPU | 20 logical CPUs; Cortex-X925, Cortex-A725 |
| CPU affinity available | 0–19 |
| CPU utilization | 3.1% over one second |
| Load average (1 / 5 / 15 min) | 0.88 / 0.82 / 0.69 |
| Host RAM total / available | 121.62 / 114.41 GiB |
| Swap used / total | 0.12 / 16.00 GiB |
| Free filesystem capacity | 182.23 GiB; home and tmp share the same filesystem capacity |
| GPU | NVIDIA GB10 |
| NVIDIA driver | 580.173.02 |
| GPU utilization / power / temperature | 5% / 12.64 W / 41°C |
| Existing GPU compute entry | rustdesk (246 MiB reported) |

`nvidia-smi` reports N/A for GPU total, used, and free memory. The JSON preserves those values as null; they must not be interpreted as zero or as an independently measured GPU capacity. Existing desktop and remote-access processes remain active, and no exclusive GPU availability is implied.

## Runtime

Python 3.12.11 at `/opt/miniforge3/bin/python3` already provides the scientific stack. NumPy, SciPy, pandas, Matplotlib, and PyTorch imported successfully with thread limits set to one. PyTorch reports CUDA availability and CUDA 13.0; the audit did not allocate CUDA tensors or run training. Package versions are preserved in the JSON. No dependency changes are necessary for a small NumPy simulation and analysis workflow.

## Recommended execution envelope

Run the initial simulations on the DGX CPU in one process, with a two-core affinity mask and one computational thread. Use `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, and `CUDA_VISIBLE_DEVICES=""` for this run only. Use nice +10; avoid nested worker pools. Choose two currently available cores from the permitted mask before starting.

Cap each job at 10 minutes wall time and aim for at most 2 GiB resident memory. For a small NumPy-only worker, a 4 GiB virtual-memory cap can provide an additional hard stop, but validate import compatibility first because address-space limits are not RSS limits. Record peak RSS, elapsed wall time, CPU affinity, all random seeds, configurations, versions, and output checksums. Stream or aggregate trial records instead of retaining all histories in memory.

Keep saved artifacts within 250 MiB initially and refuse to begin if free disk falls below 20 GiB or available host RAM falls below 16 GiB. These are conservative proposed thresholds, not controls enforced by this read-only audit. Recheck current load before execution. GPU acceleration is unnecessary for small tabular simulations; leaving compute allocation disabled avoids consuming GPU resources used by the active remote desktop session.

Do not stop other processes, change host services, reset the GPU, or modify global CUDA/Python packages. A GPU-specific experiment, if needed later, should use a separately bounded worker after the CPU results establish its value.

## Scope and evidence

The accompanying `dgx_resource_audit.json` contains sanitized measurements and proposed limits. No environment values, command lines, credentials, or process IDs are saved. No experiments or package installations were performed by the audit. No `AGENTS.md` was found in `/`, `/home`, `/home/urad`, or the task directory at audit time.
