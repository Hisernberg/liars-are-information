#!/usr/bin/env python3
"""README-friendly animated GIFs for every video in media/.

GitHub renders GIFs inline in a README but does not play MP4 files committed to
the repository, so every video gets a GIF twin under media/gif/ (the MP4s stay
for download and for the Hugging Face Space, which plays them). Each GIF is
width-limited and palette-optimised to stay well under GitHub's image limits.

    PYTHONPATH=src python experiments/make_gifs.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
OUT = MEDIA / "gif"
FF = imageio_ffmpeg.get_ffmpeg_exe()
LIMIT_MB = 8.0

# (source, output name, start seconds, duration seconds or None, fps, width)
JOBS = [
    ("film_parts/1_protocol.mp4", "film_1_protocol", 0, None, 8, 800),
    ("film_parts/2_learning.mp4", "film_2_learning", 0, None, 6, 800),
    ("film_parts/3_elimination.mp4", "film_3_slow_motion", 0, None, 6, 800),
    ("film_parts/4_results.mp4", "film_4_results", 1.5, 3, 2, 800),
    ("swarm_mmlu_gateaware_f07.mp4", "swarm_mmlu_gateaware_f07", 0, None, 4, 760),
    ("swarm_boolq_coherent_f05.mp4", "swarm_boolq_coherent_f05", 0, None, 4, 760),
    ("swarm_medqa_rushing_llm_f05.mp4", "swarm_medqa_rushing_llm_f05", 0, None, 4, 760),
    ("gate_aware_sweep.mp4", "gate_aware_sweep", 0, None, 6, 760),
    ("live_debate_mmlu.mp4", "live_debate_mmlu", 0, None, 4, 760),
    ("live_debate_boolq.mp4", "live_debate_boolq", 0, None, 4, 760),
    ("label_switching.mp4", "label_switching", 0, None, 6, 800),
    ("live_informed_arc.mp4", "live_informed_arc", 0, None, 4, 760),
    ("live_informed_boolq.mp4", "live_informed_boolq", 0, None, 4, 760),
]


def gif(src: Path, dst: Path, start: float, dur: float | None, fps: int, width: int, colors: int = 128) -> float:
    clip = ["-ss", str(start)] + (["-t", str(dur)] if dur else [])
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run([FF, "-y", "-loglevel", "error", *clip, "-i", str(src), "-vf", vf, "-loop", "0", str(dst)], check=True)
    return dst.stat().st_size / 1e6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src, name, start, dur, fps, width in JOBS:
        path = MEDIA / src
        if not path.exists():
            print("missing", src)
            continue
        dst = OUT / f"{name}.gif"
        size = gif(path, dst, start, dur, fps, width)
        while size > LIMIT_MB and width > 480:  # shrink until it fits
            width -= 120
            size = gif(path, dst, start, dur, max(fps - 1, 2), width, colors=96)
        print(f"{dst.relative_to(ROOT)}  {size:.1f} MB  ({width}px)")


if __name__ == "__main__":
    main()
