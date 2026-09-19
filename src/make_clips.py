#!/usr/bin/env python3
"""테스트 클립을 만든다. 짧고 512x512 로 맞춘다.

왜 512 인가: VAE latent 실험에서 Stable Diffusion 계열 VAE 가 8배 다운샘플하므로
512 -> 64x64 latent 가 된다. 기준선과 latent 쪽이 같은 해상도를 봐야 비교가 된다.
왜 짧은가: 비트레이트를 여러 점에서 재야 하는데 길면 실험이 안 돈다.
"""
import subprocess, sys
from pathlib import Path

SRC = [
    ("jelly", "/home/kim/Jellyfish-20-Mbps.mkv", "00:00:10", 6),
    ("bird",  "/home/kim/bird20.mkv",            "00:00:05", 6),
    ("sample","/home/kim/sample.mp4",            "00:00:30", 6),
]
out = Path("clips"); out.mkdir(exist_ok=True)

for name, src, ss, dur in SRC:
    if not Path(src).exists():
        print(f"  {name}: 원본 없음 — 건너뜀"); continue
    dst = out / f"{name}.y4m"          # 무압축 중간본. 기준선의 기준선이다
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", ss, "-i", src, "-t", str(dur),
           "-vf", "scale=512:512:force_original_aspect_ratio=increase,crop=512:512,fps=24",
           "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)
    sz = dst.stat().st_size
    print(f"  {name}: {dur}초 512x512@24fps  무압축 {sz/1e6:.1f} MB")
