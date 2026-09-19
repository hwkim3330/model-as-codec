#!/usr/bin/env python3
"""VAE 왕복만으로 얼마나 잃는가. 양자화 이전의 천장이다.

이걸 안 재면 'latent 코덱이 나쁘다'가 양자화 탓인지 VAE 탓인지 못 가른다.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M
from latent_codec import load_vae, encode, decode

dev="cuda"
vae, n = load_vae("stabilityai/sd-vae-ft-mse", dev)
print(f"VAE {n/1e6:.1f}M — 양자화 없이 float latent 그대로 왕복\n")
print(f"{'클립':<8} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>7}   (이것이 latent 방식의 상한)")
for clip in sorted(Path("clips").glob("*.y4m")):
    ref = M.read_y4m_frames(clip)
    rec = decode(vae, encode(vae, ref, dev), dev)
    m = M.compare(ref, rec)
    print(f"{clip.stem:<8} {m['psnr']:>7.2f} {m['ssim']:>6.3f} {m['lpips']:>7.4f}")
