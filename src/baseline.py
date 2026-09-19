#!/usr/bin/env python3
"""기존 코덱 기준선. 이게 있어야 나중에 무슨 주장이든 할 수 있다.

x264 / x265 / SVT-AV1 을 여러 비트레이트로 돌리고, 실제 파일 크기와
품질을 같이 낸다. '목표 비트레이트'가 아니라 실측 크기를 쓴다 -
인코더는 목표를 자주 못 맞춘다.
"""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M

CODECS = {
    "x264":  ["-c:v", "libx264",   "-preset", "medium"],
    "x265":  ["-c:v", "libx265",   "-preset", "medium", "-x265-params", "log-level=none"],
    "av1":   ["-c:v", "libsvtav1", "-preset", "6"],
}
KBPS = [1000, 500, 250, 120, 60, 30, 15]

out = Path("out"); out.mkdir(exist_ok=True)
rows = []
for clip in sorted(Path("clips").glob("*.y4m")):
    ref = M.read_y4m_frames(clip)
    dur = len(ref) / 24.0
    print(f"\n=== {clip.stem}  {len(ref)}프레임 {dur:.1f}초 ===")
    print(f"{'코덱':<6} {'목표kbps':>8} {'실제KB':>8} {'실제kbps':>9} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>7}")
    for cname, cargs in CODECS.items():
        for kb in KBPS:
            enc = out / f"{clip.stem}_{cname}_{kb}.mp4"
            dec = out / f"{clip.stem}_{cname}_{kb}.y4m"
            subprocess.run(["ffmpeg","-y","-v","error","-i",str(clip),*cargs,
                            "-b:v",f"{kb}k","-an",str(enc)], check=True)
            subprocess.run(["ffmpeg","-y","-v","error","-i",str(enc),
                            "-pix_fmt","yuv420p",str(dec)], check=True)
            tst = M.read_y4m_frames(dec)
            m = M.compare(ref, tst)
            sz = enc.stat().st_size
            real_kbps = sz*8/1000/dur
            rows.append({"clip":clip.stem,"codec":cname,"target_kbps":kb,
                         "bytes":sz,"real_kbps":real_kbps, **m})
            print(f"{cname:<6} {kb:>8} {sz/1024:>8.1f} {real_kbps:>9.1f} "
                  f"{m['psnr']:>7.2f} {m['ssim']:>6.3f} {m['lpips']:>7.4f}")
            dec.unlink(missing_ok=True)
json.dump(rows, open(out/"baseline.json","w"), indent=1)
print(f"\n{len(rows)}개 측정 -> out/baseline.json")
