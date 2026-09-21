#!/usr/bin/env python3
"""Opus(기존) vs EnCodec(모델 공유) 를 같은 지표로 비교한다.

영상 실험의 짝이다. 영상에서는 졌는데 오디오에서는 어떤가.
EnCodec 모델 크기는 전송량에 넣지 않는다 — 양쪽이 갖고 있다는 가정.
"""
import json, subprocess, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).parent))
import audio_metrics as A

OPUS_KBPS = [6, 8, 12, 16, 24, 32, 48]
ENC_BW    = [1.5, 3.0, 6.0, 12.0, 24.0]   # EnCodec 24kHz 지원 대역폭

out = Path("out"); out.mkdir(exist_ok=True)
tmp = Path("out/_a"); tmp.mkdir(exist_ok=True)

from transformers import EncodecModel, AutoProcessor
dev = "cuda"
enc = EncodecModel.from_pretrained("facebook/encodec_24khz").to(dev).eval()
proc = AutoProcessor.from_pretrained("facebook/encodec_24khz")
nparam = sum(p.numel() for p in enc.parameters())
print(f"EnCodec 24kHz: {nparam/1e6:.1f}M 파라미터 ({nparam*4/1e6:.0f} MB fp32)")
print("모델 크기는 전송량에 안 들어간다 — 양쪽이 갖고 있다는 가정\n")

rows = []
for wav in sorted(Path("audio").glob("*.wav")):
    ref, sr = A.read(wav)
    dur = len(ref)/sr
    print(f"=== {wav.stem}  {dur:.1f}초 {sr}Hz ===")
    print(f"{'방식':<16} {'kbps':>7} {'mel':>7} {'msSTFT':>8} {'SI-SDR':>8}")

    for kb in OPUS_KBPS:
        o = tmp/f"{wav.stem}_{kb}.opus"; d = tmp/f"{wav.stem}_{kb}.wav"
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(wav),"-c:a","libopus",
                        "-b:a",f"{kb}k","-ar","24000","-ac","1",str(o)],check=True)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(o),"-ar",str(sr),
                        "-ac","1",str(d)],check=True)
        t,_ = A.read(d)
        m = A.compare(ref,t,sr)
        real = o.stat().st_size*8/1000/dur
        rows.append({"clip":wav.stem,"method":"opus","target_kbps":kb,
                     "real_kbps":real,**m})
        print(f"{'opus':<16} {real:>7.1f} {m['mel']:>7.3f} {m['msstft']:>8.3f} {m['sisdr']:>8.2f}")

    x = torch.from_numpy(ref)[None,None].to(dev)
    for bw in ENC_BW:
        with torch.no_grad():
            e = enc.encode(x, bandwidth=bw)
            y = enc.decode(e.audio_codes, e.audio_scales)[0]
        t = y[0,0].cpu().numpy()
        m = A.compare(ref,t,sr)
        # 실제 전송량: 코드북 인덱스 개수 × 10비트(코드북 1024)
        codes = e.audio_codes           # (1, 1, n_q, T)
        nbits = codes.shape[-2]*codes.shape[-1]*10
        real = nbits/1000/dur
        rows.append({"clip":wav.stem,"method":"encodec","target_kbps":bw,
                     "real_kbps":real,"n_q":int(codes.shape[-2]),
                     "enc_params":nparam,**m})
        print(f"{'encodec '+str(bw):<16} {real:>7.1f} {m['mel']:>7.3f} {m['msstft']:>8.3f} {m['sisdr']:>8.2f}")
    print()
json.dump(rows, open(out/"audio.json","w"), indent=1)
print(f"{len(rows)}개 측정 -> out/audio.json")
