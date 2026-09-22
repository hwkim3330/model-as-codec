#!/usr/bin/env python3
"""실험 9 — 음악을 원래 형식(48 kHz 스테레오)에서 재다.

실험 8 은 24 kHz 모노였고, 그래서 "음악 코덱 비교라고 부를 수 없다" 고 적었다.
여기서는 음악용으로 학습된 EnCodec 48kHz(스테레오 네이티브)를 쓰고 Opus 도
48 kHz 스테레오로 돌린다.

비트레이트 회계:
  opus     파일 크기 (Ogg 컨테이너 포함 — 다른 방식에 유리하게 잡히는 쪽이다)
  encodec  코드 개수 x 10 bit (코드북 1024) + 청크마다 scale 1개 x 32 bit
           normalize=True 라 scale 을 빼면 복원이 안 되므로 반드시 센다
  dac      44.1 kHz 모노 모델이라 채널별로 따로 돌린다 = 코드 2배.
           조인트 스테레오를 못 쓰므로 DAC 에 불리한 조건이고, 그렇게 적는다
"""
import json, subprocess, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, soundfile as sf

dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path("out"); tmp = out/"_48"; tmp.mkdir(parents=True, exist_ok=True)

from metrics48 import mel_st, msstft_st, compare

def resample(x, sr_from, sr_to, tag):
    if sr_from == sr_to: return x
    p, q = tmp/f"{tag}_a.wav", tmp/f"{tag}_b.wav"
    sf.write(p, x, sr_from)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(p),"-ar",str(sr_to),str(q)], check=True)
    y,_ = sf.read(q, dtype="float32")
    return y if y.ndim>1 else y[:,None]

# ---- 모델 ------------------------------------------------------------------
from transformers import EncodecModel, DacModel
enc = EncodecModel.from_pretrained("facebook/encodec_48khz").to(dev).eval()
dac = DacModel.from_pretrained("descript/dac_44khz").to(dev).eval()
print(f"  encodec_48khz  {sum(p.numel() for p in enc.parameters())/1e6:.1f}M  "
      f"48kHz 스테레오 네이티브  bw {enc.config.target_bandwidths}")
print(f"  dac_44khz      {sum(p.numel() for p in dac.parameters())/1e6:.1f}M  "
      f"44.1kHz 모노 (채널별 2회)\n")

rows = []
for wav in sorted(Path("audio48").glob("*.wav")):
    ref, sr = sf.read(wav, dtype="float32")        # (n,2) @48k
    dur = len(ref)/sr
    print(f"=== {wav.stem}  {dur:.1f}초 48kHz 스테레오 ===")
    print(f"{'방식':<20}{'kbps':>8}{'mel':>8}{'msSTFT':>9}")

    # --- Opus 48k 스테레오
    for kb in [16, 32, 64, 96, 128]:
        o = tmp/f"{wav.stem}_{kb}.opus"; d = tmp/f"{wav.stem}_{kb}.wav"
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(wav),"-c:a","libopus",
                        "-b:a",f"{kb}k","-ar","48000","-ac","2",str(o)], check=True)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(o),"-ar","48000","-ac","2",str(d)], check=True)
        t,_ = sf.read(d, dtype="float32")
        m = compare(ref, t, sr); r = o.stat().st_size*8/1000/dur
        rows.append({"clip":wav.stem,"method":"opus","cfg":kb,"kbps":r,**m})
        print(f"{'opus '+str(kb)+'k':<20}{r:>8.1f}{m['mel']:>8.3f}{m['msstft']:>9.3f}")

    # --- EnCodec 48k 스테레오
    x = torch.from_numpy(ref.T)[None].to(dev)      # (1,2,n)
    for bw in enc.config.target_bandwidths:
        with torch.no_grad():
            e = enc.encode(x, bandwidth=bw)
            y = enc.decode(e.audio_codes, e.audio_scales)[0]
        c = e.audio_codes                           # (chunks,B,n_q,T)
        nb = int(np.prod(c.shape[2:]) * c.shape[0] * 10)     # 코드 10 bit
        nb += int(c.shape[0]) * 32                           # 청크마다 scale float32
        t = y[0].T.cpu().numpy()
        m = compare(ref, t, sr); r = nb/1000/dur
        rows.append({"clip":wav.stem,"method":"encodec48","cfg":bw,"kbps":r,**m})
        print(f"{'encodec48 '+str(bw):<20}{r:>8.1f}{m['mel']:>8.3f}{m['msstft']:>9.3f}")

    # --- DAC 44.1k, 채널별
    r441 = resample(ref, sr, 44100, wav.stem)
    for nq in [3, 6, 9]:
        chans, nb = [], 0
        with torch.no_grad():
            for ch in range(r441.shape[1]):
                xi = torch.from_numpy(r441[:,ch])[None,None].to(dev)
                ei = dac.encode(xi, n_quantizers=nq)
                yi = dac.decode(ei.quantized_representation).audio_values
                chans.append(yi[0].squeeze().cpu().numpy())
                nb += ei.audio_codes[:, :nq].numel() * 10
        y441 = np.stack(chans, 1)
        t = resample(y441, 44100, sr, wav.stem+"_d")
        m = compare(ref, t, sr); r = nb/1000/dur
        rows.append({"clip":wav.stem,"method":"dac44","cfg":nq,"kbps":r,**m})
        print(f"{'dac44 nq='+str(nq):<20}{r:>8.1f}{m['mel']:>8.3f}{m['msstft']:>9.3f}")
    print()

json.dump(rows, open(out/"music48.json","w"), indent=1)
print(f"{len(rows)}개 -> out/music48.json")
