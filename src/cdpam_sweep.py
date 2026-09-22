#!/usr/bin/env python3
"""실험 10b — 전 구간을 CDPAM 으로 다시 매긴다.

mel 과 (고친) msSTFT 가 절약 배수를 5.7x 대 19.3x 로 3.4배 다르게 낸다.
둘 다 스펙트럼 L1 계열이라 어느 쪽이 사람 귀에 가까운지 자기들끼리는 못 가른다.
CDPAM 은 사람의 '어느 쪽이 원본에 가까운가' 판단으로 학습된 거리이므로
심판으로 쓴다. (22.05 kHz 모노로 동작하는 한계는 그대로 적는다.)
"""
import json, subprocess, warnings, sys
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from contextlib import contextmanager
import cdpam

@contextmanager
def _legacy():
    o = torch.load
    torch.load = lambda *a, **k: o(*a, **{**k, "weights_only": False})
    try: yield
    finally: torch.load = o

dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path("out"); tmp = out/"_c"; tmp.mkdir(parents=True, exist_ok=True)

def i16_22k(x, sr, tag):
    a, b = tmp/f"{tag}_a.wav", tmp/f"{tag}_b.wav"
    sf.write(a, x, sr)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(a),"-ar","22050","-ac","1",
                    "-c:a","pcm_s16le",str(b)], check=True)
    y,_ = sf.read(b, dtype="int16"); return y[None]

from transformers import EncodecModel, DacModel
enc = EncodecModel.from_pretrained("facebook/encodec_48khz").to(dev).eval()
dac = DacModel.from_pretrained("descript/dac_44khz").to(dev).eval()
with _legacy(): M = cdpam.CDPAM(dev=dev)

rows = []
for wav in sorted(Path("audio48").glob("*.wav")):
    ref, sr = sf.read(wav, dtype="float32"); dur = len(ref)/sr
    ri = i16_22k(ref, sr, "ref")
    print(f"=== {wav.stem} ===")
    def score(y, meth, cfg, kb):
        n = min(len(ref), len(y))
        d = float(M.forward(ri, i16_22k(y[:n], sr, "y")))
        rows.append({"clip":wav.stem,"method":meth,"cfg":cfg,"kbps":kb,"cdpam":d})
        print(f"  {meth+' '+str(cfg):<18}{kb:>8.1f}{d:>10.4f}")

    for kb in [16,32,64,96,128]:
        o,d = tmp/f"o{kb}.opus", tmp/f"o{kb}.wav"
        sf.write(tmp/"i.wav", ref, sr)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(tmp/"i.wav"),"-c:a","libopus",
                        "-b:a",f"{kb}k","-ar","48000","-ac","2",str(o)],check=True)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(o),"-ar","48000","-ac","2",str(d)],check=True)
        y,_ = sf.read(d, dtype="float32"); score(y,"opus",kb,o.stat().st_size*8/1000/dur)

    x = torch.from_numpy(ref.T)[None].to(dev)
    for bw in enc.config.target_bandwidths:
        with torch.no_grad():
            e = enc.encode(x, bandwidth=bw); y = enc.decode(e.audio_codes, e.audio_scales)[0]
        c = e.audio_codes
        nb = int(np.prod(c.shape[2:])*c.shape[0]*10) + int(c.shape[0])*32
        score(y[0].T.cpu().numpy(), "encodec48", bw, nb/1000/dur)

    sf.write(tmp/"r.wav", ref, sr)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(tmp/"r.wav"),"-ar","44100",str(tmp/"r441.wav")],check=True)
    r441,_ = sf.read(tmp/"r441.wav", dtype="float32")
    for nq in [3,6,9]:
        chans, nb = [], 0
        with torch.no_grad():
            for ch in range(r441.shape[1]):
                xi = torch.from_numpy(r441[:,ch])[None,None].to(dev)
                ei = dac.encode(xi, n_quantizers=nq)
                chans.append(dac.decode(ei.quantized_representation).audio_values[0].squeeze().cpu().numpy())
                nb += ei.audio_codes[:, :nq].numel()*10
        sf.write(tmp/"d.wav", np.stack(chans,1), 44100)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(tmp/"d.wav"),"-ar","48000",str(tmp/"d48.wav")],check=True)
        y,_ = sf.read(tmp/"d48.wav", dtype="float32"); score(y,"dac44",nq,nb/1000/dur)
json.dump(rows, open(out/"cdpam_sweep.json","w"), indent=1)
print(f"\n{len(rows)}개 -> out/cdpam_sweep.json")
