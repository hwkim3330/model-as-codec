#!/usr/bin/env python3
"""실험 10 — 사람 판단으로 학습된 지표로 다시 매기고, 청취 시험용 파일을 굽는다.

실험 9 에서 mel 과 msSTFT 가 서로 다른 순위를 냈다 (pop 10.2x 대 4.9x,
piano 의 DAC 는 방향까지 반대). 스펙트럼 L1 이 사람 귀를 대신하지 못한다는 뜻이다.

여기서 두 가지를 한다.
  1) CDPAM 으로 다시 잰다. 사람의 '어느 쪽이 원본에 가까운가' 판단 약 50만 건으로
     학습된 거리다. 22.05 kHz 모노로 동작하므로 고역 판정에는 한계가 있다.
  2) 실제 청취 시험(MUSHRA 식)에 쓸 오디오를 굽는다. 기준/앵커/각 코덱을
     같은 길이로 맞춰 FLAC 로 낸다. 사람이 직접 듣는 건 내가 대신할 수 없다.

MUSHRA 관례대로 **저역통과 3.5 kHz 앵커**를 넣는다. 청취자가 척도를 어떻게
쓰는지 보정하는 기준점이고, 이게 없으면 점수를 해석할 수 없다.
"""
import json, subprocess, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, soundfile as sf
import sys; sys.path.insert(0, str(Path(__file__).parent))
from metrics48 import mel_st, msstft_st

import cdpam
from contextlib import contextmanager

@contextmanager
def _allow_legacy_pickle():
    """CDPAM 의 체크포인트는 PyTorch 2.6 이전 형식이라 weights_only=False 가 필요하다.
    파일은 pip 로 설치한 cdpam 배포판에 들어 있는 것이고, 모델 생성 한 번에만
    푼다. 다른 torch.load 는 기본값(안전) 그대로 둔다."""
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})
    try: yield
    finally: torch.load = orig

dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path("out"); listen = Path("listen"); listen.mkdir(exist_ok=True)
tmp = out/"_p"; tmp.mkdir(parents=True, exist_ok=True)

def to22k_mono_i16(x, sr, tag):
    """CDPAM 입력 규격: 22.05 kHz 모노 int16. ffmpeg 로 미리 맞춘다."""
    a, b = tmp/f"{tag}_a.wav", tmp/f"{tag}_b.wav"
    sf.write(a, x, sr)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(a),"-ar","22050","-ac","1",
                    "-c:a","pcm_s16le",str(b)], check=True)
    y, _ = sf.read(b, dtype="int16")
    return y[None]

# ---- 조건 정의 --------------------------------------------------------------
# 청취 시험은 비트레이트를 대충 맞춰야 의미가 있다. 26~32 kbps 대로 모은다.
from transformers import EncodecModel, DacModel
enc = EncodecModel.from_pretrained("facebook/encodec_48khz").to(dev).eval()
dac = DacModel.from_pretrained("descript/dac_44khz").to(dev).eval()

def dec_opus(ref, sr, kb, tag):
    o, d = tmp/f"{tag}.opus", tmp/f"{tag}_o.wav"
    sf.write(tmp/f"{tag}_i.wav", ref, sr)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(tmp/f"{tag}_i.wav"),"-c:a","libopus",
                    "-b:a",f"{kb}k","-ar","48000","-ac","2",str(o)],check=True)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(o),"-ar","48000","-ac","2",str(d)],check=True)
    y,_ = sf.read(d, dtype="float32")
    return y, o.stat().st_size*8/1000/(len(ref)/sr)

def dec_enc48(ref, sr, bw, tag):
    x = torch.from_numpy(ref.T)[None].to(dev)
    with torch.no_grad():
        e = enc.encode(x, bandwidth=bw); y = enc.decode(e.audio_codes, e.audio_scales)[0]
    c = e.audio_codes
    nb = int(np.prod(c.shape[2:])*c.shape[0]*10) + int(c.shape[0])*32
    return y[0].T.cpu().numpy(), nb/1000/(len(ref)/sr)

def dec_dac44(ref, sr, nq, tag):
    a,b = tmp/f"{tag}_r.wav", tmp/f"{tag}_r441.wav"
    sf.write(a, ref, sr)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(a),"-ar","44100",str(b)],check=True)
    r,_ = sf.read(b, dtype="float32"); chans=[]; nb=0
    with torch.no_grad():
        for ch in range(r.shape[1]):
            xi = torch.from_numpy(r[:,ch])[None,None].to(dev)
            ei = dac.encode(xi, n_quantizers=nq)
            chans.append(dac.decode(ei.quantized_representation).audio_values[0].squeeze().cpu().numpy())
            nb += ei.audio_codes[:, :nq].numel()*10
    sf.write(tmp/f"{tag}_d.wav", np.stack(chans,1), 44100)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(tmp/f"{tag}_d.wav"),"-ar","48000",
                    str(tmp/f"{tag}_d48.wav")],check=True)
    y,_ = sf.read(tmp/f"{tag}_d48.wav", dtype="float32")
    return y, nb/1000/(len(ref)/sr)

def anchor(ref, sr, tag):
    """MUSHRA 표준 앵커: 3.5 kHz 저역통과."""
    a,b = tmp/f"{tag}_ai.wav", tmp/f"{tag}_ao.wav"
    sf.write(a, ref, sr)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(a),"-af","lowpass=f=3500",str(b)],check=True)
    y,_ = sf.read(b, dtype="float32")
    return y, None

CONDS = [("opus32","opus",32), ("encodec48_24","enc",24.0), ("dac44_9","dac",9),
         ("opus16","opus",16), ("encodec48_6","enc",6.0), ("anchor35","anchor",None)]

with _allow_legacy_pickle():
    m = cdpam.CDPAM(dev=dev)
rows = []
for wav in sorted(Path("audio48").glob("*.wav")):
    ref, sr = sf.read(wav, dtype="float32"); dur = len(ref)/sr
    sf.write(listen/f"{wav.stem}__reference.flac", ref, sr)
    ri = to22k_mono_i16(ref, sr, "ref")
    print(f"=== {wav.stem} ===")
    print(f"{'조건':<16}{'kbps':>8}{'CDPAM':>9}{'mel':>8}{'msSTFT':>9}")
    for name, kind, cfg in CONDS:
        tag = f"{wav.stem}_{name}"
        if   kind=="opus":   y, kb = dec_opus(ref, sr, cfg, tag)
        elif kind=="enc":    y, kb = dec_enc48(ref, sr, cfg, tag)
        elif kind=="dac":    y, kb = dec_dac44(ref, sr, cfg, tag)
        else:                y, kb = anchor(ref, sr, tag)
        n = min(len(ref), len(y)); y = y[:n]
        sf.write(listen/f"{wav.stem}__{name}.flac", y, sr)
        d = float(m.forward(ri, to22k_mono_i16(y, sr, "y")))
        mm = mel_st(ref[:n], y, sr); ms = msstft_st(ref[:n], y)
        rows.append({"clip":wav.stem,"cond":name,"kbps":kb,"cdpam":d,"mel":mm,"msstft":ms})
        print(f"{name:<16}{(f'{kb:.1f}' if kb else '-'):>8}{d:>9.4f}{mm:>8.3f}{ms:>9.3f}")
    print()
json.dump(rows, open(out/"perceptual.json","w"), indent=1)
print(f"{len(rows)}개 -> out/perceptual.json,  청취용 FLAC {len(list(listen.glob('*.flac')))}개 -> listen/")
