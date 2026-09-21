#!/usr/bin/env python3
"""2026 현재의 신경망 오디오 코덱들과 Opus 를 나란히 놓는다.

EnCodec(2022) 만으로는 현재 상태를 대표하지 못한다. 저비트레이트에서
Mimi 가, 고비트레이트에서 DAC 이 더 낫다는 보고가 있어 직접 확인한다.

모델 크기는 전송량에 넣지 않되 **반드시 같이 적는다** — 공유해야 할
'악기'가 클수록 손익분기가 늦게 온다.
"""
import json, subprocess, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
import audio_metrics as A

dev = "cuda"
out = Path("out"); tmp = Path("out/_s"); tmp.mkdir(parents=True, exist_ok=True)
rows = []

def resample(x, sr_from, sr_to, tmpdir, tag):
    if sr_from == sr_to: return x
    a = tmpdir/f"{tag}_in.wav"; b = tmpdir/f"{tag}_out.wav"
    sf.write(a, x, sr_from)
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(a),"-ar",str(sr_to),"-ac","1",str(b)],check=True)
    y,_ = sf.read(b, dtype="float32")
    return y

# ---- 모델 적재 -------------------------------------------------------------
from transformers import EncodecModel, MimiModel, DacModel
models = {}
try:
    m = EncodecModel.from_pretrained("facebook/encodec_24khz").to(dev).eval()
    models["encodec"] = dict(m=m, sr=24000, n=sum(p.numel() for p in m.parameters()),
                             bws=[1.5,3.0,6.0,12.0,24.0])
except Exception as e: print("encodec 실패:", str(e)[:60])
try:
    m = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
    models["mimi"] = dict(m=m, sr=24000, n=sum(p.numel() for p in m.parameters()),
                          nqs=[4,8,16,32])          # RVQ 층수로 비트레이트 조절
except Exception as e: print("mimi 실패:", str(e)[:60])
try:
    m = DacModel.from_pretrained("descript/dac_24khz").to(dev).eval()
    models["dac"] = dict(m=m, sr=24000, n=sum(p.numel() for p in m.parameters()), nqs=[2,4,8,16,32])
except Exception as e: print("dac 실패:", str(e)[:60])

for k,v in models.items():
    print(f"  {k:<9} {v['n']/1e6:>6.1f}M 파라미터  ({v['n']*4/1e6:>5.0f} MB fp32)")
print()

# ---- 측정 -----------------------------------------------------------------
for wav in sorted(Path("audio").glob("*.wav")):
    ref24, sr = A.read(wav)
    dur = len(ref24)/sr
    print(f"=== {wav.stem}  {dur:.1f}초 ===")
    print(f"{'방식':<18} {'kbps':>7} {'mel':>7} {'msSTFT':>8}")

    for kb in [6,12,24,48]:
        o = tmp/f"{wav.stem}_{kb}.opus"; d = tmp/f"{wav.stem}_{kb}.wav"
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(wav),"-c:a","libopus",
                        "-b:a",f"{kb}k","-ar","24000","-ac","1",str(o)],check=True)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(o),"-ar","24000","-ac","1",str(d)],check=True)
        t,_ = A.read(d); m = A.compare(ref24,t,sr)
        r = o.stat().st_size*8/1000/dur
        rows.append({"clip":wav.stem,"method":"opus","kbps":r,**m})
        print(f"{'opus '+str(kb)+'k':<18} {r:>7.1f} {m['mel']:>7.3f} {m['msstft']:>8.3f}")

    for name, cfg in models.items():
        x = torch.from_numpy(ref24)[None,None].to(dev)
        if name == "encodec":
            for bw in cfg["bws"]:
                with torch.no_grad():
                    e = cfg["m"].encode(x, bandwidth=bw)
                    y = cfg["m"].decode(e.audio_codes, e.audio_scales)[0]
                c = e.audio_codes
                nb = c.shape[-2]*c.shape[-1]*10
                t = y[0,0].cpu().numpy(); m = A.compare(ref24,t,sr)
                r = nb/1000/dur
                rows.append({"clip":wav.stem,"method":"encodec","kbps":r,
                             "params":cfg["n"],**m})
                print(f"{'encodec '+str(bw):<18} {r:>7.1f} {m['mel']:>7.3f} {m['msstft']:>8.3f}")
        else:
            for nq in cfg["nqs"]:
                try:
                    with torch.no_grad():
                        if name == "mimi":
                            e = cfg["m"].encode(x, num_quantizers=nq)
                            y = cfg["m"].decode(e.audio_codes)[0]
                            c = e.audio_codes
                        else:
                            e = cfg["m"].encode(x, n_quantizers=nq)
                            y = cfg["m"].decode(e.quantized_representation).audio_values
                            c = e.audio_codes[:, :nq]
                    nb = c.shape[-2]*c.shape[-1]*10
                    t = y[0].squeeze().cpu().numpy(); m = A.compare(ref24,t,sr)
                    r = nb/1000/dur
                    rows.append({"clip":wav.stem,"method":name,"nq":nq,"kbps":r,
                                 "params":cfg["n"],**m})
                    print(f"{name+' nq='+str(nq):<18} {r:>7.1f} {m['mel']:>7.3f} {m['msstft']:>8.3f}")
                except Exception as ex:
                    print(f"{name+' nq='+str(nq):<18} 실패: {str(ex)[:50]}")
    print()
json.dump(rows, open(out/"audio_sota.json","w"), indent=1)
print(f"{len(rows)}개 -> out/audio_sota.json")
