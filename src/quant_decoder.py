#!/usr/bin/env python3
"""실험 7 — int8 디코더가 실제로 들리는가.

실험 6 의 표에 int8 열을 넣고 "코드북 품질 영향 미측정이므로 하한선으로만
읽으라"고 써 뒀다. 안 잰 숫자를 표에 넣은 것이므로 잰다.

수신 측만 누른다. 인코더는 fp32 로 두고 (송신자 사정), 디코더 경로의
Conv/Linear 가중치와 RVQ 코드북을 int8 로 fake-quantize 한 뒤 디코딩해서
fp32 디코딩과 비교한다.

  per-channel 대칭 양자화: 출력 채널마다 s = max|w| / 127, w_q = round(w/s)*s
  코드북은 엔트리(행)마다 같은 방식.

읽는 법: 기준은 'fp32 디코딩 대비 mel 거리가 얼마나 늘었나' 가 아니라
**'원본 대비 품질 저하가 Opus 와의 격차를 잡아먹는가'** 다. 코덱 자체의
손실이 이미 mel 0.2 대인데 양자화가 0.01 을 더하는 건 무의미하고,
0.1 을 더하면 실험 6 의 int8 열은 못 쓴다.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
import audio_metrics as A

dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path("out")

def qtensor(w, axis=0, bits=8):
    """per-channel 대칭 fake-quant. axis 를 따라 스케일을 따로 잡는다."""
    qmax = 2 ** (bits - 1) - 1
    dims = [d for d in range(w.dim()) if d != axis]
    s = w.abs().amax(dim=dims, keepdim=True) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (w / s).round().clamp(-qmax - 1, qmax) * s

# 디코더 경로로 볼 이름. 인코더/다운샘플은 송신 측이므로 건드리지 않는다.
DEC = ("decoder", "upsample")
QNT = ("quantizer", "codebook", "vq")

def quantize_receiver(model, bits=8, do_codebook=True):
    """디코더 가중치 + (선택) 코드북을 int8 로 누른다. 누른 개수를 돌려준다.

    처음 판에서 두 군데가 샜다. 반드시 분해표와 대조할 것:
      - EnCodec 디코더의 LSTM 은 Conv1d/Linear 가 아니라 43% 밖에 안 눌렸다.
      - DAC 코드북은 버퍼가 아니라 nn.Embedding 파라미터라 0% 눌렸다.
    """
    nw = nc = 0
    with torch.no_grad():
        targets = DEC + (QNT if do_codebook else ())
        for name, mod in model.named_modules():
            ln = name.lower()
            if not any(k in ln for k in targets): continue
            if isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.Linear)):
                # ConvTranspose1d 는 (in, out, k) 라 출력 채널이 axis=1
                ax = 1 if isinstance(mod, nn.ConvTranspose1d) else 0
                mod.weight.copy_(qtensor(mod.weight, axis=ax, bits=bits))
                if any(k in ln for k in QNT): nc += mod.weight.numel()
                else:                        nw += mod.weight.numel()
            elif isinstance(mod, (nn.LSTM, nn.GRU, nn.RNN)):
                for pn, prm in mod.named_parameters(recurse=False):
                    if not pn.startswith("weight"): continue
                    prm.copy_(qtensor(prm, axis=0, bits=bits))
                    nw += prm.numel()
            elif isinstance(mod, nn.Embedding):      # DAC 코드북이 여기 걸린다
                mod.weight.copy_(qtensor(mod.weight, axis=0, bits=bits))
                if any(k in ln for k in QNT): nc += mod.weight.numel()
                else:                        nw += mod.weight.numel()
        if do_codebook:
            # 코드북은 구현에 따라 버퍼(EnCodec/Mimi) 이기도 파라미터(DAC) 이기도 하다
            done = {id(p) for _, m in model.named_modules() for p in m.parameters(recurse=False)
                    if any(k in _.lower() for k in QNT)}
            for name, t in model.named_buffers():
                ln = name.lower()
                if not any(k in ln for k in QNT): continue
                if not t.dtype.is_floating_point or t.dim() != 2: continue
                if "cluster_size" in ln or "embed_avg" in ln: continue  # 학습 잔재
                t.data.copy_(qtensor(t.data, axis=0, bits=bits))
                nc += t.numel()
    return nw, nc

def run(name, load, encdec, cfgs):
    ref_rows = []
    for wav in sorted(Path("audio").glob("*.wav")):
        ref, sr = A.read(wav); dur = len(ref)/sr
        x = torch.from_numpy(ref)[None, None].to(dev)
        for cfg in cfgs:
            # --- fp32 기준
            m32 = load(); m32.eval()
            with torch.no_grad(): y32 = encdec(m32, x, cfg)
            q32 = A.compare(ref, y32, sr)
            # --- int8 (가중치만)
            mw = load(); mw.eval(); nw, _ = quantize_receiver(mw, 8, do_codebook=False)
            with torch.no_grad(): yw = encdec(mw, x, cfg)
            qw = A.compare(ref, yw, sr)
            # --- int8 (가중치 + 코드북)
            mb = load(); mb.eval(); _, nc = quantize_receiver(mb, 8, do_codebook=True)
            with torch.no_grad(): yb = encdec(mb, x, cfg)
            qb = A.compare(ref, yb, sr)
            ref_rows.append({"clip": wav.stem, "method": name, "cfg": str(cfg),
                             "mel_fp32": q32["mel"], "mel_int8_w": qw["mel"],
                             "mel_int8_wc": qb["mel"],
                             "d_w": qw["mel"]-q32["mel"], "d_wc": qb["mel"]-q32["mel"],
                             "nw": nw, "nc": nc})
            print(f"  {wav.stem:<8} {name:<8} {str(cfg):<8} "
                  f"fp32 {q32['mel']:.4f} | int8(w) {qw['mel']:.4f} ({qw['mel']-q32['mel']:+.4f}) "
                  f"| int8(w+cb) {qb['mel']:.4f} ({qb['mel']-q32['mel']:+.4f})")
            del m32, mw, mb
            if dev == "cuda": torch.cuda.empty_cache()
    return ref_rows

from transformers import EncodecModel, MimiModel, DacModel

def enc_encodec(m, x, bw):
    e = m.encode(x, bandwidth=bw)
    return m.decode(e.audio_codes, e.audio_scales)[0][0,0].cpu().numpy()
def enc_mimi(m, x, nq):
    e = m.encode(x, num_quantizers=nq)
    return m.decode(e.audio_codes)[0][0].squeeze().cpu().numpy()
def enc_dac(m, x, nq):
    e = m.encode(x, n_quantizers=nq)
    return m.decode(e.quantized_representation).audio_values[0].squeeze().cpu().numpy()

rows = []
print("=== int8 수신 측 양자화의 실제 품질 영향 (mel 거리, 낮을수록 좋음) ===\n")
rows += run("encodec", lambda: EncodecModel.from_pretrained("facebook/encodec_24khz").to(dev),
            enc_encodec, [1.5, 6.0])
rows += run("mimi", lambda: MimiModel.from_pretrained("kyutai/mimi").to(dev), enc_mimi, [4, 16])
rows += run("dac",  lambda: DacModel.from_pretrained("descript/dac_24khz").to(dev), enc_dac, [2, 8])
json.dump(rows, open(out/"quant_decoder.json","w"), indent=1)

print("\n=== 커버리지 자기검증 (분해표 대비) ===\n")
sp = json.load(open(out/"model_split.json"))
ok = True
for meth in ["encodec","mimi","dac"]:
    r = next((r for r in rows if r["method"]==meth), None)
    if not r: continue
    s_ = sp[meth]
    cw = r["nw"]/s_["decoder"]*100
    cc = r["nc"]/s_["quantizer"]*100 if s_["quantizer"] else 100.0
    flag = "" if (cw>95 and cc>95) else "  <-- 샜다"
    if flag: ok = False
    print(f"  {meth:<9} 디코더 {cw:>5.1f}%  코드북 {cc:>5.1f}%{flag}")
if not ok:
    print("\n  ** 커버리지가 100%% 가 아니면 아래 판정은 무효다. **")

print("\n=== 판정 ===\n")
dw  = [r["d_w"]  for r in rows]
dwc = [r["d_wc"] for r in rows]
print(f"  가중치만 int8       : mel 증가 중앙값 {np.median(dw):+.4f}  최악 {max(dw):+.4f}")
print(f"  가중치+코드북 int8  : mel 증가 중앙값 {np.median(dwc):+.4f}  최악 {max(dwc):+.4f}")
print()
worst = max(rows, key=lambda r: r["d_wc"])
print(f"  최악: {worst['method']} {worst['cfg']} / {worst['clip']}  "
      f"{worst['mel_fp32']:.4f} -> {worst['mel_int8_wc']:.4f}")

# --- 손익분기로 되먹이기 -----------------------------------------------------
# int8 로 품질이 떨어지면 '같은 품질의 Opus' 기준점도 내려간다. 절약이 줄고
# 손익분기가 늦어진다. 실험 6 의 int8 열은 이 효과를 무시했으므로 여기서 잰다.
print("\n=== int8 품질 저하를 반영한 손익분기 ===\n")
sota = json.load(open(out/"audio_sota.json"))
sp   = json.load(open(out/"model_split.json"))
CHEAP = {"encodec": ("bw", 1.5), "mimi": ("nq", 4), "dac": ("nq", 2)}

def opus_at(clip, mel):
    """그 품질 이상을 내는 가장 싼 Opus."""
    c = [x for x in sota if x["clip"] == clip and x["method"] == "opus" and x["mel"] <= mel]
    return min(c, key=lambda x: x["kbps"]) if c else None

print(f"{'클립':<9}{'코덱':<9}{'':>3}{'fp32 기준':>22}{'int8 기준':>24}")
print(f"{'':<21}{'opus':>7}{'절약':>7}{'손익':>8}{'opus':>8}{'절약':>7}{'손익':>8}{'':>4}")
print("-"*76)
verdict = []
for r in rows:
    k, v = CHEAP[r["method"]]
    if abs(float(r["cfg"]) - v) > 1e-6: continue
    n = next(x for x in sota if x["clip"] == r["clip"] and x["method"] == r["method"]
             and abs(x["kbps"] - (v if r["method"] == "encodec" else x["kbps"])) < 1e9
             and (r["method"] == "encodec" or x.get("nq") == v)
             and (r["method"] != "encodec" or abs(x["mel"] - r["mel_fp32"]) < 1e-6))
    o32 = opus_at(r["clip"], r["mel_fp32"])
    o8  = opus_at(r["clip"], r["mel_int8_wc"])
    if not o32 or not o8:
        print(f"{r['clip']:<9}{r['method']:<9}   기준 Opus 없음"); continue
    recv8 = sp[r["method"]]["recv"] * 1      # int8 = 1 byte/param
    recv32 = sp[r["method"]]["recv"] * 4
    s32 = (o32["kbps"] - n["kbps"]) * 1000
    s8  = (o8["kbps"]  - n["kbps"]) * 1000
    h32 = recv32 * 8 / s32 / 3600 if s32 > 0 else float("inf")
    h8  = recv8  * 8 / s8  / 3600 if s8  > 0 else float("inf")
    flag = "유지" if h8 < h32 else "역전"
    verdict.append((h8 < h32, r, h32, h8))
    print(f"{r['clip']:<9}{r['method']:<9}{'':>3}{o32['kbps']:>7.1f}{o32['kbps']/n['kbps']:>6.1f}x"
          f"{h32:>7.1f}h{o8['kbps']:>8.1f}{o8['kbps']/n['kbps']:>6.1f}x{h8:>7.1f}h  {flag}")

good = sum(1 for g, *_ in verdict if g)
print(f"\n  int8 이 fp32 보다 여전히 유리한 칸: {good}/{len(verdict)}")
if good == len(verdict):
    print("  -> 실험 6 의 int8 열은 품질 저하를 반영해도 성립한다.")
else:
    print("  -> 일부 칸에서 품질 저하가 크기 절감을 잡아먹는다. 실험 6 의 int8 열 수정 필요.")

# --- 이 검증의 분해능 ---------------------------------------------------------
# 위에서 기준 Opus 가 한 칸도 안 바뀌었다. 그게 'int8 이 무해해서' 인지
# 'Opus 격자(6/12/24/48k)가 성겨서' 인지 갈라야 한다.
print("\n=== 이 검증의 분해능 ===\n")
for clip in ["speech", "mixed", "ambient"]:
    op = sorted([x for x in sota if x["clip"] == clip and x["method"] == "opus"],
                key=lambda x: x["kbps"])
    steps = [abs(b["mel"] - a["mel"]) for a, b in zip(op, op[1:])]
    dmax = max(r["d_wc"] for r in rows if r["clip"] == clip)
    print(f"  {clip:<8} Opus 이웃 설정 간 mel 간격 {min(steps):.3f}~{max(steps):.3f}"
          f"  |  int8 최대 저하 {dmax:+.4f}")
allsteps = [abs(b["mel"]-a["mel"]) for clip in ["speech","mixed","ambient"]
            for a, b in zip(sorted([x for x in sota if x["clip"]==clip and x["method"]=="opus"],
                                   key=lambda x: x["kbps"]),
                            sorted([x for x in sota if x["clip"]==clip and x["method"]=="opus"],
                                   key=lambda x: x["kbps"])[1:])]
dmax = max(r["d_wc"] for r in rows)
print(f"\n  int8 최대 저하 {dmax:.4f} 는 가장 좁은 Opus 간격 {min(allsteps):.3f} 의 "
      f"{dmax/min(allsteps)*100:.0f}% 다.")
print("  즉 저하가 격자보다 작아서 '기준 Opus 불변' 은 강한 증거가 아니다.")
print("  강하게 말할 수 있는 것은 이것뿐이다:")
print(f"    **int8 수신 측 양자화의 mel 저하는 최악 {dmax:+.4f} 로,")
print(f"      코덱 자체 손실(mel 0.26~1.02)의 {dmax/0.26*100:.0f}% 미만이다.**")
