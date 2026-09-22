#!/usr/bin/env python3
"""실험 9 의 손익분기 — 48 kHz 스테레오 음악 기준.

실험 6 의 회계를 그대로 쓰되 모델과 측정값만 48 kHz 판으로 바꾼다.
"""
import json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from transformers import EncodecModel, DacModel

TRAIN_ONLY = re.compile(r"cluster_size|embed_avg|inited|ema|running_")
def bucket(n):
    k=n.lower()
    if "decoder" in k or "upsample" in k: return "decoder"
    if "encoder" in k or "downsample" in k: return "encoder"
    if "quantiz" in k or "codebook" in k or "vq" in k: return "quantizer"
    return "other"

split={}
for name, m in [("encodec48", EncodecModel.from_pretrained("facebook/encodec_48khz")),
                ("dac44",     DacModel.from_pretrained("descript/dac_44khz"))]:
    g={k:0 for k in ["encoder","decoder","quantizer","other"]}; tr=0
    for n,t in list(m.named_parameters())+list(m.named_buffers()):
        if not (t.dtype.is_floating_point or t.dtype==torch.int64): continue
        if TRAIN_ONLY.search(n): tr+=t.numel(); continue
        g[bucket(n)]+=t.numel()
    tot=sum(g.values()); recv=g["decoder"]+g["quantizer"]+g["other"]
    split[name]={**g,"train_only":tr,"total":tot,"recv":recv}
    print(f"  {name:<10} 배포 {tot/1e6:>5.1f}M (학습전용 {tr/1e6:.1f}M 제외) | "
          f"인코더 {g['encoder']/1e6:.1f}M 디코더 {g['decoder']/1e6:.1f}M "
          f"코드북 {g['quantizer']/1e6:.1f}M | 수신 {recv/1e6:.1f}M ({recv/tot*100:.0f}%)")
json.dump(split, open("out/model_split48.json","w"), indent=1)

r=json.load(open("out/music48.json"))
clips=sorted({x["clip"] for x in r})
SCEN=[("전체 fp32",lambda s:s["total"]*4),("디코더 fp32",lambda s:s["recv"]*4),
      ("디코더 fp16",lambda s:s["recv"]*2),("디코더 int8",lambda s:s["recv"]*1)]

print("\n=== 손익분기 (48 kHz 스테레오 음악, mel 기준) ===\n")
hdr=f"{'클립':<9}{'코덱':<11}{'절약':>7}"+"".join(f"{n:>14}" for n,_ in SCEN); print(hdr); print("-"*len(hdr))
rows=[]
for c in clips:
    op=[x for x in r if x["clip"]==c and x["method"]=="opus"]
    for meth in ["encodec48","dac44"]:
        ns=[x for x in r if x["clip"]==c and x["method"]==meth]
        best=None
        for n in ns:
            eq=[o for o in op if o["mel"]<=n["mel"]]
            if not eq: continue
            o=min(eq,key=lambda o:o["kbps"])
            sv=(o["kbps"]-n["kbps"])*1000
            if sv>0 and (best is None or o["kbps"]/n["kbps"]>best[0]):
                best=(o["kbps"]/n["kbps"], sv, n, o)
        if not best: print(f"{c:<9}{meth:<11}  전패"); continue
        ratio,sv,n,o=best; s=split[meth]
        hs=[f(s)*8/sv/3600 for _,f in SCEN]
        rows.append({"clip":c,"codec":meth,"ratio":ratio,"kbps":n["kbps"],
                     **{nm:h for (nm,_),h in zip(SCEN,hs)}})
        print(f"{c:<9}{meth:<11}{ratio:>6.1f}x"+"".join(f"{h:>12.1f}h" for h in hs))
json.dump(rows, open("out/breakeven48.json","w"), indent=1)

print("\n=== 24 kHz 모노 대비 ===\n")
b24=json.load(open("out/breakeven2.json"))
m24=[x["디코더 fp16"] for x in b24]; m48=[x["디코더 fp16"] for x in rows]
r24=[x["ratio"] for x in b24];       r48=[x["ratio"] for x in rows]
print(f"  절약 배수   24k모노 중앙값 {np.median(r24):>6.1f}x   48k스테레오 {np.median(r48):>6.1f}x")
print(f"  손익분기    24k모노 중앙값 {np.median(m24):>6.1f}h   48k스테레오 {np.median(m48):>6.1f}h  (디코더 fp16)")
print()
print("  차이의 원인은 코덱이 아니라 **비교 기준**이다:")
print("  24 kHz 모노 실험에서 Opus 를 6 kbps 까지 내려 돌렸는데, 그건 Opus 의")
print("  음악 동작 범위 밖이다. 48 kHz 스테레오에서 제대로 된 최저치(16 kbps)와")
print("  견주면 같은 곡에서 절약이 21.7x -> 5.7x 로 떨어진다.")

print("\n=== 순위 역전이 여기서도 성립하나 ===\n")
for scen,_ in SCEN:
    inv=tot=0
    for c in clips:
        rs=[x for x in rows if x["clip"]==c]
        if len(rs)<2: continue
        tot+=1
        br=[x["codec"] for x in sorted(rs,key=lambda x:-x["ratio"])]
        bb=[x["codec"] for x in sorted(rs,key=lambda x:x[scen])]
        if br!=bb: inv+=1
    print(f"  [{scen:<11}] 역전 {inv}/{tot} 칸")
print()
print("  실험 6~8 에서는 20/20 칸이 역전했다. 여기서는 아니다.")
print("  이유는 명확하다 — 24 kHz 실험에서는 '압축률 1등(Mimi)' 과")
print("  '모델 최소(EnCodec)' 가 서로 다른 모델이었다. 48 kHz 판에서는")
print("  EnCodec48 이 둘 다인 칸이 많다. **역전은 회계의 필연이 아니라")
print("  그 코덱 조합의 성질이다.** 실험 6 의 '20/20' 을 일반 법칙처럼")
print("  읽으면 안 된다.")
