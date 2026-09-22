#!/usr/bin/env python3
"""손익분기 재계산 — 수신자는 디코더만 있으면 된다.

breakeven.py 는 체크포인트 전체를 모델 비용으로 넣었다. 그런데 단방향 전송
(방송/스트리밍/저장) 에서 수신 측은 **인코더를 쓰지 않는다**. 인코더를 빼면
손익분기가 앞당겨진다. 얼마나?

계산 과정에서 기존 MODEL_BYTES 를 검산했다. EnCodec 은 파라미터가 14.9M
뿐인데 체크포인트가 93.1 MB 다. 차이는 **RVQ 코드북 버퍼** 였다 - 파라미터가
아니라 버퍼로 등록돼 있어 named_parameters() 에 안 잡힌다. 버퍼까지 넣으면
fp32 합이 on-disk 크기와 정확히 맞는다 (93.1 / 384.6 / 298.6 MB).

그리고 그 버퍼 중 cluster_size / embed_avg 는 **학습용 EMA 상태**라 배포에는
필요 없다. EnCodec 기준 4.2M 개가 그냥 빠진다.
"""
import json
from pathlib import Path

# 주의: 이 json 은 quant_decoder.py 의 버킷 감사에서 mimi downsample(인코더)/
# upsample(디코더) 을 갈라 고친 판이다. 다시 만들 땐 그 분류를 유지할 것.
split = json.load(open("out/model_split.json"))
audio = json.load(open("out/audio_sota.json"))

# 세 가지 배포 시나리오
# fp16 은 추론에서 사실상 무손실이라 "안전한" 시나리오, int8 은 보정이 필요한
# 공격적 시나리오다. 코드북까지 int8 로 누르는 건 품질 영향이 미검증이므로
# 하한선으로만 읽어야 한다.
SCEN = [
    ("전체 fp32",   lambda s: s["total"] * 4),
    ("디코더 fp32", lambda s: s["recv"]  * 4),
    ("디코더 fp16", lambda s: s["recv"]  * 2),
    ("디코더 int8", lambda s: s["recv"]  * 1),
]

def pairs():
    """클립x코덱마다 (가장 싼 신경망 설정, 같은 품질의 가장 싼 opus) 를 찾는다."""
    for clip in ["speech", "mixed", "ambient"]:
        opus = [x for x in audio if x["clip"] == clip and x["method"] == "opus"]
        for meth in ["mimi", "dac", "encodec"]:
            ms = [x for x in audio if x["clip"] == clip and x["method"] == meth]
            if not ms: continue
            n  = min(ms, key=lambda x: x["kbps"])
            eq = [x for x in opus if x["mel"] <= n["mel"]]
            if not eq: continue
            o = min(eq, key=lambda x: x["kbps"])
            save = (o["kbps"] - n["kbps"]) * 1000
            if save <= 0: continue
            yield clip, meth, n, o, save

print("=== 모델 구성 ===\n")
print(f"{'코덱':<9} {'전체':>9} {'인코더':>9} {'디코더+VQ':>11} {'수신 비중':>9}")
print("-" * 52)
for k, s in split.items():
    print(f"{k:<9} {s['total']/1e6:>8.1f}M {s['encoder']/1e6:>8.1f}M "
          f"{s['recv']/1e6:>10.1f}M {s['recv']/s['total']*100:>8.0f}%"
          + (f"   (학습전용 {s['train_only']/1e6:.1f}M 제외)" if s.get('train_only') else ""))

print("\n\n=== 손익분기 (시간 분량) ===\n")
hdr = f"{'클립':<9} {'코덱':<9} {'절약':>7} " + "".join(f"{n:>14}" for n, _ in SCEN)
print(hdr); print("-" * len(hdr))

rows = []
for clip, meth, n, o, save in pairs():
    s = split[meth]
    hs = [f(s) * 8 / save / 3600 for _, f in SCEN]
    rows.append({"clip": clip, "codec": meth, "ratio": o["kbps"] / n["kbps"],
                 "neural_kbps": n["kbps"], "opus_kbps": o["kbps"],
                 **{nm: h for (nm, _), h in zip(SCEN, hs)}})
    print(f"{clip:<9} {meth:<9} {o['kbps']/n['kbps']:>6.1f}x "
          + "".join(f"{h:>12.1f}h" for h in hs))

json.dump(rows, open("out/breakeven2.json", "w"), indent=1)

print("\n\n=== 순위가 뒤집히는가? ===\n")
for scen, _ in SCEN:
    for clip in ["speech", "mixed", "ambient"]:
        rs = [r for r in rows if r["clip"] == clip]
        if len(rs) < 2: continue
        by_ratio = [r["codec"] for r in sorted(rs, key=lambda r: -r["ratio"])]
        by_be    = [r["codec"] for r in sorted(rs, key=lambda r: r[scen])]
        flag = "뒤집힘" if by_ratio != by_be else "일치  "
        print(f"  [{scen:<11}] {clip:<8} 압축률순 {' > '.join(by_ratio):<24}"
              f" 손익분기순 {' > '.join(by_be):<24} {flag}")
    print()

print("=== 결론 ===\n")
enc = [r for r in rows if r["codec"] == "encodec"]
if enc:
    r = min(enc, key=lambda r: r["디코더 fp16"])
    print(f"  가장 유리한 조건 (EnCodec, {r['clip']}, 디코더 fp16 = "
          f"{split['encodec']['recv']*2/1e6:.0f} MB):")
    print(f"    손익분기 {r['디코더 fp16']*60:.0f}분.")
    r2 = max(rows, key=lambda r: r["전체 fp32"])
    print(f"  가장 불리한 조건 ({r2['codec']}, {r2['clip']}, 전체 fp32):")
    print(f"    손익분기 {r2['전체 fp32']:.0f}시간.")
    print()
    lo = min(r[n] for r in rows for n, _ in SCEN)
    hi = max(r[n] for r in rows for n, _ in SCEN)
    print(f"  같은 실험인데 손익분기가 {lo:.1f}시간 ~ {hi:.1f}시간, {hi/lo:.0f}배 벌어진다.")
    print("  '신경망 코덱이 실용적이냐'는 질문의 답은 배포 조건에 전적으로 달려 있고,")
    print("  압축률만 보고하는 표로는 답할 수 없다.")
    print(f"  다만 **순위 역전은 시나리오 {len(SCEN)}개 x 클립 3개 = {len(SCEN)*3}칸 전부에서 살아남는다.**")
print()
