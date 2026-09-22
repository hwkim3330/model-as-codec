#!/usr/bin/env python3
"""손익분기 — 모델을 공유하는 코덱이 언제부터 실제로 이득인가.

코덱 논문은 rate-distortion 곡선만 보고한다. 디코더가 300 MB 라는 사실은
표에 없다. 그런데 그 코덱을 쓰려면 양쪽이 먼저 300 MB 를 가져야 한다.

정직한 단위는 '비트레이트' 가 아니라 **총 전송량** 이다:

    총량(N) = 모델크기 + N × 편당전송량

이 함수가 기존 코덱의 N × 편당전송량 과 만나는 지점이 손익분기다.
그 지점을 시간 단위로 내면 해석이 된다 - '몇 시간 분량부터 이득인가'.
"""
import json, sys
from pathlib import Path

# ⚠ 이 스크립트는 체크포인트 전체를 모델 비용으로 넣는다. 수신 측은 디코더만
#   필요하고 학습용 EMA 버퍼도 빼야 한다 -> src/breakeven2.py (실험 6) 가 정본.
MODEL_BYTES = {          # 실제 체크포인트 크기 (fp32 safetensors)
    "encodec":  93.1e6,
    "dac":     298.7e6,
    "mimi":    384.6e6,
}
VAE_BYTES = 334.6e6

def load(p):
    return json.load(open(p)) if Path(p).exists() else []

audio = load("out/audio_sota.json")

print("=== 오디오: 같은 품질에서 손익분기 ===\n")
print(f"{'클립':<9} {'코덱':<8} {'모델':>8} {'절약':>10} {'손익분기':>12} {'':>4}")
print("-"*62)
rows=[]
for clip in ["speech","mixed","ambient"]:
    opus=[x for x in audio if x["clip"]==clip and x["method"]=="opus"]
    for meth in ["mimi","dac","encodec"]:
        ms=[x for x in audio if x["clip"]==clip and x["method"]==meth]
        if not ms: continue
        n=min(ms,key=lambda x:x["kbps"])              # 가장 싼 설정
        eq=[x for x in opus if x["mel"]<=n["mel"]]     # 같거나 더 나은 opus
        if not eq: continue
        o=min(eq,key=lambda x:x["kbps"])
        save_bps=(o["kbps"]-n["kbps"])*1000            # 초당 절약 비트
        if save_bps<=0: continue
        sec=MODEL_BYTES[meth]*8/save_bps               # 손익분기(초)
        rows.append({"clip":clip,"codec":meth,"model_MB":MODEL_BYTES[meth]/1e6,
                     "neural_kbps":n["kbps"],"opus_kbps":o["kbps"],
                     "ratio":o["kbps"]/n["kbps"],"breakeven_hours":sec/3600})
        print(f"{clip:<9} {meth:<8} {MODEL_BYTES[meth]/1e6:>7.0f}MB "
              f"{o['kbps']/n['kbps']:>9.1f}x {sec/3600:>10.1f}시간")
print()
print("=== 뒤집어 보면 ===\n")
for r in rows:
    if r["codec"]!="mimi": continue
    print(f"  {r['clip']}: Mimi 가 Opus 대비 {r['ratio']:.0f}배 절약한다고 하지만,")
    print(f"    {r['model_MB']:.0f} MB 를 먼저 보내야 하므로")
    print(f"    **{r['breakeven_hours']:.0f}시간** 분량을 넘겨야 본전이다.")
    for scen,h in [("팟캐스트 1편(1시간)",1),("음악앨범 1장(1시간)",1),
                   ("하루치 통화(2시간)",2),("오디오북 1권(10시간)",10)]:
        v = "이득" if h>r["breakeven_hours"] else "손해"
        print(f"      {scen:<22} {v}")
    print()
json.dump(rows, open("out/breakeven.json","w"), indent=1)

print("=== 영상 ===\n")
print(f"  VAE 디코더 {VAE_BYTES/1e6:.0f} MB")
print("  그런데 실험 1 에서 VAE latent 은 같은 품질에 **더 많은** 비트를 썼다.")
print("  편당 절약이 음수이므로 **손익분기가 존재하지 않는다.**")
print("  아무리 많이 보내도 본전이 안 된다.")
