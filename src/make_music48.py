#!/usr/bin/env python3
"""실험 9 용 음악 클립 — 48 kHz 스테레오, 원래 형식 그대로.

실험 8 은 24 kHz 모노였다. 실제 음악 배포는 48 kHz 스테레오이고, 음악
코덱(EnCodec 48kHz)도 거기에 맞춰 학습돼 있다. 24 k 모노 결과를 음악 코덱
비교라고 부를 수 없다고 적었으므로 제대로 다시 만든다.

곡은 성격이 갈리게 골랐다. 실험 8 에서 '음악이냐'가 아니라 **고역 에너지**가
승패를 가른다는 게 드러났으므로, 대역분포가 서로 다른 것들로 고른다.
"""
import subprocess
from pathlib import Path
import numpy as np, soundfile as sf

D = "/tmp/claude-1000/-home-kim/9f21bc40-3f4d-43a2-be33-bc56cd93e4c3/scratchpad/music"
SRC = [
    # 이름          원본                                   시작   설명
    ("piano",   f"{D}/jamendo-352723.mp3",                  40, "피아노 왈츠 — 고역 거의 없음"),
    ("vocal",   f"{D}/jamendo-264026.mp3",                  40, "아카펠라 하모니 — 가장 밝다"),
    ("pop",     f"{D}/jamendo-274107.mp3",                  40, "보컬 팝 — 균형"),
    ("sax",     f"{D}/jamendo-291896.mp3",                  40, "색소폰/트롬본 — 저역+고역"),
    ("latin",   f"{D}/jamendo-282973.mp3",                  40, "라틴 — 중역 지배"),
    ("game",    "/home/kim/kaizo_webgame/assets/strikeback/music/sierra.mp3",     30, "게임 BGM — 넓은 고역"),
    ("gameld",  "/home/kim/kaizo_webgame/assets/strikeback/music/Connection.mp3", 30, "게임 BGM — 라우드 마스터링"),
]
out = Path("audio48"); out.mkdir(exist_ok=True)
print(f"{'이름':<9}{'RMS L/R':>16}{'스테레오폭':>10}  대역 (0.2/0.8/3/8/16k %)   설명")
print("-" * 96)
for name, src, ss, desc in SRC:
    if not Path(src).exists():
        print(f"  {name}: 원본 없음 {src}"); continue
    dst = out / f"{name}.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(ss),"-i",src,"-t","10",
                    "-ac","2","-ar","48000","-c:a","pcm_s16le",str(dst)], check=True)
    x, sr = sf.read(dst, dtype="float32")          # (n, 2)
    # 무음/모노위장 검사 — 실험 8 의 무음 사고 재발 방지
    rms = np.sqrt((x**2).mean(0))
    width = float(np.abs(x[:,0]-x[:,1]).mean() / (np.abs(x).mean()+1e-12))
    m = x.mean(1)
    S = np.abs(np.fft.rfft(m*np.hanning(len(m)))); fq = np.fft.rfftfreq(len(m), 1/sr)
    bands = [(20,200),(200,800),(800,3000),(3000,8000),(8000,16000)]
    e = np.array([S[(fq>=a)&(fq<b)].sum() for a,b in bands]); e = e/e.sum()
    warn = ""
    if rms.min() < 1e-4: warn = "  ** 한쪽 채널이 무음 **"
    elif width < 0.01:   warn = "  ** 사실상 모노 **"
    print(f"{name:<9}{rms[0]:>7.4f}/{rms[1]:<8.4f}{width:>9.3f}   "
          + " ".join(f"{p*100:>4.0f}" for p in e) + f"   {desc}{warn}")
