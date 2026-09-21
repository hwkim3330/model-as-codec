#!/usr/bin/env python3
"""오디오 테스트 클립. 24 kHz 모노로 맞춘다 — EnCodec 24kHz 모델 기준.

세 종류를 쓴다. 코덱은 내용에 따라 성적이 크게 갈리므로 하나로는 안 된다.
  speech  사람 목소리   — 신경망 코덱이 가장 강한 영역
  music   음악          — 조화 구조가 있어 모델이 배울 것이 많다
  mixed   영상 사운드트랙 — 말+음악+효과음이 섞인 실제 콘텐츠
"""
import subprocess, sys
from pathlib import Path

#  주의: 이 머신에 음악 파일이 없어서 음악은 시험하지 못했다.
#  video2.mp4 의 해당 구간은 완전 무음(RMS 0.000000)이었고, 그걸
#  모르고 돌렸다가 msSTFT 가 수백만으로 나와서 알아챘다.
#  입력부터 확인할 것.
SRC = [
    ("speech", "/home/kim/supertonic-3-assets/audio_samples/keld_reference.wav", "00:00:00", 10),
    ("mixed",  "/home/kim/sample.mp4",                  "00:00:10", 10),
    ("ambient","/home/kim/talkinghead-ref/audio/murmur.mp3", "00:00:10", 10),
]
out = Path("audio"); out.mkdir(exist_ok=True)
for name, src, ss, dur in SRC:
    if not Path(src).exists():
        print(f"  {name}: 원본 없음"); continue
    dst = out / f"{name}.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-ss",ss,"-i",src,"-t",str(dur),
                    "-ac","1","-ar","24000","-c:a","pcm_s16le",str(dst)], check=True)
    sz = dst.stat().st_size
    print(f"  {name}: {dur}초 24kHz 모노  {sz/1024:.0f} KB  (무압축 {sz*8/dur/1000:.0f} kbps)")
