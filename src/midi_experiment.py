#!/usr/bin/env python3
"""이 저장소의 중심 비유를 직접 잰다.

MIDI 는 슬룻이 주장한 구조가 실제로 동작하는 유일하게 확립된 사례다.
작은 '키'(악보)와 거대한 '디코더'(사운드폰트). 그 회계를 숫자로 낸다.

  1. MIDI 를 사운드폰트로 렌더링 -> 이것을 원본으로 삼는다
  2. 그 원본을 Opus / EnCodec 으로 압축
  3. MIDI 파일 크기와 나란히 놓는다

주의: 이건 공정한 비교가 아니다. MIDI 는 '그 사운드폰트로 렌더링한
바로 그 소리' 만 재현할 수 있고, 다른 사운드폰트를 쓰면 다른 소리가 난다.
Opus/EnCodec 은 임의의 파형을 받는다. 그 비대칭이 요점이다.
"""
import json, subprocess, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import audio_metrics as A

SF   = "/usr/share/sounds/sf2/TimGM6mb.sf2"
SF_BIG = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
MIDIS = ["/home/kim/microduck-runtime-reference/sounds/scores/duck_strut.mid",
         "/home/kim/microduck-runtime-reference/sounds/scores/outer_wilds.mid"]
OPUS_KBPS = [6, 12, 24, 48, 96]
ENC_BW    = [1.5, 3.0, 6.0, 12.0, 24.0]

out = Path("out"); out.mkdir(exist_ok=True)
tmp = Path("out/_m"); tmp.mkdir(exist_ok=True)

from transformers import EncodecModel
dev = "cuda"
enc = EncodecModel.from_pretrained("facebook/encodec_24khz").to(dev).eval()
np_enc = sum(p.numel() for p in enc.parameters())

rows = []
for mp in MIDIS:
    mp = Path(mp)
    if not mp.exists(): continue
    midi_bytes = mp.stat().st_size

    # 렌더링 — 이것이 '원본'
    raw = tmp/f"{mp.stem}.wav"
    subprocess.run(["fluidsynth","-ni","-F",str(raw),"-r","24000","-q",SF,str(mp)],
                   check=True, capture_output=True)
    # 모노 24k 로 정규화, 최대 20초
    ref_p = tmp/f"{mp.stem}_ref.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-i",str(raw),"-t","20",
                    "-ac","1","-ar","24000","-c:a","pcm_s16le",str(ref_p)],check=True)
    ref, sr = A.read(ref_p)
    if np.sqrt((ref**2).mean()) < 1e-4:
        print(f"  {mp.stem}: 렌더링이 무음 — 건너뜀"); continue
    dur = len(ref)/sr
    midi_kbps = midi_bytes*8/1000/dur

    print(f"\n=== {mp.stem}  {dur:.1f}초 ===")
    print(f"  MIDI 악보      {midi_bytes:>7,} 바이트  = {midi_kbps:>6.2f} kbps")
    print(f"  사운드폰트     {Path(SF).stat().st_size:>7,} 바이트  ({Path(SF).name})")
    print(f"  무압축 PCM     {len(ref)*2:>7,} 바이트  = 384 kbps")
    print(f"\n  {'방식':<14} {'바이트':>9} {'kbps':>7} {'mel':>7}  MIDI 대비")
    print(f"  {'MIDI+SF':<14} {midi_bytes:>9,} {midi_kbps:>7.2f} {0.0:>7.3f}  기준(완전 일치)")

    for kb in OPUS_KBPS:
        o = tmp/f"{mp.stem}_{kb}.opus"; d = tmp/f"{mp.stem}_{kb}.wav"
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(ref_p),"-c:a","libopus",
                        "-b:a",f"{kb}k","-ar","24000","-ac","1",str(o)],check=True)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(o),"-ar",str(sr),"-ac","1",str(d)],check=True)
        t,_ = A.read(d); m = A.compare(ref,t,sr)
        sz = o.stat().st_size
        rows.append({"piece":mp.stem,"method":"opus","kbps":sz*8/1000/dur,
                     "bytes":sz,"midi_bytes":midi_bytes,**m})
        print(f"  {'opus '+str(kb)+'k':<14} {sz:>9,} {sz*8/1000/dur:>7.1f} {m['mel']:>7.3f}  {sz/midi_bytes:>6.1f}배")

    x = torch.from_numpy(ref)[None,None].to(dev)
    for bw in ENC_BW:
        with torch.no_grad():
            e = enc.encode(x, bandwidth=bw)
            y = enc.decode(e.audio_codes, e.audio_scales)[0]
        t = y[0,0].cpu().numpy(); m = A.compare(ref,t,sr)
        c = e.audio_codes
        sz = int(c.shape[-2]*c.shape[-1]*10/8)
        rows.append({"piece":mp.stem,"method":"encodec","kbps":sz*8/1000/dur,
                     "bytes":sz,"midi_bytes":midi_bytes,"enc_params":np_enc,**m})
        print(f"  {'encodec '+str(bw):<14} {sz:>9,} {sz*8/1000/dur:>7.1f} {m['mel']:>7.3f}  {sz/midi_bytes:>6.1f}배")

json.dump(rows, open(out/"midi.json","w"), indent=1)
print(f"\n{len(rows)}개 측정 -> out/midi.json")
