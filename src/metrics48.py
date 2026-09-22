#!/usr/bin/env python3
"""48 kHz 스테레오용 지표. music48.py 와 perceptual.py 가 함께 쓴다.

music48.py 에서 떼어냈다 — 거기서 import 하면 모듈 최상위의 측정 루프가
통째로 다시 돈다. (latent_codec.py 에서 같은 실수를 한 적이 있다.)
"""
import numpy as np

def _stft(x, n):
    hop = n//4; w = np.hanning(n).astype(np.float32)
    m = 1 + (len(x)-n)//hop
    if m < 1: return np.zeros((n//2+1,1), np.float32)
    return np.abs(np.stack([np.fft.rfft(x[i*hop:i*hop+n]*w) for i in range(m)], 1))

def _melfb(sr, n_fft, n_mels=80, fmin=20):
    fmax = sr/2
    h2m = lambda f: 2595*np.log10(1+f/700); m2h = lambda m: 700*(10**(m/2595)-1)
    pts = m2h(np.linspace(h2m(fmin), h2m(fmax), n_mels+2))
    b = np.floor((n_fft+1)*pts/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1), np.float32)
    for i in range(n_mels):
        l,c,r = b[i],b[i+1],b[i+2]
        if c>l: fb[i,l:c] = (np.arange(l,c)-l)/(c-l)
        if r>c: fb[i,c:r] = (r-np.arange(c,r))/(r-c)
    return fb

def mel_st(a, b, sr, n_fft=2048, n_mels=80):
    """채널별 mel L1 의 평균. 48 kHz 라 창을 키웠다."""
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    fb = _melfb(sr, n_fft, n_mels); vals = []
    for ch in range(a.shape[1]):
        A = np.log(fb @ _stft(a[:,ch], n_fft) + 1e-5)
        B = np.log(fb @ _stft(b[:,ch], n_fft) + 1e-5)
        m = min(A.shape[1], B.shape[1]); vals.append(np.abs(A[:,:m]-B[:,:m]).mean())
    return float(np.mean(vals))

def msstft_st(a, b, scales=(512,1024,2048,4096)):
    """Multi-scale STFT 거리 = 스펙트럼 수렴 + 로그 크기 L1.

    처음엔 선형 크기 L1 만 썼는데, 그러면 **고역을 0 으로 만드는 게 벌을 안 받는다**
    — 고역은 원래 크기가 작아서 지워도 절대 오차가 거의 안 는다. 그 결과
    3.5 kHz 저역통과 앵커가 7곡 중 6곡에서 '최고 품질' 로 나왔다.
    MUSHRA 앵커를 넣어 두지 않았으면 못 잡았을 것이다.

    로그 항이 그걸 잡는다. log(0+eps) 가 크게 음수라 지우는 쪽이 크게 벌받는다.
    Yamamoto et al. 2020 의 표준 형태다.
    """
    n = min(len(a), len(b)); a, b = a[:n], b[:n]; vals=[]
    for ch in range(a.shape[1]):
        for s in scales:
            A,B = _stft(a[:,ch],s), _stft(b[:,ch],s)
            m = min(A.shape[1],B.shape[1]); A,B = A[:,:m], B[:,:m]
            sc  = np.linalg.norm(A-B) / (np.linalg.norm(A)+1e-9)      # 스펙트럼 수렴
            mag = np.abs(np.log(A+1e-7) - np.log(B+1e-7)).mean()      # 로그 크기 L1
            vals.append(sc + mag)
    return float(np.mean(vals))

def compare(a, b, sr): return {"mel": mel_st(a,b,sr), "msstft": msstft_st(a,b)}

