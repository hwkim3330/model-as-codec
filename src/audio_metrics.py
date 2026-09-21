#!/usr/bin/env python3
"""오디오 품질 지표.

파형 차이(SI-SDR)는 코덱 비교에 거의 쓸모가 없다 — 신경망 코덱은 파형을
맞추지 않고 '같게 들리는' 것을 만들기 때문에 위상이 어긋나면 SI-SDR 이
무너진다. 그래서 스펙트럼 거리를 주로 본다.

  Mel distance        멜 스펙트로그램 L1. 사람 귀와 대체로 맞는다
  Multi-scale STFT    여러 창 크기로 본 스펙트럼 거리. 코덱 논문 표준
  SI-SDR              참고용. 신경망 코덱에는 불리하게 나온다는 것을 알고 본다
"""
import numpy as np, soundfile as sf

def read(p):
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    return x, sr

def _stft(x, n):
    hop = n // 4
    w = np.hanning(n).astype(np.float32)
    m = 1 + (len(x) - n) // hop
    if m < 1: return np.zeros((n//2+1, 1))
    F = np.empty((n//2+1, m), np.complex64)
    for i in range(m):
        F[:, i] = np.fft.rfft(x[i*hop:i*hop+n] * w)
    return np.abs(F)

def _mel_fb(sr, n_fft, n_mels=64, fmin=20, fmax=None):
    fmax = fmax or sr/2
    def hz2mel(f): return 2595*np.log10(1+f/700)
    def mel2hz(m): return 700*(10**(m/2595)-1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels+2))
    bins = np.floor((n_fft+1)*pts/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1), np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i+1], bins[i+2]
        if c > l: fb[i, l:c] = (np.arange(l, c)-l)/(c-l)
        if r > c: fb[i, c:r] = (r-np.arange(c, r))/(r-c)
    return fb

def mel_dist(a, b, sr, n_fft=1024, n_mels=64):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    fb = _mel_fb(sr, n_fft, n_mels)
    A = np.log(fb @ _stft(a, n_fft) + 1e-5)
    B = np.log(fb @ _stft(b, n_fft) + 1e-5)
    m = min(A.shape[1], B.shape[1])
    return float(np.abs(A[:, :m] - B[:, :m]).mean())

def ms_stft(a, b, scales=(256, 512, 1024, 2048)):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    tot = []
    for s in scales:
        A, B = _stft(a, s), _stft(b, s)
        m = min(A.shape[1], B.shape[1])
        A, B = A[:, :m], B[:, :m]
        sc = np.linalg.norm(A-B)/(np.linalg.norm(A)+1e-8)
        mg = np.abs(np.log(A+1e-5)-np.log(B+1e-5)).mean()
        tot.append(sc+mg)
    return float(np.mean(tot))

def si_sdr(ref, est):
    n = min(len(ref), len(est)); ref, est = ref[:n], est[:n]
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref)/(np.dot(ref, ref)+1e-12)
    t = a*ref; e = est - t
    return float(10*np.log10((np.dot(t,t)+1e-12)/(np.dot(e,e)+1e-12)))

def compare(ref, tst, sr):
    return {"mel": mel_dist(ref, tst, sr),
            "msstft": ms_stft(ref, tst),
            "sisdr": si_sdr(ref, tst)}
