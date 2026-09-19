#!/usr/bin/env python3
"""품질 지표. PSNR/SSIM 은 참고용이고, 실제 판단은 LPIPS 로 한다.

왜 PSNR 을 믿지 않는가: 생성 코덱은 픽셀을 맞추지 않고 '그럴듯한 것'을 만든다.
PSNR 은 그걸 심하게 벌주고, 반대로 흐릿하게 뭉갠 영상에는 관대하다.
둘 다 낼 것 - 그 둘이 갈리는 지점 자체가 정보다.
"""
import numpy as np, torch, cv2

_lpips = None
def _get_lpips(dev):
    global _lpips
    if _lpips is None:
        import lpips as L
        _lpips = L.LPIPS(net='alex').to(dev).eval()
    return _lpips

def read_y4m_frames(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if max_frames and len(fr) >= max_frames: break
    cap.release()
    return np.stack(fr) if fr else np.zeros((0,))

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 99.0 if mse == 0 else 10*np.log10(255.0**2/mse)

def ssim(a, b):
    from scipy.ndimage import uniform_filter
    A = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    B = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    mA, mB = uniform_filter(A, 7), uniform_filter(B, 7)
    sA = uniform_filter(A*A, 7) - mA*mA
    sB = uniform_filter(B*B, 7) - mB*mB
    sAB = uniform_filter(A*B, 7) - mA*mB
    return float(np.mean(((2*mA*mB+C1)*(2*sAB+C2))/((mA*mA+mB*mB+C1)*(sA+sB+C2))))

def lpips_dist(A, B, dev='cuda', batch=8):
    """A,B: (N,H,W,3) uint8 -> 평균 LPIPS (낮을수록 지각적으로 가깝다)"""
    m = _get_lpips(dev); out = []
    for i in range(0, len(A), batch):
        a = torch.from_numpy(A[i:i+batch]).permute(0,3,1,2).float().to(dev)/127.5 - 1
        b = torch.from_numpy(B[i:i+batch]).permute(0,3,1,2).float().to(dev)/127.5 - 1
        with torch.no_grad(): out.append(m(a, b).flatten().cpu().numpy())
    return float(np.concatenate(out).mean())

def compare(ref, tst, dev='cuda'):
    n = min(len(ref), len(tst))
    ref, tst = ref[:n], tst[:n]
    return {
        "frames": n,
        "psnr": float(np.mean([psnr(ref[i], tst[i]) for i in range(n)])),
        "ssim": float(np.mean([ssim(ref[i], tst[i]) for i in range(n)])),
        "lpips": lpips_dist(ref, tst, dev),
    }
