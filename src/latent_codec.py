#!/usr/bin/env python3
"""VAE latent 을 '악보'로 쓰는 코덱. MIDI 의 영상판.

    프레임 512x512x3  --VAE-->  latent 4x64x64  --양자화--> 보낸다
                      <--VAE--                              복원

VAE 가 사운드폰트다. 양쪽이 이미 갖고 있다고 가정하고, 추가 전송량만 센다.
그 가정이 이 실험의 전부이므로 모델 크기를 항상 같이 적는다.

latent 을 그냥 저장하면 오히려 커진다(4*64*64*2바이트 = 32KB/프레임).
그래서 세 단계를 거친다:
  1. 채널별 스케일 잡고 정수로 양자화 (비트수는 인자)
  2. 시간축 차분 - 연속 프레임의 latent 은 매우 비슷하다
  3. 무손실 엔트로피 부호화 (여기서는 zlib. 실제로는 산술부호화가 낫다)

3번을 실제 코덱 수준으로 만들지 않았으므로 여기서 나오는 비트레이트는
**낙관적이지 않다** - 제대로 하면 더 줄어든다. 즉 하한이 아니라 상한이다.
"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import metrics as M

def load_vae(name, dev):
    from diffusers import AutoencoderKL
    v = AutoencoderKL.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    n = sum(p.numel() for p in v.parameters())
    return v, n

@torch.no_grad()
def encode(vae, frames, dev, batch=8):
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i+batch]).permute(0,3,1,2).float().to(dev)/127.5 - 1
        out.append(vae.encode(x.half()).latent_dist.mean.float().cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def decode(vae, lat, dev, batch=8):
    out = []
    for i in range(0, len(lat), batch):
        z = torch.from_numpy(lat[i:i+batch]).to(dev).half()
        y = vae.decode(z).sample.float().cpu().numpy()
        out.append(((y.transpose(0,2,3,1)+1)*127.5).clip(0,255).astype(np.uint8))
    return np.concatenate(out)

def pack(lat, bits):
    """채널별 대칭 양자화 + 시간축 차분 + zlib. (바이트수, 복원된 latent)"""
    qmax = (1 << (bits-1)) - 1
    s = np.abs(lat).max(axis=(0,2,3), keepdims=True) / qmax
    s[s == 0] = 1e-8
    q = np.clip(np.round(lat/s), -qmax, qmax).astype(np.int16)
    d = q.copy(); d[1:] = q[1:] - q[:-1]          # 시간축 차분
    blob = zlib.compress(d.astype(np.int16).tobytes(), 9)
    nbytes = len(blob) + s.size*4                  # 스케일도 보내야 한다
    return nbytes, (q.astype(np.float32) * s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--bits", type=int, nargs="+", default=[8,6,5,4,3])
    a = ap.parse_args()
    dev = "cuda"

    vae, nparam = load_vae(a.vae, dev)
    print(f"VAE: {a.vae}   {nparam/1e6:.1f}M 파라미터  (fp16 기준 {nparam*2/1e6:.0f} MB)")
    print("이 크기는 전송량에 안 들어간다 — 양쪽이 이미 갖고 있다는 가정이다\n")

    rows = []
    for clip in sorted(Path("clips").glob("*.y4m")):
        ref = M.read_y4m_frames(clip)
        dur = len(ref)/24.0
        lat = encode(vae, ref, dev)
        print(f"=== {clip.stem}  {len(ref)}프레임  latent {lat.shape[1:]} ===")
        print(f"{'bits':>5} {'KB':>8} {'kbps':>9} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>7}")
        for b in a.bits:
            nb, deq = pack(lat, b)
            rec = decode(vae, deq, dev)
            m = M.compare(ref, rec)
            kbps = nb*8/1000/dur
            rows.append({"clip":clip.stem,"method":"vae-latent","bits":b,
                         "bytes":nb,"real_kbps":kbps,"vae_params":nparam, **m})
            print(f"{b:>5} {nb/1024:>8.1f} {kbps:>9.1f} {m['psnr']:>7.2f} {m['ssim']:>6.3f} {m['lpips']:>7.4f}")
        print()
    json.dump(rows, open("out/latent.json","w"), indent=1)
    print(f"{len(rows)}개 측정 -> out/latent.json")


if __name__ == "__main__":
    main()
