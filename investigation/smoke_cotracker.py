"""Smoke test: load CoTracker3 offline, run on a synthetic moving-blob video,
confirm MPS works and inspect output shapes + visibility/confidence semantics."""
import sys
import numpy as np
import torch

HUB = "/Users/helsens/.cache/torch/hub/facebookresearch_co-tracker_main"
sys.path.insert(0, HUB)

torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", dev, "| torch", torch.__version__)

# synthetic textured video with a smoothly translating gaussian blob + static texture
T, H, W = 24, 256, 256
ys, xs = np.mgrid[0:H, 0:W]
rng = np.random.RandomState(0)
texture = rng.rand(H, W).astype(np.float32)  # static background texture (trackable)
vid = np.zeros((T, H, W), np.float32)
for t in range(T):
    cx, cy = 60 + 5 * t, 128 + 2 * t
    blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 18.0 ** 2)))
    vid[t] = 0.6 * texture + blob
vid = (vid - vid.min()) / (vid.max() - vid.min())
v = torch.from_numpy(vid)[None, :, None].repeat(1, 1, 3, 1, 1) * 255.0  # (1,T,3,H,W)

print("listing hub entrypoints...")
try:
    print(torch.hub.list("facebookresearch/co-tracker", trust_repo=True))
except Exception as e:
    print("hub.list failed:", repr(e))

def load(name):
    try:
        return torch.hub.load("facebookresearch/co-tracker", name, trust_repo=True)
    except Exception as e:
        print(f"github load {name} failed: {e!r}; trying local")
        return torch.hub.load(HUB, name, source="local", trust_repo=True)

model = load("cotracker3_offline").to(dev).eval()
v = v.to(dev)
with torch.no_grad():
    tracks, vis = model(v, grid_size=15)
print("tracks:", tuple(tracks.shape), tracks.dtype, "| vis:", tuple(vis.shape), vis.dtype)
vf = vis.float()
print("vis range [%.3f, %.3f] mean %.3f  unique<=4: %s"
      % (vf.min(), vf.max(), vf.mean(), torch.unique(vf)[:4].tolist()))
# displacement of tracked points start->end (should follow blob/texture)
d = (tracks[0, -1] - tracks[0, 0])
print("median per-point displacement (px):", float(d.norm(dim=-1).median()))
print("OK: cotracker3 offline runs on", dev)
