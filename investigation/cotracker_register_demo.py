"""
Empirical test of CoTracker3 on REAL zebrafish data (one z-plane over time):
  1) Does it track the nuclei-like features through large drift + bleaching?
  2) Extract the GRADED confidence (bypassing the predictor's boolean threshold).
  3) Can the tracks drive XY registration? Compare to phase-correlation baseline.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python investigation/cotracker_register_demo.py
"""
import sys
import json
import numpy as np
import tifffile
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUB = "/Users/helsens/.cache/torch/hub/facebookresearch_co-tracker_main"
sys.path.insert(0, HUB)
OUT = "investigation/out"

# ---- params ----
Z_PLANE = 33          # best-focus plane (from inspect_data.py)
CHAN = 0              # best-texture channel
T_START, T_LEN, T_STEP = 0, 64, 1
GRID = 30             # grid_size -> GRID**2 query points
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device={dev} plane=z{Z_PLANE} ch{CHAN} frames [{T_START}:{T_START+T_LEN*T_STEP}:{T_STEP}] grid={GRID}")

# ---- load one z-plane over time, normalize per-frame to 8-bit, RGB ----
ARR = tifffile.memmap("Defective_somite.tif")  # (T,Z,C,Y,X)
frames = list(range(T_START, T_START + T_LEN * T_STEP, T_STEP))
def norm8(img):
    a, b = np.percentile(img, [1, 99.5])
    return np.clip((img.astype(np.float32) - a) / max(b - a, 1), 0, 1).astype(np.float32)
vid_g = np.stack([norm8(ARR[t, Z_PLANE, CHAN]) for t in frames])  # (T,H,W) in [0,1]
T, H, W = vid_g.shape
video = torch.from_numpy(vid_g)[None, :, None].repeat(1, 1, 3, 1, 1) * 255.0  # (1,T,3,H,W)
video = video.to(dev)

# ---- load CoTracker3 offline + monkeypatch to capture SOFT vis/conf ----
predictor = torch.hub.load(HUB, "cotracker3_offline", source="local", trust_repo=True).to(dev)
predictor.model.eval()
raw = {}
_orig = predictor.model.forward
def _patched(*a, **k):
    out = _orig(*a, **k)
    raw["out"] = out  # (coords, vis_soft, conf_soft, train_data)
    return out
predictor.model.forward = _patched

with torch.no_grad():
    tracks, vis_bool = predictor(video, grid_size=GRID, grid_query_frame=0, backward_tracking=False)
tracks = tracks[0].cpu().numpy()            # (T,N,2) in pixel coords (x,y)
vis_bool = vis_bool[0].cpu().numpy()        # (T,N) bool
N = tracks.shape[1]

# graded confidence (soft) from the raw model output
conf_soft = vis_soft = None
try:
    out = raw["out"]
    vis_soft = out[1][0, :, :N].float().cpu().numpy()
    conf_soft = out[2][0, :, :N].float().cpu().numpy()
    print("SOFT scores captured: vis range [%.3f,%.3f] conf range [%.3f,%.3f]"
          % (vis_soft.min(), vis_soft.max(), conf_soft.min(), conf_soft.max()))
except Exception as e:
    print("could not capture soft confidence:", repr(e))

print("tracks", tracks.shape, "vis_bool frac visible per-frame: "
      f"t0={vis_bool[0].mean():.2f} mid={vis_bool[T//2].mean():.2f} last={vis_bool[-1].mean():.2f}")

# ---- registration from tracks: similarity transform frame t -> frame 0 ----
def ncc_masked(a, b, m):
    a, b = a[m], b[m]
    if a.size < 50:
        return np.nan
    a = a - a.mean(); b = b - b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return np.nan
    return float(np.mean((a / a.std()) * (b / b.std())))

ref = vid_g[0]
score = conf_soft if conf_soft is not None else vis_bool.astype(np.float32)
ncc_raw, ncc_ct, ncc_pc = [], [], []
pc_cum = np.array([0.0, 0.0])
win = cv2.createHanningWindow((W, H), cv2.CV_32F)
prev = ref.astype(np.float32)
n_pts_used = []
for ti in range(T):
    cur = vid_g[ti].astype(np.float32)
    # --- raw (no registration) ---
    full = np.ones((H, W), bool)
    ncc_raw.append(ncc_masked(cur, ref, full))
    # --- CoTracker similarity: map points@t -> points@0 ---
    good = (vis_bool[ti] & vis_bool[0])
    if conf_soft is not None:
        good &= (conf_soft[ti] > 0.5)
    src = tracks[ti, good]   # positions at frame t
    dst = tracks[0, good]    # positions at frame 0
    n_pts_used.append(int(good.sum()))
    reg = None
    if good.sum() >= 8:
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is not None:
            reg = cv2.warpAffine(cur, M, (W, H), flags=cv2.INTER_LINEAR)
    if reg is not None:
        m = reg > 0
        ncc_ct.append(ncc_masked(reg, ref, m))
    else:
        ncc_ct.append(np.nan)
    # --- phase-correlation cumulative translation baseline ---
    (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
    pc_cum += [dx, dy]
    Mt = np.array([[1, 0, -pc_cum[0]], [0, 1, -pc_cum[1]]], np.float32)
    pcreg = cv2.warpAffine(cur, Mt, (W, H), flags=cv2.INTER_LINEAR)
    mpc = pcreg > 0
    ncc_pc.append(ncc_masked(pcreg, ref, mpc))
    prev = cur

ncc_raw, ncc_ct, ncc_pc = map(np.array, (ncc_raw, ncc_ct, ncc_pc))
def mn(x): return float(np.nanmean(x))
print("\n=== mean NCC vs frame0 (higher=better aligned) ===")
print(f"  raw (no reg)        : {mn(ncc_raw):.3f}")
print(f"  CoTracker similarity: {mn(ncc_ct):.3f}  (pts used: med={int(np.median(n_pts_used))}/{N})")
print(f"  phase-corr translate: {mn(ncc_pc):.3f}")

# ---- plots ----
plt.figure(figsize=(8, 4))
plt.plot(ncc_raw, label="raw"); plt.plot(ncc_ct, label="CoTracker sim")
plt.plot(ncc_pc, label="phasecorr"); plt.xlabel("frame"); plt.ylabel("NCC vs t0")
plt.legend(); plt.title("Registration quality over time"); plt.tight_layout()
plt.savefig(f"{OUT}/ncc_vs_time.png", dpi=110); plt.close()

if conf_soft is not None:
    plt.figure(figsize=(8, 4))
    plt.plot(vis_bool.mean(1), label="frac visible (bool)")
    plt.plot(conf_soft.mean(1), label="mean confidence (soft)")
    plt.plot(vis_soft.mean(1), label="mean visibility (soft)")
    plt.xlabel("frame"); plt.ylabel("score"); plt.legend()
    plt.title("CoTracker3 visibility/confidence over time"); plt.tight_layout()
    plt.savefig(f"{OUT}/conf_vs_time.png", dpi=110); plt.close()

# ---- track overlay on frame 0, mid, last (colored by confidence) ----
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for k, ti in enumerate([0, T // 2, T - 1]):
    ax[k].imshow(vid_g[ti], cmap="gray")
    c = score[ti] if score.ndim == 2 else np.ones(N)
    sc = ax[k].scatter(tracks[ti, :, 0], tracks[ti, :, 1], c=c, s=6,
                       cmap="plasma", vmin=0, vmax=1)
    ax[k].set_title(f"frame {ti}"); ax[k].axis("off")
plt.colorbar(sc, ax=ax, fraction=0.02, label="confidence")
plt.suptitle("CoTracker3 tracks on ch0 z%d (color=confidence)" % Z_PLANE)
plt.savefig(f"{OUT}/tracks_overlay.png", dpi=110, bbox_inches="tight"); plt.close()

json.dump(dict(
    frames=[T_START, T_LEN, T_STEP], plane=Z_PLANE, chan=CHAN, N=N,
    ncc_raw=mn(ncc_raw), ncc_cotracker=mn(ncc_ct), ncc_phasecorr=mn(ncc_pc),
    frac_visible_last=float(vis_bool[-1].mean()),
    median_pts_used=int(np.median(n_pts_used)),
    soft_conf_available=conf_soft is not None,
), open(f"{OUT}/cotracker_demo_summary.json", "w"), indent=2)
print(f"\nPNGs + summary written to {OUT}/")
