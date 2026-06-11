"""
Characterize the RAW unregistered timepoints (channel 1, nuclear marker) around the
heat-shock contraction. Works on downsampled volumes (XY//4, z every 4) to stay in RAM.
Goal: SEE the contraction + measure per-frame motion magnitude (the feasibility number)
and how fast volume-to-volume similarity drops across the event.
"""
import glob, json
import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "investigation/out"
XY_DS, Z_DS = 4, 4                      # downsample factors for the overview
files = sorted(glob.glob("t00*_Channel 1.tif"))
labels = [f.split("_")[0] for f in files]   # t0001, t0014...t0018
print("files:", labels)

def load_ds(f):
    with tifffile.TiffFile(f) as t:      # compressed -> read needed pages, decompress
        pages = t.pages
        idx = list(range(0, len(pages), Z_DS))
        pl0 = pages[0].asarray()
        h = (pl0.shape[0]//XY_DS)*XY_DS; w = (pl0.shape[1]//XY_DS)*XY_DS
        out = np.empty((len(idx), h//XY_DS, w//XY_DS), np.float32)
        for i, z in enumerate(idx):
            pl = pages[z].asarray()[:h, :w].astype(np.float32)
            out[i] = pl.reshape(h//XY_DS, XY_DS, w//XY_DS, XY_DS).mean((1, 3))
    return out

vols = [load_ds(f) for f in files]
print("downsampled vol shape:", vols[0].shape)

def norm(v):
    lo, hi = np.percentile(v, [1, 99.5]); return np.clip((v-lo)/max(hi-lo,1),0,1)
def ncc(a, b):
    a=a.ravel().astype(np.float64); b=b.ravel().astype(np.float64)
    a-=a.mean(); b-=b.mean(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

# --- intensity / bleaching ---
print("\n=== intensity per timepoint ===")
for lab, v in zip(labels, vols):
    print(f"  {lab}: mean={v.mean():8.1f}  p99={np.percentile(v,99):8.1f}")

# --- consecutive similarity + motion (xy from MIP phase-corr, z from z-profile) ---
print("\n=== consecutive transitions (downsampled voxels; xN for full-res XY) ===")
trans = {}
for i in range(len(vols)-1):
    a, b = vols[i], vols[i+1]
    na, nb = norm(a), norm(b)
    mip_a, mip_b = na.max(0), nb.max(0)
    (dy, dx), err, _ = phase_cross_correlation(mip_a, mip_b, upsample_factor=2)
    zp_a, zp_b = na.mean((1,2)), nb.mean((1,2))
    dz = np.argmax(np.correlate(zp_a-zp_a.mean(), zp_b-zp_b.mean(), "full")) - (len(zp_a)-1)
    c = ncc(na, nb)
    trans[f"{labels[i]}->{labels[i+1]}"] = dict(dx=float(dx), dy=float(dy), dz=int(dz),
                                                xy_shift=float(np.hypot(dx,dy)), ncc=c)
    print(f"  {labels[i]}->{labels[i+1]}: xy_shift={np.hypot(dx,dy):5.1f}ds px "
          f"(~{np.hypot(dx,dy)*XY_DS:5.1f} full px)  dz~{dz}  NCC={c:.3f}")
print(f"  baseline {labels[0]}->{labels[1]} (far apart) NCC shown above for context")

# --- montages: mid-z slice and MIP across the 6 timepoints ---
midz = vols[0].shape[0]//2
for tag, proj in [("midz", lambda v: v[midz]), ("mip", lambda v: v.max(0))]:
    fig, ax = plt.subplots(1, len(vols), figsize=(3*len(vols), 3.2))
    for k, (lab, v) in enumerate(zip(labels, vols)):
        ax[k].imshow(norm(proj(v)), cmap="gray"); ax[k].set_title(lab); ax[k].axis("off")
    plt.suptitle(f"channel1 nuclei — {tag} (XY/{XY_DS})"); plt.tight_layout()
    plt.savefig(f"{OUT}/raw_{tag}.png", dpi=110); plt.close()

# --- consecutive difference images (MIP) to localize the contraction ---
fig, ax = plt.subplots(1, len(vols)-1, figsize=(3*(len(vols)-1), 3.2))
for i in range(len(vols)-1):
    d = norm(vols[i+1].max(0)) - norm(vols[i].max(0))
    ax[i].imshow(d, cmap="bwr", vmin=-0.5, vmax=0.5)
    ax[i].set_title(f"{labels[i+1]}-{labels[i]}"); ax[i].axis("off")
plt.suptitle("MIP difference between consecutive frames (red/blue = change)")
plt.tight_layout(); plt.savefig(f"{OUT}/raw_diff.png", dpi=110); plt.close()

json.dump(dict(shape_full=[215,2304,2304], xy_ds=XY_DS, z_ds=Z_DS,
               transitions=trans), open(f"{OUT}/raw_summary.json","w"), indent=2)
print(f"\nsaved raw_midz.png raw_mip.png raw_diff.png + raw_summary.json to {OUT}/")
