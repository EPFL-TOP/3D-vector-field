"""Quantify + visualize the co-moving result: NCC(frame_k vs anchor) for raw vs
co-moving, and a mid-z montage. Run after compute_fields.py."""
import numpy as np, tifffile, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "investigation/out"
LAB = ["t0014","t0015","t0016","t0017","t0018"]

co = tifffile.imread(f"{OUT}/comoving_t14anchor.tif").astype(np.float32)  # (T,Z,Y,X)
rw = tifffile.imread(f"{OUT}/raw_uncorrected.tif").astype(np.float32)

def ncc(a,b):
    a=a.ravel().astype(np.float64); b=b.ravel().astype(np.float64)
    a-=a.mean(); b-=b.mean(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

print("=== NCC vs anchor (t14): how 'fixed' does each frame look? ===")
print(f"{'frame':>7}{'raw':>8}{'comoving':>10}")
res={}
for k,lab in enumerate(LAB):
    r=ncc(rw[k],rw[0]); c=ncc(co[k],co[0])
    res[lab]=dict(raw=r,comoving=c); print(f"{lab:>7}{r:>8.3f}{c:>10.3f}")
json.dump(res, open(f"{OUT}/comoving_ncc.json","w"), indent=2)

z = co.shape[1]//2
fig,ax=plt.subplots(2,len(LAB),figsize=(3*len(LAB),6))
for k,lab in enumerate(LAB):
    ax[0,k].imshow(rw[k,z],cmap="gray"); ax[0,k].set_title(f"raw {lab}"); ax[0,k].axis("off")
    ax[1,k].imshow(co[k,z],cmap="gray"); ax[1,k].set_title(f"comoving {lab}"); ax[1,k].axis("off")
ax[0,0].set_ylabel("RAW"); ax[1,0].set_ylabel("CO-MOVING")
plt.suptitle("mid-z: raw (top) vs co-moving stabilized to t14 (bottom)")
plt.tight_layout(); plt.savefig(f"{OUT}/comoving_montage.png",dpi=110)
print(f"\nsaved {OUT}/comoving_montage.png and comoving_ncc.json")
