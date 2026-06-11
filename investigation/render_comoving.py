"""
Render a co-moving movie from PRECOMPUTED fields (investigation/fields/), using an
INCREMENTAL composed displacement field (one resample per frame, O(T), less blur than
chained resampling). Outputs a pseudo z-MIP movie (raw | co-moving) + montage + NCC.
Reuses the t14->t18 fields; no re-registration.
"""
import glob
import numpy as np
import tifffile
import SimpleITK as sitk
import imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS, OUT, XY_DS = "investigation/fields", "investigation/out", 4
SPACING = (0.347*XY_DS, 0.347*XY_DS, 1.5)
LAB = ["t0014", "t0015", "t0016", "t0017", "t0018"]
files = {l: [f for f in glob.glob("t00*_Channel 1.tif") if l in f][0] for l in LAB}

def load(f):
    with tifffile.TiffFile(f) as t:
        pages = t.pages; pl0 = pages[0].asarray()
        h = (pl0.shape[0]//XY_DS)*XY_DS; w = (pl0.shape[1]//XY_DS)*XY_DS
        out = np.empty((len(pages), h//XY_DS, w//XY_DS), np.float32)
        for z in range(len(pages)):
            pl = pages[z].asarray()[:h, :w].astype(np.float32)
            out[z] = pl.reshape(h//XY_DS, XY_DS, w//XY_DS, XY_DS).mean((1, 3))
    lo, hi = np.percentile(out, [1, 99.7]); return np.clip((out-lo)/max(hi-lo,1),0,1).astype(np.float32)

print("loading volumes...")
raw = {l: load(files[l]) for l in LAB}
def sitkimg(v): im = sitk.GetImageFromArray(np.ascontiguousarray(v.astype(np.float32))); im.SetSpacing(SPACING); return im
refgrid = sitkimg(raw[LAB[0]])

pairs = [(LAB[i], LAB[i+1]) for i in range(len(LAB)-1)]
rigid, demons = {}, {}
for a, b in pairs:
    rigid[(a,b)] = sitk.ReadTransform(f"{FIELDS}/rigid_{a}_{b}.tfm")
    df = sitk.ReadImage(f"{FIELDS}/demons_{a}_{b}.nrrd")
    demons[(a,b)] = sitk.DisplacementFieldTransform(sitk.Cast(df, sitk.sitkVectorFloat64))

def pair_composite(a, b):                 # applies demons then rigid (matches m_def=moving(T_r(T_d(x))))
    c = sitk.CompositeTransform(3); c.AddTransform(rigid[(a,b)]); c.AddTransform(demons[(a,b)]); return c
def to_field(tf):
    f = sitk.TransformToDisplacementFieldFilter(); f.SetReferenceImage(refgrid)
    return sitk.DisplacementFieldTransform(f.Execute(tf))

# incremental composed field phi_k (anchor->k): phi_k = P_{k-1} o phi_{k-1}
print("composing co-moving stack (incremental composed field)...")
comoving = [raw[LAB[0]]]; M = None
for k in range(1, len(LAB)):
    P = pair_composite(LAB[k-1], LAB[k])
    if M is None:
        phi = P
    else:
        c = sitk.CompositeTransform(3); c.AddTransform(P); c.AddTransform(M); phi = c  # M applied first
    M = to_field(phi)
    warped = sitk.Resample(sitkimg(raw[LAB[k]]), refgrid, M, sitk.sitkLinear, 0.0)
    comoving.append(sitk.GetArrayFromImage(warped))
    print(f"  {LAB[k]} -> anchor")

def ncc(a, b):
    a=a.ravel().astype(np.float64); b=b.ravel().astype(np.float64); a-=a.mean(); b-=b.mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
print("\n=== NCC vs anchor t14 (composed-field render) ===")
print(f"{'frame':>7}{'raw':>8}{'comoving':>10}")
for k,l in enumerate(LAB):
    print(f"{l:>7}{ncc(raw[l],raw[LAB[0]]):>8.3f}{ncc(comoving[k],comoving[0]):>10.3f}")

# pseudo z-MIP movie (raw | comoving) + montage
def pmip(v):
    m = v.max(0); lo,hi = np.percentile(m,[1,99]); return (np.clip((m-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8)
rm = [pmip(raw[l]) for l in LAB]; cm = [pmip(c) for c in comoving]
sep = np.full((rm[0].shape[0], 6), 255, np.uint8)
frames = [np.hstack([r, sep, c]) for r, c in zip(rm, cm)]
imageio.mimsave(f"{OUT}/comoving_pmip.gif", frames, duration=0.7, loop=0)
fig, ax = plt.subplots(2, len(LAB), figsize=(3*len(LAB), 6))
for k,l in enumerate(LAB):
    ax[0,k].imshow(rm[k], cmap="gray"); ax[0,k].set_title(f"raw {l}"); ax[0,k].axis("off")
    ax[1,k].imshow(cm[k], cmap="gray"); ax[1,k].set_title(f"comoving {l}"); ax[1,k].axis("off")
plt.suptitle("pseudo z-MIP: raw (top) vs co-moving stabilized to t14 (bottom)")
plt.tight_layout(); plt.savefig(f"{OUT}/comoving_pmip_montage.png", dpi=110)
print(f"\nwrote {OUT}/comoving_pmip.gif (raw|comoving) and comoving_pmip_montage.png")
