"""
STAGE-1 PROTOTYPE: estimate a dense 3D displacement field between two consecutive
RAW volumes across the heat-shock contraction (t0015 -> t0016, the hardest pair).

Pipeline: per-volume normalize + histogram-match (handle signal drop) ->
3D rigid pre-align (NCC, multi-res) to absorb the ~40px+20z gross motion ->
diffeomorphic Demons for the dense, invertible (composable) deformation field.

Reports: NCC raw/rigid/deformable (valid-overlap), field magnitude, Jacobian
(folding check). Saves a mid-z visualization + the displacement field stats.
Works at XY//4, full Z (keeps the tricky z axis).

Run: PYTORCH... not needed. python investigation/prototype_field.py
"""
import glob, json
import numpy as np
import tifffile
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "investigation/out"
XY_DS = 4
SPACING = (1.39, 1.39, 1.5)   # (x,y,z) working assumption: orig xy~0.347um *4 ~=1.39; z=1.5um. CALIBRATION TBC.
FIX_F = "t0015_Channel 1.tif"
MOV_F = "t0016_Channel 1.tif"

def load(f):
    with tifffile.TiffFile(f) as t:
        pages = t.pages
        pl0 = pages[0].asarray()
        h = (pl0.shape[0]//XY_DS)*XY_DS; w = (pl0.shape[1]//XY_DS)*XY_DS
        out = np.empty((len(pages), h//XY_DS, w//XY_DS), np.float32)
        for z in range(len(pages)):
            pl = pages[z].asarray()[:h, :w].astype(np.float32)
            out[z] = pl.reshape(h//XY_DS, XY_DS, w//XY_DS, XY_DS).mean((1, 3))
    return out

def norm01(v):
    lo, hi = np.percentile(v, [1, 99.7]); return np.clip((v-lo)/max(hi-lo,1),0,1).astype(np.float32)
def to_sitk(v):
    im = sitk.GetImageFromArray(v); im.SetSpacing(SPACING); return im
def ncc(a, b, m):
    a=a.astype(np.float64).ravel(); b=b.astype(np.float64).ravel(); mm=m.ravel()>0.5
    a,b=a[mm],b[mm]; a-=a.mean(); b-=b.mean(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

print("loading volumes...")
f_np, m_np = norm01(load(FIX_F)), norm01(load(MOV_F))
print("vol shape", f_np.shape)
fixed, moving = to_sitk(f_np), to_sitk(m_np)
# histogram-match moving->fixed to counter the t16 signal drop (demons needs intensity correspondence)
moving = sitk.HistogramMatching(moving, fixed, numberOfHistogramLevels=256,
                                numberOfMatchPoints=15, thresholdAtMeanIntensity=True)
ones = sitk.GetImageFromArray(np.ones_like(m_np)); ones.SetSpacing(SPACING)

def resample(mov, ref, tf, fill=0.0, interp=sitk.sitkLinear):
    return sitk.Resample(mov, ref, tf, interp, fill, mov.GetPixelID())

# --- 3D rigid pre-align (NCC metric, multi-resolution) ---
init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                         sitk.CenteredTransformInitializerFilter.GEOMETRY)
R = sitk.ImageRegistrationMethod()
R.SetMetricAsCorrelation(); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, seed=1)
R.SetInterpolator(sitk.sitkLinear)
R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 300, gradientMagnitudeTolerance=1e-6)
R.SetOptimizerScalesFromPhysicalShift()
R.SetShrinkFactorsPerLevel([8,4,2,1]); R.SetSmoothingSigmasPerLevel([3,2,1,0])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
R.SetInitialTransform(init, inPlace=True); R.Execute(fixed, moving)
tx = init.GetParameters()
m_rigid = resample(moving, fixed, init)
mask_r = sitk.GetArrayFromImage(resample(ones, fixed, init, 0.0, sitk.sitkNearestNeighbor))
rigid_np = sitk.GetArrayFromImage(m_rigid)
print(f"rigid translation (x,y,z) phys = ({tx[3]:.1f},{tx[4]:.1f},{tx[5]:.1f}); "
      f"rot(rad)=({tx[0]:.3f},{tx[1]:.3f},{tx[2]:.3f})")

# --- diffeomorphic Demons for the dense deformation field (on rigid-aligned moving) ---
demons = sitk.DiffeomorphicDemonsRegistrationFilter()
demons.SetNumberOfIterations(80)
demons.SetStandardDeviations(1.5)          # Gaussian smoothing of the field = regularization
field = demons.Execute(fixed, m_rigid)     # displacement-field image (vector)
dft = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
m_def = resample(m_rigid, fixed, dft)
def_np = sitk.GetArrayFromImage(m_def)

# --- metrics ---
raw_ncc   = ncc(f_np, m_np, mask_r)
rigid_ncc = ncc(f_np, rigid_np, mask_r)
def_ncc   = ncc(f_np, def_np, mask_r)
fa = sitk.GetArrayFromImage(field)         # (z,y,x,3) in physical units
mag = np.linalg.norm(fa, axis=-1)
jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field))
fold = float((jac < 0).mean())
print(f"\nNCC valid-overlap:  raw={raw_ncc:.3f}  rigid={rigid_ncc:.3f}  +demons={def_ncc:.3f}")
print(f"demons field magnitude (phys units): mean={mag.mean():.2f} p99={np.percentile(mag,99):.2f} max={mag.max():.2f}")
print(f"Jacobian det: min={jac.min():.3f} median={np.median(jac):.3f} folding(frac<0)={fold:.4f}")

# --- visualization at mid-z ---
z = f_np.shape[0]//2
def n(x): lo,hi=np.percentile(x,[1,99]); return np.clip((x-lo)/max(hi-lo,1),0,1)
fig, ax = plt.subplots(2, 4, figsize=(15, 7.2))
panels = [("fixed t15", n(f_np[z])), ("moving t16 (raw)", n(m_np[z])),
          ("rigid", n(rigid_np[z])), ("rigid+demons", n(def_np[z]))]
for i,(t_,im) in enumerate(panels):
    ax[0,i].imshow(im, cmap="gray"); ax[0,i].set_title(t_); ax[0,i].axis("off")
# checkerboards vs fixed
for i,(t_,im) in enumerate(panels):
    cb=n(f_np[z]).copy(); s=48
    for yy in range(0,cb.shape[0],s):
        for xx in range(0,cb.shape[1],s):
            if ((yy//s)+(xx//s))%2==0: cb[yy:yy+s,xx:xx+s]=im[yy:yy+s,xx:xx+s]
    ax[1,i].imshow(cb,cmap="gray"); ax[1,i].set_title("checker vs fixed"); ax[1,i].axis("off")
# overwrite last bottom panel with field quiver + magnitude
ax[1,3].clear()
ax[1,3].imshow(mag[z], cmap="magma");
st=20
yy,xx=np.mgrid[0:mag.shape[1]:st,0:mag.shape[2]:st]
ax[1,3].quiver(xx,yy, fa[z,::st,::st,0], fa[z,::st,::st,1], color="cyan", scale=200, width=0.003)
ax[1,3].set_title("demons field (mag+quiver)"); ax[1,3].axis("off")
plt.suptitle("Stage-1 dense field: t0015 -> t0016 (contraction), mid-z")
plt.tight_layout(); plt.savefig(f"{OUT}/prototype_field.png", dpi=110); plt.close()

json.dump(dict(pair="t0015->t0016", vol_shape=list(f_np.shape), spacing=SPACING,
               rigid_translation=[tx[3],tx[4],tx[5]], rigid_rot=[tx[0],tx[1],tx[2]],
               ncc_raw=raw_ncc, ncc_rigid=rigid_ncc, ncc_demons=def_ncc,
               field_mag_mean=float(mag.mean()), field_mag_p99=float(np.percentile(mag,99)),
               field_mag_max=float(mag.max()), jac_min=float(jac.min()), folding_frac=fold),
          open(f"{OUT}/prototype_field_summary.json","w"), indent=2)
print(f"\nsaved prototype_field.png + summary to {OUT}/")
