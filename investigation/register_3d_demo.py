"""
Prove the 3D intensity-based registration backbone on REAL volumes.
Register volume(t0) <- volume(t_k) for several time gaps, staged:
  rigid (Euler3D) -> affine -> deformable (BSpline), Mattes-MI metric
  (MI is robust to photobleaching). Report 3D NCC before/after each stage.

Run: python investigation/register_3d_demo.py
"""
import json
import numpy as np
import tifffile
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHAN = 0
SPACING = (1.0, 1.0, 1.5)   # (x,y,z) microns; xy pixel size unknown -> assume 1.0, z=1.5 from metadata
GAPS = [5, 10, 40]          # small gaps = what a sequential t->t+1 pipeline uses; 40 = hard direct case
OUT = "investigation/out"

ARR = tifffile.memmap("Defective_somite.tif")  # (T,Z,C,Y,X)

def vol(t):
    return np.asarray(ARR[t, :, CHAN], dtype=np.float32)  # (Z,Y,X)

def to_sitk(v):
    img = sitk.GetImageFromArray(v)  # numpy (z,y,x) -> sitk (x,y,z)
    img.SetSpacing(SPACING)
    return img

def ncc(a, b, mask=None):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    if mask is not None:
        m = mask.ravel() > 0.5
        a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def make_reg(metric_bins=50, iters=200):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(metric_bins)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, seed=1234)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=iters,
                                    convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R

def resample(moving, ref, tf, fill):
    return sitk.Resample(moving, ref, tf, sitk.sitkLinear, float(fill), moving.GetPixelID())

results = {}
viz = None
for gap in GAPS:
    f_np, m_np = vol(0), vol(gap)
    fixed, moving = to_sitk(f_np), to_sitk(m_np)
    bg = float(np.median(m_np))            # background fill (NOT 0) -> no hard voids for MI
    ones = sitk.GetImageFromArray(np.ones_like(m_np)); ones.SetSpacing(SPACING)
    stages = {}

    # --- rigid ---
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = make_reg(iters=300); R.SetInitialTransform(init, inPlace=True)
    R.Execute(fixed, moving); rigid = init
    rigid_np = sitk.GetArrayFromImage(resample(moving, fixed, rigid, bg))
    mask_r = sitk.GetArrayFromImage(resample(ones, fixed, rigid, 0.0))
    stages["rigid"] = ncc(f_np, rigid_np, mask_r)
    stages["raw@rigid_overlap"] = ncc(f_np, m_np, mask_r)  # fair baseline over same voxels

    # --- affine: initialize from rigid (no intermediate zero-fill) ---
    aff_init = sitk.AffineTransform(3)
    aff_init.SetCenter(rigid.GetCenter())
    aff_init.SetTranslation(rigid.GetTranslation())
    aff_init.SetMatrix(rigid.GetMatrix())
    R = make_reg(iters=300); R.SetInitialTransform(aff_init, inPlace=True)
    R.Execute(fixed, moving); affine = aff_init
    aff_np = sitk.GetArrayFromImage(resample(moving, fixed, affine, bg))
    mask_a = sitk.GetArrayFromImage(resample(ones, fixed, affine, 0.0))
    stages["affine"] = ncc(f_np, aff_np, mask_a)

    # --- deformable BSpline on top of affine (affine as moving-initial) ---
    bsp_np = aff_np
    try:
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(32)
        R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.10, seed=7)
        R.SetInterpolator(sitk.sitkLinear)
        bsp = sitk.BSplineTransformInitializer(fixed, [6, 6, 4])
        R.SetInitialTransform(bsp, inPlace=True)
        R.SetMovingInitialTransform(affine)          # compose: bspline refines on top of affine
        R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=80)
        R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        bspline = R.Execute(fixed, moving)
        composite = sitk.CompositeTransform([affine, bspline])  # affine first, then bspline
        bsp_np = sitk.GetArrayFromImage(resample(moving, fixed, composite, bg))
        mask_b = sitk.GetArrayFromImage(resample(ones, fixed, composite, 0.0))
        stages["bspline"] = ncc(f_np, bsp_np, mask_b)
    except Exception as e:
        stages["bspline"] = None
        print(f"  [gap {gap}] bspline failed: {e!r}")

    results[gap] = stages
    print(f"gap t0 vs t{gap}: " + "  ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}=NA" for k, v in stages.items()))

    if gap == GAPS[-1]:
        z = 33
        viz = (f_np[z], m_np[z], rigid_np[z], aff_np[z], bsp_np[z], gap)

# ---- visualization of mid-z slice through the stages ----
if viz is not None:
    f, m, r, a, b, gap = viz
    def n(x):
        lo, hi = np.percentile(x, [1, 99]); return np.clip((x - lo) / max(hi - lo, 1), 0, 1)
    titles = [f"fixed t0 z33", f"moving t{gap} (raw)", "rigid", "affine", "bspline"]
    ims = [f, m, r, a, b]
    fig, ax = plt.subplots(2, 5, figsize=(16, 6.5))
    for i, (im, ti) in enumerate(zip(ims, titles)):
        ax[0, i].imshow(n(im), cmap="gray"); ax[0, i].set_title(ti); ax[0, i].axis("off")
        # checkerboard vs fixed to reveal misalignment
        cb = n(f).copy(); step = 30
        for yy in range(0, cb.shape[0], step):
            for xx in range(0, cb.shape[1], step):
                if ((yy // step) + (xx // step)) % 2 == 0:
                    cb[yy:yy+step, xx:xx+step] = n(im)[yy:yy+step, xx:xx+step]
        ax[1, i].imshow(cb, cmap="gray"); ax[1, i].set_title("checker vs fixed"); ax[1, i].axis("off")
    plt.suptitle(f"3D registration stages, mid-z slice (t0 vs t{gap})")
    plt.tight_layout(); plt.savefig(f"{OUT}/register_3d_stages.png", dpi=110); plt.close()

json.dump(results, open(f"{OUT}/register_3d_summary.json", "w"), indent=2)
print("\nsaved", f"{OUT}/register_3d_stages.png", "and summary json")
