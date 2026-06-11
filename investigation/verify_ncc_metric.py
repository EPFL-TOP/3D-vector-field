"""Verify the synthesis agent's refinement: NCC metric + RegularStepGradientDescent
fixes the t10 rigid divergence seen with Mattes-MI. Rigid-only, fast."""
import numpy as np, tifffile, SimpleITK as sitk
CHAN, SPACING = 0, (1.0, 1.0, 1.5)
ARR = tifffile.memmap("Defective_somite.tif")
def vol(t): return np.asarray(ARR[t, :, CHAN], np.float32)
def norm(v):  # per-volume percentile normalize (counter bleaching) before NCC metric
    lo, hi = np.percentile(v, [1, 99.5]); return np.clip((v-lo)/max(hi-lo,1),0,1).astype(np.float32)
def to_sitk(v):
    img = sitk.GetImageFromArray(v); img.SetSpacing(SPACING); return img
def ncc(a,b,m):
    a=a.ravel().astype(np.float64); b=b.ravel().astype(np.float64); mm=m.ravel()>0.5
    a,b=a[mm],b[mm]; a-=a.mean(); b-=b.mean()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

for gap in [10, 40]:
    f_np, m_np = norm(vol(0)), norm(vol(gap))
    fixed, moving = to_sitk(f_np), to_sitk(m_np)
    ones = sitk.GetImageFromArray(np.ones_like(m_np)); ones.SetSpacing(SPACING)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()                       # NCC (same-modality)
    R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
        numberOfIterations=300, gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4,2,1]); R.SetSmoothingSigmasPerLevel([2,1,0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=True); R.Execute(fixed, moving)
    rg = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, init, sitk.sitkLinear,
                                              float(np.median(m_np)), moving.GetPixelID()))
    mask = sitk.GetArrayFromImage(sitk.Resample(ones, fixed, init, sitk.sitkNearestNeighbor,
                                                0.0, ones.GetPixelID()))
    tx = init.GetParameters()[3:6]
    print(f"t0 vs t{gap}: raw@overlap={ncc(f_np,m_np,mask):.3f}  rigid={ncc(f_np,rg,mask):.3f} "
          f" translation(x,y,z)um=({tx[0]:.1f},{tx[1]:.1f},{tx[2]:.1f})")
