"""
STAGE 1 — sequential dense 3D displacement fields across the contraction (t14->t18).

For each consecutive pair (fixed=t, moving=t+1): per-volume normalize + histogram-match,
3D rigid pre-align (NCC, multi-res), diffeomorphic Demons -> store rigid transform + dense
demons field (compressed) + QC. Then chained-pullback compose all frames into the t14 anchor
frame and write a co-moving 4D stack + the raw 4D stack as ImageJ TIFFs you can scroll in
Fiji/napari.

Working resolution XY//4 (full Z). Calibration: xy=0.347um, z=1.5um -> ds spacing (1.388,1.388,1.5).
"""
import glob, json, os
import numpy as np
import tifffile
import SimpleITK as sitk

OUT = "investigation/out"
FIELDS = "investigation/fields"
os.makedirs(FIELDS, exist_ok=True)
XY_DS = 4
SPACING = (0.347*XY_DS, 0.347*XY_DS, 1.5)    # (x,y,z) microns at working resolution
LABELS = ["t0014", "t0015", "t0016", "t0017", "t0018"]
files = [f for f in sorted(glob.glob("t00*_Channel 1.tif")) if any(l in f for l in LABELS)]
print("pairs over:", [f.split('_')[0] for f in files])

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
def sitkimg(v):
    im = sitk.GetImageFromArray(np.ascontiguousarray(v)); im.SetSpacing(SPACING); return im
def ncc(a, b, m):
    a=a.astype(np.float64).ravel(); b=b.astype(np.float64).ravel(); mm=m.ravel()>0.5
    a,b=a[mm],b[mm]
    if a.size<50: return float("nan")
    a-=a.mean(); b-=b.mean(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

print("loading volumes (XY//%d, full Z)..." % XY_DS)
raw = {lab: norm01(load(f)) for lab, f in zip(LABELS, files)}
shape = raw[LABELS[0]].shape
print("vol shape", shape)
ones = sitkimg(np.ones(shape, np.float32))

def register_pair(fix_np, mov_np):
    fixed = sitkimg(fix_np); moving = sitkimg(mov_np)
    moving = sitk.HistogramMatching(moving, fixed, 256, 15, True)
    # rigid (NCC, multi-res); level-8 helps capture the large z jump
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
    m_rigid = sitk.Resample(moving, fixed, init, sitk.sitkLinear, 0.0)
    mask = sitk.GetArrayFromImage(sitk.Resample(ones, fixed, init, sitk.sitkNearestNeighbor, 0.0))
    # diffeomorphic demons for the dense, invertible deformation field
    dem = sitk.DiffeomorphicDemonsRegistrationFilter()
    dem.SetNumberOfIterations(60); dem.SetStandardDeviations(1.5)
    field = dem.Execute(fixed, m_rigid)
    dft = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    m_def = sitk.Resample(m_rigid, fixed, dft, sitk.sitkLinear, 0.0)
    # QC
    rig_np = sitk.GetArrayFromImage(m_rigid); def_np = sitk.GetArrayFromImage(m_def)
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field))
    mag = np.linalg.norm(sitk.GetArrayFromImage(field), axis=-1)
    qc = dict(ncc_raw=ncc(fix_np, mov_np, mask), ncc_rigid=ncc(fix_np, rig_np, mask),
              ncc_demons=ncc(fix_np, def_np, mask),
              rigid_xyz_um=[init.GetParameters()[i] for i in (3,4,5)],
              field_mag_p99_um=float(np.percentile(mag,99)), field_mag_max_um=float(mag.max()),
              jac_min=float(jac.min()), folding_frac=float((jac<0).mean()))
    return init, dft, qc

# --- sequential pairwise registration ---
rigids, demons = {}, {}
QC = {}
for i in range(len(LABELS)-1):
    a, b = LABELS[i], LABELS[i+1]
    print(f"\nregistering {a} -> {b} ...")
    rig, dft, qc = register_pair(raw[a], raw[b])
    rigids[(a,b)] = rig; demons[(a,b)] = dft; QC[f"{a}->{b}"] = qc
    print(f"  NCC raw={qc['ncc_raw']:.3f} rigid={qc['ncc_rigid']:.3f} +demons={qc['ncc_demons']:.3f} | "
          f"rigid(x,y,z)um={[round(v,1) for v in qc['rigid_xyz_um']]} | "
          f"field p99={qc['field_mag_p99_um']:.1f}um folding={qc['folding_frac']:.4f}")
    sitk.WriteImage(demons[(a,b)].GetDisplacementField(), f"{FIELDS}/demons_{a}_{b}.nrrd", True)
    sitk.WriteTransform(rig, f"{FIELDS}/rigid_{a}_{b}.tfm")

# --- chained pullback: bring every frame into the t14 anchor frame ---
print("\ncomposing co-moving stack (anchor = t14)...")
anchor = LABELS[0]
def pull_one_step(img_np, pair):  # resample img (in frame b) into frame a: apply rigid then demons
    img = sitkimg(img_np); fixed = sitkimg(raw[pair[0]])
    img = sitk.Resample(img, fixed, rigids[pair], sitk.sitkLinear, 0.0)
    img = sitk.Resample(img, fixed, demons[pair], sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(img)

comoving = [raw[anchor]]
for k in range(1, len(LABELS)):
    img = raw[LABELS[k]]
    for j in range(k-1, -1, -1):           # pull k -> k-1 -> ... -> 0(anchor)
        img = pull_one_step(img, (LABELS[j], LABELS[j+1]))
    comoving.append(img)
    print(f"  {LABELS[k]} -> anchor done")

def to_u8(stack): return (np.clip(np.stack(stack),0,1)*255).astype(np.uint8)
tifffile.imwrite(f"{OUT}/comoving_t14anchor.tif", to_u8(comoving), imagej=True,
                 metadata={"axes":"TZYX"})
tifffile.imwrite(f"{OUT}/raw_uncorrected.tif", to_u8([raw[l] for l in LABELS]), imagej=True,
                 metadata={"axes":"TZYX"})
json.dump(QC, open(f"{OUT}/compute_fields_qc.json","w"), indent=2)
print(f"\nWROTE:\n  {OUT}/comoving_t14anchor.tif  (corrected, scroll T to check stability)\n"
      f"  {OUT}/raw_uncorrected.tif     (raw, for comparison)\n"
      f"  {FIELDS}/  per-pair fields+rigid\n  {OUT}/compute_fields_qc.json")
