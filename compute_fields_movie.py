#!/usr/bin/env python
"""
Full-movie 3D displacement-field pipeline (validated method: SimpleITK rigid Euler3D + NCC
metric + diffeomorphic Demons). CPU-only, parallel across independent consecutive pairs.

  Stage A  register every consecutive pair -> compact float16 composed field (rigid o demons),
           resumable (skips pairs already done), parallel via --workers.
  Stage B  render a co-moving movie by composing fields into the anchor frame (incremental,
           O(T)), write a pseudo z-MIP movie (raw | co-moving) + per-frame NCC-vs-anchor QC.

Designed to run on the Windows server / cluster WHERE THE DATA LIVES (no GPU needed for the
field step; GPUs are for later segmentation). Example:

  python compute_fields_movie.py --input "D:/embryo/t*_Channel 1.tif" --out D:/embryo/out \
      --xy-ds 4 --workers 16 --demons-iters 60 --sigma 1.5 --reanchor 0

Notes:
  * --xy-ds N downsamples XY by N for field ESTIMATION (estimate-low-res / apply-full-res is
    standard); Z is kept full (it carries the large through-plane motion). Voxel size becomes
    (0.347*N, 0.347*N, 1.5) um.
  * --reanchor 0 = single global anchor (frame 0). >0 = reset the reference every K frames to
    bound drift accumulation on long movies (a reference jump appears at each reset).
  * Fields are float16 .nrrd, ~ (3,Z,Y,X)*2 bytes each. They already COMPOSE rigid+demons, so
    they are also directly reusable later as the cell-tracking motion prior.
"""
import argparse, glob, json, os
import numpy as np
import tifffile
import SimpleITK as sitk

XY_PITCH_UM, Z_PITCH_UM = 0.347, 1.5

def load_vol(path, xy_ds):
    with tifffile.TiffFile(path) as t:
        pages = t.pages; pl0 = pages[0].asarray()
        h = (pl0.shape[0]//xy_ds)*xy_ds; w = (pl0.shape[1]//xy_ds)*xy_ds
        out = np.empty((len(pages), h//xy_ds, w//xy_ds), np.float32)
        for z in range(len(pages)):
            pl = pages[z].asarray()[:h, :w].astype(np.float32)
            out[z] = pl.reshape(h//xy_ds, xy_ds, w//xy_ds, xy_ds).mean((1, 3)) if xy_ds > 1 else pl
    lo, hi = np.percentile(out, [1, 99.7])
    return np.clip((out-lo)/max(hi-lo, 1), 0, 1).astype(np.float32)

def spacing(xy_ds):
    return (XY_PITCH_UM*xy_ds, XY_PITCH_UM*xy_ds, Z_PITCH_UM)

def sitkimg(v, xy_ds):
    im = sitk.GetImageFromArray(np.ascontiguousarray(v.astype(np.float32))); im.SetSpacing(spacing(xy_ds)); return im

def ncc(a, b, m=None):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    if m is not None:
        mm = m.ravel() > 0.5; a, b = a[mm], b[mm]
    if a.size < 50: return float("nan")
    a -= a.mean(); b -= b.mean(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def register_pair(fix_np, mov_np, xy_ds, iters, sigma):
    """Return (composed_field_image[float32 vector], qc dict). Field maps fixed(t)->moving(t+1)."""
    fixed, moving = sitkimg(fix_np, xy_ds), sitkimg(mov_np, xy_ds)
    moving = sitk.HistogramMatching(moving, fixed, 256, 15, True)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation(); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 300, gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([8, 4, 2, 1]); R.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=True); R.Execute(fixed, moving)
    m_rigid = sitk.Resample(moving, fixed, init, sitk.sitkLinear, 0.0)
    ones = sitkimg(np.ones_like(fix_np), xy_ds)
    mask = sitk.GetArrayFromImage(sitk.Resample(ones, fixed, init, sitk.sitkNearestNeighbor, 0.0))
    dem = sitk.DiffeomorphicDemonsRegistrationFilter()
    dem.SetNumberOfIterations(iters); dem.SetStandardDeviations(sigma)
    dfield = dem.Execute(fixed, m_rigid)
    dft = sitk.DisplacementFieldTransform(sitk.Cast(dfield, sitk.sitkVectorFloat64))
    # compose rigid o demons (applies demons then rigid) and bake to ONE field on the fixed grid
    comp = sitk.CompositeTransform(3); comp.AddTransform(init); comp.AddTransform(dft)
    tf = sitk.TransformToDisplacementFieldFilter(); tf.SetReferenceImage(fixed)
    composed = tf.Execute(comp)
    m_def = sitk.Resample(moving, fixed, sitk.DisplacementFieldTransform(sitk.Cast(composed, sitk.sitkVectorFloat64)),
                          sitk.sitkLinear, 0.0)
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(dfield))
    qc = dict(ncc_raw=ncc(fix_np, mov_np, mask),
              ncc_rigid=ncc(fix_np, sitk.GetArrayFromImage(m_rigid), mask),
              ncc_composed=ncc(fix_np, sitk.GetArrayFromImage(m_def), mask),
              rigid_xyz_um=[init.GetParameters()[i] for i in (3, 4, 5)],
              folding_frac=float((jac < 0).mean()))
    return composed, qc

def read_vol(cache_path, files, i, xy_ds):
    """Volume getter: from the zarr cache if built, else decode the TIFF."""
    if cache_path:
        import zarr
        return np.asarray(zarr.open(cache_path, mode="r")[i], dtype=np.float32)
    return load_vol(files[i], xy_ds)

def build_cache(files, cache_path, xy_ds):
    """Read each (large) TIFF ONCE, downsample+normalize, store in a zarr (T,Z,Y,X) float16.
    Resumable via a .done.json marker -> safe to re-run / interrupt."""
    import zarr
    marker = cache_path + ".done.json"
    done = set(json.load(open(marker))) if os.path.exists(marker) else set()
    v0 = load_vol(files[0], xy_ds); Z, Y, X = v0.shape
    if os.path.exists(cache_path):
        z = zarr.open(cache_path, mode="r+")
    else:
        z = zarr.open(cache_path, mode="w", shape=(len(files), Z, Y, X), chunks=(1, Z, Y, X), dtype="float16")
    for i, f in enumerate(files):
        if i in done:
            continue
        z[i] = (v0 if i == 0 else load_vol(f, xy_ds)).astype("float16")
        done.add(i); json.dump(sorted(done), open(marker, "w"))
        print(f"  cached {i:04d} {os.path.basename(f)}")

def _worker(a):
    i, files, fpath, qpath, xy_ds, iters, sigma, cache_path = a
    if os.path.exists(fpath) and os.path.exists(qpath):
        return i, json.load(open(qpath))
    field, qc = register_pair(read_vol(cache_path, files, i, xy_ds),
                              read_vol(cache_path, files, i + 1, xy_ds), xy_ds, iters, sigma)
    sitk.WriteImage(sitk.Cast(field, sitk.sitkVectorFloat32), fpath, True)  # compressed; float16 not in nrrd vec
    json.dump(qc, open(qpath, "w"), indent=2)
    return i, qc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help='glob, e.g. "D:/embryo/t*_Channel 1.tif"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--xy-ds", type=int, default=4)
    ap.add_argument("--demons-iters", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--reanchor", type=int, default=0, help="reset reference every K frames (0=global)")
    ap.add_argument("--render-ds", type=int, default=1, help="extra XY downsample for the movie only")
    ap.add_argument("--cache-dir", default=None,
                    help="if set, pre-downsample each volume ONCE into a zarr cache here (ideally a LOCAL SSD), "
                         "then register+render from the cache -> reads each huge TIFF once instead of twice "
                         "(big win when --input is on a slow/network drive)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    assert len(files) >= 2, f"need >=2 timepoints, got {len(files)} from {args.input!r}"
    os.makedirs(f"{args.out}/fields", exist_ok=True)
    print(f"{len(files)} timepoints; {len(files)-1} pairs; xy_ds={args.xy_ds}; workers={args.workers}")

    # --- optional: build a one-pass zarr volume cache (reads each huge TIFF once) ---
    cache_path = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = os.path.join(args.cache_dir, f"volcache_ds{args.xy_ds}.zarr")
        print(f"building volume cache at {cache_path} (one read per file)...")
        build_cache(files, cache_path, args.xy_ds)

    # --- Stage A: parallel pairwise registration (resumable) ---
    jobs = [(i, files, f"{args.out}/fields/field_{i:04d}.nrrd",
             f"{args.out}/fields/field_{i:04d}.json", args.xy_ds, args.demons_iters, args.sigma, cache_path)
            for i in range(len(files)-1)]
    from multiprocessing import Pool
    qcs = {}
    with Pool(args.workers) as pool:
        for i, qc in pool.imap_unordered(_worker, jobs):
            qcs[i] = qc
            print(f"  pair {i:04d} {os.path.basename(files[i])} -> +1 : "
                  f"NCC raw={qc['ncc_raw']:.3f} rigid={qc['ncc_rigid']:.3f} comp={qc['ncc_composed']:.3f} "
                  f"fold={qc['folding_frac']:.4f}")
    json.dump({str(k): qcs[k] for k in sorted(qcs)}, open(f"{args.out}/qc_per_pair.json", "w"), indent=2)

    # --- Stage B: co-moving render (incremental composed field) + pseudo z-MIP movie + QC ---
    import imageio.v2 as imageio
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    def fld(i):  # field_i maps frame i -> i+1
        return sitk.DisplacementFieldTransform(sitk.Cast(sitk.ReadImage(f"{args.out}/fields/field_{i:04d}.nrrd"),
                                                          sitk.sitkVectorFloat64))
    def mip8(v):
        m = v.max(0); lo, hi = np.percentile(m, [1, 99]); return (np.clip((m-lo)/max(hi-lo, 1e-6), 0, 1)*255).astype(np.uint8)

    anchor = args.anchor
    ref = sitkimg(read_vol(cache_path, files, anchor, args.xy_ds), args.xy_ds)
    tff = sitk.TransformToDisplacementFieldFilter(); tff.SetReferenceImage(ref)
    raw_mip, co_mip, resets = [], [], []   # resets = rendered-list indices where the reference jumped
    M = None  # composed field anchor->k
    for k in range(args.anchor, len(files)):
        v = read_vol(cache_path, files, k, args.xy_ds)
        is_reset = (k == args.anchor) or (args.reanchor and (k - anchor) % args.reanchor == 0)
        if is_reset:
            if k != args.anchor:
                resets.append(len(co_mip))
            anchor = k; ref = sitkimg(v, args.xy_ds); tff.SetReferenceImage(ref); M = None
            warped = v
        else:
            P = fld(k-1)
            if M is None:
                phi = P
            else:
                c = sitk.CompositeTransform(3); c.AddTransform(P); c.AddTransform(M); phi = c
            M = sitk.DisplacementFieldTransform(tff.Execute(phi))
            warped = sitk.GetArrayFromImage(sitk.Resample(sitkimg(v, args.xy_ds), ref, M, sitk.sitkLinear, 0.0))
        raw_mip.append(mip8(v)); co_mip.append(mip8(warped))
        print(f"  render frame {k:04d}")

    sep = np.full((raw_mip[0].shape[0], 6), 255, np.uint8)
    frames = [np.hstack([r, sep, c]) for r, c in zip(raw_mip, co_mip)]
    if args.render_ds > 1:
        frames = [f[::args.render_ds, ::args.render_ds] for f in frames]
    imageio.mimsave(f"{args.out}/comoving_pmip.gif", frames, duration=0.4, loop=0)
    try:                                   # robust MP4 via ffmpeg (needs imageio-ffmpeg in the env)
        import imageio_ffmpeg  # noqa: F401
        w = imageio.get_writer(f"{args.out}/comoving_pmip.mp4", fps=6, codec="libx264", macro_block_size=1)
        for f in frames:
            w.append_data(np.stack([f]*3, -1))
        w.close()
    except Exception as e:
        print("mp4 skipped (conda install imageio-ffmpeg):", e)

    # QC: consecutive-frame NCC = "how stable does it look when scrolling" (1.0 = no apparent motion)
    def consec(mips, skip):
        out = [np.nan]
        for j in range(1, len(mips)):
            out.append(np.nan if j in skip else ncc(mips[j], mips[j-1]))
        return out
    raw_c, co_c = consec(raw_mip, set()), consec(co_mip, set(resets))
    plt.figure(figsize=(11, 4))
    plt.plot(raw_c, label="raw frame-to-frame", alpha=0.6)
    plt.plot(co_c, label="co-moving frame-to-frame", lw=2)
    for r in resets:
        plt.axvline(r, color="gray", ls=":", lw=0.6)
    plt.ylim(0, 1.02); plt.xlabel("frame"); plt.ylabel("consecutive pseudo-MIP NCC")
    plt.legend(); plt.title("scroll stability (higher = barely moves between frames; dotted = reanchor reset)")
    plt.tight_layout(); plt.savefig(f"{args.out}/comoving_stability.png", dpi=110)
    import json as _json
    _json.dump({"raw_consec": raw_c, "comoving_consec": co_c, "resets": resets},
               open(f"{args.out}/comoving_stability.json", "w"))
    print(f"\nWROTE {args.out}/comoving_pmip.gif (+ .mp4), comoving_stability.png, qc_per_pair.json, fields/")

if __name__ == "__main__":
    main()
